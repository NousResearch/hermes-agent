// Product receipts belong to the compilers, not PM dependency receipts or profiles.
import { createHash } from 'node:crypto'
import { existsSync, readFileSync, readdirSync, statSync, writeFileSync } from 'node:fs'
import { join, resolve } from 'node:path'
import { parseArgs } from 'node:util'
import { isMain } from './frontend-common.mjs'

const receiptName = 'hermes-build.json'
const workspaces = { tui: 'ui-tui', web: 'web', desktop: 'apps/desktop' }
const generated = new Set(['node_modules', 'dist', 'build', 'release', '.cache', '.git', 'coverage', 'test-results', 'playwright-report'])

// OS file-manager metadata is never a build input, and it can land in ANY hashed
// tree (source, output or a prepared dir) the moment the checkout is opened in
// Finder or Explorer mid-build — .DS_Store, AppleDouble ._* sidecars and
// .localized on macOS, Thumbs.db and Desktop.ini on Windows. Skip it everywhere
// so it can't flip a freshness hash and abort `hermes update` (#122632, #122803).
const osMetadata = name => {
  const base = name.split('/').pop()
  return base === '.DS_Store' || base === '.localized'
    || base === 'Thumbs.db' || base === 'Desktop.ini' || base.startsWith('._')
}

// buildTui bundles these source roots (including the Ink source alias), not
// the workspaces' documentation, test runners or other product recipes.
const tuiInputs = [
  ...['ui-tui', 'ui-tui/packages/hermes-ink', 'apps/shared'].flatMap(root => [
    `${root}/src`, `${root}/package.json`, `${root}/tsconfig.json`,
  ]),
  'tsconfig.json', 'package.json', 'package-lock.json', '.npmrc', 'pm/lock.json',
  'scripts/build/tui.mjs', 'scripts/build/frontend-common.mjs', 'scripts/build/freshness.mjs',
]

// manifest, when given, records every visited entry's state (missing / directory / content
// digest) under a label so a later walk can name what changed; the aggregate hash is unchanged.
// OS metadata is skipped before the manifest write, so it is neither hashed nor named.
function treeHash(root, inputs, skip, contents = () => true, manifest = null, label = '') {
  const hash = createHash('sha256')
  function visit(name) {
    if (osMetadata(name) || skip(name)) return
    const file = join(root, name)
    hash.update(name.replaceAll('\\\\', '/')).update('\\0')
    const key = `${label}${name.replaceAll('\\\\', '/')}`
    if (!existsSync(file)) { hash.update('missing\\0'); manifest?.set(key, 'missing'); return }
    if (statSync(file).isDirectory()) {
      hash.update('directory\\0')
      manifest?.set(key, 'directory')
      for (const child of readdirSync(file).sort()) visit(`${name}/${child}`)
    } else {
      const body = contents(name) ? readFileSync(file) : 'file'
      hash.update(body).update('\\0')
      manifest?.set(key, createHash('sha256').update(body).digest('hex'))
    }
  }
  for (const input of inputs) visit(input)
  return hash.digest('hex')
}

export function sourceHash(source, product, manifest = null) {
  const workspace = workspaces[product]
  if (!workspace) throw new Error(`Unknown frontend product: ${product}`)
  return treeHash(source, product === 'tui' ? tuiInputs : [
    workspace, 'apps/shared', 'package.json', 'package-lock.json', '.npmrc', 'pm/lock.json',
    'scripts/build',
    'scripts/generate-icons.mjs', 'scripts/generate_icons.py',
    // The root install-stamp.json is runtime identity rewritten after every install; desktop's
    // baked stamp is a prepared input.
    'assets', 'pyproject.toml', 'uv.lock',
  ], name => {
    // Build scripts are inputs; workspace build directories are outputs.
    const parts = name.split('/')
    return (!name.startsWith('scripts/') && parts.some(part => generated.has(part)))
      || (product === 'tui' && (parts.includes('__tests__') || /\.(test|spec)(-d)?\.[cm]?[jt]sx?$/.test(name)))
      || parts.some(part => part.startsWith('.dist-') || part.startsWith('.staging-') || part === '__pycache__')
      || name.endsWith('.tsbuildinfo') || name.endsWith('.pyc')
  }, () => true, manifest)
}

function outputHash(out) {
  // Native binaries can be signed after compilation. Their ABI validation is
  // owned by native preparation; renderer/main/preload bytes must stay intact.
  return treeHash(out, readdirSync(out).sort(), name => name === receiptName || name === '.hermes-product',
    name => !name.split('/').includes('node_modules') && !name.startsWith('native/'))
}

// The desktop install stamp is a real prepared input — buildDesktop bakes its bytes
// into electron-main.mjs — but write-build-stamp.mjs rewrites its `builtAt` clock on
// every build (#123308). Hashing the whole file let a second build racing this one
// kill it with "inputs changed" for a difference the build machinery itself made.
// Hash the stamp's provenance identity instead: commit/payload/tag must still
// invalidate, and only the clock is ignored.
//
// The clock is NOT dropped, only excluded from this comparison. It is recorded
// separately in the receipt as `stampClock` — the value the bake actually used —
// because the output really does embed it: outputHash covers dist/ at build time,
// but electron-builder's extraResources copies the live stamp file into the bundle
// afterwards (electron-builder.config.cjs), so a restamp landing between the two
// would ship a bundle whose Resources/install-stamp.json disagrees with the baked
// main, and detectBundleSwap (bundle-swap.ts) would report a swap for a bundle
// that was never replaced. The path differs per packager (the desktop workspace
// stamp, the release bundle's own stamp, a Nix store path); the race does not.
const stampClockFields = new Set(['builtAt'])

// Key order is not part of provenance: write-build-stamp emits one fixed literal
// order today, but a cosmetic re-serialisation must not force a rebuild.
function stampIdentity(identity) {
  return Object.fromEntries(Object.entries(identity)
    .filter(([key]) => !stampClockFields.has(key))
    .sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0)))
}

// Missing and non-JSON stamps fall back to the whole-file tree hash rather than
// failing: a missing input must keep reading as one distinct hash, never as a
// throw, so a build that has not stamped yet stays "not current" instead of
// aborting. Only a parsable stamp object can have its clock removed.
// The manifest records that same identity digest, under the same `stamp:.` key a
// raw walk would use, so a clock-only rewrite is not named as a changed input.
function stampContentHash(path, manifest = null, label = '') {
  const parsed = readStamp(path)
  if (!parsed) return treeHash(resolve(path), ['.'], () => false, () => true, manifest, label)
  const digest = createHash('sha256').update(JSON.stringify(stampIdentity(parsed))).digest('hex')
  manifest?.set(`${label}.`, digest)
  return digest
}

function readStamp(path) {
  let identity = null
  try {
    identity = JSON.parse(readFileSync(path))
  } catch { return null }
  return identity && typeof identity === 'object' && !Array.isArray(identity) ? identity : null
}

/** The clock the output bakes, or null when the stamp is absent/unparsable. */
function stampClock(prepared = {}) {
  const path = prepared.stamp
  if (!path) return null
  const identity = readStamp(resolve(path))
  return identity ? (identity.builtAt ?? null) : null
}

// The per-entry manifest rides on the inputs object as a non-enumerable property: the
// receipt, JSON equality and every existing reader see exactly the shape they always did.
const manifestKey = Symbol('manifest')

export function buildInputs(source, product, prepared = {}) {
  const manifest = new Map()
  const inputs = {
    sourceHash: sourceHash(source, product, manifest),
    prepared: Object.entries(prepared).sort().map(([name, path]) => ({
      name, path: resolve(path), hash: name === 'stamp'
        ? stampContentHash(resolve(path), manifest, `${name}:`)
        : treeHash(resolve(path), ['.'], () => false, () => true, manifest, `${name}:`),
    })),
  }
  Object.defineProperty(inputs, manifestKey, { value: manifest, enumerable: false })
  return inputs
}

function preparedPaths(inputs) {
  return Object.fromEntries(inputs.prepared.map(({ name, path }) => [name, path]))
}

const changedEntriesShown = 20

/** Entries whose state differs between two manifests, ordered for a stable message. */
export function changedInputs(before, after) {
  const names = new Set([...before.keys(), ...after.keys()])
  const changed = []
  for (const name of [...names].sort()) {
    const was = before.get(name), now = after.get(name)
    if (was === now) continue
    const kind = was === undefined ? 'added'
      : now === undefined ? 'removed'
      : describe(was) === describe(now) ? 'changed'
      : `${describe(was)} -> ${describe(now)}`
    changed.push(`${name} (${kind})`)
  }
  return changed
}

function describe(state) {
  return state === 'missing' || state === 'directory' ? state : 'file'
}

/** The bounded, sorted list of differing entries for a refusal message. */
export function describeChangedInputs(changed, limit = changedEntriesShown) {
  if (!changed.length) return 'no differing entries identified'
  const shown = changed.slice(0, limit)
  const more = changed.length - shown.length
  return `${shown.join(', ')}${more > 0 ? ` and ${more} more` : ''}`
}

// The clock the compiler BAKED, not the clock on disk now. desktop.mjs must pass
// the clock it read at bundleElectronMain: re-reading the stamp here would record
// a restamp that landed in between as the baked value, and productCurrent would
// then certify a product whose packaged clock disagrees with electron-main.mjs.
// Undefined (tui/web, which have no stamp) is dropped from the receipt, which is
// exactly the "predates this" shape productCurrent already tolerates.
export function recordProduct({ source, product, out, inputs, stampClock }) {
  const current = buildInputs(source, product, preparedPaths(inputs))
  if (JSON.stringify(current) !== JSON.stringify(inputs)) {
    // The inputs differ from the pre-compilation snapshot. Name the differing
    // entries; the caller works out who wrote them.
    const detail = inputs[manifestKey]
      ? describeChangedInputs(changedInputs(inputs[manifestKey], current[manifestKey]))
      : 'no per-entry manifest was captured for these inputs'
    throw new Error(`Build inputs changed during compilation; retry the build: ${detail}`)
  }
  writeFileSync(join(out, receiptName), JSON.stringify({
    schema: 1, product, platform: process.platform, arch: process.arch, node: process.versions.node,
    inputs, stampClock, outputHash: outputHash(out),
  }) + '\n')
}

export function productCurrent({ source, product, out, prepared }) {
  try {
    const saved = JSON.parse(readFileSync(join(out, receiptName), 'utf8'))
    const paths = prepared ?? preparedPaths(saved.inputs)
    return saved.schema === 1 && saved.product === product
      && saved.platform === process.platform && saved.arch === process.arch
      && saved.node === process.versions.node
      // The pre-build gate, unlike recordProduct, must also see the clock. A restamp
      // that lands after the build (a racing write-build-stamp, or a second builder
      // that reached extraResources first) leaves dist baking one builtAt while the
      // live stamp says another, and packaging would then ship a bundle that
      // detectBundleSwap reports as replaced. A receipt with no recorded clock
      // predates this and stays comparable by hash alone. That branch is not a
      // packaging hole: the release path (scripts/bundles/desktop.py) never consults
      // this gate, it always stamps and compiles, and the source launchers that do
      // consult it reuse the dist already on disk rather than packaging a new bundle.
      && (saved.stampClock === undefined || saved.stampClock === stampClock(paths))
      && JSON.stringify(saved.inputs) === JSON.stringify(buildInputs(source, product, paths))
      && saved.outputHash === outputHash(out)
  } catch { return false }
}

if (isMain(import.meta.url)) {
  const { values } = parseArgs({ options: {
    source: { type: 'string' }, product: { type: 'string' }, out: { type: 'string' },
  } })
  if (!values.source || !values.product || !values.out) throw new Error('--source, --product and --out are required')
  console.log(JSON.stringify(productCurrent(values)))
}
