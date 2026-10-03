// Product receipts belong to the compilers, not PM dependency receipts or profiles.
import { createHash } from 'node:crypto'
import { existsSync, readFileSync, readdirSync, statSync, writeFileSync } from 'node:fs'
import { join, resolve } from 'node:path'
import { parseArgs } from 'node:util'
import { isMain } from './frontend-common.mjs'

const receiptName = 'hermes-build.json'
const workspaces = { tui: 'ui-tui', web: 'web', desktop: 'apps/desktop' }
// Another product's output under a workspace is never this product's input.
const generated = new Set(['node_modules', 'dist', 'dist-webapp', 'build', 'release', '.cache', '.git', 'coverage', 'test-results', 'playwright-report'])

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

// The browser renderer compiles Desktop and its shared package in a prepared
// workspace. Release metadata (pyproject.toml, uv.lock, icons) is not its input.
const webappInputs = [
  'apps/desktop', 'apps/shared', 'package.json', 'package-lock.json', '.npmrc', 'pm/lock.json',
  'scripts/build/webapp.mjs', 'scripts/build/node-deps.mjs', 'scripts/build/frontend-common.mjs',
  'scripts/build/freshness.mjs',
]

function productInputs(product) {
  if (product === 'tui') return tuiInputs
  if (product === 'webapp') return webappInputs
  const workspace = workspaces[product]
  if (!workspace) throw new Error(`Unknown frontend product: ${product}`)
  return [
    workspace, 'apps/shared', 'package.json', 'package-lock.json', '.npmrc', 'pm/lock.json',
    'scripts/build',
    'scripts/generate-icons.mjs', 'scripts/generate_icons.py',
    // The root install-stamp.json is runtime identity rewritten after every install; desktop's
    // baked stamp is a prepared input.
    'assets', 'pyproject.toml', 'uv.lock',
  ]
}

function skipInput(product) {
  return name => {
    // Build scripts are inputs; workspace build directories are outputs.
    const parts = name.split('/')
    return (!name.startsWith('scripts/') && parts.some(part => generated.has(part)))
      || (product === 'tui' && (parts.includes('__tests__') || /\.(test|spec)(-d)?\.[cm]?[jt]sx?$/.test(name)))
      || parts.some(part => part.startsWith('.dist-') || part.startsWith('.staging-') || part === '__pycache__')
      || name.endsWith('.tsbuildinfo') || name.endsWith('.pyc')
  }
}

// Pre-order walk: a directory is reported before its sorted children.
function walkTree(root, inputs, skip, onEntry) {
  function visit(name) {
    if (osMetadata(name) || skip(name)) return
    const file = join(root, name)
    if (!existsSync(file)) return onEntry(name, 'missing')
    if (!statSync(file).isDirectory()) return onEntry(name, 'file')
    onEntry(name, 'directory')
    for (const child of readdirSync(file).sort()) visit(`${name}/${child}`)
  }
  for (const input of inputs) visit(input)
}

function treeHash(root, inputs, skip, contents = () => true) {
  const hash = createHash('sha256')
  walkTree(root, inputs, skip, (name, kind) => {
    hash.update(name.replaceAll('\\', '/')).update('\0')
    if (kind === 'file') hash.update(contents(name) ? readFileSync(join(root, name)) : 'file').update('\0')
    else hash.update(`${kind}\0`)
  })
  return hash.digest('hex')
}

export function sourceHash(source, product) {
  return treeHash(source, productInputs(product), skipInput(product))
}

/** The present entries a product's source hash covers, for mirroring a prepared workspace. */
export function sourceEntries(source, product) {
  const entries = []
  walkTree(source, productInputs(product), skipInput(product), (name, kind) => {
    if (kind !== 'missing') entries.push({ name, kind })
  })
  return entries
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
function stampContentHash(path) {
  const parsed = readStamp(path)
  return parsed ? createHash('sha256').update(JSON.stringify(stampIdentity(parsed))).digest('hex')
    : treeHash(resolve(path), ['.'], () => false)
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

export function buildInputs(source, product, prepared = {}) {
  return {
    sourceHash: sourceHash(source, product),
    prepared: Object.entries(prepared).sort().map(([name, path]) => ({
      name, path: resolve(path), hash: name === 'stamp' ? stampContentHash(resolve(path)) : treeHash(resolve(path), ['.'], () => false),
    })),
  }
}

function preparedPaths(inputs) {
  return Object.fromEntries(inputs.prepared.map(({ name, path }) => [name, path]))
}

// The clock the compiler BAKED, not the clock on disk now. desktop.mjs must pass
// the clock it read at bundleElectronMain: re-reading the stamp here would record
// a restamp that landed in between as the baked value, and productCurrent would
// then certify a product whose packaged clock disagrees with electron-main.mjs.
// Undefined (tui/web, which have no stamp) is dropped from the receipt, which is
// exactly the "predates this" shape productCurrent already tolerates.
export function recordProduct({ source, product, out, inputs, stampClock }) {
  if (JSON.stringify(buildInputs(source, product, preparedPaths(inputs))) !== JSON.stringify(inputs)) {
    throw new Error('Build inputs changed during compilation; retry the build')
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
