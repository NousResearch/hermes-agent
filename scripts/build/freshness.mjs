// Product receipts belong to the compilers, not PM dependency receipts or profiles.
import { createHash } from 'node:crypto'
import { existsSync, readFileSync, readdirSync, statSync, writeFileSync } from 'node:fs'
import { join, resolve } from 'node:path'
import { parseArgs } from 'node:util'
import { isMain } from './frontend-common.mjs'

const receiptName = 'hermes-build.json'
const workspaces = { tui: 'ui-tui', web: 'web', desktop: 'apps/desktop' }
const generated = new Set(['node_modules', 'dist', 'build', 'release', '.cache', '.git', 'coverage', 'test-results', 'playwright-report'])

function treeHash(root, inputs, skip, contents = () => true) {
  const hash = createHash('sha256')
  function visit(name) {
    if (skip(name)) return
    const file = join(root, name)
    hash.update(name.replaceAll('\\', '/')).update('\0')
    if (!existsSync(file)) { hash.update('missing\0'); return }
    if (statSync(file).isDirectory()) {
      hash.update('directory\0')
      for (const child of readdirSync(file).sort()) visit(`${name}/${child}`)
    } else {
      hash.update(contents(name) ? readFileSync(file) : 'file').update('\0')
    }
  }
  for (const input of inputs) visit(input)
  return hash.digest('hex')
}

export function sourceHash(source, product) {
  const workspace = workspaces[product]
  if (!workspace) throw new Error(`Unknown frontend product: ${product}`)
  return treeHash(source, [
    workspace, 'apps/shared', 'package.json', 'package-lock.json', '.npmrc', 'pm/lock.json',
    'scripts/build',
    ...(product === 'tui' ? [] : ['scripts/generate-icons.mjs', 'scripts/generate_icons.py',
      'assets', 'pyproject.toml', 'uv.lock', 'install-stamp.json']),
  ], name => {
    // Build scripts are inputs; workspace build directories are outputs.
    const parts = name.split('/')
    return (!name.startsWith('scripts/') && parts.some(part => generated.has(part)))
      || parts.some(part => part.startsWith('.dist-') || part.startsWith('.staging-') || part === '__pycache__')
      || name.endsWith('.tsbuildinfo') || name.endsWith('.pyc')
  })
}

function outputHash(out) {
  // Native binaries can be signed after compilation. Their ABI validation is
  // owned by native preparation; renderer/main/preload bytes must stay intact.
  return treeHash(out, readdirSync(out).sort(), name => name === receiptName || name === '.hermes-product',
    name => !name.split('/').includes('node_modules'))
}

export function buildInputs(source, product, prepared = {}) {
  return {
    sourceHash: sourceHash(source, product),
    prepared: Object.entries(prepared).sort().map(([name, path]) => ({
      name, path: resolve(path), hash: treeHash(resolve(path), ['.'], () => false),
    })),
  }
}

function preparedPaths(inputs) {
  return Object.fromEntries(inputs.prepared.map(({ name, path }) => [name, path]))
}

export function recordProduct({ source, product, out, inputs }) {
  if (JSON.stringify(buildInputs(source, product, preparedPaths(inputs))) !== JSON.stringify(inputs)) {
    throw new Error('Build inputs changed during compilation; retry the build')
  }
  writeFileSync(join(out, receiptName), JSON.stringify({
    schema: 1, product, platform: process.platform, arch: process.arch, node: process.versions.node,
    inputs, outputHash: outputHash(out),
  }) + '\n')
}

export function productCurrent({ source, product, out, prepared }) {
  try {
    const saved = JSON.parse(readFileSync(join(out, receiptName), 'utf8'))
    return saved.schema === 1 && saved.product === product
      && saved.platform === process.platform && saved.arch === process.arch
      && saved.node === process.versions.node
      && JSON.stringify(saved.inputs) === JSON.stringify(buildInputs(source, product, prepared ?? preparedPaths(saved.inputs)))
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
