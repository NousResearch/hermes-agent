import { existsSync, lstatSync, mkdirSync, mkdtempSync, readFileSync, realpathSync, renameSync, rmSync, writeFileSync } from 'node:fs'
import { createRequire } from 'node:module'
import path from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'
import { parseArgs } from 'node:util'

export const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..')

// Look only in the prepared workspace, not a parent checkout's dependency tree.
// Package symlinks themselves are valid inputs (including Nix store references).
export function workspaceTool(source, workspace, name) {
  for (const root of [path.join(source, workspace), source]) {
    const manifest = path.join(root, 'node_modules', name, 'package.json')
    if (existsSync(manifest)) {
      return createRequire(path.join(root, 'package.json')).resolve(name)
    }
  }
  throw new Error(`Missing prepared ${name} dependency in ${workspace} or ${source}/node_modules`)
}

function canonical(file) {
  if (existsSync(file)) return realpathSync(file)
  return path.join(canonical(path.dirname(file)), path.basename(file))
}

function contains(parent, child) {
  const relative = path.relative(parent, child)
  return relative === '' || (!relative.startsWith(`..${path.sep}`) && relative !== '..' && !path.isAbsolute(relative))
}

const productMarker = '.hermes-product'
const productOwner = 'hermes-frontend-product-v1\n'
const developerOutputs = ['ui-tui/dist', 'hermes_cli/web_dist', 'apps/desktop/dist', 'apps/desktop/build/native-deps']

function developerOutput(source, out) {
  return source && developerOutputs.some(name => path.resolve(out) === path.join(path.resolve(source), name))
}

// A destination name alone does not confer ownership of its existing contents.
function requireOwnedOutput(out, source) {
  const stat = lstatSync(out, { throwIfNoEntry: false })
  if (!stat) return
  if (stat.isSymbolicLink() || !stat.isDirectory()) throw new Error(`Output must be a directory, not a file or symlink: ${out}`)
  // These exact npm destinations belong to their compiler. Other paths require
  // explicit ownership before an existing directory can be replaced.
  if (developerOutput(source, out)) return
  const marker = path.join(out, productMarker)
  if (!lstatSync(marker, { throwIfNoEntry: false })?.isFile() || readFileSync(marker, 'utf8') !== productOwner) {
    throw new Error(`Output already exists and is not builder-owned; choose a fresh output directory: ${out}`)
  }
}

export function productOutput(source, out, inputs) {
  if (!source || !out) throw new Error('source and output paths are required')
  const src = realpathSync(path.resolve(source))
  const dest = canonical(path.resolve(out))
  if (contains(dest, src) || inputs.some(input => {
    const protectedPath = canonical(path.join(src, input))
    return contains(dest, protectedPath) || contains(protectedPath, dest)
  })) throw new Error(`Output must not overlap build inputs: ${out}`)
  // In-tree products have explicit homes; everything else under source is an
  // input, even when this particular compiler does not read it.
  const generated = developerOutput(src, dest)
    || ['.build', 'apps/desktop/build/products'].some(name => {
      const root = path.join(src, name)
      return dest !== root && contains(root, dest)
    })
  if (contains(src, dest) && !generated) throw new Error(`Output must be outside source or in a supported generated destination: ${out}`)
  requireOwnedOutput(out, src)
  return { source: src, out: dest }
}

// Never delete the last successful product before a compiler succeeds. The
// staging and backup directories are siblings so publication stays on one FS.
export function publishDirectory(staged, out, { source } = {}) {
  // The destination may have been occupied while the compiler was running.
  requireOwnedOutput(out, source)
  writeFileSync(path.join(staged, productMarker), productOwner)
  const backup = `${staged}.previous`
  const previous = existsSync(out)
  if (previous) renameSync(out, backup)
  try {
    renameSync(staged, out)
  } catch (error) {
    if (previous) renameSync(backup, out)
    throw error
  }
  if (previous) rmSync(backup, { recursive: true, force: true })
}

export async function withProduct(out, compile, { source } = {}) {
  requireOwnedOutput(out, source)
  mkdirSync(path.dirname(out), { recursive: true })
  const scratch = mkdtempSync(path.join(path.dirname(out), `.${path.basename(out)}-build-`))
  const product = path.join(scratch, 'product')
  mkdirSync(product)
  try {
    await compile(product, scratch)
    publishDirectory(product, out, { source })
  } finally {
    rmSync(scratch, { recursive: true, force: true })
  }
}

export function isMain(url) {
  return process.argv[1] && url === pathToFileURL(path.resolve(process.argv[1])).href
}

export function frontendArgs(args, extra = {}) {
  const { values } = parseArgs({ args, options: { source: { type: 'string' }, out: { type: 'string' }, ...extra } })
  if (!values.source || !values.out) throw new Error('--source and --out are required for product builds')
  return values
}
