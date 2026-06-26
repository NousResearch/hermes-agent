'use strict'

/**
 * Stage JavaScript dependencies for electron-builder packaging.
 *
 * Workspace dedup hoists npm packages into the root `node_modules/`, which
 * electron-builder's default file collector (when `files:` is explicitly set
 * in package.json) cannot reach.  The result: packaged builds ship with no
 * JS dependencies and any `require()` of an npm package in the main process
 * fails at runtime ("Cannot find module '<name>'").
 *
 * Rather than restructure the workspace dedup or balloon the package with
 * the whole node_modules tree, we copy ONLY the runtime-essential JS deps
 * into apps/desktop/node_modules/ and let the `files` list in package.json
 * pick them up from there.
 *
 * Runs as part of `npm run build`. Idempotent -- always re-stages on each
 * build to pick up dep updates.
 *
 * Layout note: we copy the full package tree (package + all transitive deps)
 * so that Node's module resolution finds everything the package needs at
 * runtime.  Only packages that are `require()`d in the Electron main process
 * (electron/*.cjs) are staged -- renderer deps are bundled by Vite.
 */

const fs = require('node:fs')
const path = require('node:path')

const APP_ROOT = path.resolve(__dirname, '..')
const REPO_ROOT = path.resolve(APP_ROOT, '..', '..')

// Packages that the Electron main process requires() at runtime.
// Keep this list minimal -- only add packages that are require()d in
// electron/*.cjs and are NOT bundled by Vite.
const JS_DEPS = [
  'simple-git',
]

function rmrf(target) {
  fs.rmSync(target, { recursive: true, force: true })
}

function ensureDir(target) {
  fs.mkdirSync(target, { recursive: true })
}

function walk(root) {
  const results = []
  const stack = [root]
  while (stack.length) {
    const current = stack.pop()
    let entries
    try {
      entries = fs.readdirSync(current, { withFileTypes: true })
    } catch {
      continue
    }
    for (const entry of entries) {
      const full = path.join(current, entry.name)
      if (entry.isDirectory()) {
        stack.push(full)
      } else if (entry.isFile()) {
        results.push(full)
      }
    }
  }
  return results
}

/**
 * Resolve the full transitive dependency tree of a package by reading its
 * package.json and recursing into dependencies.  Returns a Set of package
 * names (the package itself + all transitive deps).
 */
function resolveTransitiveDeps(pkgName, seen = new Set()) {
  if (seen.has(pkgName)) return seen
  seen.add(pkgName)

  const pkgJsonPath = path.join(REPO_ROOT, 'node_modules', pkgName, 'package.json')
  if (!fs.existsSync(pkgJsonPath)) return seen

  let pkg
  try {
    pkg = JSON.parse(fs.readFileSync(pkgJsonPath, 'utf-8'))
  } catch {
    return seen
  }

  const deps = { ...pkg.dependencies, ...pkg.peerDependencies }
  for (const dep of Object.keys(deps || {})) {
    resolveTransitiveDeps(dep, seen)
  }

  return seen
}

function stageOne(pkgName) {
  const from = path.join(REPO_ROOT, 'node_modules', pkgName)
  const to = path.join(APP_ROOT, 'node_modules', pkgName)

  if (!fs.existsSync(from)) {
    throw new Error(
      `stage-js-deps: source missing at ${from}.  Run \`npm install\` ` +
        `at the workspace root first.`
    )
  }

  rmrf(to)
  ensureDir(to)

  const files = walk(from)
  let copied = 0
  for (const abs of files) {
    const rel = path.relative(from, abs)
    const dest = path.join(to, rel)
    ensureDir(path.dirname(dest))
    fs.copyFileSync(abs, dest)
    copied += 1
  }
  console.log(`[stage-js-deps] node_modules/${pkgName}: ${copied} files`)
}

function main() {
  // Collect all transitive deps across all listed packages
  const allDeps = new Set()
  for (const pkg of JS_DEPS) {
    resolveTransitiveDeps(pkg, allDeps)
  }

  // Stage each dep into apps/desktop/node_modules/
  for (const pkg of allDeps) {
    stageOne(pkg)
  }

  console.log(`[stage-js-deps] staged ${allDeps.size} packages`)
}

main()
