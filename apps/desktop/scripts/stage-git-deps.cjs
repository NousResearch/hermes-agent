'use strict'

/**
 * stage-git-deps.cjs — stage pure-JS runtime node_modules for packaging.
 *
 * WHY THIS EXISTS
 * ---------------
 * The packaged app ships NO node_modules: main.cjs is meant to be bundled
 * (scripts/bundle-electron-main.mjs) and the electron-builder `files:` list
 * deliberately omits node_modules. That works for everything main.cjs pulls
 * in directly — but `electron/git-review-ops.cjs` is loaded as a SEPARATE,
 * unbundled sibling file (main.cjs require()s it at runtime), so its
 * `require('simple-git')` stays a bare runtime require with nothing to
 * resolve. Result: the app dies on launch with
 *
 *   Error: Cannot find module 'simple-git'
 *   Require stack: .../app.asar/electron/git-review-ops.cjs
 *
 * `simple-git` is workspace-hoisted into the repo-root node_modules/, which
 * electron-builder's file collector cannot reach once `files:` is set.
 *
 * THE FIX
 * -------
 * Mirror scripts/stage-native-deps.cjs: copy ONLY simple-git and its
 * production dependency closure into apps/desktop/build/extra-node-modules/,
 * then ship that subtree via extraResources mapped to `node_modules`. It
 * lands at <resources>/node_modules/, which is on Node's module-resolution
 * walk as it exits the app.asar boundary into the real parent directory — so
 * `require('simple-git')` from inside the asar resolves with NO code change.
 *
 * The dependency closure is computed dynamically from each package.json's
 * `dependencies`, so it stays correct if simple-git's deps change. Modules
 * are resolved from the repo-root node_modules first (hoisted), falling back
 * to the app-local node_modules.
 *
 * Runs as part of `npm run build`. Idempotent — re-stages on every build.
 */

const fs = require('node:fs')
const path = require('node:path')

const APP_ROOT = path.resolve(__dirname, '..')
const REPO_ROOT = path.resolve(APP_ROOT, '..', '..')
const STAGE_ROOT = path.join(APP_ROOT, 'build', 'extra-node-modules')

// Entry points whose production closure we want in the packaged app.
const ROOTS = ['simple-git']

// Where to look for an installed module, in priority order.
const SEARCH_ROOTS = [
  path.join(REPO_ROOT, 'node_modules'),
  path.join(APP_ROOT, 'node_modules')
]

function rmrf(target) {
  fs.rmSync(target, { recursive: true, force: true })
}

function ensureDir(target) {
  fs.mkdirSync(target, { recursive: true })
}

function findModuleDir(name) {
  for (const root of SEARCH_ROOTS) {
    const candidate = path.join(root, name)
    if (fs.existsSync(path.join(candidate, 'package.json'))) {
      return candidate
    }
  }
  return null
}

// Recursively copy a directory, skipping any nested node_modules (deps are
// resolved and staged flat at the top level, matching the hoisted layout).
function copyModule(srcDir, destDir) {
  ensureDir(destDir)
  for (const entry of fs.readdirSync(srcDir, { withFileTypes: true })) {
    if (entry.isDirectory() && entry.name === 'node_modules') continue
    const src = path.join(srcDir, entry.name)
    const dest = path.join(destDir, entry.name)
    if (entry.isDirectory()) {
      copyModule(src, dest)
    } else if (entry.isFile() || entry.isSymbolicLink()) {
      fs.copyFileSync(src, dest)
    }
  }
}

function readDeps(moduleDir) {
  try {
    const pkg = JSON.parse(fs.readFileSync(path.join(moduleDir, 'package.json'), 'utf8'))
    return Object.keys(pkg.dependencies || {})
  } catch {
    return []
  }
}

function main() {
  rmrf(STAGE_ROOT)
  ensureDir(STAGE_ROOT)

  const seen = new Set()
  const queue = [...ROOTS]
  let staged = 0

  while (queue.length) {
    const name = queue.shift()
    if (seen.has(name)) continue
    seen.add(name)

    const moduleDir = findModuleDir(name)
    if (!moduleDir) {
      throw new Error(
        `stage-git-deps: dependency "${name}" not found in any node_modules ` +
          `(${SEARCH_ROOTS.join(', ')}). Run \`npm install\` at the workspace root first.`
      )
    }

    copyModule(moduleDir, path.join(STAGE_ROOT, name))
    staged += 1

    for (const dep of readDeps(moduleDir)) {
      if (!seen.has(dep)) queue.push(dep)
    }
  }

  console.log(
    `[stage-git-deps] ${path.relative(APP_ROOT, STAGE_ROOT)}: staged ${staged} modules ` +
      `(${[...seen].sort().join(', ')})`
  )
}

main()
