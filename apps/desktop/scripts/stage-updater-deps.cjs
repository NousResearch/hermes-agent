'use strict'

/**
 * Stage electron-updater (+ its full dependency closure) for electron-builder
 * packaging.
 *
 * WHY THIS EXISTS
 * ---------------
 * The shell self-updater (`electron/shell-updater.cjs`, wired from
 * `electron/main.cjs`) `require('electron-updater')` at runtime. In packaged
 * builds that require THREW with `Cannot find module 'electron-updater'`, so
 * `autoUpdater` fell back to null and the shell self-updater was disabled from
 * 0.16.1 onward -- it never checked for, downloaded, or surfaced an update
 * (real user report, ~/.apexnodes/logs/desktop.log:
 *   [shell-update] electron-updater unavailable (disabled): Cannot find module 'electron-updater'
 *   [shell-update] disabled (dev / unpackaged build)
 * ).
 *
 * ROOT CAUSE (two compounding reasons, both verified against app-builder-lib
 * 26.15.3 source):
 *   1. `build.files` in package.json is an explicit white-list that does not
 *      list node_modules, so the app-source file matcher never copies it.
 *   2. `scripts/before-build.cjs` returns `false`. In electron-builder that
 *      sets `_nodeModulesHandledExternally = true` (packager.js), which makes
 *      `platformPackager.js` SKIP `computeNodeModuleFileSets` entirely -- the
 *      collector that would otherwise walk the production dependency tree
 *      (including workspace-root-hoisted deps). Result: ZERO node_modules land
 *      in the asar, so no production dependency (electron-updater included) is
 *      ever packaged, regardless of hoisting.
 * Because of (2), simply adding `node_modules/**` to `files` does nothing --
 * empirically verified (asar still had 0 node_modules entries).
 *
 * Workspace dedup additionally hoists electron-updater to the repo-root
 * node_modules, so even the app matcher couldn't reach it.
 *
 * FIX
 * ---
 * Mirror the existing `scripts/stage-native-deps.cjs` pattern (which solves the
 * identical unreachable-hoisted-dep problem for node-pty): copy the runtime
 * files of electron-updater AND its entire transitive dependency closure into
 * `apps/desktop/build/updater-deps/node_modules/` and ship that subtree via
 * extraResources. `electron/main.cjs` falls back to require()-ing from
 * `process.resourcesPath/updater-deps/node_modules/electron-updater` when the
 * normal require fails (dev mode never reaches the fallback).
 *
 * The closure is computed with require.resolve -- NOT a hand-maintained list --
 * so a new transitive dependency can never be silently dropped (which would
 * reintroduce "Cannot find module"). The nested layout of version-pinned deps
 * (e.g. electron-updater/node_modules/fs-extra) is preserved so Node's resolver
 * finds the right version at runtime.
 *
 * Runs as part of `npm run build`. Idempotent -- always re-stages.
 */

const fs = require('node:fs')
const path = require('node:path')

const APP_ROOT = path.resolve(__dirname, '..')
const REPO_ROOT = path.resolve(APP_ROOT, '..', '..')
const STAGE_ROOT = path.join(APP_ROOT, 'build', 'updater-deps')
// The node_modules tree is nested under an intermediate `vendor/` dir on
// purpose. electron-builder's extraResources copier (app-builder-lib
// util/filter.js) hard-rejects any path whose relative name is exactly
// `node_modules` at the TOP level of the copied `from` dir -- so if we staged
// build/updater-deps/node_modules/* directly, electron-builder would create an
// empty updater-deps/ and silently drop the whole subtree (verified). Putting
// it at build/updater-deps/vendor/node_modules/* makes the top-level child
// `vendor` (allowed); the deeper `vendor/node_modules` matches the `**/*`
// extraResources pattern and ships. Node's require() still needs the
// `node_modules` naming to resolve the closure, which this preserves.
const STAGE_NODE_MODULES = path.join(STAGE_ROOT, 'vendor', 'node_modules')

// Manifest of the staged closure, written NEXT TO vendor/ so it ships via the
// same `build/updater-deps` -> `updater-deps` extraResources mapping (lands at
// <Resources>/updater-deps/updater-deps-manifest.json in the packaged app). The
// post-pack integrity gate (scripts/assert-updater-deps.cjs, run by BOTH the
// mac and win workflows) reads it to verify every package that was staged
// actually survived into the packaged tree -- catching a dropped/short-circuited
// copy on either platform without a hand-maintained package list.
const MANIFEST_NAME = 'updater-deps-manifest.json'
const STAGE_MANIFEST = path.join(STAGE_ROOT, MANIFEST_NAME)

// Entry package whose full production-dependency closure we stage.
const ROOT_PACKAGE = 'electron-updater'

// Per-package files we do NOT ship -- source maps, TS typings, and READMEs are
// dead weight at runtime. Everything else (JS, JSON, LICENSE) is copied so the
// packages remain self-describing and legally intact.
const EXCLUDE_EXT = new Set(['.map', '.ts', '.md', '.markdown'])
const EXCLUDE_DIR = new Set(['.github', '.bin'])

function rmrf(target) {
  fs.rmSync(target, { recursive: true, force: true })
}

function ensureDir(target) {
  fs.mkdirSync(target, { recursive: true })
}

/**
 * Resolve the on-disk directory of a package as seen FROM `fromDir`, following
 * Node's real module resolution (so hoisted-to-root and version-pinned nested
 * copies both resolve correctly).
 */
function resolvePkgDir(name, fromDir) {
  try {
    return path.dirname(require.resolve(`${name}/package.json`, { paths: [fromDir] }))
  } catch {
    return null
  }
}

/**
 * Walk the production dependency graph starting at ROOT_PACKAGE. Returns a Map
 * of packageName -> absolute source dir. `dependencies` only (never
 * devDependencies) -- matches what actually gets require()'d at runtime.
 */
function collectClosure() {
  const dirs = new Map()
  const visit = (name, fromDir) => {
    if (dirs.has(name)) return
    const dir = resolvePkgDir(name, fromDir)
    if (!dir) {
      throw new Error(
        `stage-updater-deps: cannot resolve "${name}" from ${fromDir}. ` +
          `Run \`npm install\` at the workspace root first.`
      )
    }
    dirs.set(name, dir)
    let pkg
    try {
      pkg = require(path.join(dir, 'package.json'))
    } catch {
      return
    }
    for (const dep of Object.keys(pkg.dependencies || {})) {
      visit(dep, dir)
    }
  }
  visit(ROOT_PACKAGE, REPO_ROOT)
  return dirs
}

function walkFiles(root) {
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
      if (entry.isDirectory()) {
        // Skip nested node_modules here -- each dependency in the closure is
        // staged at the top level of the staged tree from its OWN resolved
        // dir, so we never want a package's bundled node_modules duplicated
        // inside it (that path is handled by the closure walk instead).
        if (entry.name === 'node_modules') continue
        if (EXCLUDE_DIR.has(entry.name)) continue
        stack.push(path.join(current, entry.name))
      } else if (entry.isFile()) {
        results.push(path.join(current, entry.name))
      }
    }
  }
  return results
}

/**
 * Where a package lands in the staged tree. The source dir's path after its
 * OUTERMOST `node_modules/` boundary is preserved verbatim, so version-pinned
 * nested copies keep their nesting and Node resolves the correct version:
 *   .../node_modules/electron-updater/node_modules/fs-extra
 *     -> <stage>/node_modules/electron-updater/node_modules/fs-extra
 *   .../node_modules/semver
 *     -> <stage>/node_modules/semver
 * Preserving the FULL nested path (not just the last segment) means two
 * different versions of the same dep in the closure never collide.
 */
function stagedDestFor(name, srcDir) {
  const marker = `${path.sep}node_modules${path.sep}`
  const idx = srcDir.indexOf(marker)
  // Relative path from the outermost node_modules boundary. Falls back to the
  // package name if the dep somehow lives outside a node_modules dir.
  const rel = idx >= 0 ? srcDir.slice(idx + marker.length) : name
  return path.join(STAGE_NODE_MODULES, rel)
}

/**
 * The package's main-entry file, relative to the package dir, resolved to the
 * REAL file require() would load -- not the raw package.json `main` string.
 * Raw mains are legal without an extension or with a `./` prefix (ms ships
 * `"./index"`, tiny-typed-emitter `"lib/index"`); the integrity gate does a
 * literal isFile() on the recorded path, so recording the raw string made the
 * gate's very first real run (0.16.6) fail on paths like `ms/./index` that
 * require() resolves but the filesystem doesn't know. Resolution happens HERE,
 * at stage time, where the source files are on disk: require.resolve() on the
 * package dir applies Node's full main/index resolution. If even that can't
 * produce a file inside the package (exotic exports-only package), return null
 * and let the gate fall back to the dir-presence check for that package.
 */
function entryRelFor(srcDir) {
  try {
    // realpath BOTH sides before relativizing: require.resolve returns the
    // real path, so a symlinked srcDir (macOS /var -> /private/var, pnpm-style
    // layouts) would otherwise relativize into ../.. garbage.
    const realSrc = fs.realpathSync(path.resolve(srcDir))
    const abs = require.resolve(realSrc)
    const rel = path.relative(realSrc, abs)
    // A main outside its own package dir would make the manifest lie; treat as
    // unresolvable rather than recording a path the gate can't join safely.
    if (rel.startsWith('..') || path.isAbsolute(rel)) return null
    return rel
  } catch {
    return null
  }
}

function copyPackage(name, srcDir) {
  const destPkgDir = stagedDestFor(name, srcDir)
  const files = walkFiles(srcDir)
  let copied = 0
  for (const abs of files) {
    if (EXCLUDE_EXT.has(path.extname(abs).toLowerCase())) continue
    const rel = path.relative(srcDir, abs)
    const dest = path.join(destPkgDir, rel)
    ensureDir(path.dirname(dest))
    fs.copyFileSync(abs, dest)
    copied += 1
  }
  return copied
}

// POSIX-normalized path of the package dir, relative to STAGE_NODE_MODULES, so
// the manifest is byte-identical whether staged on macOS or Windows and the
// gate can join it with either separator.
function stagedRelDir(name, srcDir) {
  return path
    .relative(STAGE_NODE_MODULES, stagedDestFor(name, srcDir))
    .split(path.sep)
    .join('/')
}

function main() {
  rmrf(STAGE_ROOT)
  ensureDir(STAGE_NODE_MODULES)

  const closure = collectClosure()
  let total = 0
  const manifestPackages = []
  for (const [name, srcDir] of [...closure].sort((a, b) => a[0].localeCompare(b[0]))) {
    total += copyPackage(name, srcDir)
    const dir = stagedRelDir(name, srcDir)
    const entryRel = entryRelFor(srcDir)
    manifestPackages.push({
      name,
      dir,
      // Package-main relative to vendor/node_modules, POSIX-joined -- the
      // REAL resolved file the gate re-checks exists in the packaged tree.
      // null = no resolvable main (gate then relies on the dir check alone).
      entry: entryRel ? `${dir}/${entryRel.split(path.sep).join('/')}` : null,
    })
  }

  // Sanity gate: the packaged app is broken if the entry package or its own
  // package.json/main is missing. Fail the build loudly rather than ship a
  // dud that silently disables self-update again.
  const entryPkgJson = path.join(STAGE_NODE_MODULES, ROOT_PACKAGE, 'package.json')
  if (!fs.existsSync(entryPkgJson)) {
    throw new Error(`stage-updater-deps: staged ${ROOT_PACKAGE} is missing package.json`)
  }
  const entryMain = require(entryPkgJson).main || 'index.js'
  const entryMainPath = path.join(STAGE_NODE_MODULES, ROOT_PACKAGE, entryMain)
  if (!fs.existsSync(entryMainPath)) {
    throw new Error(
      `stage-updater-deps: staged ${ROOT_PACKAGE} main entry missing (${entryMain})`
    )
  }

  // Emit the closure manifest for the post-pack integrity gate. `root` names
  // the entry package so the gate can hard-require its presence explicitly.
  fs.writeFileSync(
    STAGE_MANIFEST,
    `${JSON.stringify({ root: ROOT_PACKAGE, packages: manifestPackages }, null, 2)}\n`
  )

  console.log(
    `[stage-updater-deps] staged ${closure.size} packages, ${total} files ` +
      `into ${path.relative(APP_ROOT, STAGE_ROOT)} ` +
      `(+ ${MANIFEST_NAME})`
  )
}

// Testable seam (stage-updater-deps.test.cjs unit-tests the entry resolution
// that broke the 0.16.6 gate); running as a script stages as before.
module.exports = { entryRelFor }

if (require.main === module) {
  main()
}
