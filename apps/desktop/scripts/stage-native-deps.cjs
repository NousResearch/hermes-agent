'use strict'

/**
 * Stage native node-modules dependencies for electron-builder packaging.
 *
 * Workspace dedup hoists `node-pty` into the root `node_modules/`, which
 * electron-builder's default file collector (when `files:` is explicitly set
 * in package.json) cannot reach.  The result: packaged builds ship with no
 * .node binaries and PTY initialization fails at runtime ("PTY support is
 * unavailable").
 *
 * Rather than restructure the workspace dedup (would require nohoist /
 * package.json shenanigans and risk breaking dev) or balloon the package
 * with the whole node_modules tree, we copy ONLY the runtime-essential
 * files of the native dep into apps/desktop/build/native-deps/ and ship
 * THAT subtree via extraResources.  main.cjs falls back to require()-ing
 * from process.resourcesPath when the hoisted-root require fails.
 *
 * Runs as part of `npm run build`. Idempotent -- always re-stages on each
 * build to pick up native binary updates.
 *
 * Layout note: upstream node-pty (microsoft/node-pty 1.x) is N-API based
 * and ships its prebuilts under `prebuilds/<platform>-<arch>/` instead of
 * `build/Release/`.  Its runtime resolver (lib/utils.js) checks
 * build/Release first and falls through to the per-arch prebuilds dir, so
 * shipping only the latter is sufficient for packaged runs.  Per-arch
 * staging keeps the resource bundle lean -- we only need the target
 * arch's prebuilt, not all of them.
 */

const fs = require('node:fs')
const path = require('node:path')

const APP_ROOT = path.resolve(__dirname, '..')
const REPO_ROOT = path.resolve(APP_ROOT, '..', '..')
const STAGE_ROOT = path.join(APP_ROOT, 'build', 'native-deps')

// The target arch may be overridden by electron-builder via npm_config_arch
// (e.g. `npm run dist -- --arm64`); fall back to the build host's arch.
const TARGET_ARCH = process.env.npm_config_arch || process.arch
const TARGET_PLATFORM = process.platform

// Modules to stage. The "from" path is the hoisted location in the workspace
// root; "to" is the layout we want inside build/native-deps/.  The "include"
// globs (relative to "from") select the runtime-essential files.  Anything
// outside the include list is left behind (source, deps/, scripts/, etc.).
const NATIVE_DEPS = [
  {
    from: path.join(REPO_ROOT, 'node_modules', 'node-pty'),
    to: path.join(STAGE_ROOT, 'node-pty'),
    include: [
      'package.json',
      'lib/*.js',
      'lib/**/*.js',
      'build/Release/*.node',
      // Per-arch runtime payload. Explicit file types so we don't ship the
      // ~25 MB of .pdb debug symbols that prebuild-install bundles for
      // Windows crash analysis -- not used at runtime, would just bloat
      // the installer.
      `prebuilds/${TARGET_PLATFORM}-${TARGET_ARCH}/*.node`,
      `prebuilds/${TARGET_PLATFORM}-${TARGET_ARCH}/*.dll`,
      `prebuilds/${TARGET_PLATFORM}-${TARGET_ARCH}/*.exe`,
      `prebuilds/${TARGET_PLATFORM}-${TARGET_ARCH}/spawn-helper`,
      `prebuilds/${TARGET_PLATFORM}-${TARGET_ARCH}/conpty/*`
    ]
  }
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

// Match a relative path against simple ** and * glob patterns. Implementation
// is intentionally tiny -- the include lists are small and don't need full
// minimatch support.
function matchGlob(rel, pattern) {
  const r = rel.replace(/\\/g, '/')
  const re = new RegExp(
    '^' +
      pattern
        .replace(/\\/g, '/')
        .replace(/[.+^${}()|[\]\\]/g, '\\$&')
        .replace(/\*\*/g, '__DOUBLE_STAR__')
        .replace(/\*/g, '[^/]*')
        .replace(/__DOUBLE_STAR__/g, '.*') +
      '$'
  )
  return re.test(r)
}

function stageOne(spec) {
  if (!fs.existsSync(spec.from)) {
    throw new Error(
      `stage-native-deps: source missing at ${spec.from}.  Run \`npm install\` ` +
        `at the workspace root first.`
    )
  }
  rmrf(spec.to)
  ensureDir(spec.to)

  const files = walk(spec.from)
  let copied = 0
  for (const abs of files) {
    const rel = path.relative(spec.from, abs)
    const included = spec.include.some(g => matchGlob(rel, g))
    if (!included) continue
    const dest = path.join(spec.to, rel)
    ensureDir(path.dirname(dest))
    fs.copyFileSync(abs, dest)
    // node-pty's darwin spawn-helper and the Windows helper binaries
    // (OpenConsole.exe, winpty-agent.exe) are invoked via posix_spawn /
    // CreateProcess at runtime, so they must remain executable in the
    // staged tree.  fs.copyFileSync preserves source mode on POSIX, but we
    // re-assert +x defensively for the darwin spawn-helper (no extension
    // means a stripped mode would be silently broken at runtime).
    if (path.basename(rel) === 'spawn-helper' && process.platform !== 'win32') {
      try { fs.chmodSync(dest, 0o755) } catch { /* best-effort */ }
    }
    copied += 1
  }
  console.log(`[stage-native-deps] ${path.relative(APP_ROOT, spec.to)}: ${copied} files`)
}

// Pure-JS runtime modules that the Electron main process require()s directly
// (currently simple-git, used by electron/git-review-ops.cjs).  Workspace dedup
// hoists these into the repo-root node_modules where electron-builder's file
// collector can't see them -- same root cause as node-pty above -- but they
// have no .node binary, so the include-glob staging doesn't apply.  Instead we
// resolve each module's full dependency closure and copy the package dirs into
// build/native-deps/node_modules/ as a flat tree (no version conflicts in these
// closures, so flat resolution is sufficient).  A `files` entry in package.json
// (from: build/native-deps/node_modules, to: node_modules) copies that tree into
// the asar at /node_modules, so a plain require('simple-git') resolves in
// packaged builds exactly as it does in dev.  (extraResources can't be used for
// this -- electron-builder strips directories named node_modules out of
// extraResources/extraFiles copies; the asar `files` collection does not.)
const JS_MODULE_CLOSURES = ['simple-git']

// Locate a package's on-disk directory the way Node's resolver does -- walk up
// the directory tree from `fromDir` checking each node_modules/<name> -- and
// confirm via package.json existence rather than require.resolve(). Packages
// with an `exports` map (e.g. simple-git) reject require.resolve of their
// package.json with ERR_PACKAGE_PATH_NOT_EXPORTED, so we probe the filesystem.
function resolvePackageDir(name, fromDir) {
  const candidates = []
  let cur = fromDir
  while (cur) {
    candidates.push(path.join(cur, 'node_modules', name))
    const parent = path.dirname(cur)
    if (parent === cur) break
    cur = parent
  }
  candidates.push(path.join(REPO_ROOT, 'node_modules', name))
  for (const candidate of candidates) {
    if (fs.existsSync(path.join(candidate, 'package.json'))) return candidate
  }
  throw new Error(`unresolved: ${name}`)
}

// Walk the dependency graph of the given roots, resolving each package to its
// on-disk directory.  Returns a Map<name, absoluteDir>.
function collectClosure(rootNames) {
  const dirs = new Map()
  const stack = rootNames.map(name => ({ name, fromDir: REPO_ROOT }))
  while (stack.length) {
    const { name, fromDir } = stack.pop()
    if (dirs.has(name)) continue
    let dir
    try {
      dir = resolvePackageDir(name, fromDir)
    } catch {
      throw new Error(
        `stage-native-deps: cannot resolve "${name}" from ${fromDir}.  Run ` +
          `\`npm install\` at the workspace root first.`
      )
    }
    dirs.set(name, dir)
    const pkg = require(path.join(dir, 'package.json'))
    for (const dep of Object.keys(pkg.dependencies || {})) {
      if (!dirs.has(dep)) stack.push({ name: dep, fromDir: dir })
    }
  }
  return dirs
}

// Recursively copy a package directory, skipping its own node_modules (the full
// closure is staged flat alongside it) and .bin shims.  These are small pure-JS
// packages, so a plain file copy is cheap.
function copyPackageDir(src, dest) {
  ensureDir(dest)
  for (const entry of fs.readdirSync(src, { withFileTypes: true })) {
    if (entry.name === 'node_modules' || entry.name === '.bin') continue
    const from = path.join(src, entry.name)
    const to = path.join(dest, entry.name)
    if (entry.isDirectory()) {
      copyPackageDir(from, to)
    } else if (entry.isFile()) {
      fs.copyFileSync(from, to)
    }
  }
}

function stageModuleClosures() {
  if (!JS_MODULE_CLOSURES.length) return
  const dirs = collectClosure(JS_MODULE_CLOSURES)
  const modulesRoot = path.join(STAGE_ROOT, 'node_modules')
  for (const [name, dir] of dirs) {
    copyPackageDir(dir, path.join(modulesRoot, name))
  }
  console.log(
    `[stage-native-deps] node_modules: ${dirs.size} packages ` +
      `(${JS_MODULE_CLOSURES.join(', ')} + transitive deps)`
  )
}

function main() {
  rmrf(STAGE_ROOT)
  ensureDir(STAGE_ROOT)
  for (const spec of NATIVE_DEPS) {
    stageOne(spec)
  }
  stageModuleClosures()
}

main()
