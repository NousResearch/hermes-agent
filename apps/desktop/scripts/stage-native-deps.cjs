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

// Pure-JS runtime dependencies that the packaged electron main require()s but
// that workspace dedup hoists into the repo-root node_modules -- out of reach
// of electron-builder's file collector, exactly like node-pty above.  Unlike
// node-pty there is no native binary to select; we stage each package's whole
// directory into build/native-deps/vendor/node_modules/<name> so the dep's own
// internal require()s resolve against a real node_modules tree, and the
// requiring file (electron/git-review-ops.cjs) falls back to that path via
// process.resourcesPath when the normal require() fails.  See issue #52735
// (packaged app crashed at launch on `Cannot find module 'simple-git'`).
//
// The closure is resolved at stage time by walking dependencies +
// optionalDependencies, so a simple-git version bump that pulls in a new
// transitive dep can't silently re-introduce the crash.
//
// Layout note: the closure lands in build/native-deps/vendor/node_modules/,
// NOT build/native-deps/node_modules/.  electron-builder's file collector
// hard-drops a `node_modules` directory that sits at the ROOT of an
// extraResources copy (app-builder-lib/out/util/filter.js: `if (relative ===
// "node_modules") return false`), but keeps a NESTED one.  Nesting under
// `vendor/` makes node_modules a subdirectory so it survives packing; the
// require() fallback in git-review-ops.cjs resolves the matching
// vendor/node_modules path.
const JS_DEP_ROOTS = ['simple-git']
const JS_DEP_STAGE_ROOT = path.join(STAGE_ROOT, 'vendor', 'node_modules')

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

// Resolve a package's directory by name, searching the repo-root node_modules
// first (where workspace dedup hoists everything) and then the requiring
// package's own node_modules for any non-hoisted nested copy.
//
// We deliberately do NOT use require.resolve(`${name}/package.json`): packages
// with an "exports" map that doesn't list "./package.json" (e.g. simple-git
// 3.x) make that subpath unresolvable under Node's exports enforcement
// (ERR_PACKAGE_PATH_NOT_EXPORTED), which fails on CI even though it happened to
// work locally.  Instead resolve the package's main entry (exports-aware) and
// walk up to the directory whose package.json's "name" matches.
function resolvePkgDir(name, fromDir) {
  const searchPaths = [fromDir, REPO_ROOT, path.join(REPO_ROOT, 'node_modules')]
  let entry
  try {
    entry = require.resolve(name, { paths: searchPaths })
  } catch {
    return null
  }
  // Walk up from the resolved entry file to the package root: the first
  // ancestor dir whose package.json declares this package's name.
  let dir = path.dirname(entry)
  while (true) {
    const pjPath = path.join(dir, 'package.json')
    try {
      const pj = JSON.parse(fs.readFileSync(pjPath, 'utf8'))
      if (pj.name === name) {
        return dir
      }
    } catch {
      // no package.json here (or unreadable) — keep walking up
    }
    const parent = path.dirname(dir)
    if (parent === dir) {
      return null
    }
    dir = parent
  }
}

// Walk dependencies + optionalDependencies from each root package and return
// the set of resolved package directories in the runtime closure.  Keyed by
// package name so a dep reached via two paths is staged once.
function resolveJsClosure(roots) {
  const closure = new Map() // name -> absolute package dir
  const stack = roots.map(name => ({ name, fromDir: REPO_ROOT }))
  while (stack.length) {
    const { name, fromDir } = stack.pop()
    if (closure.has(name)) continue
    const dir = resolvePkgDir(name, fromDir)
    if (!dir) {
      throw new Error(
        `stage-native-deps: could not resolve '${name}' for the simple-git ` +
          `closure.  Run \`npm install\` at the workspace root first.`
      )
    }
    closure.set(name, dir)
    let pj
    try {
      pj = JSON.parse(fs.readFileSync(path.join(dir, 'package.json'), 'utf8'))
    } catch {
      continue
    }
    const deps = { ...(pj.dependencies || {}), ...(pj.optionalDependencies || {}) }
    for (const depName of Object.keys(deps)) {
      stack.push({ name: depName, fromDir: dir })
    }
  }
  return closure
}

// Stage the resolved JS dependency closure into build/native-deps/vendor/node_modules/
// so the packaged app (and the nix output) can require() it from
// process.resourcesPath when the hoisted-root require() isn't reachable.  Each
// package is copied whole (minus node_modules/ — the closure is flattened so
// every dep already has its own top-level entry) into a real node_modules
// layout, which keeps the deps' own internal require()s working unchanged.
function stageJsClosure(roots) {
  const closure = resolveJsClosure(roots)
  rmrf(JS_DEP_STAGE_ROOT)
  ensureDir(JS_DEP_STAGE_ROOT)
  let staged = 0
  for (const [name, fromDir] of closure) {
    const dest = path.join(JS_DEP_STAGE_ROOT, name)
    ensureDir(path.dirname(dest))
    // Copy the package directory but skip any nested node_modules/ — the
    // closure is flattened, so nested copies would just bloat the bundle.
    fs.cpSync(fromDir, dest, {
      recursive: true,
      filter: src => path.basename(src) !== 'node_modules'
    })
    staged += 1
  }
  console.log(
    `[stage-native-deps] vendor/node_modules/: ${staged} package(s) ` +
      `(${[...closure.keys()].sort().join(', ')})`
  )
}

function main() {
  rmrf(STAGE_ROOT)
  ensureDir(STAGE_ROOT)
  for (const spec of NATIVE_DEPS) {
    stageOne(spec)
  }
  stageMimalloc()
  stageJsClosure(JS_DEP_ROOTS)
}

function stageMimalloc() {
  const platformKey =
    TARGET_PLATFORM === 'darwin' ? 'darwin' : TARGET_PLATFORM === 'win32' ? 'win32' : 'linux'
  const subdir = `${platformKey}-${TARGET_ARCH}`
  const destDir = path.join(STAGE_ROOT, 'mimalloc', subdir)
  const libName =
    TARGET_PLATFORM === 'darwin'
      ? 'libmimalloc.dylib'
      : TARGET_PLATFORM === 'win32'
        ? 'mimalloc-redirect.dll'
        : 'libmimalloc.so'

  const candidates = []
  if (TARGET_PLATFORM === 'linux') {
    candidates.push(
      `/usr/lib/${TARGET_ARCH}-linux-gnu/${libName}`,
      `/usr/lib/${TARGET_ARCH}-linux-gnu/libmimalloc.so.2`,
      `/usr/lib/${libName}`,
      `/usr/local/lib/${libName}`
    )
  } else if (TARGET_PLATFORM === 'darwin') {
    candidates.push(
      '/opt/homebrew/lib/libmimalloc.dylib',
      '/usr/local/lib/libmimalloc.dylib'
    )
  }

  for (const src of candidates) {
    if (!src || !fs.existsSync(src)) continue
    ensureDir(destDir)
    fs.copyFileSync(src, path.join(destDir, libName))
    console.log(`[stage-native-deps] mimalloc: staged ${src} -> ${destDir}`)
    return
  }
  console.warn(
    '[stage-native-deps] mimalloc library not found on build host — ' +
      'dashboard allocator preload will be skipped until libmimalloc is staged'
  )
}

main()
