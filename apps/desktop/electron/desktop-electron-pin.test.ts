/**
 * Regression: the desktop Electron dependency must be an exact, consistent pin.
 *
 * The Windows desktop install failed at "Building desktop app" because Electron
 * changed its install mechanism mid patch-series:
 *
 *     electron 40.9.3 .. 40.10.2  -> @electron/get@^2 + extract-zip@^2  (pure JS)
 *     electron 40.10.3 / 40.10.4  -> @electron/get@^5 +
 *                                    @electron-internal/extract-zip@^1 (native napi)
 *
 * ``apps/desktop/package.json`` declared ``electronVersion: 40.9.3`` (the tested,
 * JS-extract build) but pinned the dependency loosely as ``electron: ^40.9.3``.
 * ``npm ci`` then resolved 40.10.3/40.10.4 — the new *native* extract-zip whose
 * win32-x64 binding fails to ``dlopen`` on some Windows hosts
 * (``ERR_DLOPEN_FAILED loading index.win32-x64-msvc.node``).
 *
 * These tests lock the contract that prevents that drift, without hard-coding the
 * specific version (which is allowed to move):
 *
 * 1. the Electron dependency is an *exact* version (Electron Builder needs the
 *    installed binary to match ``electronVersion`` / ``electronDist``), and
 * 2. the dependency, ``build.electronVersion``, and the resolved lockfile entry
 *    all agree — so ``npm ci`` installs exactly what the build packages.
 *
 * A third pin lives in the *root* ``package.json``: ``allowScripts``. Its keys are
 * version-exact (``"electron@41.10.7": true``), so bumping Electron without moving
 * the key silently orphans Electron's postinstall — ``npm install`` still exits 0
 * with only an ``npm warn install-scripts`` line, the binary is never downloaded,
 * and the failure surfaces much later as a missing ``Electron.app`` at package or
 * launch time. The last two tests lock that pin to the other two.
 */

import assert from 'node:assert/strict'
import fs from 'node:fs'
import path from 'node:path'

import { test } from 'vitest'

const REPO_ROOT = path.resolve(__dirname, '..', '..', '..')
const DESKTOP_PKG = path.join(REPO_ROOT, 'apps', 'desktop', 'package.json')
const ROOT_LOCK = path.join(REPO_ROOT, 'package-lock.json')
const ROOT_PKG = path.join(REPO_ROOT, 'package.json')

// An exact semver: digits.digits.digits with an optional prerelease/build tag,
// but NO range operators (^ ~ > < = * x || spaces || -range).
const EXACT_SEMVER = /^\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?$/

function desktopPkg(): Record<string, unknown> {
  assert.ok(fs.existsSync(DESKTOP_PKG), `missing ${DESKTOP_PKG}`)

  return JSON.parse(fs.readFileSync(DESKTOP_PKG, 'utf-8'))
}

function electronSpec(pkg: Record<string, unknown>): string {
  for (const section of ['dependencies', 'devDependencies'] as const) {
    const deps = (pkg[section] ?? {}) as Record<string, string>
    const spec = deps['electron']

    if (spec) {
      return spec
    }
  }

  assert.fail('electron is not listed in apps/desktop dependencies')
}

test('electron dependency is exactly pinned', () => {
  const spec = electronSpec(desktopPkg())
  assert.match(
    spec,
    EXACT_SEMVER,
    `electron must be pinned to an exact version, got "${spec}". ` +
      'A range (^/~) lets npm ci resolve a newer Electron whose postinstall ' +
      'may differ from the one the build was validated against.'
  )
})

test('electron dependency matches build.electronVersion', () => {
  const pkg = desktopPkg()
  const spec = electronSpec(pkg)
  const build = (pkg.build ?? {}) as Record<string, unknown>
  const builderVersion = build.electronVersion as string | undefined
  assert.ok(builderVersion, 'build.electronVersion is missing')
  assert.equal(
    spec,
    builderVersion,
    `electron dependency ("${spec}") must equal build.electronVersion ` +
      `("${builderVersion}"); otherwise electron-builder packages a different ` +
      'version than npm installs into electronDist.'
  )
})

test('lockfile resolves the pinned electron', () => {
  if (!fs.existsSync(ROOT_LOCK)) {
    return
  } // skip if lockfile not present

  const spec = electronSpec(desktopPkg())
  const lock = JSON.parse(fs.readFileSync(ROOT_LOCK, 'utf-8'))
  const packages = (lock.packages ?? {}) as Record<string, { version?: string }>

  const resolved = Object.entries(packages)
    .filter(([key]) => key.endsWith('node_modules/electron'))
    .map(([, meta]) => meta.version)
    .filter((v): v is string => !!v)

  assert.ok(resolved.length > 0, 'no electron entry found in package-lock.json')

  for (const v of resolved) {
    assert.equal(
      v,
      spec,
      `package-lock.json resolves electron to ${v}, but the pin is "${spec}"; ` +
        'run `npm install --package-lock-only` so `npm ci` stays consistent.'
    )
  }
})

/** Root ``allowScripts`` map: keys are ``name@version`` or a bare ``name``. */
function allowScripts(): Record<string, boolean> {
  assert.ok(fs.existsSync(ROOT_PKG), `missing ${ROOT_PKG}`)
  const pkg = JSON.parse(fs.readFileSync(ROOT_PKG, 'utf-8'))

  return (pkg.allowScripts ?? {}) as Record<string, boolean>
}

/** Split ``"electron@41.10.7"`` -> ``['electron', '41.10.7']``; a bare name yields a null version. */
function splitAllowKey(key: string): [string, string | null] {
  const at = key.lastIndexOf('@')

  // No '@', or a leading '@' with no second one, means the key is a bare package
  // name (covers every installed version) rather than a version-exact pin.
  if (at <= 0) {
    return [key, null]
  }

  return [key.slice(0, at), key.slice(at + 1)]
}

/** Versions of ``name`` present in the root lockfile. */
function lockedVersions(name: string): string[] {
  if (!fs.existsSync(ROOT_LOCK)) {
    return []
  }

  const lock = JSON.parse(fs.readFileSync(ROOT_LOCK, 'utf-8'))
  const packages = (lock.packages ?? {}) as Record<string, { version?: string }>

  return [
    ...new Set(
      Object.entries(packages)
        .filter(([key]) => key.endsWith(`node_modules/${name}`))
        .map(([, meta]) => meta.version)
        .filter((v): v is string => !!v)
    )
  ]
}

test('allowScripts pins the same electron as the dependency', () => {
  const spec = electronSpec(desktopPkg())

  const keys = Object.keys(allowScripts()).filter(k => splitAllowKey(k)[0] === 'electron')

  assert.ok(
    keys.length > 0,
    'electron has no allowScripts entry in the root package.json; its postinstall ' +
      'downloads the binary, so npm would skip it and leave no Electron.app.'
  )

  const versions = keys.map(k => splitAllowKey(k)[1])
  assert.ok(
    versions.includes(spec) || versions.includes(null),
    `allowScripts pins electron@[${versions.join(', ')}] but the dependency is ` +
      `"${spec}". allowScripts keys are version-exact, so a bump that misses this ` +
      "key orphans electron's postinstall: npm install still succeeds (only an " +
      '`npm warn install-scripts` line), the binary is never downloaded, and the ' +
      'build fails later with a missing Electron.app.'
  )
})

test('every allowScripts entry covers the installed version', () => {
  // Guards the general case behind the electron test above: any version-exact
  // allowScripts key that drifts from the lockfile silently disables that
  // package's install scripts. Extra keys for versions that are no longer
  // installed are fine — only an *uncovered installed* version is a problem.
  const byName = new Map<string, Set<string | null>>()

  for (const key of Object.keys(allowScripts())) {
    const [name, version] = splitAllowKey(key)
    const versions = byName.get(name) ?? new Set<string | null>()
    versions.add(version)
    byName.set(name, versions)
  }

  for (const [name, versions] of byName) {
    // A bare-name key covers every version of that package.
    if (versions.has(null)) {
      continue
    }

    for (const installed of lockedVersions(name)) {
      assert.ok(
        versions.has(installed),
        `package-lock.json installs ${name}@${installed}, but allowScripts only ` +
          `pins ${name}@[${[...versions].join(', ')}]. Add "${name}@${installed}" ` +
          `to allowScripts in the root package.json (or drop the stale key) so ` +
          `its install scripts still run.`
      )
    }
  }
})
