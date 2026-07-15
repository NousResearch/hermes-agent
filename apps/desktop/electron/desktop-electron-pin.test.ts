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
 */

import assert from 'node:assert/strict'
import fs from 'node:fs'
import path from 'node:path'

import { test } from 'vitest'

const REPO_ROOT = path.resolve(__dirname, '..', '..', '..')
const DESKTOP_PKG = path.join(REPO_ROOT, 'apps', 'desktop', 'package.json')
const ROOT_LOCK = path.join(REPO_ROOT, 'package-lock.json')

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
