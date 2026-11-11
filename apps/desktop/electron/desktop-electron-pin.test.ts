/**
 * Regression: the desktop Electron dependency must be an exact, consistent pin.
 *
 * The Windows desktop install failed at "Building desktop app" because Electron
 * changed its install mechanism mid patch-series:
 *
 *     electron 40.9.3 .. 40.10.2  -> @electron/get@^2 + extract-zip@^2  (pure JS)
 *     electron 40.10.3 and later  -> @electron/get@^5 +
 *                                    @electron-internal/extract-zip@^1 (native napi)
 *
 * ``apps/desktop/package.json`` declared ``electronVersion: 40.9.3`` (the tested,
 * JS-extract build) but pinned the dependency loosely as ``electron: ^40.9.3``.
 * npm then resolved 40.10.3/40.10.4 — the new *native* extract-zip whose
 * win32 binding imports VCRUNTIME140.dll and fails to ``dlopen`` on hosts without
 * the VC++ runtime (``ERR_DLOPEN_FAILED loading index.win32-x64-msvc.node``).
 * scripts/build/electron-runtime.mjs now supplies that DLL.
 *
 * These tests lock the contract that prevents that drift, without hard-coding the
 * specific version (which is allowed to move):
 *
 * 1. the Electron dependency is an *exact* version (Electron Builder needs the
 *    installed binary to match ``electronVersion`` / ``electronDist``), and
 * 2. the dependency and the independently resolved lockfile entry agree — so a frozen install installs exactly
 *    what the build packages.
 */

import assert from 'node:assert/strict'
import fs from 'node:fs'
import path from 'node:path'

import { test } from 'vitest'
import { parse } from 'yaml'

const REPO_ROOT = path.resolve(__dirname, '..', '..', '..')
const DESKTOP_PKG = path.join(REPO_ROOT, 'apps', 'desktop', 'package.json')
const ROOT_LOCK = path.join(REPO_ROOT, 'pnpm-lock.yaml')

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
      'A range (^/~) lets a lockfile update resolve a newer Electron whose postinstall ' +
      'may differ from the one the build was validated against.'
  )
})

interface LockDependency {
  specifier: string
  version: string
}

interface Lockfile {
  importers?: Record<string, Partial<Record<'dependencies' | 'devDependencies', Record<string, LockDependency>>>>
  packages?: Record<string, unknown>
}

test('lockfile resolves the pinned electron', () => {
  const spec = electronSpec(desktopPkg())
  const lock = parse(fs.readFileSync(ROOT_LOCK, 'utf-8')) as Lockfile
  const importer = lock.importers?.['apps/desktop']
  const entry = importer?.devDependencies?.electron ?? importer?.dependencies?.electron

  assert.ok(entry, 'no apps/desktop electron entry found in pnpm-lock.yaml')

  // A locked version carries its peer resolution as a suffix: 44.9.3(supports-color@10.2.2).
  const resolved = [
    entry.version.replace(/\(.*$/, ''),
    ...Object.keys(lock.packages ?? {})
      .filter(key => key.startsWith('electron@'))
      .map(key => key.slice('electron@'.length))
  ]

  for (const v of resolved) {
    assert.equal(
      v,
      spec,
      `pnpm-lock.yaml resolves electron to ${v}, but the pin is "${spec}"; ` +
        'run `pnpm install --lockfile-only` so frozen installs stay consistent.'
    )
  }
})
