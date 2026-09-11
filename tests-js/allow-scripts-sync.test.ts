/**
 * Invariants tying ``allowScripts`` to the lockfile it gates.
 *
 * npm's ``allowScripts`` allowlist is keyed by exact ``name@version``, so an
 * entry silently stops matching the moment that dependency is bumped. Nothing
 * else in the build notices: npm downgrades the blocked script to a warning
 * buried in install output, and the failure only surfaces much later as a
 * missing native artifact.
 *
 * That has now bitten twice on Windows. ``get-windows`` was added to
 * ``apps/desktop`` without an allow entry, so its node-pre-gyp install script
 * never downloaded the win32 binding and ``hermes desktop`` died in
 * ``stage-native-deps``. In the same window, a CVE sweep moved Electron to
 * 40.10.6 and left the ``electron@40.10.2`` pin behind, blocking Electron's
 * own postinstall on any clean install.
 *
 * Two contracts keep the allowlist honest:
 *
 * - Every versioned pin names a version the lockfile actually resolves, so a
 *   dependency bump that orphans its pin fails here instead of in a user's
 *   build.
 * - Every package the lockfile marks as having an install script is covered
 *   by a decision — allowed at its exact version, or denied by name.
 *
 * A bare-name key (no ``@version``) is a deliberate standing decision that
 * survives version bumps, which is how ``unicode-animations: false`` stays a
 * permanent denial.
 */

import assert from 'node:assert/strict'
import fs from 'node:fs'
import path from 'node:path'

import { describe, test } from 'vitest'

const REPO_ROOT = path.resolve(__dirname, '..')

const MANIFESTS = [
  { name: 'root', dir: '.' },
  { name: 'website', dir: 'website' }
]

function manifestLabel(dir: string): string {
  return path.join(dir === '.' ? '' : dir, 'package.json')
}

interface LockPackage {
  name?: string
  version?: string
  hasInstallScript?: boolean
}

function readJson(filePath: string): Record<string, unknown> {
  return JSON.parse(fs.readFileSync(filePath, 'utf-8'))
}

function packageNameFor(lockPath: string, entry: LockPackage): string {
  return entry.name ?? lockPath.split('node_modules/').pop() ?? lockPath
}

/** Every version of every package the lockfile installs, keyed by name. */
function installedVersions(lock: Record<string, unknown>): Map<string, Set<string>> {
  const versions = new Map<string, Set<string>>()

  for (const [lockPath, entry] of Object.entries(
    (lock.packages ?? {}) as Record<string, LockPackage>
  )) {
    if (!lockPath || !entry.version) {
      continue
    }

    const name = packageNameFor(lockPath, entry)
    const seen = versions.get(name) ?? new Set<string>()

    seen.add(entry.version)
    versions.set(name, seen)
  }

  return versions
}

function splitPin(key: string): { name: string; version: string } | null {
  // Scoped packages carry a leading @, so match the LAST @ as the separator.
  const match = key.match(/^(.+)@([^@]+)$/)

  return match ? { name: match[1], version: match[2] } : null
}

describe.each(MANIFESTS)('$name allowScripts', ({ dir }) => {
  const manifestPath = path.join(REPO_ROOT, dir, 'package.json')
  const lockPath = path.join(REPO_ROOT, dir, 'package-lock.json')
  const label = manifestLabel(dir)

  test('every versioned pin matches a version in the lockfile', () => {
    if (!fs.existsSync(lockPath)) {
      return
    }

    const allow = (readJson(manifestPath).allowScripts ?? {}) as Record<string, boolean>
    const versions = installedVersions(readJson(lockPath))
    const stale: string[] = []

    for (const key of Object.keys(allow)) {
      const pin = splitPin(key)

      if (!pin) {
        continue
      }

      const installed = versions.get(pin.name)

      if (!installed?.has(pin.version)) {
        stale.push(`  "${key}" — lockfile resolves ${pin.name} to ${installed ? [...installed].join(', ') : '<nothing>'}`)
      }
    }

    assert.deepEqual(
      stale,
      [],
      `Stale allowScripts entries in ${label}:\n${stale.join('\n')}\n` +
        "npm matches these by exact version, so each package's install script is " +
        'silently blocked. Update the pin to the installed version, or drop the ' +
        'entry if the dependency is gone.'
    )
  })

  test('every package with an install script has an allowScripts decision', () => {
    if (!fs.existsSync(lockPath)) {
      return
    }

    const allow = (readJson(manifestPath).allowScripts ?? {}) as Record<string, boolean>
    const lock = readJson(lockPath)
    const uncovered: string[] = []

    for (const [entryPath, entry] of Object.entries(
      (lock.packages ?? {}) as Record<string, LockPackage>
    )) {
      if (!entryPath || !entry.hasInstallScript) {
        continue
      }

      const name = packageNameFor(entryPath, entry)

      if (!(`${name}@${entry.version}` in allow) && !(name in allow)) {
        uncovered.push(`  ${name}@${entry.version}`)
      }
    }

    assert.deepEqual(
      uncovered,
      [],
      `Packages with install scripts and no allowScripts decision in ${label}:\n` +
        `${uncovered.join('\n')}\n` +
        'npm blocks these. Add "<name>@<version>": true to allow, or "<name>": false ' +
        'to deny permanently.'
    )
  })
})
