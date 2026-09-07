/**
 * Tests for the Python 3.12-only version probes in backend-probes.ts.
 *
 * Operational Hermes runtime is exactly Python 3.12. Every interpreter
 * candidate is accepted only after a read-only version probe proves
 * sys.version_info[:2] == (3, 12); unsupported or unproven candidates fail
 * closed. These tests exercise the actual pure selection behavior and
 * rejection reasons, not source text.
 *
 * Run with: npx vitest run src/... or node --test (see package.json).
 */

import assert from 'node:assert/strict'

import { test } from 'vitest'

import {
  isSupportedHermesPython,
  isSupportedPythonVersion,
  parsePythonVersionFromProbeOutput
} from './backend-probes'

import { resolveVenvHermesCommand } from './windows-hermes-path'

test('isSupportedPythonVersion accepts only 3.12', () => {
  assert.equal(isSupportedPythonVersion(3, 12), true)
  assert.equal(isSupportedPythonVersion(3, 11), false)
  assert.equal(isSupportedPythonVersion(3, 13), false)
  assert.equal(isSupportedPythonVersion(3, 14), false)
  assert.equal(isSupportedPythonVersion(3, 10), false)
  assert.equal(isSupportedPythonVersion(2, 7), false)
})

test('parsePythonVersionFromProbeOutput parses major.minor', () => {
  assert.deepEqual(parsePythonVersionFromProbeOutput('3.12'), [3, 12])
  assert.deepEqual(parsePythonVersionFromProbeOutput('3.12.11'), [3, 12])
  assert.deepEqual(parsePythonVersionFromProbeOutput('  3.12\n'), [3, 12])
  assert.deepEqual(parsePythonVersionFromProbeOutput('Python 3.12.13'), [3, 12])
  assert.equal(parsePythonVersionFromProbeOutput(''), null)
  assert.equal(parsePythonVersionFromProbeOutput('nope'), null)
})

test('isSupportedHermesPython fails closed for missing binary', () => {
  assert.equal(isSupportedHermesPython(''), false)
  assert.equal(
    isSupportedHermesPython('/definitely/not/a/real/python-anywhere'),
    false
  )
})

test('isSupportedHermesPython proves the host Node is not Python 3.12', () => {
  // Node exits non-zero on `-c 'import sys...'`, so the probe is unproven
  // and must fail closed rather than being accepted.
  assert.equal(isSupportedHermesPython(process.execPath), false)
})

test('resolveVenvHermesCommand fails closed when the version probe rejects', () => {
  const deps = {
    isWindows: true,
    isCommandScript: () => false,
    fileExists: () => true,
    directoryExists: () => false,
    canImportHermesCli: () => true,
    isSupportedHermesPython: () => false,
    getVenvPython: (venvRoot: string) => `${venvRoot}/Scripts/python.exe`,
    getVenvSitePackagesEntries: () => [],
    buildDesktopBackendEnv: () => ({ FAKE_ENV: '1' }),
    hermesHome: '/fake/hermes-home',
    resolvePath: (...segments: string[]) => segments.join('/').replace(/\/+/g, '/'),
    dirname: (p: string) => p.slice(0, p.lastIndexOf('/')) || '/',
    basename: (p: string) => p.slice(p.lastIndexOf('/') + 1),
    rememberLog: () => {}
  }
  assert.equal(
    resolveVenvHermesCommand('/root/venv/Scripts/hermes.exe', ['serve'], deps),
    null
  )
})

test('resolveVenvHermesCommand accepts a 3.12-proven venv python', () => {
  const deps = {
    isWindows: true,
    isCommandScript: () => false,
    fileExists: () => true,
    directoryExists: () => false,
    canImportHermesCli: () => true,
    isSupportedHermesPython: () => true,
    getVenvPython: (venvRoot: string) => `${venvRoot}/Scripts/python.exe`,
    getVenvSitePackagesEntries: () => [],
    buildDesktopBackendEnv: () => ({ FAKE_ENV: '1' }),
    hermesHome: '/fake/hermes-home',
    resolvePath: (...segments: string[]) => segments.join('/').replace(/\/+/g, '/'),
    dirname: (p: string) => p.slice(0, p.lastIndexOf('/')) || '/',
    basename: (p: string) => p.slice(p.lastIndexOf('/') + 1),
    rememberLog: () => {}
  }
  const result = resolveVenvHermesCommand('/root/venv/Scripts/hermes.exe', ['serve'], deps)
  assert.ok(result)
  assert.equal(result?.command, '/root/venv/Scripts/python.exe')
})
