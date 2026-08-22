import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import test from 'node:test'
import { fileURLToPath } from 'node:url'

import {
  PACKAGED_MAIN_MODULE,
  READY_TOKENS,
  assertBackendReadyArtifactSourceAcceptsBothTokens,
  assertPackagedBackendReadyArtifact,
  extractPackagedMainSource,
  resolvePackagedAsarPath,
  unpackedPathForAsar
} from './backend-ready-artifact.mjs'

const here = path.dirname(fileURLToPath(import.meta.url))
const BACKEND_READY_SOURCE = path.resolve(here, '..', 'electron', 'backend-ready.ts')

function readBackendReadySource() {
  return fs.readFileSync(BACKEND_READY_SOURCE, 'utf8')
}

function staleDashboardOnlySource() {
  const source = readBackendReadySource()
  const stale = source.replace(
    'HERMES_(?:BACKEND|DASHBOARD)_READY',
    'HERMES_DASHBOARD_READY'
  )
  assert.notEqual(stale, source, 'test fixture failed to create the stale parser')
  return stale
}

test('backend-ready artifact source accepts both desktop readiness tokens', () => {
  assert.doesNotThrow(() => {
    assertBackendReadyArtifactSourceAcceptsBothTokens(
      readBackendReadySource(),
      PACKAGED_MAIN_MODULE
    )
  })
})

test('artifact guard rejects a stale dashboard-only readiness parser', () => {
  assert.throws(
    () => assertBackendReadyArtifactSourceAcceptsBothTokens(staleDashboardOnlySource()),
    /HERMES_BACKEND_READY/
  )
})

test('resolvePackagedAsarPath handles Windows/Linux and macOS app layouts', () => {
  assert.equal(
    resolvePackagedAsarPath({ appOutDir: '/tmp/win-unpacked', electronPlatformName: 'win32' }),
    path.join('/tmp/win-unpacked', 'resources', 'app.asar')
  )
  assert.equal(
    resolvePackagedAsarPath({ appOutDir: '/tmp/linux-unpacked', electronPlatformName: 'linux' }),
    path.join('/tmp/linux-unpacked', 'resources', 'app.asar')
  )
  assert.equal(
    resolvePackagedAsarPath({
      appOutDir: '/tmp/mac-arm64',
      electronPlatformName: 'darwin',
      packager: { appInfo: { productFilename: 'Hermes' } }
    }),
    path.join('/tmp/mac-arm64', 'Hermes.app', 'Contents', 'Resources', 'app.asar')
  )
  assert.equal(
    resolvePackagedAsarPath({
      appOutDir: '/tmp/Hermes.app',
      electronPlatformName: 'darwin'
    }),
    path.join('/tmp/Hermes.app', 'Contents', 'Resources', 'app.asar')
  )
})

test('extractPackagedMainSource reads the packaged Electron main bundle from app.asar', () => {
  const tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-ready-artifact-'))
  const asarPath = path.join(tempRoot, 'app.asar')
  const expectedSource = readBackendReadySource()
  const calls = []
  const fakeAsar = {
    extractFile(archive, file) {
      calls.push({ archive, file })
      return Buffer.from(expectedSource, 'utf8')
    }
  }

  try {
    fs.writeFileSync(asarPath, 'fake asar archive', 'utf8')
    assert.equal(extractPackagedMainSource(asarPath, { asarModule: fakeAsar }), expectedSource)
    assert.deepEqual(calls, [{ archive: asarPath, file: PACKAGED_MAIN_MODULE }])
  } finally {
    fs.rmSync(tempRoot, { recursive: true, force: true })
  }
})

test('extractPackagedMainSource prefers an unpacked Electron main bundle', () => {
  const tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-ready-artifact-'))
  const asarPath = path.join(tempRoot, 'app.asar')
  const unpackedPath = unpackedPathForAsar(asarPath)
  const expectedSource = readBackendReadySource()
  const fakeAsar = {
    extractFile() {
      throw new Error('asar should not be consulted when unpacked bundle exists')
    }
  }

  try {
    fs.writeFileSync(asarPath, 'fake asar archive', 'utf8')
    fs.mkdirSync(path.dirname(unpackedPath), { recursive: true })
    fs.writeFileSync(unpackedPath, expectedSource, 'utf8')
    assert.equal(extractPackagedMainSource(asarPath, { asarModule: fakeAsar }), expectedSource)
  } finally {
    fs.rmSync(tempRoot, { recursive: true, force: true })
  }
})

test('assertPackagedBackendReadyArtifact validates the extracted parser marker', () => {
  const tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-ready-artifact-'))
  const asarPath = path.join(tempRoot, 'app.asar')
  const fakeAsar = {
    extractFile() {
      return Buffer.from(readBackendReadySource(), 'utf8')
    }
  }

  try {
    fs.writeFileSync(asarPath, 'fake asar archive', 'utf8')
    const result = assertPackagedBackendReadyArtifact(asarPath, { asarModule: fakeAsar })
    assert.equal(result.asarPath, asarPath)
    assert.equal(result.module, PACKAGED_MAIN_MODULE)
    assert.deepEqual(result.tokens, READY_TOKENS)
  } finally {
    fs.rmSync(tempRoot, { recursive: true, force: true })
  }
})
