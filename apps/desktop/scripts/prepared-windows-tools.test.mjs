import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { test } from 'vitest'
import { ensureWindowsBundleTools } from './windows-bundle-tools.mjs'
import { azureSignFile } from './sign-msix.mjs'

test('signing validates once per packager, including concurrent file hooks, never process-wide', async () => {
  const previous = process.env.HERMES_PREPARED_PACKAGING
  const previousDotnet = process.env.DOTNET_ROOT
  process.env.HERMES_PREPARED_PACKAGING = 'operation-selection.json'
  let admissions = 0
  const signed = []
  const dependencies = {
    ensureTools: async () => { admissions++; return { dotnetRoot: 'prepared-dotnet' } },
    loadManager: async () => class {
      async initialize() {}
      async signFile({ path: file }) { signed.push(file) }
    },
  }
  const resources = os.tmpdir()
  const packager = { config: { toolsets: { winCodeSign: { url: `file://${path.join(resources, 'prepared-packaging-tools/winCodeSign')}` } } }, buildResourcesDir: resources, platformOptions: {} }
  try {
    await Promise.all(['first.exe', 'second.msix'].map(file => azureSignFile(file, packager, dependencies)))
    assert.equal(admissions, 1)
    await azureSignFile('third.msix', { ...packager }, dependencies)
    assert.equal(admissions, 2)
    assert.deepEqual(signed, ['first.exe', 'second.msix', 'third.msix'])
  } finally {
    if (previous === undefined) delete process.env.HERMES_PREPARED_PACKAGING
    else process.env.HERMES_PREPARED_PACKAGING = previous
    if (previousDotnet === undefined) delete process.env.DOTNET_ROOT
    else process.env.DOTNET_ROOT = previousDotnet
  }
})

test('the per-file Azure signer admits the same prepared selection before constructing its manager', async () => {
  const previous = process.env.HERMES_PREPARED_PACKAGING
  process.env.HERMES_PREPARED_PACKAGING = path.join(os.tmpdir(), 'absent-signing-selection.json')
  try {
    await assert.rejects(azureSignFile('output.msix', { config: {}, buildResourcesDir: os.tmpdir() }), /run preparation again/)
  } finally {
    if (previous === undefined) delete process.env.HERMES_PREPARED_PACKAGING
    else process.env.HERMES_PREPARED_PACKAGING = previous
  }
})

test('a prepared signing selection cannot fall through to a supplier', async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'prepared-windows-'))
  try {
    await assert.rejects(ensureWindowsBundleTools({
      prepared: path.join(root, 'missing.json'), signing: true, config: {},
      load: async () => { throw new Error('supplier must not run') },
    }), /run preparation again/i)
  } finally {
    fs.rmSync(root, { recursive: true, force: true })
  }
})
