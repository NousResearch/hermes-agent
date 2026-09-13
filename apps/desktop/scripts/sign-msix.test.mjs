import assert from 'node:assert/strict'
import fs from 'node:fs'
import { createRequire } from 'node:module'
import path from 'node:path'
import { pathToFileURL } from 'node:url'
import { test } from 'vitest'

import { azureConfigFromEnv, shouldSignFile } from './sign-msix.mjs'

const require = createRequire(import.meta.url)

test('shouldSignFile admits only .msix and .msixbundle artifacts', () => {
  assert.equal(shouldSignFile('release/Hermes-0.17.0-win32-arm64.msix'), true)
  assert.equal(shouldSignFile('release/Hermes-0.17.0-win32-x64.msix'), true)
  assert.equal(shouldSignFile('release/Hermes-0.17.0-win32-arm64.msixbundle'), true)
  // Case-insensitive — artifactName could emit .MSIX on some host.
  assert.equal(shouldSignFile('release/Hermes-0.17.0-win32-arm64.MSIX'), true)
})

test('shouldSignFile rejects the Store-submission variant (Partner Center re-signs)', () => {
  // The Store- msix manifest Publisher is the Partner Center publisher ID
  // (CN=EE6D86E4-...), which no signable cert subject can match — ATS
  // cannot customize CN and CA/B requires the legal entity name — so
  // SignerSign would fail 0x8007000B. Partner Center signs on ingestion.
  assert.equal(
    shouldSignFile('release/Store-HermesBundled-0.28.0-canary.20260828211829-win-x64.msix'),
    false
  )
  assert.equal(
    shouldSignFile('release/Store-HermesBundled-0.28.0-canary.20260828211829-win-arm64.msixbundle'),
    false
  )
  // The out-of-store artifacts keep the only signature Windows validates.
  assert.equal(
    shouldSignFile('release/HermesBundled-0.28.0-canary.20260828211829-win-x64.msix'),
    true
  )
})

test('shouldSignFile rejects every non-package file the hook is asked to sign', () => {
  // The app exe and any payload binary are covered by the package's block
  // map — signing them is wasted round-trips and would break the hash if
  // done after makeappx packs the package.
  assert.equal(shouldSignFile('release/win-unpacked/Hermes.exe'), false)
  assert.equal(shouldSignFile('C:/work/hermes-agent/release/win-unpacked/Hermes.exe'), false)
  assert.equal(shouldSignFile('release/Hermes-0.17.0-win32-arm64.nsis.exe'), false)
  assert.equal(shouldSignFile('release/Hermes-0.17.0-win32-arm64.msixupload'), false)
  assert.equal(shouldSignFile('release/Hermes-0.17.0-win32-arm64.dll'), false)
  assert.equal(shouldSignFile(''), false)
})

test('azureConfigFromEnv composes the Azure signing config from the environment', () => {
  assert.deepEqual(
    azureConfigFromEnv({
      AZURE_SIGN_ENDPOINT: 'https://cus.codesigning.azure.net',
      AZURE_SIGN_ACCOUNT: 'codesign2',
      AZURE_SIGN_PROFILE: 'hermesagent',
      AZURE_SIGN_PUBLISHER: 'CN=Nous Research Inc.'
    }),
    {
      type: 'azure',
      endpoint: 'https://cus.codesigning.azure.net',
      codeSigningAccountName: 'codesign2',
      certificateProfileName: 'hermesagent',
      publisherName: 'CN=Nous Research Inc.'
    }
  )
  // Missing vars stay undefined — the manager's ctor handles that.
  assert.deepEqual(azureConfigFromEnv({}), {
    type: 'azure',
    endpoint: undefined,
    codeSigningAccountName: undefined,
    certificateProfileName: undefined,
    publisherName: undefined
  })
})

// ─── builder-bump tripwire ──────────────────────────────────────────────────
// sign-msix.mjs deep-imports app-builder-lib's WindowsSignAzureManager by
// file path (the package exports map exposes only "." and "./internal", so
// the dist path is reached the same way run-electron-builder.mjs finds the
// CLI: resolve the entry, walk up to the package root, then direct-file
// import). If an electron-builder/app-builder-lib bump moves or renames the
// class, THIS test is what fails in js-tests — before a release build does.
// 60s timeout: the direct-file import drags in app-builder-lib's whole
// module graph (toolsets, electronGet, vm), which takes ~30s on a cold
// disk cache — far past vitest's 5s default.
test('the real WindowsSignAzureManager resolves and constructs against a packager shim', { timeout: 60_000 }, async () => {
  const entry = require.resolve('app-builder-lib')
  let root = path.dirname(entry)
  while (!fs.existsSync(path.join(root, 'package.json'))) {
    const parent = path.dirname(root)
    assert.notEqual(parent, root, 'app-builder-lib package root not found')
    root = parent
  }
  const mod = await import(
    pathToFileURL(path.join(root, 'dist', 'codeSign', 'win', 'windowsSignAzureManager.js')).href
  )
  assert.equal(typeof mod.WindowsSignAzureManager, 'function')

  // Constructor contract (verified against 27.0.0-alpha.6): reads
  // packager.platformOptions.sign (throws unless type === 'azure'),
  // packager.config.toolsets (optional), and packager.buildResourcesDir
  // (only stored — WineVmManager's constructor touches no filesystem).
  const mgr = new mod.WindowsSignAzureManager({
    platformOptions: {
      sign: {
        type: 'azure',
        endpoint: 'x',
        codeSigningAccountName: 'y',
        certificateProfileName: 'z',
        publisherName: 'p'
      }
    },
    config: {},
    buildResourcesDir: '/tmp'
  })
  assert.equal(typeof mgr.signFile, 'function')
  assert.equal(typeof mgr.initialize, 'function')
})
