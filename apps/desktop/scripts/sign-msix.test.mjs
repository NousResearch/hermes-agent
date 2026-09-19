import assert from 'node:assert/strict'
import fs from 'node:fs'
import { createRequire } from 'node:module'
import path from 'node:path'
import os from 'node:os'
import { pathToFileURL } from 'node:url'
import { test, vi } from 'vitest'

import { azureConfigFromEnv, azureSignFile, shouldSignFile } from './sign-msix.mjs'

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

test('azureSignFile calls the real supplier through a prepared local toolset', async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'sign-supplier-'))
  const lib = path.dirname(path.dirname(require.resolve('app-builder-lib')))
  const { VmManager } = await import(pathToFileURL(path.join(lib, 'dist/vm/vm.js')).href)
  const { WineVmManager } = await import(pathToFileURL(path.join(lib, 'dist/vm/WineVm.js')).href)
  const native = process.platform === 'win32' ? VmManager : WineVmManager
  const calls = []
  // Only the final native execution is intercepted; manager init, dispatch and metadata are real.
  const execution = vi.spyOn(native.prototype, 'exec').mockImplementation(async (file, args) => { calls.push({ file, args }) })
  const kits = path.join(root, 'kits')
  const kit = path.join(kits, process.arch === 'ia32' ? 'x86' : 'x64')
  fs.mkdirSync(kit, { recursive: true })
  const metadata = path.join(root, 'metadata.json')
  for (const [name, value] of Object.entries({ AZURE_SIGN_ENDPOINT: 'https://test.invalid', AZURE_SIGN_ACCOUNT: 'account', AZURE_SIGN_PROFILE: 'profile', AZURE_SIGN_PUBLISHER: 'CN=Fixture' })) vi.stubEnv(name, value)
  try {
    await azureSignFile(path.join(root, 'fixture.msix'), {
      platformOptions: { sign: { type: 'signtool' } },
      config: { toolsets: { winCodeSign: { url: `file://${kits}` } } },
      buildResourcesDir: root,
      getTempFile: async () => metadata
    })
    assert.deepEqual(JSON.parse(fs.readFileSync(metadata, 'utf8')), { Endpoint: 'https://test.invalid', CodeSigningAccountName: 'account', CertificateProfileName: 'profile' })
    assert.equal(calls.length, 1)
    assert.equal(calls[0].file, path.join(kit, 'signtool.exe'))
    assert.deepEqual(calls[0].args.slice(0, 7), ['sign', '/fd', 'SHA256', '/tr', 'http://timestamp.acs.microsoft.com', '/td', 'SHA256'])
    assert.ok(calls[0].args[calls[0].args.indexOf('/dlib') + 1].endsWith('Azure.CodeSigning.Dlib.dll'))
    assert.ok(calls[0].args.at(-1).endsWith('fixture.msix'))
  } finally {
    execution.mockRestore(); vi.unstubAllEnvs()
    fs.rmSync(root, { recursive: true, force: true })
  }
})
