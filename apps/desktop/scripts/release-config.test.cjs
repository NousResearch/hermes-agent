const assert = require('node:assert/strict')
const path = require('node:path')
const test = require('node:test')

const configPath = path.resolve(__dirname, '..', 'electron-builder.release.cjs')
const requiredEnvironment = {
  AZURE_TRUSTED_SIGNING_ENDPOINT: 'https://example.codesigning.azure.net/',
  AZURE_TRUSTED_SIGNING_ACCOUNT_NAME: 'nous-signing',
  AZURE_TRUSTED_SIGNING_CERTIFICATE_PROFILE_NAME: 'hermes-release',
  WINDOWS_SIGNING_PUBLISHER_NAME: 'Nous Research'
}

test('release config enables mandatory Azure signing without changing local config', () => {
  Object.assign(process.env, requiredEnvironment)
  delete require.cache[configPath]
  const config = require(configPath)

  assert.equal(config.forceCodeSigning, true)
  assert.equal(config.win.signAndEditExecutable, true)
  assert.deepEqual(config.win.azureSignOptions, {
    endpoint: requiredEnvironment.AZURE_TRUSTED_SIGNING_ENDPOINT,
    codeSigningAccountName: requiredEnvironment.AZURE_TRUSTED_SIGNING_ACCOUNT_NAME,
    certificateProfileName: requiredEnvironment.AZURE_TRUSTED_SIGNING_CERTIFICATE_PROFILE_NAME,
    publisherName: requiredEnvironment.WINDOWS_SIGNING_PUBLISHER_NAME,
    fileDigest: 'SHA256',
    timestampDigest: 'SHA256',
    timestampRfc3161: 'http://timestamp.acs.microsoft.com'
  })

  const packageJson = require('../package.json')
  assert.equal(packageJson.build.win.signAndEditExecutable, false)
  assert.equal(packageJson.build.win.azureSignOptions, undefined)
})

test('release config rejects a missing signing variable', () => {
  const previous = process.env.WINDOWS_SIGNING_PUBLISHER_NAME
  delete process.env.WINDOWS_SIGNING_PUBLISHER_NAME
  delete require.cache[configPath]

  assert.throws(() => require(configPath), /requires WINDOWS_SIGNING_PUBLISHER_NAME/)

  process.env.WINDOWS_SIGNING_PUBLISHER_NAME = previous
})
