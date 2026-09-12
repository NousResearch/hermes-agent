const fs = require('node:fs')
const path = require('node:path')

const packageJson = JSON.parse(fs.readFileSync(path.join(__dirname, 'package.json'), 'utf8'))

const releaseVariables = {
  endpoint: 'AZURE_TRUSTED_SIGNING_ENDPOINT',
  codeSigningAccountName: 'AZURE_TRUSTED_SIGNING_ACCOUNT_NAME',
  certificateProfileName: 'AZURE_TRUSTED_SIGNING_CERTIFICATE_PROFILE_NAME',
  publisherName: 'WINDOWS_SIGNING_PUBLISHER_NAME'
}

const azureSignOptions = Object.fromEntries(
  Object.entries(releaseVariables).map(([property, environmentName]) => {
    const value = process.env[environmentName]?.trim()
    if (!value) {
      throw new Error(`Official Windows release signing requires ${environmentName}`)
    }
    return [property, value]
  })
)

module.exports = {
  ...packageJson.build,
  forceCodeSigning: true,
  win: {
    ...packageJson.build.win,
    signAndEditExecutable: true,
    azureSignOptions: {
      ...azureSignOptions,
      fileDigest: 'SHA256',
      timestampDigest: 'SHA256',
      timestampRfc3161: 'http://timestamp.acs.microsoft.com'
    }
  },
  nsis: {
    ...packageJson.build.nsis,
    warningsAsErrors: true
  }
}
