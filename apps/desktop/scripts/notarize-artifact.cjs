const fs = require('node:fs')
const os = require('node:os')
const path = require('node:path')
const { execFile } = require('node:child_process')

function run(command, args) {
  return new Promise((resolve, reject) => {
    execFile(command, args, (error, stdout, stderr) => {
      if (error) {
        // Intentionally omit args from the rejection message: callers pass
        // notarization credentials (key id, issuer, key file path) here, and
        // surfacing them in error output would land in CI logs.
        reject(new Error(`${command} failed: ${stderr?.trim() || stdout?.trim() || error.message}`))
        return
      }
      resolve()
    })
  })
}

function inlineKeyLooksValid(value) {
  return value.includes('BEGIN PRIVATE KEY') && value.includes('END PRIVATE KEY')
}

function resolveApiKeyPath(rawValue) {
  const value = String(rawValue || '').trim()
  if (!value) return { keyPath: '', cleanup: () => {} }

  if (fs.existsSync(value)) {
    return { keyPath: value, cleanup: () => {} }
  }

  if (!inlineKeyLooksValid(value)) {
    throw new Error('APPLE_API_KEY must be a file path or inline .p8 key content')
  }

  const tempPath = path.join(os.tmpdir(), `hermes-notary-${Date.now()}-${process.pid}.p8`)
  fs.writeFileSync(tempPath, value, 'utf8')
  return {
    keyPath: tempPath,
    cleanup: () => fs.rmSync(tempPath, { force: true })
  }
}

async function main() {
  const artifactPath = process.argv[2]
  if (!artifactPath || !fs.existsSync(artifactPath)) {
    throw new Error(`Missing artifact to notarize: ${artifactPath || '(none)'}`)
  }

  const profile = String(process.env.APPLE_NOTARY_PROFILE || '').trim()
  if (profile) {
    await run('xcrun', ['notarytool', 'submit', artifactPath, '--keychain-profile', profile, '--wait'])
    await run('xcrun', ['stapler', 'staple', '-v', artifactPath])
    return
  }

  const keyId = String(process.env.APPLE_API_KEY_ID || '').trim()
  const issuer = String(process.env.APPLE_API_ISSUER || '').trim()
  const rawApiKey = process.env.APPLE_API_KEY
  if (!rawApiKey || !keyId || !issuer) {
    throw new Error('APPLE_API_KEY, APPLE_API_KEY_ID, and APPLE_API_ISSUER are required')
  }

  const { keyPath, cleanup } = resolveApiKeyPath(rawApiKey)
  try {
    await run('xcrun', ['notarytool', 'submit', artifactPath, '--key', keyPath, '--key-id', keyId, '--issuer', issuer, '--wait'])
    await run('xcrun', ['stapler', 'staple', '-v', artifactPath])
  } finally {
    cleanup()
  }
}

main().catch(() => {
  console.error('Notarization failed. Check configuration and command output in secure CI logs.')
  process.exit(1)
})
