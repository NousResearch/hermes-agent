const fs = require('node:fs')
const os = require('node:os')
const path = require('node:path')
const { execFile } = require('node:child_process')

const { signAndVerify } = require('./codesign-mac.cjs')
const { build: BUILD_CONFIG } = require('../package.json')

function run(command, args) {
  return new Promise((resolve, reject) => {
    execFile(command, args, (error, stdout, stderr) => {
      if (error) {
        reject(
          new Error(
            `${command} ${args.join(' ')} failed: ${stderr?.trim() || stdout?.trim() || error.message}`
          )
        )
        return
      }
      resolve({ stdout, stderr })
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
    cleanup: () => {
      try {
        fs.rmSync(tempPath, { force: true })
      } catch {
        // Best-effort cleanup.
      }
    }
  }
}

exports.default = async function notarize(context) {
  const { electronPlatformName, appOutDir, packager } = context
  if (electronPlatformName !== 'darwin') return

  const appName = packager.appInfo.productFilename
  const appPath = path.join(appOutDir, `${appName}.app`)
  if (!fs.existsSync(appPath)) {
    throw new Error(`Cannot notarize missing app bundle: ${appPath}`)
  }

  // BUILD-266: electron-builder silently skips macOS code signing when no
  // Developer ID identity is configured (no CSC_LINK/CSC_NAME and no
  // matching keychain identity, as on a bare dev machine) -- it leaves
  // whatever stale signature the Electron binary shipped with in place. That
  // shipped straight to /Applications on 2026-07-01 with
  // Contents/_CodeSignature/ absent and the generic Electron linker-signed
  // stub identifier instead of com.nousresearch.hermes, and Gatekeeper
  // rejected the app as "damaged". This afterSign hook previously only ran
  // real notarization and did nothing when notarization creds were absent,
  // so the badly-signed bundle sailed through with no error.
  //
  // signAndVerify closes that gap unconditionally, for every packaged build
  // (`npm run pack`, `dist:mac*`, CI, or a dev's manual build) regardless of
  // whether notarization credentials are configured: it ad-hoc signs the
  // bundle only if it doesn't already carry a valid, non-stub signature
  // (never clobbering a real Developer ID signature), then hard-verifies
  // the result. A verification failure throws here, which aborts the
  // electron-builder invocation -- a badly-signed .app can never become a
  // build artifact, let alone get copied into /Applications.
  await signAndVerify(appPath, { expectedIdentifier: BUILD_CONFIG.appId })

  const profile = String(process.env.APPLE_NOTARY_PROFILE || '').trim()
  if (profile) {
    const zipPath = path.join(appOutDir, `${appName}.zip`)
    await run('ditto', ['-c', '-k', '--sequesterRsrc', '--keepParent', appPath, zipPath])
    await run('xcrun', ['notarytool', 'submit', zipPath, '--keychain-profile', profile, '--wait'])
    await run('xcrun', ['stapler', 'staple', '-v', appPath])
    try {
      fs.rmSync(zipPath, { force: true })
    } catch {
      // Best-effort cleanup.
    }
    return
  }

  const keyId = String(process.env.APPLE_API_KEY_ID || '').trim()
  const issuer = String(process.env.APPLE_API_ISSUER || '').trim()
  const rawApiKey = process.env.APPLE_API_KEY
  if (!rawApiKey || !keyId || !issuer) {
    console.log(
      'Skipping notarization: APPLE_API_KEY, APPLE_API_KEY_ID, and APPLE_API_ISSUER are not fully configured.'
    )
    return
  }

  const { keyPath, cleanup } = resolveApiKeyPath(rawApiKey)
  const zipPath = path.join(appOutDir, `${appName}.zip`)
  try {
    await run('ditto', ['-c', '-k', '--sequesterRsrc', '--keepParent', appPath, zipPath])
    await run('xcrun', ['notarytool', 'submit', zipPath, '--key', keyPath, '--key-id', keyId, '--issuer', issuer, '--wait'])
    await run('xcrun', ['stapler', 'staple', '-v', appPath])
  } finally {
    try {
      fs.rmSync(zipPath, { force: true })
    } catch {
      // Best-effort cleanup.
    }
    cleanup()
  }
}
