import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { execFile } from 'node:child_process'
import { setTimeout } from 'node:timers/promises'

export function runCommand(command, args, options = {}) {
  return new Promise((resolve, reject) => {
    execFile(command, args, options, (error, stdout, stderr) => {
      if (error) {
        reject(
          Object.assign(new Error(
            `${command} ${args.join(' ')} failed: ${stderr?.trim() || stdout?.trim() || error.message}`,
            { cause: error }
          ), { code: error.code, stdout, stderr })
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

async function submitAndStaple(zipPath, appPath, auth, { run, sleep, log, now }) {
  log(`[notarize] submitting ${zipPath}. Waiting for Apple's decision.`)
  const deadline = now() + 4 * 60 * 60 * 1000
  const waitDelays = [15000, 30000, 60000]
  let resumeId
  let result
  let submitError
  for (let attempt = 0; ; attempt++) {
    const remaining = Math.ceil(deadline - now())
    if (remaining <= 0) {
      throw new Error(`Notarization ${resumeId ?? zipPath} exceeded the wait budget`, { cause: submitError })
    }
    const action = resumeId ? ['wait', resumeId] : ['submit', zipPath, '--wait']
    try {
      result = await run('xcrun', [
        'notarytool', ...action, ...auth, '--timeout', `${Math.ceil(remaining / 1000)}s`, '--output-format', 'json'
      ], { timeout: remaining + 60000 })
      submitError = undefined
      break
    } catch (error) {
      // A rejected submission can still return a JSON result and a log ID.
      result = error
      submitError = error
      const output = `${error.stdout ?? ''}\n${error.stderr ?? ''}\n${error.message}`
      if (!/NSURLErrorDomain\s+Code=-1001\b/.test(output)) break
      // The status-request URL identifies the uploaded app, unlike other UUIDs in the error.
      resumeId ??= output.match(/https:\/\/appstoreconnect\.apple\.com\/notary\/v2\/submissions\/([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})(?=[?\s,}]|$)/i)?.[1]
      const delay = waitDelays[attempt]
      if (!resumeId || delay === undefined || now() + delay >= deadline) break
      log(`[notarize] request timed out for ${resumeId}. Resuming wait in ${delay / 1000}s.`)
      await sleep(delay)
    }
  }
  let submission
  try {
    submission = JSON.parse(result.stdout)
  } catch {
    throw submitError ?? new Error(`Invalid notarytool response: ${result.stdout}`)
  }
  const { id, status } = submission ?? {}
  if (typeof id !== 'string' || !id || typeof status !== 'string') {
    throw submitError ?? new Error(`Incomplete notarytool response: ${result.stdout}`)
  }
  if (resumeId && id !== resumeId) {
    throw new Error(`notarytool returned submission ${id} while waiting for ${resumeId}`)
  }
  log(`[notarize] submission ${id}: ${status}`)
  if (submitError || status !== 'Accepted') {
    let diagnostics
    try {
      const result = await run('xcrun', ['notarytool', 'log', id, ...auth], { timeout: 120000 })
      diagnostics = result.stdout
    } catch (error) {
      diagnostics = `Could not retrieve Apple's diagnostic log: ${error.message}`
    }
    throw new Error(`Notarization ${id}: ${status}\n${diagnostics}`, { cause: submitError })
  }

  // Accepted tickets can take time to reach Apple's CloudKit lookup service.
  const delays = [15000, 30000, 60000, 120000, 120000]
  for (let attempt = 0; ; attempt++) {
    try {
      log(`[notarize] stapling attempt ${attempt + 1}/${delays.length + 1}`)
      await run('xcrun', ['stapler', 'staple', '-v', appPath], { timeout: 120000 })
      log(`[notarize] ticket stapled for submission ${id}`)
      return
    } catch (error) {
      const output = `${error.stdout ?? ''}\n${error.stderr ?? ''}\n${error.message}`
      const missingTicket = error.code === 65 && /CloudKit query[^\n]*Record not found/i.test(output)
      if (!missingTicket || attempt === delays.length) throw error
      log(`[notarize] accepted ticket not available. Retrying in ${delays[attempt] / 1000}s.`)
      await sleep(delays[attempt])
    }
  }
}

export default async function notarize(context, {
  run = runCommand, env = process.env, sleep = setTimeout, log = console.log, now = () => performance.now()
} = {}) {
  const { electronPlatformName, appOutDir, packager } = context
  if (electronPlatformName !== 'darwin') return

  const appName = packager.appInfo.productFilename
  const appPath = path.join(appOutDir, `${appName}.app`)
  if (!fs.existsSync(appPath)) {
    throw new Error(`Cannot notarize missing app bundle: ${appPath}`)
  }

  const profile = String(env.APPLE_NOTARY_PROFILE || '').trim()
  let auth
  let cleanup = () => {}
  if (profile) {
    auth = ['--keychain-profile', profile]
  } else {
    const keyId = String(env.APPLE_API_KEY_ID || '').trim()
    const issuer = String(env.APPLE_API_ISSUER || '').trim()
    const rawApiKey = env.APPLE_API_KEY
    if (!rawApiKey || !keyId || !issuer) {
      log(
        'Skipping notarization: APPLE_API_KEY, APPLE_API_KEY_ID, and APPLE_API_ISSUER are not fully configured.'
      )
      return
    }
    const key = resolveApiKeyPath(rawApiKey)
    auth = ['--key', key.keyPath, '--key-id', keyId, '--issuer', issuer]
    cleanup = key.cleanup
  }

  const zipPath = path.join(appOutDir, `${appName}.zip`)
  try {
    await run('ditto', ['-c', '-k', '--sequesterRsrc', '--keepParent', appPath, zipPath])
    await submitAndStaple(zipPath, appPath, auth, { run, sleep, log, now })
  } finally {
    try {
      fs.rmSync(zipPath, { force: true })
    } catch {
      // Best-effort cleanup.
    }
    cleanup()
  }
}
