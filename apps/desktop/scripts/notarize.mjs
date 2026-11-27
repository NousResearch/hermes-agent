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

// These NSURLError codes mean a request to Apple got no answer: it timed out (-1001), its connection
// dropped (-1005), or the runner could not reach the host (-1003, -1004, -1006, -1009), which points at
// the runner's own network rather than at Apple. None of them is a verdict, so the same request is
// repeated within its time limit.
const LOST_CONTACT = /NSURLErrorDomain\s+Code=-(?:1001|1003|1004|1005|1006|1009)\b/
const CONTACT_DELAYS = [15000, 30000, 60000, 120000]
// Stapling starts after Apple has accepted the app. A runner that still cannot reach Apple's
// ticket service after this long is not worth holding for the rest of the notarization budget.
const STAPLE_CONTACT_WINDOW = 45 * 60 * 1000

function contactDelay(attempt) {
  return CONTACT_DELAYS[Math.min(attempt, CONTACT_DELAYS.length - 1)]
}

function commandOutput(error) {
  return `${error.stdout ?? ''}\n${error.stderr ?? ''}\n${error.message}`
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

export function resolveNotaryAuth(env = process.env) {
  const profile = String(env.APPLE_NOTARY_PROFILE || '').trim()
  if (profile) {
    return { auth: ['--keychain-profile', profile], cleanup: () => {} }
  }
  const keyId = String(env.APPLE_API_KEY_ID || '').trim()
  const issuer = String(env.APPLE_API_ISSUER || '').trim()
  const rawApiKey = env.APPLE_API_KEY
  if (!rawApiKey || !keyId || !issuer) {
    return null
  }
  const key = resolveApiKeyPath(rawApiKey)
  return {
    auth: ['--key', key.keyPath, '--key-id', keyId, '--issuer', issuer],
    cleanup: key.cleanup
  }
}

export async function validateStapled(targetPath, {
  run = runCommand, sleep = setTimeout, log = console.log, now = () => performance.now(), deadline, label = 'notarize'
} = {}) {
  const validateDeadline = Math.min(deadline ?? now() + STAPLE_CONTACT_WINDOW, now() + STAPLE_CONTACT_WINDOW)
  let contactRetries = 0
  for (let attempt = 1; ; attempt++) {
    try {
      log(`[${label}] validating stapled ticket (attempt ${attempt})`)
      await run('xcrun', ['stapler', 'validate', targetPath], { timeout: 120000 })
      return
    } catch (error) {
      const output = commandOutput(error)
      if (!LOST_CONTACT.test(output)) throw error
      const delay = contactDelay(contactRetries++)
      if (now() + delay >= validateDeadline) throw error
      log(`[${label}] lost contact with Apple while validating ${targetPath}. Retrying in ${delay / 1000}s.`)
      await sleep(delay)
    }
  }
}

export const DEFAULT_WAIT_BUDGET_MS = 4 * 60 * 60 * 1000
// The CI Build and package step allows 330 minutes and starts electron-builder in its own process about two minutes
// in. performance.now() counts from that process's start, so packing and signing spend this budget too.
export const COMBINED_NOTARY_BUDGET_MS = 300 * 60 * 1000
export const DMG_VERIFY_RESERVE_MS = 15 * 60 * 1000

export function remainingNotaryBudgetMs({
  now = () => performance.now(),
  totalBudgetMs = COMBINED_NOTARY_BUDGET_MS,
  reserveMs = DMG_VERIFY_RESERVE_MS
} = {}) {
  return Math.max(0, totalBudgetMs - now() - reserveMs)
}

export async function submitAndStaple(submitPath, staplePath, auth, {
  run = runCommand,
  sleep = setTimeout,
  log = console.log,
  now = () => performance.now(),
  label = 'notarize',
  waitBudgetMs = DEFAULT_WAIT_BUDGET_MS
}) {
  log(`[${label}] submitting ${submitPath}. Waiting for Apple's decision.`)
  const deadline = now() + waitBudgetMs
  let resumeId
  let result
  let submitError
  for (let attempt = 0; ; attempt++) {
    const remaining = Math.ceil(deadline - now())
    if (remaining <= 0) {
      throw new Error(`Notarization ${resumeId ?? submitPath} exceeded the wait budget`, { cause: submitError })
    }
    const action = resumeId ? ['wait', resumeId] : ['submit', submitPath, '--wait']
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
      const output = commandOutput(error)
      if (!LOST_CONTACT.test(output)) break
      // The status-request URL identifies the uploaded app, unlike other UUIDs in the error.
      resumeId ??= output.match(/https:\/\/appstoreconnect\.apple\.com\/notary\/v2\/submissions\/([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})(?=[?\s,}]|$)/i)?.[1]
      const delay = contactDelay(attempt)
      if (!resumeId || now() + delay >= deadline) break
      log(`[${label}] lost contact with Apple while waiting for ${resumeId}. Resuming wait in ${delay / 1000}s.`)
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
  log(`[${label}] submission ${id}: ${status}`)
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
  const ticketRetryLimit = 5
  const stapleDeadline = Math.min(deadline, now() + STAPLE_CONTACT_WINDOW)
  let ticketRetries = 0
  let contactRetries = 0
  for (let attempt = 1; ; attempt++) {
    try {
      log(`[${label}] stapling attempt ${attempt}`)
      await run('xcrun', ['stapler', 'staple', '-v', staplePath], { timeout: 120000 })
      log(`[${label}] ticket stapled for submission ${id}`)
      return { id, deadline }
    } catch (error) {
      const output = commandOutput(error)
      if (error.code === 65 && /CloudKit query[^\n]*Record not found/i.test(output)) {
        if (ticketRetries === ticketRetryLimit) throw error
        const delay = contactDelay(ticketRetries++)
        log(`[${label}] accepted ticket not available. Retrying in ${delay / 1000}s.`)
        await sleep(delay)
        continue
      }
      if (!LOST_CONTACT.test(output)) throw error
      const delay = contactDelay(contactRetries++)
      if (now() + delay >= stapleDeadline) throw error
      log(`[${label}] lost contact with Apple while stapling ${id}. Retrying in ${delay / 1000}s.`)
      await sleep(delay)
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

  const resolved = resolveNotaryAuth(env)
  if (!resolved) {
    log(
      'Skipping notarization: APPLE_API_KEY, APPLE_API_KEY_ID, and APPLE_API_ISSUER are not fully configured.'
    )
    return
  }
  const { auth, cleanup } = resolved

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
