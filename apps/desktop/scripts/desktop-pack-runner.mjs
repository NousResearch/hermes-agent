import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { randomUUID } from 'node:crypto'
import { spawnSync } from 'node:child_process'
import { pathToFileURL } from 'node:url'
import { PACK_SESSION_ENV, PACK_JOURNAL_ENV, settleDesktopPack } from './desktop-pack-transaction.mjs'

function settleJournal(journal, sessionId, builderSucceeded) {
  if (!sessionId || !journal || !path.isAbsolute(journal)) throw new Error('Invalid packaging recovery identity')
  const targets = fs.readFileSync(journal, 'utf8').split('\n').filter(Boolean).map(JSON.parse)
  return settleDesktopPack({ targets, sessionId, builderSucceeded })
}

// The installer can recover after it has terminated a timed-out builder job.
// It owns the supplied journal and removes it after this process returns.
export function recoverDesktopBuilder(journal, sessionId) {
  return settleJournal(journal, sessionId, false)
}

// beforePack supplies the actual output paths, including custom output roots
// and every architecture. Guessing release/win-unpacked would miss staged builds.
export function runDesktopBuilder(executable, args, { env = process.env, spawn = spawnSync } = {}) {
  const callerSession = env[PACK_SESSION_ENV]
  const callerJournal = env[PACK_JOURNAL_ENV]
  if (Boolean(callerSession) !== Boolean(callerJournal) || (callerJournal && !path.isAbsolute(callerJournal))) {
    return { error: new Error('Packaging session and absolute journal must be supplied together'), status: 1 }
  }
  const root = callerJournal ? undefined : fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-pack-'))
  const journal = callerJournal || path.join(root, 'targets.jsonl')
  const sessionId = callerSession || randomUUID()
  let result
  let settlement
  try {
    if (root) fs.writeFileSync(journal, '')
    else if (!fs.statSync(journal).isFile()) throw new Error('Packaging journal must be a file')
    try {
      result = spawn(executable, args, {
        stdio: 'inherit',
        env: { ...env, [PACK_SESSION_ENV]: sessionId, [PACK_JOURNAL_ENV]: journal }
      })
    } catch (error) {
      result = { error, status: null }
    }
    settlement = settleJournal(journal, sessionId, !result.error && result.status === 0)
    return {
      ...result,
      status: result.error || !settlement.ok || result.status == null ? 1 : result.status,
      settlement
    }
  } catch (error) {
    // Unknown/incomplete ownership records cannot authorize destructive recovery.
    return { error, status: 1, settlement }
  } finally {
    if (root) fs.rmSync(root, { recursive: true, force: true })
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  try {
    if (process.argv.length !== 5 || process.argv[2] !== '--recover') throw new Error('Expected --recover <journal> <session>')
    const result = recoverDesktopBuilder(process.argv[3], process.argv[4])
    for (const failure of result.failures) console.error(`[desktop-pack-recovery] ${failure.reason}`)
    process.exitCode = result.ok ? 0 : 1
  } catch (error) {
    console.error(`[desktop-pack-recovery] ${error.message}`)
    process.exitCode = 1
  }
}
