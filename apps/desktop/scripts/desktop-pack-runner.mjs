import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { randomUUID } from 'node:crypto'
import { spawnSync } from 'node:child_process'
import { PACK_SESSION_ENV, PACK_JOURNAL_ENV, settleDesktopPack } from './desktop-pack-transaction.mjs'

// beforePack supplies the actual output paths, including custom output roots
// and every architecture. Guessing release/win-unpacked would miss staged builds.
export function runDesktopBuilder(executable, args, { env = process.env, spawn = spawnSync } = {}) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-pack-'))
  const journal = path.join(root, 'targets.jsonl')
  const sessionId = randomUUID()
  let result
  let settlement
  try {
    fs.writeFileSync(journal, '')
    try {
      result = spawn(executable, args, {
        stdio: 'inherit',
        env: { ...env, [PACK_SESSION_ENV]: sessionId, [PACK_JOURNAL_ENV]: journal }
      })
    } catch (error) {
      result = { error, status: null }
    }
    const targets = fs.readFileSync(journal, 'utf8').split('\n').filter(Boolean).map(JSON.parse)
    settlement = settleDesktopPack({ targets, sessionId, builderSucceeded: !result.error && result.status === 0 })
    return {
      ...result,
      status: result.error || !settlement.ok || result.status == null ? 1 : result.status,
      settlement
    }
  } catch (error) {
    // Unknown/incomplete ownership records cannot authorize destructive recovery.
    return { error, status: 1, settlement }
  } finally {
    fs.rmSync(root, { recursive: true, force: true })
  }
}
