import assert from 'node:assert/strict'

import { test } from 'vitest'

import { scrapeReadyPort, spawnLogPath } from './remote-lifecycle'

const OWNERSHIP_ID = '0123456789abcdef0123456789abcdef'
const SPAWN_NONCE = '0123456789abcdef'

function logSsh(log: string) {
  return {
    async exec(cmd: string) {
      return cmd.startsWith('cat ') ? log : ''
    }
  }
}

test('scrapeReadyPort explains a missing remote profile', async () => {
  const ssh = logSsh("Error: Profile 'operator' does not exist. Create it with: hermes profile create operator\n")

  await assert.rejects(
    () =>
      scrapeReadyPort(ssh, spawnLogPath(OWNERSHIP_ID, SPAWN_NONCE), {
        timeoutMs: 1000,
        isAlive: async () => false
      }),
    (err: any) => {
      assert.equal(err.kind, 'remote-profile-missing')
      assert.equal(err.profile, 'operator')
      assert.match(err.message, /remote Hermes profile 'operator' does not exist/)
      assert.match(err.message, /hermes profile create operator/)

      return true
    }
  )
})

test('scrapeReadyPort keeps the generic spawn failure for other exits', async () => {
  const ssh = logSsh('Traceback (most recent call last):\nRuntimeError: boom\n')

  await assert.rejects(
    () =>
      scrapeReadyPort(ssh, spawnLogPath(OWNERSHIP_ID, SPAWN_NONCE), {
        timeoutMs: 1000,
        isAlive: async () => false
      }),
    (err: any) => {
      assert.equal(err.kind, 'spawn-failed')

      return true
    }
  )
})
