import assert from 'node:assert/strict'

import { test } from 'vitest'

import { resolveTerminalConnection } from './connection-apply'

const ssh = {
  host: 'registry-box.test',
  user: 'hermes'
}

test('terminal start preserves the selected SSH target and scope', async () => {
  const target = {
    ssh,
    scope: 'connection:registry-ssh:profile:worker'
  }

  const resolved = await resolveTerminalConnection(
    () => target,
    async () => {
      throw new Error('backend fallback must not run for an active SSH target')
    }
  )

  assert.equal(resolved, target)
  assert.equal(resolved?.ssh, ssh)
  assert.equal(resolved?.scope, 'connection:registry-ssh:profile:worker')
})

test('terminal start does not invent SSH when canonical routing selects local or remote HTTP', async () => {
  let backendChecks = 0

  const resolved = await resolveTerminalConnection(
    () => null,
    async () => {
      backendChecks += 1
    }
  )

  assert.equal(resolved, null)
  assert.equal(backendChecks, 0)
})

test('terminal start re-reads the SSH target after backend startup', async () => {
  const target = {
    ssh,
    scope: 'connection:registry-ssh'
  }
  let ready = false

  const resolved = await resolveTerminalConnection(
    () => (ready ? target : 'pending'),
    async () => {
      ready = true
    }
  )

  assert.equal(resolved, target)
  assert.equal(resolved?.scope, 'connection:registry-ssh')
})
