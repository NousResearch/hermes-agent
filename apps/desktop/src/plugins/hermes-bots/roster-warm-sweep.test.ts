import assert from 'node:assert/strict'

import { test } from 'vitest'

import { warmRosterBackends } from './roster-pane-lifecycle'
import type { RosterRow } from './types'

const row = (over: Partial<RosterRow> & { name: string }): RosterRow =>
  ({ ...over }) as RosterRow

test('the roster sweep warms every local bot row through the one warm resolver', () => {
  const warmed: string[] = []

  warmRosterBackends(
    [row({ name: 'atlas' }), row({ name: 'dev' }), row({ name: 'agrovoz' })],
    name => warmed.push(name)
  )

  assert.deepEqual(warmed, ['atlas', 'dev', 'agrovoz'])
})

test('ghost, remote, connection-scoped and unnamed rows are skipped', () => {
  const warmed: string[] = []

  warmRosterBackends(
    [
      row({ name: 'ghosted', ghost: true }),
      row({ name: 'remote-bot', remoteSource: true }),
      row({ name: 'scoped-bot', connectionId: 'conn-1' }),
      row({ name: '' }),
      // null/undefined rows must not crash the sweep.
      null as unknown as RosterRow,
      row({ name: 'local-bot' })
    ],
    name => warmed.push(name)
  )

  assert.deepEqual(warmed, ['local-bot'])
})

test('a warm failure never breaks the sweep or the roster render', () => {
  const warmed: string[] = []

  warmRosterBackends(
    [row({ name: 'first' }), row({ name: 'second' }), row({ name: 'third' })],
    name => {
      if (name === 'second') {
        throw new Error('warm exploded')
      }

      warmed.push(name)
    }
  )

  assert.deepEqual(warmed, ['first', 'third'])
})
