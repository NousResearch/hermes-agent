import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { test } from 'vitest'

import { acknowledgeUpdateRelaunch, parseUpdateRelaunchAckRequest } from './windows-update-relaunch'

test('accepts only a request-bound ACK path below HERMES_HOME logs', () => {
  const home = path.join('C:\\Users\\damian', 'hermes')
  const ack = path.join(home, 'logs', 'desktop-relaunch-abc.json')
  const id = 'a'.repeat(32)

  assert.deepEqual(
    parseUpdateRelaunchAckRequest(
      [`--hermes-update-relaunch-ack=${ack}`, `--hermes-update-relaunch-request=${id}`],
      home
    ),
    { ackPath: path.resolve(ack), requestId: id }
  )
  assert.equal(
    parseUpdateRelaunchAckRequest(
      ['--hermes-update-relaunch-ack=C:\\Temp\\owned.json', `--hermes-update-relaunch-request=${id}`],
      home
    ),
    null
  )
  assert.equal(parseUpdateRelaunchAckRequest([`--hermes-update-relaunch-ack=${ack}`], home), null)
})

test('writes the cold-start PID ACK for the updater', () => {
  const home = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-relaunch-'))
  const ack = path.join(home, 'logs', 'desktop-relaunch.json')
  const id = 'b'.repeat(32)

  assert.equal(
    acknowledgeUpdateRelaunch(
      [`--hermes-update-relaunch-ack=${ack}`, `--hermes-update-relaunch-request=${id}`],
      home,
      4242
    ),
    true
  )
  assert.deepEqual(JSON.parse(fs.readFileSync(ack, 'utf8')), {
    ok: true,
    pid: 4242,
    requestId: id,
    readyAt: JSON.parse(fs.readFileSync(ack, 'utf8')).readyAt
  })
})
