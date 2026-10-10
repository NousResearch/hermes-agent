import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { test } from 'vitest'

import { activityHomes, formatElapsed, readBackgroundWork } from './background-work-status'

test('activityHomes aggregates the default and named profile roots', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-work-status-'))
  const named = path.join(root, 'profiles', 'coder')
  fs.mkdirSync(named, { recursive: true })
  assert.deepEqual(activityHomes(named).sort(), [root, named].sort())
  fs.rmSync(root, { recursive: true, force: true })
})

test('readBackgroundWork returns live leases and removes dead ones', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-work-status-'))
  const leases = path.join(root, 'cache', 'background-work')
  fs.mkdirSync(leases, { recursive: true })
  fs.writeFileSync(path.join(leases, 'live.json'), JSON.stringify({
    key: 'live', title: 'Safe task', worker: 'kanban', profile: 'coder', model: 'gpt',
    provider: 'openai-codex', pid: process.pid, started_at: 100, state: 'running'
  }))
  fs.writeFileSync(path.join(leases, 'dead.json'), JSON.stringify({
    key: 'dead', title: 'Old task', worker: 'kanban', profile: 'coder', model: 'gpt',
    pid: 99999999, started_at: 100, state: 'running'
  }))

  const items = readBackgroundWork(root, 165)

  assert.deepEqual(items.map(item => item.key), ['live'])
  assert.equal(items[0].provider, 'openai-codex')
  assert.equal(items[0].state, 'running')
  assert.equal(fs.existsSync(path.join(leases, 'dead.json')), false)
  fs.rmSync(root, { recursive: true, force: true })
})

test('readBackgroundWork orders concurrent workers deterministically', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-work-status-'))
  const leases = path.join(root, 'cache', 'background-work')
  fs.mkdirSync(leases, { recursive: true })

  for (const key of ['charlie', 'alpha', 'bravo']) {
    fs.writeFileSync(path.join(leases, `${key}.json`), JSON.stringify({
      key, title: `${key} card`, worker: 'kanban', profile: 'coder', model: 'm',
      pid: process.pid, started_at: 200, state: 'running'
    }))
  }

  for (let attempt = 0; attempt < 3; attempt += 1) {
    assert.deepEqual(readBackgroundWork(root, 300).map(item => item.key), ['alpha', 'bravo', 'charlie'])
  }

  fs.rmSync(root, { recursive: true, force: true })
})

test('formatElapsed stays compact', () => {
  assert.equal(formatElapsed(0), '0s')
  assert.equal(formatElapsed(65), '1m 05s')
  assert.equal(formatElapsed(3661), '1h 01m')
})
