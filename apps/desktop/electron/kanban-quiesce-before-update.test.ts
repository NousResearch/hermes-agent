import { strict as assert } from 'node:assert'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { test } from 'vitest'

import { quiesceKanbanWorkersForUpdate } from './kanban-quiesce-before-update'
import { markerPath, removeUpdateMarkerIfOwned, writeUpdateMarker } from './update-marker'

test('update marker ownership brackets the bounded kanban quiesce subprocess', () => {
  const home = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-kanban-update-'))
  const root = path.join(home, 'hermes-agent')
  const python = path.join(root, 'venv', 'Scripts', 'python.exe')
  fs.mkdirSync(path.dirname(python), { recursive: true })
  fs.writeFileSync(python, '')

  try {
    writeUpdateMarker(home, 4242, { kill: (() => true) as typeof process.kill })
    const calls: Array<{ command: string; args: string[]; options: any }> = []

    const result = quiesceKanbanWorkersForUpdate(root, home, {
      isWindows: true,
      execFileSync: ((command, args, options) => {
        calls.push({ command, args, options })

        return JSON.stringify({
          ok: true,
          reclaimed: [{ board: 'default', task_id: 't_deadbeef' }],
          failed: []
        })
      }) as any
    })

    assert.equal(result.ok, true)
    assert.deepEqual(result.reclaimed, [{ board: 'default', task_id: 't_deadbeef' }])
    assert.equal(calls.length, 1)
    assert.equal(calls[0].command, python)
    assert.deepEqual(calls[0].args, ['-m', 'hermes_cli.kanban_update_coordination'])
    assert.equal(calls[0].options.env.HERMES_HOME, home)
    assert.equal(removeUpdateMarkerIfOwned(home, 7), false)
    assert.equal(fs.existsSync(markerPath(home)), true)
    assert.equal(removeUpdateMarkerIfOwned(home, 4242), true)
    assert.equal(fs.existsSync(markerPath(home)), false)
  } finally {
    fs.rmSync(home, { recursive: true, force: true })
  }
})

test('nonzero quiesce exit preserves structured blocker diagnostics from stdout', () => {
  const root = path.join(os.tmpdir(), 'hermes-kanban-update-root')
  const python = path.join(root, 'venv', 'Scripts', 'python.exe')

  const failure = {
    ok: false,
    reclaimed: [{ board: 'first', task_id: 't_reclaimed' }],
    failed: [{ board: 'second', task_id: 't_blocked', error: 'dispatch lock busy' }]
  }

  const error = Object.assign(new Error('Command failed'), {
    stdout: Buffer.from(JSON.stringify(failure), 'utf8')
  })

  const result = quiesceKanbanWorkersForUpdate(root, 'C:\\hermes', {
    isWindows: true,
    existsSync: candidate => candidate === python,
    execFileSync: (() => {
      throw error
    }) as any
  })

  assert.deepEqual(result, failure)
})
