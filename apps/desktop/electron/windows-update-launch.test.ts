/**
 * Tests for electron/windows-update-launch.ts — the one-shot scheduled-task
 * launch that gets the Windows updater OUT of the desktop's Job Object.
 *
 * What this locks:
 *   1. The generated PowerShell launcher embeds env, updater path/args, ACK
 *      path and request id with correct single-quote escaping, guards against
 *      duplicate firings, and self-deletes the task.
 *   2. schtasks argv shape: forced one-shot creation plus explicit /Run.
 *   3. ACK parsing is bound to the request id and fail-closed on garbage.
 *   4. The full launch flow ACKs the updater PID (happy path), surfaces
 *      launcher-reported failures, and times out instead of hanging.
 */

import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { test } from 'vitest'

import {
  buildUpdateLaunchScript,
  launchUpdaterViaScheduledTask,
  nextStartTime,
  parseLaunchAck,
  schtasksCreateArgs,
  schtasksDeleteArgs,
  schtasksEndArgs,
  schtasksRunArgs,
  UPDATE_LAUNCH_TASK_NAME,
  updateLaunchPaths
} from './windows-update-launch'

const SCRIPT_ARGS = {
  updaterPath: 'C:\\Users\\o\'brien\\AppData\\Local\\hermes\\hermes-setup.exe',
  updaterArgs: ['--update', '--branch', 'main'],
  hermesHome: 'C:\\Users\\o\'brien\\AppData\\Local\\hermes',
  pathEnv: 'C:\\venv\\Scripts;C:\\Windows\\system32',
  ackPath: 'C:\\Users\\o\'brien\\AppData\\Local\\hermes\\logs\\update-launch-ack.json',
  logPath: 'C:\\Users\\o\'brien\\AppData\\Local\\hermes\\logs\\update-launch.log',
  requestId: 'req-123'
}

test('launcher script embeds env, updater, ack and request id with escaped quotes', () => {
  const script = buildUpdateLaunchScript(SCRIPT_ARGS)

  assert.ok(script.includes(`$env:HERMES_HOME = 'C:\\Users\\o''brien\\AppData\\Local\\hermes'`))
  assert.ok(script.includes(`$env:PATH = 'C:\\venv\\Scripts;C:\\Windows\\system32'`))
  assert.ok(script.includes(`-FilePath 'C:\\Users\\o''brien\\AppData\\Local\\hermes\\hermes-setup.exe'`))
  assert.ok(script.includes(`@('--update', '--branch', 'main')`))
  assert.ok(script.includes(`$requestId = 'req-123'`))
  assert.ok(script.includes(`$ack = 'C:\\Users\\o''brien\\AppData\\Local\\hermes\\logs\\update-launch-ack.json'`))
})

test('launcher script guards duplicates, waits for the updater, and self-deletes the task', () => {
  const script = buildUpdateLaunchScript(SCRIPT_ARGS)

  // Duplicate-firing guard is bound to the request id and exits early.
  assert.ok(script.includes(`$existing.requestId -eq $requestId`))
  assert.ok(script.includes('duplicate invocation ignored'))
  // Keeps the task instance alive while the updater runs (IgnoreNew shield).
  assert.ok(script.includes('Wait-Process -Id $p.Id'))
  // Cleans up its own one-shot task definition.
  assert.ok(script.includes(`schtasks.exe /Delete /F /TN '${UPDATE_LAUNCH_TASK_NAME}'`))
  // Failure path still writes a (negative) ACK so the desktop can react.
  assert.ok(script.includes('ok = $false'))
})

test('schtasks argv: forced one-shot create and explicit run', () => {
  const create = schtasksCreateArgs('Hermes_UpdateLaunch', 'C:\\hh\\update-launch.ps1', '12:34')

  assert.deepEqual(create.slice(0, 8), ['/Create', '/F', '/TN', 'Hermes_UpdateLaunch', '/SC', 'ONCE', '/ST', '12:34'])
  assert.equal(create[8], '/TR')
  assert.ok(create[9].startsWith('powershell.exe -NoProfile -ExecutionPolicy Bypass'))
  assert.ok(create[9].endsWith('-File "C:\\hh\\update-launch.ps1"'))

  assert.deepEqual(schtasksRunArgs('Hermes_UpdateLaunch'), ['/Run', '/TN', 'Hermes_UpdateLaunch'])
  assert.deepEqual(schtasksEndArgs('Hermes_UpdateLaunch'), ['/End', '/TN', 'Hermes_UpdateLaunch'])
  assert.deepEqual(schtasksDeleteArgs('Hermes_UpdateLaunch'), ['/Delete', '/F', '/TN', 'Hermes_UpdateLaunch'])
})

test('nextStartTime pads and wraps around midnight', () => {
  assert.equal(nextStartTime(new Date(2026, 6, 17, 9, 5, 0)), '09:07')
  assert.equal(nextStartTime(new Date(2026, 6, 17, 23, 59, 0)), '00:01')
})

test('parseLaunchAck accepts only the matching request id and sane pids', () => {
  const ok = parseLaunchAck('﻿{"ok":true,"requestId":"r1","updaterPid":4321}', 'r1')

  assert.ok(ok)
  assert.equal(ok.ok, true)
  assert.equal(ok.updaterPid, 4321)

  const failed = parseLaunchAck('{"ok":false,"requestId":"r1","error":"boom"}', 'r1')

  assert.ok(failed)
  assert.equal(failed.ok, false)
  assert.equal(failed.error, 'boom')

  assert.equal(parseLaunchAck('{"ok":true,"requestId":"other","updaterPid":1}', 'r1'), null)
  assert.equal(parseLaunchAck('not json', 'r1'), null)
  assert.equal(parseLaunchAck('', 'r1'), null)
  assert.equal(parseLaunchAck('{"ok":true,"requestId":"r1","updaterPid":"4242junk"}', 'r1')?.updaterPid, null)
})

function tempHome(): string {
  return fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-update-launch-'))
}

test('launch flow: writes launcher script, fires task, returns ACKed updater pid', async () => {
  const home = tempHome()
  const calls: string[][] = []

  const result = await launchUpdaterViaScheduledTask({
    updaterPath: 'C:\\hh\\hermes-setup.exe',
    updaterArgs: ['--update', '--branch', 'main'],
    hermesHome: home,
    pathEnv: 'C:\\x',
    timeoutMs: 3000,
    pollMs: 10,
    runCommand: async (file, args) => {
      calls.push([file, ...args])

      if (args[0] === '/Run') {
        // Simulate the launcher: read the request id back out of the script
        // the flow just wrote, then ACK with an updater pid.
        const scriptPath = calls[0][calls[0].indexOf('/TR') + 1].match(/-File "([^"]+)"/)?.[1] ?? ''

        const paths = {
          scriptPath,
          ackPath: fs
            .readFileSync(scriptPath, 'utf8')
            .match(/\$ack = '([^']+)'/)?.[1]
            .replace(/''/g, "'") ?? ''
        }

        const script = fs.readFileSync(paths.scriptPath, 'utf8')
        const requestId = /\$requestId = '([^']+)'/.exec(script)?.[1] ?? ''

        fs.writeFileSync(paths.ackPath, JSON.stringify({ ok: true, requestId, updaterPid: 777 }))
      }

      return { code: 0, stdout: '', stderr: '' }
    }
  })

  assert.deepEqual(result, { ok: true, updaterPid: 777, error: null })
  assert.equal(calls.length, 2)
  assert.equal(calls[0][0], 'schtasks.exe')
  assert.equal(calls[0][1], '/Create')
  assert.ok(calls[0].includes('ONCE'))
  assert.match(calls[0][calls[0].indexOf('/TN') + 1], /^Hermes_UpdateLaunch_[a-f0-9]+$/)
  assert.equal(calls[1][1], '/Run')
})

test('launch flow: surfaces schtasks create failure without firing the task', async () => {
  const home = tempHome()
  const calls: string[][] = []

  const result = await launchUpdaterViaScheduledTask({
    updaterPath: 'C:\\hh\\hermes-setup.exe',
    updaterArgs: ['--update'],
    hermesHome: home,
    pathEnv: 'C:\\x',
    runCommand: async (_file, args) => {
      calls.push(args)

      return { code: 1, stdout: '', stderr: 'ERROR: Access is denied.' }
    }
  })

  assert.equal(result.ok, false)
  assert.match(result.error ?? '', /schtasks create failed/)
  assert.equal(calls.length, 2)
  assert.equal(calls[1][0], '/Delete')
})

test('launch flow: surfaces launcher-reported failure and stale-ack timeout', async () => {
  const home = tempHome()

  const reported = await launchUpdaterViaScheduledTask({
    updaterPath: 'C:\\hh\\hermes-setup.exe',
    updaterArgs: ['--update'],
    hermesHome: home,
    pathEnv: 'C:\\x',
    timeoutMs: 3000,
    pollMs: 10,
    runCommand: async (_file, args) => {
      if (args[0] === '/Run') {
        const scriptPath = fs
          .readdirSync(home)
          .map(name => path.join(home, name))
          .find(candidate => candidate.endsWith('.ps1')) ?? ''

        const paths = {
          scriptPath,
          ackPath: fs
            .readFileSync(scriptPath, 'utf8')
            .match(/\$ack = '([^']+)'/)?.[1]
            .replace(/''/g, "'") ?? ''
        }

        const script = fs.readFileSync(paths.scriptPath, 'utf8')
        const requestId = /\$requestId = '([^']+)'/.exec(script)?.[1] ?? ''

        fs.writeFileSync(paths.ackPath, JSON.stringify({ ok: false, requestId, error: 'spawn denied' }))
      }

      return { code: 0, stdout: '', stderr: '' }
    }
  })

  assert.equal(reported.ok, false)
  assert.match(reported.error ?? '', /spawn denied/)

  // An ACK from a PREVIOUS request must not satisfy this run: the flow
  // deletes stale ACKs up front and re-checks the embedded request id.
  const staleHome = tempHome()
  const stalePaths = updateLaunchPaths(staleHome)

  fs.mkdirSync(path.dirname(stalePaths.ackPath), { recursive: true })
  fs.writeFileSync(stalePaths.ackPath, JSON.stringify({ ok: true, requestId: 'stale', updaterPid: 1 }))

  const timedOut = await launchUpdaterViaScheduledTask({
    updaterPath: 'C:\\hh\\hermes-setup.exe',
    updaterArgs: ['--update'],
    hermesHome: staleHome,
    pathEnv: 'C:\\x',
    timeoutMs: 100,
    pollMs: 10,
    runCommand: async () => ({ code: 0, stdout: '', stderr: '' })
  })

  assert.equal(timedOut.ok, false)
  assert.match(timedOut.error ?? '', /no launcher ACK/)
})

test('launch flow ends and deletes the one-shot task after run failure or ACK timeout', async () => {
  const runFailureCalls: string[][] = []

  const runFailed = await launchUpdaterViaScheduledTask({
    updaterPath: 'C:\\hh\\hermes-setup.exe',
    updaterArgs: ['--update'],
    hermesHome: tempHome(),
    pathEnv: 'C:\\x',
    runCommand: async (_file, args) => {
      runFailureCalls.push(args)

      return args[0] === '/Run'
        ? { code: 1, stdout: '', stderr: 'run failed' }
        : { code: 0, stdout: '', stderr: '' }
    }
  })

  assert.equal(runFailed.ok, false)
  assert.deepEqual(
    runFailureCalls.map(args => args[0]),
    ['/Create', '/Run', '/End', '/Delete']
  )

  const timeoutCalls: string[][] = []

  const timedOut = await launchUpdaterViaScheduledTask({
    updaterPath: 'C:\\hh\\hermes-setup.exe',
    updaterArgs: ['--update'],
    hermesHome: tempHome(),
    pathEnv: 'C:\\x',
    timeoutMs: 25,
    pollMs: 5,
    runCommand: async (_file, args) => {
      timeoutCalls.push(args)

      return { code: 0, stdout: '', stderr: '' }
    }
  })

  assert.equal(timedOut.ok, false)
  assert.deepEqual(
    timeoutCalls.map(args => args[0]),
    ['/Create', '/Run', '/End', '/Delete']
  )
})
