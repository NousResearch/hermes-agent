import assert from 'node:assert/strict'
import { type ChildProcess, spawn } from 'node:child_process'
import { once } from 'node:events'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { test, vi } from 'vitest'

import {
  RELAUNCH_WAITER_LOG_TAIL_BYTES,
  type RelaunchWaiterHandle,
  type SpawnWaiter,
  startRelaunchWaiter,
  waiterLogTail
} from './relaunch-waiter'

const scriptPath: string = path.resolve(import.meta.dirname, '../../scripts/update-relaunch-waiter.ps1')

async function stop(child: ChildProcess | undefined): Promise<void> {
  if (child?.pid && child.exitCode === null && child.signalCode === null) {
    const exited = once(child, 'close')
    child.kill()
    await exited
  }
}

for (const mode of [
  { name: 'ready', ready: true },
  { name: 'silent', ready: false },
  { name: 'spawn-error', ready: false, missing: true },
  { name: 'early-exit', ready: false, earlyExit: true },
  { name: 'cancel-timeout', ready: true, refuseKill: true },
  { name: 'startup-cancel-timeout', ready: false, refuseKill: true },
  { name: 'cleanup-error', ready: true, refuseCleanup: true }
]) {
  test(`waiter ownership survives its complete lifecycle (${mode.name})`, async (): Promise<void> => {
    let child: ChildProcess | undefined
    let originalKill: ChildProcess['kill'] | undefined
    let stage: string = ''
    let closed: boolean = false
    const cleanupError: Error = new Error('fixture staging cleanup refused')

    const start: SpawnWaiter = (command: string, args: string[], options: Parameters<SpawnWaiter>[2]): ChildProcess => {
      stage = options.cwd
      const readyFile: string = path.join(stage, 'ready.txt')
      assert.equal(
        command,
        path.win32.join(
          process.env.SystemRoot || process.env.SYSTEMROOT || 'C:\\Windows',
          'System32',
          'WindowsPowerShell',
          'v1.0',
          'powershell.exe'
        )
      )
      assert.deepEqual(options, { cwd: stage, detached: true, stdio: 'ignore', windowsHide: true })
      assert.notEqual(stage, path.dirname(scriptPath))
      assert.deepEqual(fs.readFileSync(path.join(stage, 'update-relaunch-waiter.ps1')), fs.readFileSync(scriptPath))
      assert.deepEqual(args, [
        '-NoProfile',
        '-NonInteractive',
        '-ExecutionPolicy',
        'Bypass',
        '-File',
        path.join(stage, 'update-relaunch-waiter.ps1'),
        '-ProcessId',
        String(process.pid),
        '-ProcessStartTimeMs',
        '1000000',
        '-IdentityName',
        'disposable-waiter-test',
        '-ReadyFile',
        readyFile,
        '-TimeoutSeconds',
        '900'
      ])
      child = spawn(
        mode.missing ? path.join(stage, 'missing.exe') : process.execPath,
        [
          '-e',
          `
        const fs = require('node:fs');
        if (process.env.TEST_EXIT === 'yes') process.exit(3);
        if (process.env.TEST_READY === 'yes') fs.writeFileSync(process.env.READY_FILE, 'ready');
        setInterval(() => {}, 1000);
      `
        ],
        {
          ...options,
          env: {
            ...process.env,
            READY_FILE: readyFile,
            TEST_READY: mode.ready ? 'yes' : 'no',
            TEST_EXIT: mode.earlyExit ? 'yes' : 'no'
          }
        }
      )
      originalKill = child.kill.bind(child)

      if (mode.refuseKill) {
        child.kill = () => false
      }

      child.once('close', () => {
        closed = true
      })

      return child
    }

    try {
      const starting: Promise<RelaunchWaiterHandle | undefined> = startRelaunchWaiter(
        {
          processId: process.pid,
          processStartTimeMs: 1_000_000,
          identityName: 'disposable-waiter-test',
          scriptPath
        },
        { spawn: start, handshakeTimeoutMs: mode.ready ? 10_000 : 2_000, cancelTimeoutMs: 2_000, pollMs: 20 }
      )

      if (!mode.ready && mode.refuseKill) {
        await assert.rejects(starting, /did not exit after cancellation/)
        assert.equal(closed, false)
        assert.equal(fs.existsSync(stage), true)

        return
      }

      const handle = await starting

      if (mode.ready) {
        assert.ok(handle && typeof handle === 'object', 'readiness must retain the cancellation handle')
        assert.equal(closed, false)

        if (mode.refuseCleanup) {
          const remove = fs.promises.rm.bind(fs.promises)
          vi.spyOn(fs.promises, 'rm').mockImplementation((file, options) => {
            return file === stage ? Promise.reject(cleanupError) : remove(file, options)
          })
        }

        const cancelled = handle.cancel()
        assert.equal(handle.cancel(), cancelled, 'concurrent cancellation shares its result')

        if (mode.refuseKill || mode.refuseCleanup) {
          await assert.rejects(cancelled, error =>
            mode.refuseCleanup
              ? error === cleanupError
              : error instanceof Error && error.message.includes('did not exit after cancellation')
          )
          assert.equal(handle.cancel(), cancelled)
          assert.equal(closed, !mode.refuseKill)
          assert.equal(fs.existsSync(stage), true)

          return
        }

        await cancelled
        await handle.cancel()
      } else {
        assert.equal(handle, undefined, 'a failed start has no live mechanism')
      }

      assert.equal(closed, true, 'the actual child has exited before the caller proceeds')
      assert.equal(fs.existsSync(stage), false)
    } finally {
      vi.restoreAllMocks()

      if (child && originalKill) {
        child.kill = originalKill
      }

      await stop(child)

      if (stage) {
        fs.rmSync(stage, { recursive: true, force: true })
      }
    }
  }, 20_000)
}

test('missing scripts refuse startup before spawning a waiter', async (): Promise<void> => {
  let spawned: boolean = false
  assert.equal(
    await startRelaunchWaiter(
      {
        processId: process.pid,
        processStartTimeMs: 1_000_000,
        identityName: 'disposable-waiter-test',
        scriptPath: path.join(scriptPath, 'absent.ps1')
      },
      {
        spawn: (): never => {
          spawned = true
          throw new Error('missing scripts must not spawn')
        }
      }
    ),
    undefined
  )
  assert.equal(spawned, false)
})

test('every failed start tells the log why it failed', async (): Promise<void> => {
  const logs: string[] = []
  const log = (message: string): void => void logs.push(message)

  await startRelaunchWaiter(
    {
      processId: process.pid,
      processStartTimeMs: 1_000_000,
      identityName: 'disposable-waiter-test',
      scriptPath: path.join(scriptPath, 'absent.ps1')
    },
    { log }
  )
  assert.match(logs.join('\n'), /cannot copy .*absent\.ps1/)

  logs.length = 0
  await startRelaunchWaiter(
    { processId: process.pid, processStartTimeMs: 1_000_000, identityName: 'disposable-waiter-test', scriptPath },
    {
      log,
      spawn: (): never => {
        throw new Error('EACCES fixture')
      }
    }
  )
  assert.match(logs.join('\n'), /spawning .*powershell\.exe threw: Error: EACCES fixture/)

  logs.length = 0
  await startRelaunchWaiter(
    { processId: process.pid, processStartTimeMs: 1_000_000, identityName: 'disposable-waiter-test', scriptPath },
    {
      log,
      pollMs: 20,
      spawn: (_command, _args, options): ChildProcess =>
        spawn(process.execPath, ['-e', 'process.exit(2)'], { ...options, stdio: 'ignore' })
    }
  )
  assert.match(logs.join('\n'), /exited before its handshake after \d+ms \(exit code 2/)

  logs.length = 0
  await startRelaunchWaiter(
    { processId: process.pid, processStartTimeMs: 1_000_000, identityName: 'disposable-waiter-test', scriptPath },
    {
      log,
      pollMs: 20,
      handshakeTimeoutMs: 200,
      spawn: (_command, _args, options): ChildProcess =>
        spawn(process.execPath, ['-e', 'setInterval(() => {}, 1000)'], { ...options, stdio: 'ignore' })
    }
  )
  assert.match(logs.join('\n'), /no handshake within 200ms \(pid \d+\); stopping it/)
}, 20_000)

test('the script log report keeps only this attempt, reads a bounded suffix, and decodes UTF-8', (): void => {
  const dir: string = fs.mkdtempSync(path.join(os.tmpdir(), 'waiter-log-'))
  const file: string = path.join(dir, 'waiter.log')

  try {
    // Older attempt's lines, then enough filler to push them past the bounded read.
    const filler: string = `t [hermes-relaunch-old] pid=1 ${'x'.repeat(200)}\n`.repeat(
      Math.ceil(RELAUNCH_WAITER_LOG_TAIL_BYTES / 200) + 5
    )

    fs.writeFileSync(
      file,
      `t [hermes-relaunch-gone] pid=2 stale line\n${filler}t [hermes-relaunch-me] pid=3 pfad C:\\Benutzer\\Jörg fehlgeschlagen\nt [hermes-relaunch-other] pid=4 not mine\n`,
      'utf8'
    )

    const report: string = waiterLogTail(file, 'hermes-relaunch-me')
    assert.match(report, /Jörg fehlgeschlagen/)
    assert.doesNotMatch(report, /not mine|stale line|\[hermes-relaunch-old\]/)
    assert.equal(waiterLogTail(file, 'hermes-relaunch-gone'), `${file}:\nno lines from this attempt`)
    // Every physical line of a multi-line script error carries the tag, so none is dropped.
    fs.writeFileSync(
      file,
      [
        't [hermes-relaunch-me] pid=3 failed: Exception: run.ps1:1',
        't [hermes-relaunch-me] pid=3 Line |',
        't [hermes-relaunch-me] pid=3  | first line second line third line',
        't [hermes-relaunch-other] pid=4 not mine'
      ].join('\n'),
      'utf8'
    )
    const multi: string = waiterLogTail(file, 'hermes-relaunch-me')
    assert.match(multi, /Line \|/)
    assert.match(multi, /third line/)
    assert.doesNotMatch(multi, /not mine/)
    assert.match(waiterLogTail(path.join(dir, 'absent.log'), 'x'), /not written/)
  } finally {
    fs.rmSync(dir, { recursive: true, force: true })
  }
})
