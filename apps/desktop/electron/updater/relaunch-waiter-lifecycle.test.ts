import assert from 'node:assert/strict'
import { type ChildProcess, spawn } from 'node:child_process'
import { once } from 'node:events'
import fs from 'node:fs'
import path from 'node:path'

import { test, vi } from 'vitest'

import { type SpawnWaiter, startRelaunchWaiter } from './relaunch-waiter'

const scriptPath = path.resolve(import.meta.dirname, '../../scripts/update-relaunch-waiter.ps1')

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
  { name: 'cancel-timeout', ready: true, refuseKill: true },
  { name: 'startup-cancel-timeout', ready: false, refuseKill: true },
  { name: 'cleanup-error', ready: true, refuseCleanup: true }
]) {
  test(`waiter ownership survives its complete lifecycle (${mode.name})`, async () => {
    let child: ChildProcess | undefined
    let originalKill: ChildProcess['kill'] | undefined
    let stage = ''
    let closed = false
    const cleanupError = new Error('fixture staging cleanup refused')

    const start: SpawnWaiter = (_command, args, options) => {
      stage = options.cwd
      const readyFile = args[args.indexOf('-ReadyFile') + 1]
      child = spawn(mode.missing ? path.join(stage, 'missing.exe') : process.execPath, ['-e', `
        const fs = require('node:fs');
        if (process.env.TEST_READY === 'yes') fs.writeFileSync(process.env.READY_FILE, 'ready');
        setInterval(() => {}, 1000);
      `], {
        ...options,
        env: { ...process.env, READY_FILE: readyFile, TEST_READY: mode.ready ? 'yes' : 'no' }
      })
      originalKill = child.kill.bind(child)

      if (mode.refuseKill) { child.kill = () => false }
      child.once('close', () => { closed = true })

      return child
    }

    try {
      const starting = startRelaunchWaiter({
        processId: process.pid,
        processStartTimeMs: Date.now(),
        identityName: 'disposable-waiter-test',
        scriptPath
      }, { spawn: start, handshakeTimeoutMs: mode.ready ? 10_000 : 2_000, cancelTimeoutMs: 2_000, pollMs: 20 })

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
          await assert.rejects(cancelled, error => mode.refuseCleanup
            ? error === cleanupError
            : error instanceof Error && error.message.includes('did not exit after cancellation'))
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

      if (child && originalKill) { child.kill = originalKill }
      await stop(child)

      if (stage) {fs.rmSync(stage, { recursive: true, force: true })}
    }
  }, 20_000)
}
