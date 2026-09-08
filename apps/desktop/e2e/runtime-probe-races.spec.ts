import { execFileSync } from 'node:child_process'
import * as fs from 'node:fs'
import * as path from 'node:path'

import { expect, test } from '@playwright/test'

import { buildAppEnv, createSandbox, launchDesktop, writeEnvFile, writeMockProviderConfig } from './fixtures'

const repo = path.resolve(import.meta.dirname, '../../..')
interface ChildRecord {
  pid: number
  argv: string[]
  parent_pid?: number
}

function records(directory: string, name: string): ChildRecord[] {
  const folder = path.join(directory, name)

  return fs.existsSync(folder)
    ? fs
        .readdirSync(folder)
        .filter(file => file.endsWith('.json'))
        .map(file => JSON.parse(fs.readFileSync(path.join(folder, file), 'utf8')))
    : []
}

// Both cases use the real renderer/preload/main IPC and provisioned Python.
// The empty namespace directory forces source-inspection fallback WITHOUT
// modifying the real checkout or venv. Python still imports the real package.
for (const owner of ['primary', 'pool'] as const) {
  test(`${owner} invalidation during serve help prevents stale backend spawn`, async () => {
    test.setTimeout(75_000)
    const python = process.env.HERMES_DESKTOP_PYTHON
    expect(python, 'Set HERMES_DESKTOP_PYTHON to a provisioned Hermes venv').toBeTruthy()
    const sandbox = createSandbox(`runtime-race-${owner}`)
    let running: Awaited<ReturnType<typeof launchDesktop>> | undefined
    const hook = path.join(sandbox.root, 'hook')
    fs.mkdirSync(hook)

    try {
      const prefix = execFileSync(python!, ['-c', 'import sys; print(sys.prefix)'], {
        encoding: 'utf8',
        env: buildAppEnv(sandbox),
        timeout: 15_000,
        windowsHide: true
      }).trim()

      const active = path.join(sandbox.hermesHome, 'hermes-agent')
      fs.mkdirSync(path.join(active, 'hermes_cli'), { recursive: true })
      fs.copyFileSync(path.join(repo, 'hermes_cli', 'main.py'), path.join(active, 'hermes_cli', 'main.py'))
      fs.symlinkSync(prefix, path.join(active, 'venv'), process.platform === 'win32' ? 'junction' : 'dir')
      fs.copyFileSync(
        path.join(import.meta.dirname, 'runtime-probe-race-sitecustomize.py'),
        path.join(hook, 'sitecustomize.py')
      )
      fs.mkdirSync(path.join(sandbox.hermesHome, 'profiles', 'race-pool'), { recursive: true })
      writeMockProviderConfig(sandbox.hermesHome, 'http://127.0.0.1:1')
      writeEnvFile(sandbox.hermesHome)

      const env = buildAppEnv(sandbox, {
        HOME: sandbox.root,
        USERPROFILE: sandbox.root,
        HERMES_DESKTOP_IS_PACKAGED: '1',
        HERMES_DESKTOP_HERMES_ROOT: '',
        HERMES_DESKTOP_HERMES: '',
        HERMES_DESKTOP_PYTHON: '',
        HERMES_PROBE_TIMEOUT_MS: '30000',
        HERMES_E2E_RUNTIME_RACE_DIR: hook,
        PYTHONPATH: [hook, repo].join(path.delimiter),
        PYTHONDONTWRITEBYTECODE: '1',
        NODE_OPTIONS: ''
      })

      delete env.HERMES_DESKTOP_DEV_SERVER
      expect(env.HERMES_DASHBOARD_SESSION_TOKEN).toBeUndefined()

      const imported = execFileSync(python!, ['-c', 'import hermes_cli; print(hermes_cli.__file__)'], {
        cwd: active,
        env,
        encoding: 'utf8',
        timeout: 15_000,
        windowsHide: true
      }).trim()

      expect(path.resolve(imported)).toBe(path.join(repo, 'hermes_cli', '__init__.py'))
      running = await launchDesktop(env)
      // Keep preload but remove automatic UI reconnects: only the explicit
      // bridge requests below own the test's connection attempts.
      await running.page.goto('about:blank')
      await running.page.waitForFunction(() => Boolean((window as any).hermesDesktop))

      const pending = running.page
        .evaluate(async scope => {
          try {
            await (window as any).hermesDesktop.getConnection(scope === 'pool' ? 'race-pool' : undefined)

            return { status: 'resolved', error: '' }
          } catch (error) {
            return { status: 'rejected', error: String(error) }
          }
        }, owner)
        .catch(error => ({ status: 'harness-error', error: String(error) }))

      await expect.poll(() => records(hook, 'entered').length, { timeout: 20_000 }).toBeGreaterThan(0)
      expect(records(hook, 'released')).toEqual([])

      // Windows venv python.exe can be a redirector that remains the worker's
      // immediate parent. Verify the real Electron-owned chain, not POSIX PPID.
      const probeParents = await running.app.evaluate(() => [
        process.pid,
        ...(process as any)
          ._getActiveHandles()
          .filter(
            (handle: any) =>
              JSON.stringify(handle.spawnargs?.slice(1)) ===
              JSON.stringify(['-m', 'hermes_cli.main', 'serve', '--help'])
          )
          .map((child: any) => child.pid)
      ])

      expect(probeParents).toContain(records(hook, 'entered')[0].parent_pid)

      if (owner === 'pool') {
        // Primary and pool share the serve-support promise. Wait for the pool's
        // own import probe to close and its promise continuations to drain, so
        // invalidation occurs at serve-help, not earlier runtime discovery.
        await expect
          .poll(
            () =>
              records(hook, 'all').filter(
                record => record.argv.at(-1) === 'import yaml; import dotenv; import hermes_cli.config'
              ).length
          )
          .toBe(2)
        await running.app.evaluate(async () => {
          const children = (process as any)
            ._getActiveHandles()
            .filter(
              (handle: any) => handle.spawnargs?.at(-1) === 'import yaml; import dotenv; import hermes_cli.config'
            )

          await Promise.all(children.map((child: any) => new Promise<void>(resolve => child.once('close', resolve))))
          await new Promise<void>(resolve => setImmediate(resolve))
        })
      }

      // Observe Node's actual spawn as well as Python startup. A post-spawn
      // ownership rejection can kill Python before sitecustomize ever executes;
      // a Python-only counter would miss that forbidden transient process.
      await running.app.evaluate((_electron, directory) => {
        const childProcess = process.getBuiltinModule('child_process')
        const fs = process.getBuiltinModule('fs')
        const path = process.getBuiltinModule('path')
        const spawn = childProcess.spawn
        childProcess.spawn = ((...args: Parameters<typeof spawn>) => {
          const child = Reflect.apply(spawn, childProcess, args)
          const options = args[2] as { env?: Record<string, string> } | undefined

          if (options?.env?.HERMES_DASHBOARD_SESSION_TOKEN) {
            const target = path.join(directory, 'spawn-attempts')
            fs.mkdirSync(target, { recursive: true })
            fs.writeFileSync(path.join(target, `${child.pid}.json`), JSON.stringify({ pid: child.pid, argv: args[1] }))
          }

          return child
        }) as typeof spawn
        process.getBuiltinModule('module').syncBuiltinESMExports()
      }, hook)

      const applied = await running.page.evaluate(
        scope =>
          (window as any).hermesDesktop.applyConnectionConfig({
            mode: 'local',
            ...(scope === 'pool' ? { profile: 'race-pool' } : {})
          }),
        owner
      )

      expect(applied.mode).toBe('local')
      expect(records(hook, 'released')).toEqual([])
      fs.writeFileSync(path.join(hook, 'release'), '')
      // Rejection is the completion barrier for the actual awaiting spawn path,
      // not an arbitrary sleep followed by an absence assertion.
      const result = await pending
      expect(result.status, result.error).toBe('rejected')
      await expect.poll(() => records(hook, 'exited').length, { timeout: 15_000 }).toBeGreaterThan(0)
      expect(records(hook, 'timed-out')).toEqual([])

      const attempts = records(hook, 'spawn-attempts').filter(
        record => owner === 'primary' || record.argv.includes('race-pool')
      )

      expect(attempts, 'invalidated attempts must not even spawn a short-lived child').toEqual([])
      const stale = records(hook, 'spawned').filter(record => owner === 'primary' || record.argv.includes('race-pool'))
      expect(stale).toEqual([])
      console.log(
        `${owner}: real serve-help exited; stale Node spawns=${attempts.length}, Python starts=${stale.length}`
      )
    } finally {
      fs.writeFileSync(path.join(hook, 'release'), '')

      if (running) {
        await running.app.close()
      }

      sandbox.cleanup()
    }
  })
}
