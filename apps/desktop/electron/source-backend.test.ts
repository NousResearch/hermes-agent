import assert from 'node:assert/strict'
import { type ChildProcess, execFileSync, spawn } from 'node:child_process'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { test, vi } from 'vitest'
import WebSocket from 'ws'

import { serveBackendArgs } from './backend-command'
import { waitForDashboardPort } from './backend-ready'
import { createSourcePythonBackend, resolveSourceInstallationBackend, type SourceBackend } from './source-backend'

interface Fixture {
  root: string
  launcher: string
  python: string
  selected: string
}

async function stop(child: ChildProcess): Promise<void> {
  if (child.exitCode !== null || child.signalCode !== null) {
    return
  }

  await new Promise<void>((resolve: () => void): void => {
    const timer: NodeJS.Timeout = setTimeout((): void => {
      child.kill('SIGKILL')
    }, 5_000)

    child.once('exit', (): void => {
      clearTimeout(timer)
      resolve()
    })
    child.kill()
  })
}

async function ping(port: number, token: string): Promise<unknown> {
  const socket: WebSocket = new WebSocket(`ws://127.0.0.1:${port}/api/ws?token=${token}`)

  try {
    return await new Promise<unknown>((resolve: (value: unknown) => void, reject: (error: Error) => void): void => {
      const timer: NodeJS.Timeout = setTimeout((): void => {
        reject(new Error('RPC timed out'))
      }, 15_000)

      socket.once('error', (error: Error): void => {
        clearTimeout(timer)
        reject(error)
      })
      socket.once('open', (): void => {
        socket.send(JSON.stringify({ jsonrpc: '2.0', id: 1, method: 'ping', params: {} }))
      })
      socket.on('message', (data: WebSocket.RawData): void => {
        const response: { id?: number; result?: unknown; error?: unknown } = JSON.parse(data.toString())

        if (response.id === 1) {
          clearTimeout(timer)
          resolve(response)
        }
      })
    })
  } finally {
    socket.terminate()
  }
}

test.skipIf(process.platform === 'win32')(
  'a PM source launcher reaches real health and RPC without adopting a legacy venv (POSIX)',
  async (): Promise<void> => {
    const temp: string = fs.mkdtempSync(path.join(os.tmpdir(), 'desktop-pm-start-'))
    const home: string = path.join(temp, 'home with spaces')

    const env: NodeJS.ProcessEnv = Object.fromEntries(
      Object.entries(process.env).filter(
        ([key]: [string, string | undefined]): boolean => !/^(HERMES_|PYTHON|UV_|VIRTUAL_ENV|XDG_)/.test(key)
      )
    )

    Object.assign(env, {
      HOME: home,
      USERPROFILE: home,
      HERMES_HOME: path.join(home, '.hermes'),
      HERMES_RUNTIME_DIR: path.join(temp, 'tools'),
      HERMES_DISABLE_LAZY_INSTALLS: '1',
      XDG_CONFIG_HOME: path.join(temp, 'config'),
      XDG_CONFIG_DIRS: path.join(temp, 'config'),
      PYTHONDONTWRITEBYTECODE: '1',
      UV_CACHE_DIR: path.join(temp, 'cache'),
      UV_OFFLINE: '1'
    })
    const python: string = process.env.HERMES_PYTHON || 'python3'
    const fixtureScript: string = path.join(import.meta.dirname, 'fixtures', 'source-backend.py')
    const token: string = 'desktop-pm-contract'
    env.HERMES_DASHBOARD_SESSION_TOKEN = token
    vi.stubEnv('HOME', home)

    try {
      const fixture: Fixture = JSON.parse(
        execFileSync(python, ['-I', fixtureScript, temp], {
          cwd: temp,
          env,
          encoding: 'utf8',
          timeout: 90_000
        })
      ) as Fixture

      assert.equal(fs.existsSync(path.join(fixture.root, 'venv')), false)
      assert.equal(fs.existsSync(path.join(fixture.root, '.venv')), false)

      const origin: { python: string; module: string; value: string } = JSON.parse(
        execFileSync(fixture.launcher, ['--run-module', 'desktop_launch_probe'], {
          cwd: temp,
          env,
          encoding: 'utf8',
          timeout: 15_000
        })
      )

      assert.equal(origin.python, fixture.python)
      assert.ok(origin.module.startsWith(`${fixture.selected}${path.sep}`))
      assert.equal(origin.value, 'selected by PM')

      for (const poisoned of [false, true]) {
        const poison: string = path.join(temp, 'legacy-python-used')

        if (poisoned) {
          for (const suffix of ['bin/python', 'Scripts/python.exe']) {
            const oldPython: string = path.join(fixture.root, 'venv', suffix)
            fs.mkdirSync(path.dirname(oldPython), { recursive: true })
            fs.writeFileSync(oldPython, `#!/bin/sh\ntouch '${poison}'\nexit 93\n`, { mode: 0o755 })
          }
        }

        const backend: SourceBackend | null = resolveSourceInstallationBackend(fixture.root, serveBackendArgs(), {
          hermesHome: env.HERMES_HOME,
          env
        })

        assert.ok(backend, 'a published, runnable PM installation must be accepted')
        assert.equal(backend.command, fixture.launcher)
        assert.equal(backend.bootstrap, false)

        const child: ChildProcess = spawn(backend.command, backend.args, {
          cwd: temp,
          env: { ...env, ...backend.env },
          shell: backend.shell,
          stdio: ['ignore', 'pipe', 'pipe']
        })

        let output: string = ''
        child.stdout?.on('data', (data: Buffer): void => {
          output += data.toString()
        })
        child.stderr?.on('data', (data: Buffer): void => {
          output += data.toString()
        })

        try {
          const port: number = (await waitForDashboardPort(child, 45_000)) as number

          const response: Response = await fetch(`http://127.0.0.1:${port}/api/health`, {
            headers: { 'X-Hermes-Session-Token': token }
          })

          assert.equal(response.status, 200, output)
          assert.deepEqual(await ping(port, token), { jsonrpc: '2.0', id: 1, result: { pong: true } })
          assert.equal(fs.existsSync(poison), false, 'the stale checkout interpreter must never execute')
          console.info('PM backend verified', { poisoned, port, health: response.status, origin })
        } catch (error: unknown) {
          throw new Error(`${String(error)}\n${output}`)
        } finally {
          await stop(child)
        }

        if (poisoned) {
          const source: SourceBackend | null = createSourcePythonBackend(fixture.root, fixture.python, ['--version'], {
            isWindows: true,
            env
          })

          assert.ok(source)
          assert.equal(source.command, fixture.python)
          assert.match(
            execFileSync(source.command, source.args, {
              cwd: source.root,
              env: { ...env, ...source.env },
              encoding: 'utf8',
              timeout: 15_000
            }),
            /Hermes/
          )
          assert.equal(fs.existsSync(poison), false)
        }
      }

      fs.unlinkSync(fixture.launcher)
      assert.equal(
        resolveSourceInstallationBackend(fixture.root, serveBackendArgs(), { hermesHome: env.HERMES_HOME, env }),
        null,
        'a missing PM command must not fall back to the stale venv'
      )
    } finally {
      fs.rmSync(temp, { recursive: true, force: true })
      vi.unstubAllEnvs()
    }
  },
  150_000
)

test('Windows console selection uses only the selected interpreter directory', (): void => {
  const temp: string = fs.mkdtempSync(path.join(os.tmpdir(), 'desktop-console-policy-'))
  const root: string = path.join(temp, 'repo')
  const selected: string = path.join(temp, 'external', 'pythonw.exe')
  const consolePython: string = path.join(path.dirname(selected), 'python.exe')

  try {
    fs.mkdirSync(path.join(root, 'venv', 'Scripts'), { recursive: true })
    fs.writeFileSync(path.join(root, 'venv', 'Scripts', 'python.exe'), 'stale')
    fs.mkdirSync(path.dirname(selected), { recursive: true })
    fs.writeFileSync(selected, '')
    assert.equal(createSourcePythonBackend(root, selected, [], { isWindows: true })?.command, selected)
    fs.writeFileSync(consolePython, '')
    const backend: SourceBackend | null = createSourcePythonBackend(root, selected, ['serve'], { isWindows: true })
    assert.equal(backend?.command, consolePython)
    assert.equal(backend?.env.PYTHONPATH, '')
    assert.equal(backend?.env.PYTHONHOME, '')
    assert.equal(createSourcePythonBackend(root, selected, [], { isWindows: false })?.command, selected)
    assert.equal(createSourcePythonBackend(root, null, []), null)
  } finally {
    fs.rmSync(temp, { recursive: true, force: true })
  }
})
