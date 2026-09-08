import { execFileSync } from 'node:child_process'
import * as fs from 'node:fs'
import * as path from 'node:path'

import { expect, test } from '@playwright/test'

import { startMockServer } from '../../../tests-js/scripts/mock-server'

import {
  buildAppEnv,
  createSandbox,
  launchDesktop,
  waitForAppReady,
  writeEnvFile,
  writeMockProviderConfig
} from './fixtures'

const repo = path.resolve(import.meta.dirname, '../../..')

/**
 * Real local active-install discovery, not a remote-connection reproduction.
 * Saved remotes bypass this resolver; this does not establish AppHangB1's cause.
 * Run after npm run build, with HERMES_DESKTOP_PYTHON set to a provisioned venv.
 * No setup/update commands or writes through the source/venv links are needed.
 */
test('runtime import probe can complete I/O with its actual Electron parent', async () => {
  test.setTimeout(90_000)
  const python = process.env.HERMES_DESKTOP_PYTHON
  expect(python, 'Set HERMES_DESKTOP_PYTHON to an existing Hermes test venv interpreter').toBeTruthy()
  const sandbox = createSandbox('runtime-probe')
  const mock = await startMockServer()
  let running: Awaited<ReturnType<typeof launchDesktop>> | undefined

  try {
    // Link only the required source package and the actual venv (Windows needs
    // its Scripts/python.exe redirector and pyvenv.cfg, not a copied executable).
    const prefix = execFileSync(python!, ['-c', 'import sys; print(sys.prefix)'], {
      encoding: 'utf8',
      env: buildAppEnv(sandbox),
      timeout: 15_000,
      windowsHide: true
    }).trim()

    const active = path.join(sandbox.hermesHome, 'hermes-agent')
    fs.mkdirSync(active)
    const linkType = process.platform === 'win32' ? 'junction' : 'dir'
    fs.symlinkSync(path.join(repo, 'hermes_cli'), path.join(active, 'hermes_cli'), linkType)
    fs.symlinkSync(prefix, path.join(active, 'venv'), linkType)
    const hook = path.join(sandbox.root, 'python-hook')
    fs.mkdirSync(hook)
    fs.copyFileSync(
      path.join(import.meta.dirname, 'runtime-probe-sitecustomize.py'),
      path.join(hook, 'sitecustomize.py')
    )
    writeMockProviderConfig(sandbox.hermesHome, mock.url)
    writeEnvFile(sandbox.hermesHome)

    const env = buildAppEnv(sandbox, {
      HOME: sandbox.root,
      USERPROFILE: sandbox.root,
      HERMES_DESKTOP_IS_PACKAGED: '1',
      HERMES_DESKTOP_HERMES_ROOT: '',
      HERMES_DESKTOP_HERMES: '',
      HERMES_DESKTOP_PYTHON: '',
      HERMES_PROBE_TIMEOUT_MS: '15000',
      HERMES_E2E_RUNTIME_PROBE_DIR: hook,
      PYTHONPATH: [hook, repo].join(path.delimiter),
      PYTHONDONTWRITEBYTECODE: '1',
      NODE_OPTIONS: ''
    })

    delete env.HERMES_DESKTOP_DEV_SERVER
    running = await launchDesktop(env)
    await expect.poll(() => fs.existsSync(path.join(hook, 'entered.json')), { timeout: 20_000 }).toBe(true)

    // The Python child is still waiting for a port. Exercise the existing
    // renderer → preload → main IPC bridge before allowing that child to exit.
    const limits = await running.page.evaluate(() =>
      (
        window as unknown as {
          hermesDesktop: { getPoolLimits: () => Promise<{ maxBackends: number }> }
        }
      ).hermesDesktop.getPoolLimits()
    )

    expect(limits.maxBackends).toBeGreaterThan(0)
    expect(fs.existsSync(path.join(hook, 'replied.json'))).toBe(false)
    // This server MUST live in Electron main, not the Playwright runner. A
    // synchronous child probe prevents main from accepting/replying and fails.
    await running.app.evaluate(async (_electron, directory) => {
      const net = process.getBuiltinModule('net')
      const fs = process.getBuiltinModule('fs')
      const path = process.getBuiltinModule('path')
      await new Promise<void>((resolve, reject) => {
        const server = net.createServer(socket => {
          socket.on('error', () => undefined)
          socket.once('data', data => {
            fs.appendFileSync(
              path.join(directory, 'accepted.jsonl'),
              JSON.stringify({
                parentPid: process.pid,
                childPid: Number(data.toString())
              }) + '\n'
            )
            socket.end(`parent:${process.pid}`)
          })
        })

        server.once('error', reject)
        server.listen(0, '127.0.0.1', () => {
          const address = server.address()

          if (!address || typeof address === 'string') {
            return reject(new Error('No TCP address'))
          }

          fs.writeFileSync(path.join(directory, 'port.tmp'), String(address.port))
          fs.renameSync(path.join(directory, 'port.tmp'), path.join(directory, 'port'))
          server.unref()
          resolve()
        })
      })
    }, hook)
    await expect.poll(() => fs.existsSync(path.join(hook, 'replied.json')), { timeout: 20_000 }).toBe(true)
    const reply = JSON.parse(fs.readFileSync(path.join(hook, 'replied.json'), 'utf8'))

    const accepted = fs
      .readFileSync(path.join(hook, 'accepted.jsonl'), 'utf8')
      .trim()
      .split('\n')
      .map(line => JSON.parse(line))

    // The launch handle's PID can differ from Electron main on Windows. Use
    // the browser process executing this callback: its I/O must stay live.
    const main = await running.app.evaluate(() => ({
      pid: process.pid,
      type: (process as NodeJS.Process & { type?: string }).type,
      electron: process.versions.electron
    }))

    expect(main.type).toBe('browser')
    expect(main.electron).toBeTruthy()
    expect(reply.home).toBe(sandbox.hermesHome)
    expect(reply.pid).not.toBe(main.pid)
    expect(reply.parent_pid).toBe(main.pid)
    expect(accepted).toContainEqual({ parentPid: reply.parent_pid, childPid: reply.pid })
    expect(fs.existsSync(path.join(hook, 'failed.txt'))).toBe(false)
    await waitForAppReady({ ...running, mock, mockUrl: mock.url, sandbox, cleanup: async () => {} })
  } finally {
    if (running) {
      await running.app.close()
    }

    await mock.close()
    sandbox.cleanup()
  }
})
