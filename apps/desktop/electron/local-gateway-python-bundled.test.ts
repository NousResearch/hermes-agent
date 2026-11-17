import fs from 'node:fs/promises'
import net from 'node:net'
import path from 'node:path'

import { expect, test, vi } from 'vitest'

import type { InstallStamp } from './install-stamp'
import { closeGatewayTicketBridges, createGatewayTicketResolver } from './local-gateway-python'
import { shortSocketTmpDir } from './local-gateway.test-helpers'
import { bundledPayload, payloadTicketRuntime } from './payload-backend'

/** A payload laid out the way the builder stamps it, the shape resolveHermesBackend() returns for
 * it (main.ts), and an owner socket that grants one ticket. */
async function bundledPayloadFixture() {
  const home = await shortSocketTmpDir('gw-bundled-')
  const resources = await shortSocketTmpDir('gw-bundled-res-')
  const userHome = await shortSocketTmpDir('gw-bundled-user-')
  const repo = path.resolve('../..')
  const python = process.env.HERMES_TEST_PYTHON || 'python3'
  const root = path.join(resources, 'agent-payload')
  await fs.mkdir(path.join(root, 'bin'), { recursive: true })
  await fs.mkdir(path.join(root, 'tools', 'python', 'bin'), { recursive: true })
  await fs.mkdir(path.join(root, 'venv', 'site-packages'), { recursive: true })
  await fs.symlink(repo, path.join(root, 'app'))
  // The CLI shim: like hermes.exe, it is not an interpreter and refuses interpreter flags.
  await fs.writeFile(
    path.join(root, 'bin', 'hermes'),
    '#!/bin/sh\necho "hermes: unrecognized arguments: $1" >&2\nexit 2\n',
    { mode: 0o755 }
  )
  const { execFileSync } = await import('node:child_process')
  const realPython = execFileSync(python, ['-c', 'import sys; print(sys.executable)'], { encoding: 'utf8' }).trim()
  await fs.symlink(realPython, path.join(root, 'tools', 'python', 'bin', 'python3'))

  const stamp = {
    schemaVersion: 1,
    commit: 'a'.repeat(40),
    commitDate: null,
    branch: null,
    builtAt: null,
    dirty: false,
    source: 'ci',
    distribution: 'desktop-app',
    updateMechanism: 'microsoft-store',
    baseVersion: null,
    displayVersion: null,
    distance: null,
    payload: 'bundled',
    tag: 'v1.0.0',
    runtime: {
      repoDir: 'app',
      toolsDir: 'tools',
      storePython: 'tools/python/bin/python3',
      sitePackages: 'venv/site-packages',
      commands: { hermes: 'bin/hermes' }
    }
  } as unknown as InstallStamp

  const payload = bundledPayload(resources, stamp)!
  const endpoint = { profile_id: home, instance_id: 'owner', runtime_protocol: 1, control_home: null }

  const backend = {
    kind: 'python',
    command: payload.shim,
    shell: false,
    root: payload.repoDir,
    env: { PYTHONPATH: '', PYTHONHOME: '', HERMES_RUNTIME_DIR: payload.toolsDir },
    ...payloadTicketRuntime(payload, { HOME: userHome })
  }

  const server = net.createServer(socket =>
    socket.once('data', () => {
      socket.end(
        JSON.stringify({ protocol: 1, id: 1, ok: true, result: { ...endpoint, ticket: 'bundled-grant' } }) + '\n'
      )
    })
  )

  await new Promise<void>(resolve => server.listen(path.join(home, 'gateway.sock'), resolve))
  await fs.chmod(path.join(home, 'gateway.sock'), 0o600)
  await fs.chmod(home, 0o700)

  const close = async () => {
    closeGatewayTicketBridges()
    await new Promise<void>(resolve => server.close(() => resolve()))

    for (const dir of [home, resources, userHome]) {
      await fs.rm(dir, { recursive: true, force: true })
    }
  }

  const mint = () =>
    createGatewayTicketResolver(
      async () => backend,
      () => home
    )(endpoint, 'native-http')

  return { payload, userHome, mint, close }
}

// #138906 (RT-21): the Store/MSIX bundled backend is `kind: 'python'` but its command is the
// payload's CLI shim (bin/hermes.exe), which rejects the ticket helper's `-P -u -c`. This runs
// the real resolver and helper process against a payload laid out the way the builder stamps it.
test.skipIf(process.platform === 'win32')(
  'a bundled payload mints tickets with its interpreter, not the CLI shim',
  async () => {
    const { mint, close } = await bundledPayloadFixture()

    try {
      expect(await mint()).toBe('bundled-grant')
    } finally {
      await close()
    }
  },
  30_000
)

// A pyc the helper writes inside a signed macOS app breaks the app's code signature, and
// Gatekeeper then holds the next update behind a prompt. A module it imports from the payload
// (sitecustomize, beside the dependencies) has no pyc yet.
test.skipIf(process.platform === 'win32')(
  'the ticket helper caches bytecode outside the bundled payload',
  async () => {
    const { payload, userHome, mint, close } = await bundledPayloadFixture()
    await fs.writeFile(path.join(payload.sitePackages, 'sitecustomize.py'), 'PAYLOAD_MODULE = 1\n')
    vi.stubEnv('PYTHONDONTWRITEBYTECODE', '')
    vi.stubEnv('PYTHONPYCACHEPREFIX', '')

    try {
      expect(await mint()).toBe('bundled-grant')
      expect(await fs.readdir(payload.sitePackages)).toEqual(['sitecustomize.py'])
      const cached = await fs.readdir(path.join(userHome, '.cache', 'hermes-pycache'), { recursive: true })
      expect(cached.filter(name => path.basename(String(name)).startsWith('sitecustomize.'))).toHaveLength(1)
    } finally {
      vi.unstubAllEnvs()
      await close()
    }
  },
  30_000
)
