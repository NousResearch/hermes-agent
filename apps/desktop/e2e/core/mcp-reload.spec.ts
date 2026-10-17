/**
 * Desktop applies a profile's MCP changes to the chat that asked (#58414): `/reload-mcp now`
 * from the composer and Capabilities ▸ Connectors ▸ Reload MCP servers both reconcile the
 * profile and commit the calling chat's new selection. A server added to config while a chat is
 * open is unreachable before the reload and callable in that same chat after it.
 *
 * The fixture servers log every message they receive with their pid. A newly added server is
 * proven by its first `initialize`; a reached tool is a `tools/call`. Servers whose config did
 * not change keep their connection (the unified gateway reconciles, it does not restart them).
 */

import { execFile, spawn } from 'node:child_process'
import fs from 'node:fs'
import path from 'node:path'
import { promisify } from 'node:util'

import { type ElectronApplication, expect, test } from '@playwright/test'
import { parse, stringify } from 'yaml'

import {
  coreAppEnv,
  createCoreSandbox,
  currentSessionId,
  launchCoreApp,
  REPO_ROOT,
  send,
  waitForInteractive,
  writeProviderHome
} from './harness'
import { startScriptedProvider } from './provider'

const FIXTURE_SERVER = path.join(REPO_ROOT, 'tests', 'e2e', 'core', 'mcp_plugins', 'mcp_fixture_server.py')

interface Inbound {
  msg?: { method?: string; params?: { arguments?: { nonce?: string } } }
  pid: number
}

function inbound(log: string): Inbound[] {
  if (!fs.existsSync(log)) {
    return []
  }

  return fs
    .readFileSync(log, 'utf8')
    .split('\n')
    .filter(Boolean)
    .map(line => JSON.parse(line) as Inbound)
}

const serverPids = (log: string) =>
  inbound(log)
    .filter(record => record.msg?.method === 'initialize')
    .map(record => record.pid)

const isAlive = (pid: number) => {
  try {
    process.kill(pid, 0)

    return true
  } catch {
    return false
  }
}

const callNonces = (log: string) =>
  inbound(log)
    .filter(record => record.msg?.method === 'tools/call')
    .map(record => record.msg?.params?.arguments?.nonce)

const fixtureServer = (name: string, python: string, log: string) => ({
  command: python,
  args: [FIXTURE_SERVER],
  connect_timeout: 60,
  env: { MCPE2E_LOG: log, MCPE2E_NAME: name, PYTHONPATH: REPO_ROOT }
})

test('reloads the profile MCP servers from the composer and from Connectors', async () => {
  const python = process.env.HERMES_E2E_PYTHON
  test.skip(!python, 'the MCP fixture server needs HERMES_E2E_PYTHON (an interpreter with the mcp SDK)')

  const provider = await startScriptedProvider()
  const sandbox = createCoreSandbox('mcp-reload')
  const alphaLog = path.join(sandbox.root, 'alpha.jsonl')
  const gammaLog = path.join(sandbox.root, 'gamma.jsonl')
  const deltaLog = path.join(sandbox.root, 'delta.jsonl')
  const configPath = path.join(sandbox.hermesHome, 'config.yaml')

  writeProviderHome(
    sandbox.hermesHome,
    provider.url,
    stringify({ mcp_servers: { alpha: fixtureServer('alpha', python!, alphaLog) } })
  )
  // Own this sandbox's gateway directly. Service discovery is outside this reload test and can see
  // the runner account's installed launchd service even with HOME redirected on macOS.
  const env = coreAppEnv(sandbox)
  const backend = spawn(python!, ['-m', 'hermes_cli.main', 'gateway', 'run', '--quiet'], {
    cwd: REPO_ROOT,
    env,
    stdio: ['ignore', 'pipe', 'pipe']
  })
  const backendLog: string[] = []
  backend.stdout.on('data', chunk => backendLog.push(chunk.toString()))
  backend.stderr.on('data', chunk => backendLog.push(chunk.toString()))
  let app: ElectronApplication | undefined

  const callServer = (server: string, nonce: string) => ({
    toolCalls: [{ name: 'tool_call', args: { calls: [{ name: `mcp__${server}__ro_probe`, arguments: { nonce } }] } }]
  })

  // The app rewrites config.yaml after launch, so edit the mapping rather than appending text.
  const addServer = (name: string, log: string) => {
    const config = parse(fs.readFileSync(configPath, 'utf8')) as { mcp_servers: Record<string, unknown> }
    config.mcp_servers[name] = fixtureServer(name, python!, log)
    fs.writeFileSync(configPath, stringify(config))
  }

  try {
    await expect
      .poll(
        async () => {
          if (backend.exitCode !== null) {
            throw new Error(`sandbox gateway exited before ready:\n${backendLog.join('')}`)
          }

          const { stdout } = await promisify(execFile)(
            python!,
            [
              '-c',
              'import sys; from hermes_cli.gateway_runtime import discover_gateway_endpoint; print(discover_gateway_endpoint(sys.argv[1]).state)',
              sandbox.hermesHome
            ],
            { cwd: REPO_ROOT, env }
          )

          return stdout.trim()
        },
        { message: 'the isolated canonical gateway is ready', timeout: 60_000 }
      )
      .toBe('ready')
    const launched = await launchCoreApp(env)
    app = launched.app
    const page = launched.page
    await waitForInteractive(app, page)
    provider.script('U1-mcpreload', [{ text: ['A1-mcpreload ready'] }])
    await send(page, 'U1-mcpreload hello')
    await expect(page.getByText('A1-mcpreload ready')).toBeVisible()
    await expect.poll(() => serverPids(alphaLog).length, { message: 'the MCP server started' }).toBeGreaterThan(0)
    const chat = await currentSessionId(page)
    const alphaPids = serverPids(alphaLog)

    addServer('gamma', gammaLog)
    provider.script('U2-mcpreload', [callServer('gamma', 'beforereload'), { text: ['A2-mcpreload tried'] }])
    await send(page, 'U2-mcpreload call gamma')
    await expect(page.getByText('A2-mcpreload tried')).toBeVisible()
    expect(callNonces(gammaLog), 'the chat reached a server added after it opened, before any reload').toEqual([])

    await send(page, '/reload-mcp now')
    await expect(page.getByText('MCP servers reloaded').first()).toBeVisible()
    await expect
      .poll(() => serverPids(gammaLog).length, { message: '/reload-mcp now started the added server' })
      .toBeGreaterThan(0)
    expect(alphaPids.every(isAlive), 'the reload kept the original server alive').toBe(true)

    provider.script('U3-mcpreload', [callServer('gamma', 'afterreload'), { text: ['A3-mcpreload called'] }])
    await send(page, 'U3-mcpreload call gamma')
    await expect(page.getByText('A3-mcpreload called')).toBeVisible()
    expect(await currentSessionId(page), 'the call ran in the chat that was open before the reload').toBe(chat)
    await expect
      .poll(() => callNonces(gammaLog), { message: 'the same chat reached the added server after the reload' })
      .toEqual(['afterreload'])

    addServer('delta', deltaLog)
    provider.script('U4-mcpreload', [callServer('delta', 'beforebutton'), { text: ['A4-mcpreload tried'] }])
    await send(page, 'U4-mcpreload call delta')
    await expect(page.getByText('A4-mcpreload tried')).toBeVisible()
    expect(callNonces(deltaLog), 'the chat reached the second added server before the button reload').toEqual([])

    await page.evaluate(() => {
      window.location.hash = '#/capabilities?tab=connectors'
    })
    await page.getByRole('button', { name: 'Reload MCP servers' }).click()
    await page.getByRole('button', { name: 'Reload now' }).click()
    await expect(page.getByText('MCP servers reloaded').first()).toBeVisible()
    await expect
      .poll(() => serverPids(deltaLog).length, { message: 'the Connectors button started the added server' })
      .toBeGreaterThan(0)
    // Session-scoped discovery may open an additional connection. The original profile connection
    // must survive either reload; counting every initialize across scopes would conflate the two.
    expect(alphaPids.every(isAlive), 'the button reload kept the original server alive').toBe(true)

    await page.evaluate(id => {
      window.location.hash = `#/${id}`
    }, chat)
    provider.script('U5-mcpreload', [callServer('delta', 'afterbutton'), { text: ['A5-mcpreload called'] }])
    await send(page, 'U5-mcpreload call delta')
    await expect(page.getByText('A5-mcpreload called')).toBeVisible()
    await expect
      .poll(() => callNonces(deltaLog), { message: 'the same chat reached the button-added server' })
      .toEqual(['afterbutton'])
  } finally {
    await app?.close()
    if (backend.exitCode === null && backend.signalCode === null) {
      const exited = new Promise<void>(resolve => backend.once('exit', () => resolve()))
      const deadline = setTimeout(() => backend.kill('SIGKILL'), 15_000)
      backend.kill('SIGTERM')
      await exited
      clearTimeout(deadline)
    }
    await provider.close()
    sandbox.cleanup()
  }
})
