/**
 * Desktop reloads the live session's MCP servers (#58414): `/reload-mcp now`
 * from the composer and Capabilities ▸ Connectors ▸ Reload MCP servers both
 * restart the profile's real MCP server, and a server added to config while a
 * chat is open becomes callable in that same chat after the reload — not
 * before it.
 *
 * Before, Desktop refused `/reload-mcp` ("not shown in the desktop slash
 * palette … use the relevant desktop control") and had no such control. The
 * fixture servers log every message they receive with their pid, so a restart
 * is a new pid sending `initialize` and a reached tool is a `tools/call`.
 */

import fs from 'node:fs'
import path from 'node:path'

import { expect, test } from '@playwright/test'
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
  const log = path.join(sandbox.root, 'alpha.jsonl')
  const gammaLog = path.join(sandbox.root, 'gamma.jsonl')
  const configPath = path.join(sandbox.hermesHome, 'config.yaml')

  writeProviderHome(
    sandbox.hermesHome,
    provider.url,
    stringify({ mcp_servers: { alpha: fixtureServer('alpha', python!, log) } })
  )
  const { app, page } = await launchCoreApp(coreAppEnv(sandbox))

  const callGamma = (nonce: string) => ({
    toolCalls: [{ name: 'tool_call', args: { calls: [{ name: 'mcp__gamma__ro_probe', arguments: { nonce } }] } }]
  })

  try {
    await waitForInteractive(app, page)
    provider.script('U1-mcpreload', [{ text: ['A1-mcpreload ready'] }])
    await send(page, 'U1-mcpreload hello')
    await expect(page.getByText('A1-mcpreload ready')).toBeVisible()
    await expect.poll(() => serverPids(log).length, { message: 'the MCP server started' }).toBeGreaterThan(0)
    const chat = await currentSessionId(page)

    const beforeSlash = serverPids(log)
    await send(page, '/reload-mcp now')
    await expect(page.getByText('MCP servers reloaded').first()).toBeVisible()
    await expect
      .poll(() => serverPids(log).length, { message: '/reload-mcp now restarted the MCP server' })
      .toBeGreaterThan(beforeSlash.length)

    // The app writes its own keys into config.yaml after launch, so edit the mapping, never append text.
    const config = parse(fs.readFileSync(configPath, 'utf8')) as { mcp_servers: Record<string, unknown> }
    config.mcp_servers.gamma = fixtureServer('gamma', python!, gammaLog)
    fs.writeFileSync(configPath, stringify(config))
    provider.script('U2-mcpreload', [callGamma('beforereload'), { text: ['A2-mcpreload tried'] }])
    await send(page, 'U2-mcpreload call gamma')
    await expect(page.getByText('A2-mcpreload tried')).toBeVisible()
    expect(callNonces(gammaLog), 'the chat reached a server added after it opened, before any reload').toEqual([])

    const beforeButton = serverPids(log)
    await page.evaluate(() => {
      window.location.hash = '#/capabilities?tab=connectors'
    })
    await page.getByRole('button', { name: 'Reload MCP servers' }).click()
    await page.getByRole('button', { name: 'Reload now' }).click()
    await expect(page.getByText('MCP servers reloaded').first()).toBeVisible()
    await expect
      .poll(() => serverPids(log).length, { message: 'the Connectors button restarted the MCP server' })
      .toBeGreaterThan(beforeButton.length)

    await page.evaluate(id => {
      window.location.hash = `#/${id}`
    }, chat)
    provider.script('U3-mcpreload', [callGamma('afterreload'), { text: ['A3-mcpreload called'] }])
    await send(page, 'U3-mcpreload call gamma')
    await expect(page.getByText('A3-mcpreload called')).toBeVisible()
    expect(await currentSessionId(page), 'the call ran in the chat that was open before the reload').toBe(chat)
    await expect
      .poll(() => callNonces(gammaLog), { message: 'the same chat reached the newly added server after the reload' })
      .toEqual(['afterreload'])
  } finally {
    await app.close()
    await provider.close()
    sandbox.cleanup()
  }
})
