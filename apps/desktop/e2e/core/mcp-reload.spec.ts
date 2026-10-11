/**
 * Desktop reloads the live session's MCP servers (#58414): `/reload-mcp now`
 * from the composer and Capabilities ▸ Connectors ▸ Reload MCP servers both
 * restart the profile's real MCP server, and say so.
 *
 * Before, Desktop refused `/reload-mcp` ("not shown in the desktop slash
 * palette … use the relevant desktop control") and had no such control. The
 * fixture server logs every message it receives with its pid, so a restart is
 * a new pid sending `initialize`.
 */

import fs from 'node:fs'
import path from 'node:path'

import { expect, test } from '@playwright/test'

import {
  coreAppEnv,
  createCoreSandbox,
  launchCoreApp,
  REPO_ROOT,
  send,
  waitForInteractive,
  writeProviderHome
} from './harness'
import { startScriptedProvider } from './provider'

const FIXTURE_SERVER = path.join(REPO_ROOT, 'tests', 'e2e', 'core', 'mcp_plugins', 'mcp_fixture_server.py')

function serverPids(log: string): number[] {
  if (!fs.existsSync(log)) {
    return []
  }

  return fs
    .readFileSync(log, 'utf8')
    .split('\n')
    .filter(Boolean)
    .map(line => JSON.parse(line) as { msg?: { method?: string }; pid: number })
    .filter(record => record.msg?.method === 'initialize')
    .map(record => record.pid)
}

test('reloads the profile MCP servers from the composer and from Connectors', async () => {
  const python = process.env.HERMES_E2E_PYTHON
  test.skip(!python, 'the MCP fixture server needs HERMES_E2E_PYTHON (an interpreter with the mcp SDK)')

  const provider = await startScriptedProvider()
  const sandbox = createCoreSandbox('mcp-reload')
  const log = path.join(sandbox.root, 'alpha.jsonl')

  const mcpServers = [
    'mcp_servers:',
    '  alpha:',
    `    command: ${JSON.stringify(python)}`,
    `    args: [${JSON.stringify(FIXTURE_SERVER)}]`,
    '    connect_timeout: 60',
    '    env:',
    `      MCPE2E_LOG: ${JSON.stringify(log)}`,
    '      MCPE2E_NAME: alpha',
    `      PYTHONPATH: ${JSON.stringify(REPO_ROOT)}`
  ].join('\n')

  writeProviderHome(sandbox.hermesHome, provider.url, mcpServers)
  const { app, page } = await launchCoreApp(coreAppEnv(sandbox))

  try {
    await waitForInteractive(app, page)
    provider.script('U1-mcpreload', [{ text: ['A1-mcpreload ready'] }])
    await send(page, 'U1-mcpreload hello')
    await expect(page.getByText('A1-mcpreload ready')).toBeVisible()
    await expect.poll(() => serverPids(log).length, { message: 'the MCP server started' }).toBeGreaterThan(0)

    const beforeSlash = serverPids(log)
    await send(page, '/reload-mcp now')
    await expect(page.getByText('MCP servers reloaded').first()).toBeVisible()
    await expect
      .poll(() => serverPids(log).length, { message: '/reload-mcp now restarted the MCP server' })
      .toBeGreaterThan(beforeSlash.length)

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
  } finally {
    await app.close()
    await provider.close()
    sandbox.cleanup()
  }
})
