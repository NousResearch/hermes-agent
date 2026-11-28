import * as fs from 'node:fs'
import * as path from 'node:path'

import { type Browser, chromium, expect, test } from '@playwright/test'

import {
  composer,
  createCoreSandbox,
  currentSessionId,
  DESKTOP_ROOT,
  freePort,
  recordWebSockets,
  sandboxProcesses,
  send,
  webappLaunchLink,
  writeProviderHome
} from './harness'
import { startScriptedProvider } from './provider'

const nonce = Math.random()
  .toString(36)
  .slice(2, 8)
  .replace(/[^a-z0-9]/g, 'x')
  .padEnd(4, 'q')

test('Webapp model pill omits Ultrafast on an unsupported model with agent.service_tier: ultrafast', async () => {
  expect(
    fs.existsSync(path.join(DESKTOP_ROOT, 'dist-webapp', 'index.html')),
    'the Webapp renderer is built (pnpm run build:webapp)'
  ).toBe(true)

  const provider = await startScriptedProvider()
  const sandbox = createCoreSandbox('pill-tier')
  writeProviderHome(sandbox.hermesHome, provider.url, 'agent:\n  service_tier: ultrafast\n')
  const port = await freePort()
  fs.appendFileSync(path.join(sandbox.hermesHome, 'config.yaml'), `\nwebapp:\n  enabled: true\n  port: ${port}\n`)
  let browser: Browser | null = null

  try {
    const link = webappLaunchLink(sandbox)
    browser = await chromium.launch({ executablePath: process.env.HERMES_E2E_CHROMIUM || undefined })
    const page = await (await browser.newContext({ viewport: { width: 1280, height: 860 } })).newPage()
    const ws = recordWebSockets(page)

    await page.goto(link)
    await expect(composer(page)).toBeVisible({ timeout: 180_000 })

    const pill = page.locator('[data-tour="model-pill"]')
    await expect(pill).toContainText('Mock Model', { timeout: 60_000 })
    await expect(pill).not.toContainText(/Ultrafast|Fast/)

    const userMarker = `U1-tier${nonce}`
    const replyMarker = `A1-tier${nonce}`
    provider.script(userMarker, [{ text: [`${replyMarker} `, 'done'] }])
    await send(page, `${userMarker} hello`, 'Enter', ws)
    await expect(page.getByText(replyMarker).first()).toBeVisible({ timeout: 120_000 })
    await expect.poll(() => currentSessionId(page)).not.toBe('')

    await expect(pill).toContainText('Mock Model')
    await expect(pill).not.toContainText(/Ultrafast|Fast/)

    await page.reload()
    await expect(composer(page)).toBeVisible({ timeout: 120_000 })
    await expect(page.getByText(replyMarker).first()).toBeVisible({ timeout: 60_000 })
    await expect(pill).toContainText('Mock Model')
    await expect(pill).not.toContainText(/Ultrafast|Fast/)
  } finally {
    await browser?.close().catch(() => undefined)

    for (const proc of sandboxProcesses(sandbox)) {
      try {
        process.kill(proc.pid, 'SIGKILL')
      } catch {
        /* already gone */
      }
    }

    await provider.close()
    sandbox.cleanup()
  }
})
