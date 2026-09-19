/**
 * E2E: the Bots roster must load when a profile.yaml carries an unquoted ISO
 * timestamp under `ui_meta` (YAML parses it as a datetime).
 *
 * On a gateway without the serialization guard, `profiles.list` never answers
 * over /api/ws and the roster spins forever (#92506). The assertion is the
 * rendered roster rows within a bounded window.
 */

import * as fs from 'node:fs'
import * as path from 'node:path'

import { startMockServer } from '../../../tests-js/scripts/mock-server'

import { writeEnvFile, writeMockProviderConfig } from '../../../tests-js/scripts/mock-provider-config'
import {
  buildAppEnv,
  createSandbox,
  launchDesktop,
  type MockBackendFixture,
  waitForAppReady
} from './fixtures'
import { expect, type Page, test } from './test'

const SHOT_DIR = process.env.BOT_LIFECYCLE_SHOT_DIR || ''

let fixture: MockBackendFixture | null = null

async function capture(page: Page, name: string): Promise<void> {
  if (!SHOT_DIR) {
    return
  }

  fs.mkdirSync(SHOT_DIR, { recursive: true })
  await page.screenshot({ path: path.join(SHOT_DIR, `${name}.png`) })
}

test.describe('Bots roster — profile with timestamp metadata', () => {
  test.beforeAll(async () => {
    test.setTimeout(300_000)
    const mock = await startMockServer()
    const sandbox = createSandbox('bot-tsmeta')
    writeMockProviderConfig(sandbox.hermesHome, mock.url)
    writeEnvFile(sandbox.hermesHome)

    for (const name of ['alpha', 'zeta']) {
      const dir = path.join(sandbox.hermesHome, 'profiles', name)
      fs.mkdirSync(dir, { recursive: true })
      writeMockProviderConfig(dir, mock.url)
      writeEnvFile(dir)
    }

    // The trigger: an unquoted ISO-8601 timestamp — yaml.safe_load → datetime.
    fs.writeFileSync(
      path.join(sandbox.hermesHome, 'profiles', 'zeta', 'profile.yaml'),
      'ui_meta:\n  hermes-bots:\n    created: 2026-08-22T00:00:00Z\n',
      'utf8'
    )

    const env = buildAppEnv(sandbox)
    const { app, page } = await launchDesktop(env)
    fixture = {
      app,
      page,
      mock,
      mockUrl: mock.url,
      sandbox,
      cleanup: async () => {
        await app.close().catch(() => undefined)
        await mock.close()
        sandbox.cleanup()
      }
    }
    await waitForAppReady(fixture, 120_000)
  })

  test.afterAll(async () => {
    await fixture?.cleanup()
  })

  test('roster rows render instead of spinning forever', async () => {
    test.setTimeout(300_000)
    const { page } = fixture!

    const tab = page
      .getByRole('button', { name: 'Bots', exact: true })
      .or(page.getByRole('tab', { name: 'Bots', exact: true }))
      .first()

    await tab.click()
    await expect(page.getByRole('button', { name: 'New bot or group chat' })).toBeVisible({ timeout: 60_000 })

    const rows = page.locator('[data-slot="bots-roster"] [data-roster-key^="local::"]')

    try {
      await expect(rows.first()).toBeVisible({ timeout: 60_000 })
    } finally {
      await capture(page, 'tsmeta-roster')
      console.log(`[probe] roster rows after 60s: ${await rows.count()}`)
    }

    await expect(page.locator('[data-roster-key="local::zeta"]').first()).toBeVisible({ timeout: 30_000 })
  })
})
