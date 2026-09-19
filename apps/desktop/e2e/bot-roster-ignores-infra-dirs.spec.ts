import fs from 'node:fs'
import path from 'node:path'

import { startMockServer } from '../../../tests-js/scripts/mock-server'

import { writeEnvFile, writeMockProviderConfig } from '../../../tests-js/scripts/mock-provider-config'
import {
  buildAppEnv,
  createSandbox,
  launchDesktop,
  type MockBackendFixture,
  waitForAppReady
} from './fixtures'
import { RealSessionBuilder } from './real-session-builder'
import { expect, test } from './test'

// A directory under profiles/ is a profile only when it carries an identity
// file. Infrastructure dirs the runtime leaves there (`sessions/`, `logs/`
// from a log rotation or cron tick) and tombstoned (deleted) profiles must not
// surface as bots in the roster (#99392, #95188).

let fixture: MockBackendFixture | null = null

async function seedBot(hermesHome: string, mockUrl: string, name: string): Promise<void> {
  const dir = path.join(hermesHome, 'profiles', name)
  fs.mkdirSync(dir, { recursive: true })
  writeMockProviderConfig(dir, mockUrl)
  writeEnvFile(dir)

  const builder = await RealSessionBuilder.start(dir)

  try {
    await builder.createSession({ title: 'Bot Chat', turns: [`Hello ${name}`] })
  } finally {
    await builder.close()
  }
}

test.beforeAll(async () => {
  const mock = await startMockServer()
  const sandbox = createSandbox('bots-infra-dirs')
  writeMockProviderConfig(sandbox.hermesHome, mock.url)
  writeEnvFile(sandbox.hermesHome)
  await seedBot(sandbox.hermesHome, mock.url, 'alpha')

  // Marker-less infrastructure dirs, exactly the shells a cron tick / log rotation leaves behind.
  for (const stray of ['sessions', 'logs']) {
    fs.mkdirSync(path.join(sandbox.hermesHome, 'profiles', stray, 'cron'), { recursive: true })
  }

  // A deleted profile: identity file present, tombstone in profiles/.deleted/<name>.
  const ghost = path.join(sandbox.hermesHome, 'profiles', 'ghost')
  fs.mkdirSync(ghost, { recursive: true })
  fs.writeFileSync(path.join(ghost, 'profile.yaml'), 'ui_meta:\n  hermes-bots: {}\n')
  fs.mkdirSync(path.join(sandbox.hermesHome, 'profiles', '.deleted'), { recursive: true })
  fs.writeFileSync(path.join(sandbox.hermesHome, 'profiles', '.deleted', 'ghost'), '')

  const { app, page } = await launchDesktop(buildAppEnv(sandbox))

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
  fixture = null
})

test('the Bots roster lists real bots only — never profiles/ infra dirs or tombstones', async () => {
  test.setTimeout(240_000)
  const page = fixture!.page

  await page
    .getByRole('button', { name: 'Bots', exact: true })
    .or(page.getByRole('tab', { name: 'Bots', exact: true }))
    .first()
    .click()
  await expect(page.getByRole('button', { name: 'New bot or group chat' })).toBeVisible()

  const roster = page.locator('[data-slot="bots-roster"]')
  await expect(roster.locator('[data-roster-key="local::alpha"]')).toBeVisible({ timeout: 30_000 })

  for (const stray of ['sessions', 'logs', 'ghost', '.deleted']) {
    await expect(roster.locator(`[data-roster-key="local::${stray}"]`)).toHaveCount(0)
  }
})
