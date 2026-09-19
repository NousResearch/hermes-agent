import fs from 'node:fs'
import path from 'node:path'

import {
  buildAppEnv,
  createSandbox,
  launchDesktop,
  type MockBackendFixture,
  waitForAppReady,
  writeEnvFile,
  writeMockProviderConfig
} from './fixtures'
import { startMockServer } from '../../../tests-js/scripts/mock-server'
import { expect, test } from './test'

// Settings' "Applies to" chips named every profile by its canonical slug
// (`alpha`, `beta`) while the Bots roster called the same agents by their Bot
// title / display name (#109686). The chip must read the presentation name the
// rest of the app uses; the slug stays the identity underneath.

let fixture: MockBackendFixture | null = null

function seedProfile(hermesHome: string, mockUrl: string, name: string, profileYaml: string): void {
  const dir = path.join(hermesHome, 'profiles', name)
  fs.mkdirSync(dir, { recursive: true })
  writeMockProviderConfig(dir, mockUrl)
  writeEnvFile(dir)
  fs.writeFileSync(path.join(dir, 'profile.yaml'), profileYaml, 'utf8')
}

test.beforeAll(async () => {
  const mock = await startMockServer()
  const sandbox = createSandbox('settings-chips')
  writeMockProviderConfig(sandbox.hermesHome, mock.url)
  writeEnvFile(sandbox.hermesHome)
  // A Bot Mode bot: the roster shows ui_meta['hermes-bots'].title.
  seedProfile(sandbox.hermesHome, mock.url, 'alpha', 'ui_meta:\n  hermes-bots:\n    title: Atlas Prime\n')
  // A renamed profile: `hermes profile rename` writes display_name.
  seedProfile(sandbox.hermesHome, mock.url, 'beta', 'display_name: Beacon\n')

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

test('Settings "Applies to" chips read the Bot title / display name, not the slug', async () => {
  test.setTimeout(300_000)
  const page = fixture!.page

  await page.evaluate(() => {
    window.location.hash = '#/settings'
  })

  // Chips carry their label as visible text (the profile rail's avatars only
  // carry it as aria-label), so filter by text to target the Settings strip.
  const chip = (label: string) => page.getByRole('button', { name: label, exact: true }).filter({ hasText: label })
  await expect(chip('Atlas Prime').first()).toBeVisible({ timeout: 60_000 })
  await expect(chip('Beacon').first()).toBeVisible({ timeout: 15_000 })
  // The slug is the identity, never the label.
  await expect(chip('alpha')).toHaveCount(0)
  await expect(chip('beta')).toHaveCount(0)
  // The root profile keeps its slug when it has neither title nor display name (control).
  await expect(chip('default').first()).toBeVisible()

  // Selecting by presentation label still scopes by the canonical slug: the
  // "applies to" note names the target by the same label as its chip.
  await chip('Atlas Prime').first().click()
  const note = page.getByRole('status').filter({ hasText: 'Atlas Prime' }).first()
  await expect(note).toBeVisible({ timeout: 10_000 })
  await expect(note).not.toContainText('alpha')

  await page.screenshot({ path: 'test-results/settings-scope-chips-bot-title.png' })
})
