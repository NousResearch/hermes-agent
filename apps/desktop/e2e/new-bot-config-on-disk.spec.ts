import { existsSync, readFileSync } from 'node:fs'
import { join } from 'node:path'

import { type MockBackendFixture, setupMockBackend, waitForAppReady } from './fixtures'
import { expect, test } from './test'

// A bot created from the New Bot dialog WITHOUT a clone source (the "inherited
// from launch profile" path) must land on disk with a usable `model:` block in
// its own config.yaml and its introduction turn must actually run — a profile
// whose config.yaml lacks the block is dead on arrival ("No LLM provider
// configured" on the first message). Verified against storage truth (the
// created profile's config.yaml under HERMES_HOME), not the toast.

let fixture: MockBackendFixture | null = null

type Page = MockBackendFixture['page']

async function openBots(page: Page): Promise<void> {
  const tab = page
    .getByRole('button', { name: 'Bots', exact: true })
    .or(page.getByRole('tab', { name: 'Bots', exact: true }))
    .first()

  await tab.click()
  await expect(page.getByRole('button', { name: 'New bot or group chat' })).toBeVisible()
}

test.beforeAll(async () => {
  fixture = await setupMockBackend()
  await waitForAppReady(fixture, 120_000)
})

// eslint-disable-next-line no-empty-pattern
test.afterEach(async ({}, info) => {
  if (!fixture) {
    return
  }

  await info.attach('native-window', { body: await fixture.page.screenshot(), contentType: 'image/png' })
  const cfg = join(fixture.sandbox.hermesHome, 'profiles', 'fresh-scout', 'config.yaml')
  await info.attach('created-config', {
    body: existsSync(cfg) ? readFileSync(cfg) : Buffer.from('<missing>'),
    contentType: 'text/plain'
  })
})

test.afterAll(async () => {
  await fixture?.cleanup()
  fixture = null
})

test('a fresh (non-clone) bot gets a runnable model block on disk', async () => {
  test.setTimeout(300_000)
  const page = fixture!.page

  await openBots(page)
  await page.getByRole('button', { name: 'New bot or group chat' }).click()
  await page.getByRole('menuitem', { name: 'New Bot' }).click()

  const dialog = page.getByRole('dialog', { name: 'New Bot' })
  await dialog.getByPlaceholder('inbox-triage').fill('fresh-scout')
  await dialog.getByPlaceholder('Inbox Triage').fill('Fresh Scout')
  await dialog.getByRole('button', { name: 'Advanced' }).click()

  // "Fresh profile" = no clone source; the model must be inherited, not cloned.
  await dialog.getByRole('combobox').filter({ hasText: /default|Fresh profile/ }).first().click()
  await page.getByRole('option', { name: 'Fresh profile (bundled skills)' }).click()
  await dialog.getByRole('button', { name: 'Create Bot' }).click()
  await expect(dialog).toBeHidden({ timeout: 60_000 })

  // Storage truth: the created profile's own config.yaml carries the model block.
  const cfgPath = join(fixture!.sandbox.hermesHome, 'profiles', 'fresh-scout', 'config.yaml')
  await expect.poll(() => existsSync(cfgPath), { timeout: 30_000 }).toBe(true)
  const cfg = readFileSync(cfgPath, 'utf8')
  expect(cfg).toMatch(/^model:/m)
  expect(cfg).toMatch(/provider:\s*mock/)
  expect(cfg).toMatch(/default:\s*mock-model/)
  // The inherited model rides a custom `providers:` gateway; its definition must travel with it
  // or the first turn dies with "Unknown provider 'mock'".
  expect(cfg).toMatch(/^providers:\n\s+mock:/m)
})
