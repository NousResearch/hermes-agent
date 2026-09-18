import fs from 'node:fs'
import path from 'node:path'

import { type MockBackendFixture, setupMockBackend, waitForAppReady } from './fixtures'
import { expect, test } from './test'

// The New Bot dialog derived the profile id with an ASCII-only slugify, so a
// CJK name like 小助手 slugged to '' and Create Bot stayed disabled (#96153).
// The id must be a deterministic ASCII token and the entered name must
// survive as the bot's display title.

let fixture: MockBackendFixture | null = null

const SHOTS = process.env.BOTS_I18N_SHOTS || ''

async function shot(page: MockBackendFixture['page'], name: string): Promise<void> {
  if (!SHOTS) {
    return
  }

  fs.mkdirSync(SHOTS, { recursive: true })
  await page.screenshot({ path: path.join(SHOTS, `${name}.png`) })
}

test.beforeAll(async () => {
  fixture = await setupMockBackend()
  await waitForAppReady(fixture, 120_000)
})

test.afterAll(async () => {
  await fixture?.cleanup()
  fixture = null
})

test('a CJK bot name enables Create Bot and lands as the display title over an ASCII profile id', async () => {
  test.setTimeout(300_000)
  const page = fixture!.page

  const tab = page
    .getByRole('button', { name: 'Bots', exact: true })
    .or(page.getByRole('tab', { name: 'Bots', exact: true }))
    .first()
  await tab.click()
  await page.getByRole('button', { name: 'New bot or group chat' }).click()
  await page.getByRole('menuitem', { name: 'New Bot' }).click()

  const dialog = page.getByRole('dialog', { name: 'New Bot' })
  await dialog.getByPlaceholder('inbox-triage').fill('小助手')

  const create = dialog.getByRole('button', { name: 'Create Bot' })
  await shot(page, 'cjk-name-typed')
  await expect(create).toBeEnabled({ timeout: 10_000 })
  await create.click()
  await expect(dialog).toBeHidden({ timeout: 30_000 })

  // Roster row carries the entered name as its title …
  const row = page.getByRole('button', { name: /^小助手 · @u5c0f-u52a9-u624b/ }).filter({ visible: true }).first()
  await expect(row).toBeVisible({ timeout: 30_000 })
  await shot(page, 'cjk-bot-row')

  // … while the backend profile id is the deterministic ASCII form.
  const profileDir = path.join(fixture!.sandbox.hermesHome, 'profiles', 'u5c0f-u52a9-u624b')
  await expect.poll(() => fs.existsSync(profileDir), { timeout: 30_000 }).toBe(true)
  const profileYaml = fs.readFileSync(path.join(profileDir, 'profile.yaml'), 'utf8')
  expect(profileYaml).toContain('小助手')
})
