import fs from 'node:fs'
import path from 'node:path'

import { setupMockBackend, waitForAppReady, writeMockProviderConfig } from './fixtures'
import { expect, test } from './test'

test('both SOUL editors explain a missing file and save it without changing personality config', async () => {
  test.setTimeout(180_000)
  const fixture = await setupMockBackend()
  const { page, sandbox, mockUrl } = fixture
  const profileName = 'soul-test'
  const profileHome = path.join(sandbox.hermesHome, 'profiles', profileName)
  fs.mkdirSync(profileHome, { recursive: true })
  writeMockProviderConfig(profileHome, mockUrl, undefined, 'personalities:\n  helper: "Keep answers brief."')
  const configPath = path.join(profileHome, 'config.yaml')
  const config = fs.readFileSync(configPath, 'utf8')
  const soulPath = path.join(profileHome, 'SOUL.md')
  const missing = /No SOUL\.md file exists for this profile/

  try {
    await waitForAppReady(fixture, 120_000)
    await page.getByRole('button', { name: 'Manage profiles…' }).click()
    await page.locator(`[data-panel-row="${profileName}"] [data-slot="row-button"]`).click()
    await expect(page.locator('.cm-content')).toBeVisible()

    // First-use initialization can seed a default SOUL.md. Remove only this
    // sandbox profile's file after initialization to exercise the missing state.
    fs.rmSync(soulPath, { force: true })
    await page.locator('[data-panel-row="default"] [data-slot="row-button"]').click()
    await page.locator(`[data-panel-row="${profileName}"] [data-slot="row-button"]`).click()
    await expect(page.locator('.cm-content')).toBeVisible()
    expect(fs.existsSync(soulPath)).toBe(false)
    await expect(page.getByText(missing)).toBeVisible()
    await page.screenshot({ path: test.info().outputPath('missing-soul.png') })

    // Dismiss without saving: the sidebar editor must report the same state,
    // not silently create a file merely because either editor was opened.
    await page.getByRole('button', { name: 'Close profiles' }).click()
    const square = page.locator('[data-slot="profile-rail"]').getByRole('button', { name: profileName, exact: true })
    await square.click({ button: 'right' })
    await page.getByRole('menuitem', { name: 'Edit SOUL.md…' }).click()
    const dialog = page.getByRole('dialog')
    await expect(dialog.getByText(missing)).toBeVisible()
    expect(fs.existsSync(soulPath)).toBe(false)
    await page.screenshot({ path: test.info().outputPath('missing-soul-dialog.png') })

    const instructions = '# Soul test\n\nGive concise answers.'
    await dialog.locator('.cm-content').fill(instructions)
    await dialog.getByRole('button', { name: 'Save SOUL.md' }).click()
    await expect(dialog).toBeHidden()
    expect(fs.readFileSync(soulPath, 'utf8')).toBe(instructions)
    expect(fs.readFileSync(configPath, 'utf8')).toBe(config)

    await page.getByRole('button', { name: 'Manage profiles…' }).click()
    await page.locator(`[data-panel-row="${profileName}"] [data-slot="row-button"]`).click()
    await expect(page.locator('.cm-content')).toContainText('Give concise answers.')
    await expect(page.getByText(missing)).toHaveCount(0)
  } finally {
    await fixture.cleanup()
  }
})
