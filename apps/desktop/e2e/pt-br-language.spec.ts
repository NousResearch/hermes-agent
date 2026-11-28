import fs from 'node:fs'
import path from 'node:path'

import { parse } from 'yaml'

import { type MockBackendFixture, setupMockBackend, waitForAppReady } from './fixtures'
import { expect, test } from './test'

let fixture: MockBackendFixture | null = null

test.afterEach(async () => {
  await fixture?.cleanup()
  fixture = null
})

test('selects Brazilian Portuguese in Appearance and keeps it after reloading', async () => {
  test.setTimeout(180_000)
  fixture = await setupMockBackend({ extraDisplayConfig: '  language: en' })
  const { page, sandbox } = fixture
  await waitForAppReady(fixture, 120_000)
  await page.evaluate(() => {
    location.hash = '#/settings?tab=config:appearance&setting=appearance.language'
  })
  await page.getByRole('button', { name: 'Switch language' }).last().click()
  await page.getByPlaceholder('Search languages…').fill('pt-br')
  await expect(page.getByRole('option', { name: /Português \(Brasil\)/ })).toHaveAttribute('aria-selected', 'true')
  await page.getByPlaceholder('Search languages…').press('Enter')
  await expect(page.locator('html')).toHaveAttribute('lang', 'pt-br')
  await expect(page.getByText('Idioma', { exact: true })).toBeVisible()
  const configPath = path.join(sandbox.hermesHome, 'config.yaml')
  await expect.poll(() => parse(fs.readFileSync(configPath, 'utf8')).display?.language).toBe('pt-br')
  await page.screenshot({ path: test.info().outputPath('settings-pt-br.png'), fullPage: true })
  await page.reload()
  await expect(page.locator('html')).toHaveAttribute('lang', 'pt-br', { timeout: 60_000 })
  await expect(page.getByText('Idioma', { exact: true })).toBeVisible()
  await page
    .getByRole('button', { name: /idioma/i })
    .last()
    .click()
  await page.locator('[cmdk-input]').last().fill('English')
  await expect(page.getByRole('option', { name: /English/ })).toHaveAttribute('aria-selected', 'true')
  await page.locator('[cmdk-input]').last().press('Enter')
  await expect(page.locator('html')).toHaveAttribute('lang', 'en')
  await expect.poll(() => parse(fs.readFileSync(configPath, 'utf8')).display?.language).toBe('en')
})
