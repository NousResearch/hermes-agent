import { expect, test } from './test'

import { type MockBackendFixture, setupMockBackend, waitForAppReady } from './fixtures'

let fixture: MockBackendFixture | null = null

const PROMPT = 'E2E conversation survives stacked tool tab switches'

test.beforeAll(async () => {
  fixture = await setupMockBackend()
  await waitForAppReady(fixture, 120_000)
})

test.afterAll(async () => {
  await fixture?.cleanup()
  fixture = null
})

test('restores the conversation after visiting the terminal tab', async () => {
  const page = fixture!.page
  const composer = page.locator('[contenteditable="true"]').first()

  await composer.click()
  await composer.type(PROMPT)
  await page.keyboard.press('Enter')
  const transcript = page.locator('[data-slot="aui_thread-viewport"]')

  await transcript.getByText(PROMPT).waitFor({ state: 'visible', timeout: 30_000 })
  await transcript.getByText(/mock inference server/).waitFor({ state: 'visible', timeout: 60_000 })

  await page.evaluate(() => {
    localStorage.setItem(
      'hermes.desktop.layoutTree.v2',
      JSON.stringify({
        active: 'workspace',
        id: 'e2e-main',
        panes: ['workspace', 'terminal', 'review'],
        type: 'group'
      })
    )
  })
  await page.reload()
  await waitForAppReady(fixture!, 120_000)
  await transcript.getByText(PROMPT).waitFor({ state: 'visible', timeout: 60_000 })

  await page.locator('[data-tree-tab="terminal"]').click()
  await expect(page.locator('[data-tree-tab="terminal"]')).toHaveAttribute('data-active', 'true')

  await page.locator('[data-tree-tab="workspace"]').click()
  await expect(transcript.getByText(PROMPT)).toBeVisible()
  await expect(page.locator('[data-slot="composer-root"]')).toBeVisible()
})
