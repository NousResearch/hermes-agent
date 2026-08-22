import { type MockBackendFixture, setupMockBackend, waitForAppReady } from './fixtures'
import { expect, test } from './test'

let fixture: MockBackendFixture | null = null

test.beforeAll(async () => {
  fixture = await setupMockBackend()
  await waitForAppReady(fixture, 120_000)
})

test.afterAll(async () => {
  await fixture?.cleanup()
  fixture = null
})

test('copies a sent user prompt without opening the edit composer', async () => {
  const page = fixture!.page
  const prompt = 'copy this prompt `verbatim` 한글'
  const composer = page.locator('[data-slot="composer-rich-input"]').first()

  await composer.fill(prompt)
  await expect(composer).toHaveText(prompt)
  await page.keyboard.press('Enter')
  await page.waitForFunction(() => (document.body.textContent ?? '').includes('mock inference server'), undefined, {
    timeout: 60_000
  })

  const userMessage = page.locator('[data-slot="aui_user-message-root"]').first()
  const copy = userMessage.getByRole('button', { name: 'Copy' })

  const readClipboard = () =>
    page.evaluate(() =>
      (window as unknown as { hermesDesktop: { readClipboard: () => Promise<string> } }).hermesDesktop.readClipboard()
    )

  const sentinel = 'clipboard-before-user-message-copy'

  await page.evaluate(
    value =>
      (
        window as unknown as { hermesDesktop: { writeClipboard: (text: string) => Promise<boolean> } }
      ).hermesDesktop.writeClipboard(value),
    sentinel
  )
  await expect.poll(readClipboard, { timeout: 10_000 }).toBe(sentinel)

  await userMessage.hover()
  await copy.click()

  await expect.poll(readClipboard, { timeout: 10_000 }).toBe(prompt)
  await expect(page.locator('[data-slot="aui_edit-composer-root"]')).toHaveCount(0)
})
