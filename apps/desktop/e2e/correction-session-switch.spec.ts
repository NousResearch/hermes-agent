/**
 * Regression coverage for a correction sent during a live response, then a
 * warm session switch away and back. The correction is an accepted user turn,
 * not an optimistic duplicate of the original prompt, and its relative place
 * in the transcript must survive the resume reconciliation.
 */

import { type TestInfo } from '@playwright/test'

import { expect, test, type Page } from './test'

import { type MockBackendFixture, setupMockBackend, waitForAppReady } from './fixtures'
import { MOCK_REPLY } from './mock-server'

const ORIGINAL_PROMPT = 'E2E original prompt must remain singular after a correction.'
const CORRECTION = 'E2E correction must stay after the original prompt.'

async function send(page: Page, text: string): Promise<void> {
  const composer = page.locator('[contenteditable="true"]').first()
  await composer.waitFor({ state: 'visible', timeout: 15_000 })
  await composer.click()
  await composer.type(text, { delay: 5 })
  await page.keyboard.press('Enter')
}

async function waitForTranscriptText(page: Page, text: string): Promise<void> {
  await page.waitForFunction(
    (expected: string) => (document.querySelector('[data-slot="aui_thread-viewport"]')?.textContent ?? '').includes(expected),
    text,
    { timeout: 30_000 },
  )
}

async function textNodeOccurrences(page: Page, text: string): Promise<number> {
  return page.evaluate((expected: string) => {
    const viewport = document.querySelector('[data-slot="aui_thread-viewport"]')
    if (!viewport) return 0

    const walker = document.createTreeWalker(viewport, NodeFilter.SHOW_TEXT)
    let count = 0
    while (walker.nextNode()) {
      if (walker.currentNode.textContent?.includes(expected)) {
        count += 1
      }
    }
    return count
  }, text)
}

async function transcriptTextOrder(page: Page): Promise<string[]> {
  return page.evaluate(() => {
    const viewport = document.querySelector('[data-slot="aui_thread-viewport"]')
    if (!viewport) return []

    return Array.from(viewport.querySelectorAll<HTMLElement>('[data-role="message"], [data-message-id]'))
      .map(message => message.textContent?.trim() ?? '')
      .filter(Boolean)
  })
}

async function openFreshDraft(page: Page): Promise<void> {
  await page.locator('[data-slot="sidebar"] button[aria-label="New session"]').first().click()
  await page.waitForFunction(
    (original: string) => !(document.querySelector('[data-slot="aui_thread-viewport"]')?.textContent ?? '').includes(original),
    ORIGINAL_PROMPT,
    { timeout: 15_000 },
  )
}

async function reopenOriginalSession(page: Page): Promise<void> {
  // The mock's first streamed token becomes the generated sidebar title, not
  // the user's original prompt. It may still be only "Hello" when we switch.
  const row = page.locator('[data-slot="sidebar"] button').filter({ hasText: MOCK_REPLY.split(' ')[0] }).first()
  await row.waitFor({ state: 'visible', timeout: 30_000 })
  await row.click()
  await waitForTranscriptText(page, ORIGINAL_PROMPT)
}

function relevantOrder(messages: string[]): string[] {
  return messages.filter(message => message.includes(ORIGINAL_PROMPT) || message.includes(CORRECTION))
}

test.describe('correction session switch', () => {
  let fixture: MockBackendFixture | null = null

  test.beforeEach(async () => {
    fixture = await setupMockBackend({
      mockServer: { holdFirstStreamForPrompt: ORIGINAL_PROMPT },
    })
    await waitForAppReady(fixture, 120_000)
  })

  test.afterEach(async () => {
    await fixture?.cleanup()
    fixture = null
  })

  test('keeps a live correction in place and does not duplicate its original prompt after switching sessions', async ({}, testInfo: TestInfo) => {
    const { mock, page } = fixture!

    await send(page, ORIGINAL_PROMPT)
    await mock.waitForHeldStream()
    await waitForTranscriptText(page, ORIGINAL_PROMPT)

    // While the original response is live, Enter routes this through
    // session.redirect. The renderer records the accepted correction once as a
    // user message after the interrupted checkpoint.
    await send(page, CORRECTION)
    await waitForTranscriptText(page, CORRECTION)

    const orderBeforeSwitch = relevantOrder(await transcriptTextOrder(page))
    expect(orderBeforeSwitch).toEqual([ORIGINAL_PROMPT, CORRECTION])
    expect(await textNodeOccurrences(page, ORIGINAL_PROMPT)).toBe(1)
    expect(await textNodeOccurrences(page, CORRECTION)).toBe(1)
    await page.screenshot({ path: testInfo.outputPath('correction-before-session-switch.png') })

    // Reproduce the observed race: switch while the correction and original
    // response are both still live, then return before the held stream settles.
    await openFreshDraft(page)
    await reopenOriginalSession(page)
    await page.waitForTimeout(500)
    await page.screenshot({ path: testInfo.outputPath('correction-after-warm-resume.png') })

    expect(relevantOrder(await transcriptTextOrder(page))).toEqual(orderBeforeSwitch)
    expect(await textNodeOccurrences(page, ORIGINAL_PROMPT)).toBe(1)
    expect(await textNodeOccurrences(page, CORRECTION)).toBe(1)

    mock.releaseHeldStream()
    await waitForTranscriptText(page, MOCK_REPLY)
  })
})