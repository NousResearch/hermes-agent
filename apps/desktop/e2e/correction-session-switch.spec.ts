/**
 * Regression coverage for a correction sent during a live response, then a
 * warm session switch away and back. The correction is an accepted user turn,
 * not an optimistic duplicate of the original prompt, and its relative place
 * in the transcript must survive the resume reconciliation.
 */

import { type TestInfo } from '@playwright/test'

import { expect, test, type Page } from './test'

import { type MockBackendFixture, setupMockBackend, waitForAppReady } from './fixtures'
import { CORRECTION_SWITCH_TRIGGER, MOCK_REPLY } from './mock-server'

const OTHER_SESSION_PROMPT = 'E2E persisted session used for a warm resume.'
const ORIGINAL_PROMPT = `${CORRECTION_SWITCH_TRIGGER}: original prompt must remain singular after a correction.`
const CORRECTION = 'E2E correction must stay after the original prompt.'
const TOOL_STARTED = 'Checking the long-running task before I continue.'
const CORRECTED_REPLY = 'The corrected task finished.'
const INFERENCE_SWITCH_TRIGGER = 'E2E_INFERENCE_SWITCH_TRIGGER'
const INFERENCE_PROMPT = `${INFERENCE_SWITCH_TRIGGER}: original inference prompt must remain singular.`
const INFERENCE_CORRECTION = `${INFERENCE_SWITCH_TRIGGER}: correction sent while inference is live.`

async function send(page: Page, text: string): Promise<void> {
  const composer = page.locator('[contenteditable="true"]').first()
  await composer.waitFor({ state: 'visible', timeout: 15_000 })
  await composer.click()
  await composer.type(text, { delay: 5 })
  await page.keyboard.press('Enter')
}

async function steer(page: Page, text: string): Promise<void> {
  const composer = page.locator('[contenteditable="true"]').first()
  const primary = page.locator('[data-slot="composer-root"] button[type="submit"]')

  await composer.waitFor({ state: 'visible', timeout: 15_000 })
  await composer.click()
  await composer.type(text, { delay: 5 })
  await expect(primary).toHaveAttribute('aria-label', /Steer/)
  await primary.click()
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

async function transcriptMessageOrder(page: Page): Promise<string[]> {
  return page.evaluate(() => {
    const viewport = document.querySelector('[data-slot="aui_thread-viewport"]')
    if (!viewport) return []

    return Array.from(viewport.querySelectorAll<HTMLElement>('[data-role="user"], [data-role="assistant"]'))
      .map(message => message.textContent?.trim() ?? '')
      .filter(Boolean)
  })
}

async function openFreshDraft(page: Page, priorSessionText: string): Promise<void> {
  await page.locator('[data-slot="sidebar"] button[aria-label="New session"]').first().click()
  await page.waitForFunction(
    (priorText: string) => !(document.querySelector('[data-slot="aui_thread-viewport"]')?.textContent ?? '').includes(priorText),
    priorSessionText,
    { timeout: 15_000 },
  )
}

async function openSidebarSession(page: Page, sidebarText: string, expectedTranscriptText: string): Promise<void> {
  const row = page.locator('[data-slot="sidebar"] button').filter({ hasText: sidebarText }).first()
  await row.waitFor({ state: 'visible', timeout: 30_000 })
  await row.click()
  await waitForTranscriptText(page, expectedTranscriptText)
}

async function reopenOriginalSession(page: Page): Promise<void> {
  // A still-running tool has not generated a final title yet, so the sidebar
  // retains the source prompt as its provisional session title.
  await openSidebarSession(page, ORIGINAL_PROMPT, ORIGINAL_PROMPT)
}

async function reopenInferenceSession(page: Page): Promise<void> {
  const row = page.locator('[data-slot="sidebar"] button').filter({ hasText: INFERENCE_PROMPT }).first()
  await row.waitFor({ state: 'visible', timeout: 30_000 })
  await row.click()
  await waitForTranscriptText(page, INFERENCE_PROMPT)
}

function relevantOrder(messages: string[]): string[] {
  return messages.filter(message => message.includes(ORIGINAL_PROMPT) || message.includes(CORRECTION))
}

function steerTurnOrder(messages: string[]): string[] {
  return messages.flatMap(message => {
    if (message.includes(ORIGINAL_PROMPT)) return [ORIGINAL_PROMPT]
    if (message.includes(CORRECTION)) return [CORRECTION]
    if (message.includes(CORRECTED_REPLY)) return [CORRECTED_REPLY]

    return []
  })
}

test.describe('correction session switch', () => {
  let fixture: MockBackendFixture | null = null

  test.beforeEach(async () => {
    fixture = await setupMockBackend({
      mockServer: { holdFirstStreamForPrompt: INFERENCE_SWITCH_TRIGGER },
    })
    await waitForAppReady(fixture, 120_000)
  })

  test.afterEach(async () => {
    await fixture?.cleanup()
    fixture = null
  })

  test('keeps a live correction in place and does not duplicate its original prompt after switching sessions', async ({}, testInfo: TestInfo) => {
    const { page } = fixture!

    // A blank draft does not exercise session hydration. Seed a real second
    // session first, matching the observed switch between two saved chats.
    await send(page, OTHER_SESSION_PROMPT)
    await waitForTranscriptText(page, MOCK_REPLY)
    await openFreshDraft(page, OTHER_SESSION_PROMPT)

    await send(page, ORIGINAL_PROMPT)
    await waitForTranscriptText(page, TOOL_STARTED)
    await waitForTranscriptText(page, ORIGINAL_PROMPT)

    // The historical session redirects while a foreground terminal task is
    // running. Use the visible Steer action to cover the real composer path.
    await steer(page, CORRECTION)
    await waitForTranscriptText(page, CORRECTION)

    const orderBeforeSwitch = relevantOrder(await transcriptTextOrder(page))
    expect(orderBeforeSwitch).toEqual([ORIGINAL_PROMPT, CORRECTION])
    expect(await textNodeOccurrences(page, ORIGINAL_PROMPT)).toBe(1)
    expect(await textNodeOccurrences(page, CORRECTION)).toBe(1)
    await page.screenshot({ path: testInfo.outputPath('correction-before-session-switch.png') })

    // Reproduce the observed race: switch to another persisted session while
    // the foreground tool is live, then return before its redirect settles.
    await openSidebarSession(page, MOCK_REPLY, OTHER_SESSION_PROMPT)
    await reopenOriginalSession(page)
    await page.waitForTimeout(500)
    await page.screenshot({ path: testInfo.outputPath('correction-after-warm-resume.png') })

    expect(relevantOrder(await transcriptTextOrder(page))).toEqual(orderBeforeSwitch)
    expect(await textNodeOccurrences(page, ORIGINAL_PROMPT)).toBe(1)
    expect(await textNodeOccurrences(page, CORRECTION)).toBe(1)

    await waitForTranscriptText(page, CORRECTED_REPLY)
    expect(steerTurnOrder(await transcriptMessageOrder(page))).toEqual([ORIGINAL_PROMPT, CORRECTION, CORRECTED_REPLY])
  })

  test('keeps an inference-time correction visible through a warm session switch', async ({}, testInfo: TestInfo) => {
    const { mock, page } = fixture!

    await send(page, OTHER_SESSION_PROMPT)
    await waitForTranscriptText(page, MOCK_REPLY)
    await openFreshDraft(page, OTHER_SESSION_PROMPT)

    await send(page, INFERENCE_PROMPT)
    await mock.waitForHeldStream()
    await waitForTranscriptText(page, INFERENCE_PROMPT)

    await send(page, INFERENCE_CORRECTION)
    await waitForTranscriptText(page, INFERENCE_CORRECTION)

    await openSidebarSession(page, MOCK_REPLY, OTHER_SESSION_PROMPT)
    await reopenInferenceSession(page)

    expect(await textNodeOccurrences(page, INFERENCE_PROMPT)).toBe(1)
    expect(await textNodeOccurrences(page, INFERENCE_CORRECTION)).toBe(1)
    await page.screenshot({ path: testInfo.outputPath('inference-correction-after-warm-resume.png') })

    mock.releaseHeldStream()
    await waitForTranscriptText(page, MOCK_REPLY)
  })
})