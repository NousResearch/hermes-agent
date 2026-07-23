/**
 * E2E test for the interim-assistant-message preservation fix (#65919).
 *
 * Reproduces the bug across all three layers (agent core → tui_gateway →
 * desktop renderer): when the agent emits assistant text alongside a tool
 * call, then completes the turn with a *different* final answer, the
 * interim text must survive in the transcript — not be wiped when
 * message.complete replaces the streaming bubble.
 *
 * The mock server walks through a multi-turn script when it sees the
 * trigger keyword:
 *
 *   Turn 1: "Let me start by planning the approach." + todo tool_call
 *   Turn 2: "Now checking the details before answering." + todo tool_call
 *   Turn 3: (no text) + todo tool_call          → NO interim (no visible text)
 *   Turn 4: "Found something interesting worth noting." + todo tool_call
 *   Turn 5: "All done! Here is the complete summary..." (final, stop)
 *
 * Two describe blocks exercise the config flag both ways:
 *
 *   display.interim_assistant_messages: true (default)
 *     → ALL interim texts AND the final text must be visible in the
 *       transcript.
 *
 *   display.interim_assistant_messages: false
 *     → only the final text is visible (no message.interim events emitted,
 *       so all streamed interim text is replaced at message.complete).
 *
 * Prerequisite: `npm run build` must have been run so dist/ exists.
 */

import { expect, test, type Page } from '@playwright/test'

import {
  type MockBackendFixture,
  setupMockBackend,
  waitForAppReady,
} from './fixtures'
import { INTERIM_TEXTS, restartMockServer } from './mock-server'

// ─── Helpers ──────────────────────────────────────────────────────────

/** Unique trigger keyword the mock server detects to switch to the script. */
const TRIGGER = 'E2E_INTERIM_TRIGGER'

/**
 * Send a message and wait for BOTH the user's message and the agent's
 * final response to appear in the transcript. Returns when the final text
 * is visible, which means message.complete has fired and the transcript
 * has settled.
 */
async function sendInterimMessage(page: Page): Promise<void> {
  const composer = page.locator('[contenteditable="true"]').first()
  await composer.waitFor({ state: 'visible', timeout: 10_000 })
  await composer.click()
  await composer.type(TRIGGER, { delay: 20 })
  await page.keyboard.press('Enter')

  // Wait for the user's trigger message to appear.
  await page.waitForFunction(
    () => (document.body.textContent ?? '').includes('E2E_INTERIM_TRIGGER'),
    undefined,
    { timeout: 15_000 },
  )

  // Wait for the agent's FINAL response (last turn). This means
  // message.complete has fired and the transcript is settled.
  await page.waitForFunction(
    (finalText) => (document.body.textContent ?? '').includes(finalText),
    INTERIM_TEXTS.finalText,
    { timeout: 90_000 },
  )

  // Give the renderer a moment to settle any final state updates
  // (hydration, session refresh) before asserting.
  await page.waitForTimeout(2000)
}

/**
 * Count how many times `text` appears as distinct text in the chat transcript
 * (excluding the session sidebar, whose session-preview label shows the
 * first streamed text as a title).
 *
 * The desktop app renders the transcript inside a
 * `[data-slot="aui_thread-viewport"]` container (from @assistant-ui/react).
 * The session sidebar's preview labels live outside that container, so
 * scoping the DOM walk to the viewport cleanly excludes them.
 */
async function countTranscriptMessagesContaining(page: Page, text: string): Promise<number> {
  return page.evaluate(
    (search) => {
      const viewport = document.querySelector('[data-slot="aui_thread-viewport"]')
      if (!viewport) {
        return 0
      }

      let count = 0
      const walker = document.createTreeWalker(
        viewport,
        NodeFilter.SHOW_ELEMENT,
        {
          acceptNode: (node) => {
            const el = node as HTMLElement
            const directText = el.textContent ?? ''
            if (!directText.includes(search)) {
              return NodeFilter.FILTER_SKIP
            }
            // Only count leaf-ish elements to avoid double-counting.
            const hasChildWithText = Array.from(el.children).some(
              (child) => (child.textContent ?? '').includes(search),
            )
            if (hasChildWithText) {
              return NodeFilter.FILTER_SKIP
            }
            return NodeFilter.FILTER_ACCEPT
          },
        },
      )
      while (walker.nextNode()) {
        count++
      }
      return count
    },
    text,
  )
}

// ─── Flag ON: interim_assistant_messages = true (default) ─────────────

test.describe('interim assistant messages — flag ON (default)', () => {
  test.describe.configure({ mode: 'serial' })

  let fixture: MockBackendFixture

  test.beforeAll(async () => {
    restartMockServer()
    fixture = await setupMockBackend()
    await waitForAppReady(fixture, 120_000)
  })

  test.afterAll(async () => {
    await fixture?.cleanup()
  })

  test('all interim texts survive alongside the final response', async () => {
    const page = fixture.page
    await sendInterimMessage(page)

    // Every interim text (turns with visible text + tool calls) must be
    // present in the transcript as its own sealed message — NOT wiped by
    // message.complete.
    for (const interimText of INTERIM_TEXTS.interims) {
      await expect
        .poll(
          () => countTranscriptMessagesContaining(page, interimText),
          { timeout: 15_000, message: `interim text "${interimText}" should be visible` },
        )
        .toBeGreaterThanOrEqual(1)
    }

    // The final text must also be visible.
    await expect
      .poll(
        () => countTranscriptMessagesContaining(page, INTERIM_TEXTS.finalText),
        { timeout: 15_000, message: 'final text should be visible' },
      )
      .toBeGreaterThanOrEqual(1)
  })
})

// ─── Flag OFF: interim_assistant_messages = false ────────────────────

test.describe('interim assistant messages — flag OFF', () => {
  test.describe.configure({ mode: 'serial' })

  let fixture: MockBackendFixture

  test.beforeAll(async () => {
    restartMockServer()
    fixture = await setupMockBackend({
      extraDisplayConfig: '  interim_assistant_messages: false',
    })
    await waitForAppReady(fixture, 120_000)
  })

  test.afterAll(async () => {
    await fixture?.cleanup()
  })

  test('only the final response is visible; all interim texts are wiped', async () => {
    const page = fixture.page
    await sendInterimMessage(page)

    // The final text must be visible.
    await expect
      .poll(
        () => countTranscriptMessagesContaining(page, INTERIM_TEXTS.finalText),
        { timeout: 15_000, message: 'final text should be visible' },
      )
      .toBeGreaterThanOrEqual(1)

    // NONE of the interim texts should be visible — with the flag off,
    // the tui_gateway never installs interim_assistant_callback, so no
    // message.interim events are emitted. All streamed interim text is
    // accumulated into the streaming bubble and replaced by
    // message.complete.
    for (const interimText of INTERIM_TEXTS.interims) {
      const count = await countTranscriptMessagesContaining(page, interimText)
      expect(
        count,
        `interim text "${interimText}" should NOT be visible when flag is off`,
      ).toBe(0)
    }
  })
})
