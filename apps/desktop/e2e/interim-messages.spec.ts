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
 *       settled transcript.
 *
 *   display.interim_assistant_messages: false
 *     → no message.interim events are emitted, so no sealed interim bubbles
 *       are created while streaming. Since the post-turn stored-history
 *       reconcile (sessions.changed → reconcileActiveTranscript, commit
 *       1a2b0ca8cb) converges the visible transcript to the persisted
 *       transcript — which has ALWAYS contained the mid-turn commentary as
 *       real assistant rows (that is what a resume shows, flag or no flag) —
 *       the settled DOM shows the whole turn as ONE assistant message
 *       containing commentary + final. The flag governs live sealing only.
 *       The test pins that converged single-message shape: every text
 *       appears exactly once, inside a single assistant message root.
 *
 * Prerequisite: `npm run build` must have been run so dist/ exists.
 */

import { expect, type Page, test } from '@playwright/test'

import {
  type MockBackendFixture,
  setupMockBackend,
  waitForAppReady,
} from './fixtures'
import { INTERIM_TEXTS, restartMockServer } from '../../../tests-js/scripts/mock-server'

// ─── Helpers ──────────────────────────────────────────────────────────

/**
 * Auto session titling (feat f726090d48, 2026-08-08) issues an auxiliary
 * `title_generation` LLM call against the SAME provider as the chat turn.
 * The mock server counts every completion request as a script turn, so the
 * title call races the chat turn and steals a scripted interim turn (the
 * stolen turn's text then never streams to the transcript). Disable the
 * model-backed title upgrade — the instant derived title needs no LLM call —
 * so the mock's script indices line up with real chat turns again.
 */
const DISABLE_AUTO_TITLE = 'auxiliary:\n  title_generation:\n    enabled: false'

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
  // (hydration, stored-history reconcile, session refresh) before asserting.
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

/** Count assistant message roots in the settled transcript. */
async function countAssistantMessageRoots(page: Page): Promise<number> {
  return page.evaluate(() => {
    const viewport = document.querySelector('[data-slot="aui_thread-viewport"]')

    return viewport
      ? viewport.querySelectorAll('[data-slot="aui_assistant-message-root"]').length
      : 0
  })
}

// ─── Flag ON: interim_assistant_messages = true (default) ─────────────

test.describe('interim assistant messages — flag ON (default)', () => {
  test.describe.configure({ mode: 'serial' })

  let fixture: MockBackendFixture

  test.beforeAll(async () => {
    restartMockServer()
    fixture = await setupMockBackend({ extraConfig: DISABLE_AUTO_TITLE })
    await waitForAppReady(fixture, 120_000)
  })

  test.afterAll(async () => {
    await fixture?.cleanup()
  })

  test('all interim texts survive alongside the final response', async () => {
    const page = fixture.page
    await sendInterimMessage(page)

    // Every interim text (turns with visible text + tool calls) must be
    // present in the settled transcript — NOT wiped by message.complete.
    // (Live, each seals as its own bubble; the post-turn stored-history
    // reconcile then converges the turn into one assistant message that
    // still carries all of them.)
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

    // No duplicates: the reconcile must CONVERGE (replace the sealed live
    // bubbles), never render a stored copy alongside a live one.
    for (const text of [...INTERIM_TEXTS.interims, INTERIM_TEXTS.finalText]) {
      const count = await countTranscriptMessagesContaining(page, text)
      expect(count, `"${text}" must not be duplicated after reconcile`).toBe(1)
    }
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
      extraConfig: DISABLE_AUTO_TITLE,
    })
    await waitForAppReady(fixture, 120_000)
  })

  test.afterAll(async () => {
    await fixture?.cleanup()
  })

  test('settled transcript converges to stored history as a single turn message', async () => {
    const page = fixture.page
    await sendInterimMessage(page)

    // The final text must be visible.
    await expect
      .poll(
        () => countTranscriptMessagesContaining(page, INTERIM_TEXTS.finalText),
        { timeout: 15_000, message: 'final text should be visible' },
      )
      .toBeGreaterThanOrEqual(1)

    // With the flag off, the tui_gateway never installs
    // interim_assistant_callback, so no message.interim events fire and no
    // sealed interim bubbles are created while streaming. After
    // message.complete, the stored-history reconcile (sessions.changed →
    // reconcileActiveTranscript) converges the view to the persisted
    // transcript, which contains the mid-turn commentary as real assistant
    // rows — exactly what a resume of this session would show. Pin that
    // converged shape: ONE assistant message root for the whole turn…
    await expect
      .poll(
        () => countAssistantMessageRoots(page),
        { timeout: 15_000, message: 'the settled turn should render as one assistant message' },
      )
      .toBe(1)

    // …containing every commentary text and the final text exactly once.
    for (const text of [...INTERIM_TEXTS.interims, INTERIM_TEXTS.finalText]) {
      await expect
        .poll(
          () => countTranscriptMessagesContaining(page, text),
          { timeout: 15_000, message: `"${text}" should appear exactly once in the converged turn` },
        )
        .toBe(1)
    }
  })
})
