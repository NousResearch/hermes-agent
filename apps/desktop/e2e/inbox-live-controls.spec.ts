/**
 * Live inbox controls — real gateway, real tools, real Action Center.
 *
 * Companion to inbox-live-approval.spec.ts (same one-app-per-test harness). Covers the
 * panel's newer controls end to end:
 *  - a live clarify question letters its options A/B/C and offers the type-your-own row
 *    (lettered one past the last choice), answerable from the panel;
 *  - a live batch clarify stages picks and typed answers per question;
 *  - a live goal can be paused and resumed straight from the panel.
 *
 * Output: .inbox-work/live-approval-evidence/ (shared with the approval spec).
 */

import fs from 'node:fs'
import path from 'node:path'
import {
  BATCH_CLARIFY_QUESTIONS,
  BATCH_CLARIFY_TRIGGER,
  INBOX_CLARIFY_CHOICES,
  INBOX_CLARIFY_QUESTION,
  INBOX_CLARIFY_TRIGGER,
  MOCK_REPLY
} from '../../../tests-js/scripts/mock-server'
import { type MockBackendFixture, setupMockBackend, waitForAppReady } from './fixtures'
import { expect, test, type Page } from './test'

const OUTPUT = path.resolve(import.meta.dirname, '../../../.inbox-work/live-approval-evidence')
const OTHER_PLACEHOLDER = 'Other (type your answer)'

test.describe.configure({ mode: 'serial' })

async function withApp(
  name: string,
  extraConfig: string | null,
  body: (fixture: MockBackendFixture, page: Page) => Promise<void>
): Promise<void> {
  fs.mkdirSync(OUTPUT, { recursive: true })
  const fixture = await setupMockBackend(extraConfig === null ? {} : { extraConfig })
  const tag = name.replace(/[^a-z0-9]+/gi, '-').slice(0, 48)

  try {
    await waitForAppReady(fixture, 120_000)
    await body(fixture, fixture.page)
  } finally {
    try {
      const logs = path.join(fixture.sandbox.hermesHome, 'logs')

      for (const log of ['agent.log', 'gui.log']) {
        const src = path.join(logs, log)

        if (fs.existsSync(src)) {
          fs.copyFileSync(src, path.join(OUTPUT, `${tag}-${log}`))
        }
      }
    } catch {
      /* diagnostics only */
    }

    await fixture.cleanup()
  }
}

async function shot(page: Page, name: string): Promise<void> {
  await page.screenshot({ path: path.join(OUTPUT, name) })
}

/** Record the answer frames the panel sends, so assertions read the real wire payload. */
async function installAnswerTap(page: Page): Promise<void> {
  await page.evaluate(() => {
    const send = WebSocket.prototype.send

    ;(window as any).__inboxAnswers = [] as string[]

    WebSocket.prototype.send = function (data) {
      try {
        const frame = JSON.parse(String(data))

        if (frame?.method === 'request.answer' || frame?.method === 'clarify.lock') {
          ;(window as any).__inboxAnswers.push(JSON.stringify(frame.params))
        }
      } catch {
        /* non-JSON frame */
      }

      return send.call(this, data)
    }
  })
}

async function sendPrompt(page: Page, text: string): Promise<void> {
  const composer = page.locator('[contenteditable="true"]').first()

  await composer.waitFor({ state: 'visible', timeout: 20_000 })
  await composer.click()
  await composer.type(text, { delay: 5 })
  await composer.press('Enter')
}

async function openInbox(page: Page): Promise<void> {
  await page.keyboard.press('Escape')
  await page.waitForTimeout(200)

  // The chip's own label — a loose matcher also clicks a sidebar row whose message
  // text happens to contain the trigger name.
  await page.getByRole('button', { name: /^(Action Center|Action Center — \d+ need attention)$/ }).first().click()
  await expect(page.getByRole('heading', { name: 'Action Center' })).toBeVisible()
}

test('a live clarify letters options A–C and the type-your-own row answers it', async () => {
  test.setTimeout(300_000)

  await withApp('clarify-letters', null, async (_fixture, page) => {
    await installAnswerTap(page)
    await sendPrompt(page, `Ask me about the surface. ${INBOX_CLARIFY_TRIGGER}`)
    await openInbox(page)

    const row = page.locator('[data-panel-row]').first()

    await expect(row).toBeVisible({ timeout: 15_000 })
    await row.click()

    // The chat still mounts the same card beneath the overlay, so scope assertions to
    // the panel surface or every match is ambiguous.
    const panel = page.locator('[data-overlay-surface]')

    // The live question is answerable in place.
    await expect(panel.getByText(INBOX_CLARIFY_QUESTION)).toBeVisible({ timeout: 15_000 })

    // Options carry their letters, and the type-your-own row takes the next one (D).
    for (const [index, choice] of INBOX_CLARIFY_CHOICES.entries()) {
      const letter = String.fromCharCode(65 + index)
      const option = panel.getByRole('button').filter({ hasText: choice }).first()

      await expect(option).toBeVisible()
      await expect(option).toHaveText(new RegExp(`^${letter}`))
    }

    const otherRow = panel.getByPlaceholder(OTHER_PLACEHOLDER)
    await expect(otherRow).toBeVisible()
    await expect(panel.getByText('D', { exact: true })).toBeVisible()
    await shot(page, 'live-controls-1-clarify-lettered.png')

    // Answer with text the options don't offer — the row exists for exactly this.
    await otherRow.fill('Terminal, but over SSH')
    await panel.getByRole('button', { name: 'Submit', exact: true }).click()

    await expect.poll(() => page.evaluate(() => (window as any).__inboxAnswers.join(' | ')), {
      timeout: 15_000
    }).toContain('Terminal, but over SSH')
    await shot(page, 'live-controls-2-clarify-typed-answer.png')

    // The answer resolved the request: the question retires from the panel, and the
    // turn resumes to the mock's canned reply. (The answer reaches the model as a
    // tool result, so `receivedPrompts` — which records the latest user message —
    // is not the right witness here; the resolved request and the continued turn are.)
    await expect(panel.getByText(INBOX_CLARIFY_QUESTION)).toHaveCount(0, { timeout: 30_000 })
    await page.keyboard.press('Escape')
    await expect(page.getByText(MOCK_REPLY, { exact: true }).first()).toBeVisible({ timeout: 60_000 })
    await shot(page, 'live-controls-2b-clarify-resolved.png')
  })
})

test('a live batch clarify stages picks and typed answers per question', async () => {
  test.setTimeout(300_000)

  await withApp('batch-typed', null, async (_fixture, page) => {
    await installAnswerTap(page)
    await sendPrompt(page, `Ask the two questions. ${BATCH_CLARIFY_TRIGGER}`)
    await openInbox(page)

    const row = page.locator('[data-panel-row]').first()

    await expect(row).toBeVisible({ timeout: 15_000 })
    await row.click()

    const panel = page.locator('[data-overlay-surface]')

    for (const entry of BATCH_CLARIFY_QUESTIONS) {
      await expect(panel.getByText(entry.question)).toBeVisible({ timeout: 15_000 })
    }

    // Both questions offer the type-your-own row, single-select included.
    const otherRows = panel.getByPlaceholder(OTHER_PLACEHOLDER)
    await expect(otherRows).toHaveCount(2)

    // q1: typed answer; q2: a pick. One confirm submits the batch.
    await otherRows.first().fill('Something else entirely')
    await panel.getByRole('button', { name: /Night/ }).click()
    await shot(page, 'live-controls-3-batch-staged.png')

    await panel.getByRole('button', { name: 'Submit answers', exact: true }).click()

    await expect.poll(() => page.evaluate(() => (window as any).__inboxAnswers.join(' | ')), {
      timeout: 15_000
    }).toContain('Something else entirely')

    // One confirm resolves the whole batch: both questions retire, and the turn
    // resumes to the mock's canned reply.
    for (const entry of BATCH_CLARIFY_QUESTIONS) {
      await expect(panel.getByText(entry.question)).toHaveCount(0, { timeout: 30_000 })
    }

    await page.keyboard.press('Escape')
    await expect(page.getByText(MOCK_REPLY, { exact: true }).first()).toBeVisible({ timeout: 60_000 })
    await shot(page, 'live-controls-4-batch-resolved.png')
  })
})

test('a live goal can be paused and resumed from the inbox panel', async () => {
  test.setTimeout(420_000)

  await withApp('goal-controls', null, async (_fixture, page) => {
    // Prime the session first: the composer chrome (and its Create-automation menu) is
    // reachable in a fresh chat, but the same flow as automation-local.spec.ts is the
    // proven path, so walk it identically.
    const prompt = 'Write a short greeting for the inbox controls test'

    await sendPrompt(page, 'Hello. This is the inbox controls test.')
    await expect(page.getByText(MOCK_REPLY, { exact: true })).toBeVisible({ timeout: 30_000 })

    // Create a goal the same way a person does: composer → Create automation.
    await page.getByRole('button', { name: 'Add files and actions', exact: true }).first().click()
    await page.getByRole('menuitem', { name: /Create automation/ }).click()
    await expect(page.getByRole('dialog')).toBeVisible()
    await page.getByLabel('Goal prompt', { exact: true }).fill(prompt)
    await page.getByRole('button', { name: 'Start goal', exact: true }).click()
    await expect(page.getByRole('dialog')).not.toBeVisible({ timeout: 30_000 })

    await openInbox(page)

    const row = page.locator('[data-panel-row]').first()

    await expect(row).toBeVisible({ timeout: 15_000 })
    await row.click()

    const panel = page.locator('[data-overlay-surface]')

    // The panel reaches the same control the composer's goal card has.
    const pause = panel.getByRole('button', { name: 'Pause goal', exact: true })

    await expect(pause).toBeVisible({ timeout: 15_000 })
    await shot(page, 'live-controls-5-goal-in-inbox.png')

    await pause.click()
    const resume = panel.getByRole('button', { name: 'Resume goal', exact: true })

    await expect(resume).toBeVisible({ timeout: 30_000 })
    await shot(page, 'live-controls-6-goal-paused.png')

    // And back — the same control in both directions.
    await resume.click()
    await expect(panel.getByRole('button', { name: 'Pause goal', exact: true })).toBeVisible({ timeout: 30_000 })
    await shot(page, 'live-controls-7-goal-resumed.png')
  })
})
