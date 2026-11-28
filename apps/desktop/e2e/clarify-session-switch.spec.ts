/**
 * A question card the agent is still waiting on survives a session switch.
 *
 * #139189: with a clarify pending, switching to another saved chat and back
 * left the turn running with no card to answer (a window reload brought it
 * back, the next switch lost it again). The backend keeps the request open the
 * whole time and lists it in `open_requests` on every activate, so the card
 * must be there again after the switch, keep what was picked on it, and still
 * answer the turn.
 *
 * The chat in that report was 145 stored rows long. When a reload restores such
 * a chat as a tab, the open clarify is re-delivered while `session.resume` is
 * still in flight; the tab used to keep that lone projected card as its whole
 * transcript, dropping the stored history, and the card remounted (losing its
 * picks) once an older page loaded above it. The long-history cases cover that.
 */

import {
  BATCH_CLARIFY_QUESTIONS,
  BATCH_CLARIFY_TRIGGER,
  LONG_HISTORY_TRIGGER,
  MOCK_REPLY,
  REAL_CLARIFY_QUESTIONS,
  REAL_CLARIFY_TRIGGER
} from '../../../tests-js/scripts/mock-server'

import { type MockBackendFixture, type MockBackendOptions, setupMockBackend, waitForAppReady } from './fixtures'
import { expect, type Page, test } from './test'

const OTHER_SESSION_PROMPT = 'E2E saved chat to switch to while a question waits.'
const CLARIFY_PROMPT = `${BATCH_CLARIFY_TRIGGER}: the card must survive a switch away and back.`
const REAL_CLARIFY_PROMPT = `${REAL_CLARIFY_TRIGGER}: so the researcher finished, what next?`
// A question that waits minutes for its answer (the shipped 90 s default would end it first).
const SLOW_ANSWER_CONFIG = 'agent:\n  clarify_timeout: 600\n'

// Inactive tabs stay mounted under a data-pane-hidden ancestor; only the visible surface counts.
const SURFACE = '[data-composer-target]:not([data-pane-hidden] [data-composer-target])'

const activeSurface = (page: Page) => page.locator(SURFACE).last()

async function send(page: Page, text: string): Promise<void> {
  const composer = activeSurface(page).locator('[contenteditable="true"]').first()
  await composer.waitFor({ state: 'visible', timeout: 15_000 })
  await composer.click()
  // One insert, not per-key typing: a re-render mid-type (a title or a compression landing) drops keys.
  await page.keyboard.insertText(text)
  await page.keyboard.press('Enter')
}

async function waitForTranscriptText(page: Page, text: string, timeout = 30_000): Promise<void> {
  await page.waitForFunction(
    ([expected, surfaceSelector]: [string, string]) => {
      const surfaces = document.querySelectorAll(surfaceSelector)
      const active = surfaces[surfaces.length - 1]

      return (active?.querySelector('[data-slot="aui_thread-viewport"]')?.textContent ?? '').includes(expected)
    },
    [text, SURFACE] as [string, string],
    { timeout }
  )
}

async function openSidebarSession(page: Page, sidebarText: string, transcriptText: string): Promise<void> {
  // A reload can list a chat as "Untitled session" until its title is stored. The chat opened here is the
  // older one, so it is the lower of the matching rows; the transcript check below confirms the pick.
  const label = new RegExp(`${sidebarText.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}|Untitled session`)
  const row = page.locator('[data-slot="sidebar"] button').filter({ hasText: label }).last()
  await row.waitFor({ state: 'visible', timeout: 30_000 })
  await row.click()
  await waitForTranscriptText(page, transcriptText)
}

/** Close the other chat's tab: a plain sidebar click then reuses the waiting chat's tab, so its
 *  pane really unmounts and coming back goes through a resume instead of a tab focus. */
async function closeOtherTab(page: Page, { cardVisible = true } = {}): Promise<void> {
  const tabs = page.locator('[data-slot="pane-tab"][data-closeable]').filter({ visible: true })

  const otherTab = page
    .locator('[data-slot="pane-tab"][data-closeable]:not([data-active="true"])')
    .filter({ visible: true })
    .first()

  await expect(tabs).toHaveCount(2, { timeout: 10_000 })
  await otherTab.click({ button: 'middle' })
  await expect(tabs).toHaveCount(1, { timeout: 10_000 })
  await expect(visibleCard(page)).toHaveCount(cardVisible ? 1 : 0)
}

const visibleCard = (page: Page) => page.locator('form[data-clarify-batch]').filter({ visible: true })

test.describe('pending clarify across a session switch', () => {
  // Backend boot plus two saved chats and two switch rounds outlast the 90 s default.
  test.setTimeout(240_000)

  let fixture: MockBackendFixture | null = null

  async function boot(options?: MockBackendOptions): Promise<Page> {
    fixture = await setupMockBackend(options)
    await waitForAppReady(fixture, 120_000)

    return fixture.page
  }

  test.afterEach(async () => {
    await fixture?.cleanup()
    fixture = null
  })

  async function clarifyInSecondTab(page: Page, { awaitCard = true, prompt = CLARIFY_PROMPT } = {}): Promise<void> {
    // A real second saved chat, so the switch goes through session activation.
    await send(page, OTHER_SESSION_PROMPT)
    await waitForTranscriptText(page, MOCK_REPLY)
    // "+" opens a new tab beside the chat; wait for its empty surface before typing into it.
    await page.locator('[data-slot="sidebar"] button[aria-label="New session"]').first().click()
    await page.waitForFunction(
      ([prior, surfaceSelector]: [string, string]) => {
        const surfaces = document.querySelectorAll(surfaceSelector)
        const active = surfaces[surfaces.length - 1]

        return (
          surfaces.length > 0 &&
          !(active?.querySelector('[data-slot="aui_thread-viewport"]')?.textContent ?? '').includes(prior)
        )
      },
      [OTHER_SESSION_PROMPT, SURFACE] as [string, string],
      { timeout: 15_000 }
    )

    await send(page, prompt)
    await waitForTranscriptText(page, prompt)

    if (awaitCard) {
      await expect(visibleCard(page)).toHaveCount(1, { timeout: 60_000 })
    }
  }

  async function switchAwayAndBack(
    page: Page,
    rounds: number,
    rowText: RegExp = /E2E_BATCH_CLARIFY_TRIGGER|Untitled session/
  ): Promise<void> {
    for (let round = 1; round <= rounds; round++) {
      await test.step(`switch away and back (${round})`, async () => {
        await openSidebarSession(page, OTHER_SESSION_PROMPT, MOCK_REPLY)
        await expect(visibleCard(page)).toHaveCount(0)

        // After a reload the waiting chat may list as "Untitled session" (no title yet).
        const row = page.locator('[data-slot="sidebar"] button').filter({ hasText: rowText }).first()

        await row.click()
        await expect(visibleCard(page)).toHaveCount(1, { timeout: 15_000 })
      })
    }
  }

  async function answerCard(page: Page, questions: { choices: string[] }[] = BATCH_CLARIFY_QUESTIONS): Promise<void> {
    // The card that came back is live: answering it finishes the waiting turn.
    const card = visibleCard(page)

    const confirm = card.locator('button[type="submit"]')

    for (const entry of questions) {
      await card.getByRole('button', { name: entry.choices[0]!, exact: false }).first().click()
    }

    await expect(confirm).toBeEnabled()
    await confirm.click()

    const settled = page.locator('[data-clarify-settled]').filter({ visible: true })
    await expect(settled).toHaveCount(1, { timeout: 30_000 })
    await expect(settled.getByText(questions[0]!.choices[0]!, { exact: true })).toBeVisible()
    await expect(visibleCard(page)).toHaveCount(0)
  }

  for (const reload of [false, true]) {
    test(`switching tabs away and back keeps the card live${reload ? ' after a window reload' : ''}`, async () => {
      const page = await boot()

      await clarifyInSecondTab(page)

      if (reload) {
        await test.step('reload the window (the card comes back from open_requests)', async () => {
          await page.reload()
          await waitForAppReady(fixture!, 120_000)
          // The reload may restore the other tab as the active one; open the waiting chat.
          await page
            .locator('[data-slot="sidebar"] button')
            .filter({ hasText: /E2E_BATCH_CLARIFY_TRIGGER|Untitled session/ })
            .first()
            .click()
          await expect(visibleCard(page)).toHaveCount(1, { timeout: 30_000 })
          await waitForTranscriptText(page, CLARIFY_PROMPT)
        })
      }

      await switchAwayAndBack(page, 2)
      await answerCard(page)
    })
  }

  for (const reload of [false, true]) {
    test(`in a chat longer than one history page, the card survives a switch${reload ? ' after a reload' : ''}`, async () => {
      const page = await boot({ extraConfig: SLOW_ANSWER_CONFIG })
      const row = /E2E_LONG_HISTORY_TRIGGER|Untitled session/

      await send(page, OTHER_SESSION_PROMPT)
      await waitForTranscriptText(page, MOCK_REPLY)
      await page.locator('[data-slot="sidebar"] button[aria-label="New session"]').first().click()
      await page.waitForFunction(
        ([prior, surfaceSelector]: [string, string]) => {
          const surfaces = document.querySelectorAll(surfaceSelector)
          const active = surfaces[surfaces.length - 1]

          return (
            surfaces.length > 0 &&
            !(active?.querySelector('[data-slot="aui_thread-viewport"]')?.textContent ?? '').includes(prior)
          )
        },
        [OTHER_SESSION_PROMPT, SURFACE] as [string, string],
        { timeout: 15_000 }
      )

      // ~130 stored rows before the question, like the 145-row chat in the report.
      await send(page, `${LONG_HISTORY_TRIGGER}: read every file before deciding.`)
      await waitForTranscriptText(page, 'Long history done.', 180_000)
      await send(page, REAL_CLARIFY_PROMPT)
      await expect(visibleCard(page)).toHaveCount(1, { timeout: 60_000 })

      if (reload) {
        await page.reload()
        await waitForAppReady(fixture!, 120_000)
        await page.locator('[data-slot="sidebar"] button').filter({ hasText: row }).first().click()
        await expect(visibleCard(page)).toHaveCount(1, { timeout: 30_000 })
        // The restored chat keeps its history, not only the waiting card.
        await waitForTranscriptText(page, REAL_CLARIFY_PROMPT)
      }

      await switchAwayAndBack(page, 2, row)
      // Every pick must still be staged when Confirm is pressed: a remounted card drops them.
      await answerCard(page, REAL_CLARIFY_QUESTIONS)
    })
  }

  for (const sameTab of [false, true]) {
    test(`a question that arrives while the chat is in the background shows on return (${sameTab ? 'same tab' : 'tab switch'})`, async () => {
      const page = await boot({ mockServer: { holdFirstCompletionContaining: BATCH_CLARIFY_TRIGGER } })

      await clarifyInSecondTab(page, { awaitCard: false })
      // The clarify turn is parked at the mock; leave the chat before the question exists.
      await fixture!.mock.waitForHeldCompletion()

      if (sameTab) {
        await closeOtherTab(page, { cardVisible: false })
      }

      await openSidebarSession(page, OTHER_SESSION_PROMPT, MOCK_REPLY)
      fixture!.mock.releaseHeldStream()
      // Let the question land on the background chat (the released turn asks it at once).
      await page.waitForTimeout(3_000)

      await page
        .locator('[data-slot="sidebar"] button')
        .filter({ hasText: /E2E_BATCH_CLARIFY_TRIGGER|Untitled session/ })
        .first()
        .click()
      await expect(visibleCard(page)).toHaveCount(1, { timeout: 15_000 })

      await switchAwayAndBack(page, 2)
      await answerCard(page)
    })
  }
})
