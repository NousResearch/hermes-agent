/**
 * E2E clarify-card durability — the field failure, end to end.
 *
 * Reported symptom: a long-running session showed "Needs your input" while the
 * live question had no answerable card anywhere in the transcript. The backend
 * was still blocked on `clarify.respond`, so the turn could never continue.
 *
 * The sequence below is the production shape of that report, not a shortcut:
 *
 *   1. a durable session with 741+ persisted rows, seeded through the real TUI
 *      gateway and agent loop (mock inference provider only);
 *   2. a COLD resume in the desktop, which mints a new runtime session id for
 *      that stored conversation — the identity rebind;
 *   3. a live blocking clarify raised before any result persists;
 *   4. a switch away to another chat and back, which runs activation plus the
 *      authoritative newest-tail hydration/graft over a transcript whose
 *      running clarify row is (correctly) not in persisted history;
 *   5. the mounted-DOM predicate: exactly ONE `[data-slot="clarify-inline"]`
 *      card with an ENABLED answer control;
 *   6. a real answer through that card, settling the original request and
 *      letting the turn continue.
 *
 * Store state and row counts are deliberately not the acceptance here. The
 * decisive assertion is a mounted, enabled, answerable card.
 *
 * Long-session precondition (scope note). The precondition this spec enforces
 * is the one the reported failure actually needs: a real store holding 741+
 * persisted rows, cold-resumed by the real desktop, whose transcript has
 * hydrated its authoritative NEWEST tail. Rendering the *oldest* seeded turn is
 * expressly NOT a precondition — full-history backfill is a separate transcript
 * concern with its own defect (ADJACENT-TRANSCRIPT-BACKFILL-001, deferred), and
 * driving it here only measured that other surface instead of clarify.
 */

import {
  buildAppEnv,
  createSandbox,
  launchDesktop,
  type Sandbox,
  waitForAppReady,
  writeEnvFile,
  writeMockProviderConfig,
} from './fixtures'
import { BLOCKING_CLARIFY_CONTINUATION, BLOCKING_CLARIFY_QUESTION, BLOCKING_CLARIFY_TRIGGER, type MockServer, startMockServer } from './mock-server'
import { RealSessionBuilder } from './real-session-builder'
import { type ElectronApplication, expect, type Page, test } from './test'

const SESSION_TITLE = 'E2E clarify durability long session'
const MIN_PERSISTED_MESSAGES = 741

// Each turn persists a user row and an assistant row, so this clears the floor
// above with room for the gateway's own bookkeeping rows.
const HISTORY_TURNS = Array.from(
  { length: 371 },
  (_unused, index) => `E2E durability turn ${index}: recheck the sandbox credential boundary`,
)

const NEWEST_SEEDED_TEXT = HISTORY_TURNS[HISTORY_TURNS.length - 1]

interface DurabilityFixture {
  app: ElectronApplication
  mock: MockServer
  page: Page
  persistedMessageCount: number
  sandbox: Sandbox
  cleanup: () => Promise<void>
}

let fixture: DurabilityFixture | null = null

async function setupSeededDesktop(): Promise<DurabilityFixture> {
  const mock = await startMockServer()
  const sandbox = createSandbox('clarify-durability')
  writeMockProviderConfig(sandbox.hermesHome, mock.url)
  writeEnvFile(sandbox.hermesHome)

  const builder = await RealSessionBuilder.start(sandbox.hermesHome)
  let persistedMessageCount = 0

  try {
    const created = await builder.createSession({ title: SESSION_TITLE, turns: HISTORY_TURNS })
    persistedMessageCount = await builder.countPersistedMessages(created.sessionId)
  } finally {
    await builder.close()
  }

  const { app, page } = await launchDesktop(buildAppEnv(sandbox))

  return {
    app,
    mock,
    page,
    persistedMessageCount,
    sandbox,
    cleanup: async () => {
      await app.close().catch(() => undefined)
      await mock.close()
      sandbox.cleanup()
    },
  }
}

/**
 * The mounted live clarify form and the card inside it. Not the store, not the
 * tool row — the DOM. A settled clarify renders outside a form, so requiring
 * the form is what makes this "live and answerable" rather than "on screen".
 */
const liveClarifyForm = (page: Page) => page.locator('form:has([data-slot="clarify-inline"])')
const liveClarifyCard = (page: Page) => liveClarifyForm(page).locator('[data-slot="clarify-inline"]')

/**
 * The sidebar's clarify attention state, read from the product's own control
 * rather than from the store: the session dot renders `role="status"` with the
 * `Needs your input` label exactly while a clarify blocks the turn.
 */
const needsInputAttention = (page: Page) =>
  page.locator('[data-slot="sidebar"] [role="status"][aria-label="Needs your input"]')

/** Text currently rendered in the transcript viewport. */
async function viewportText(page: Page): Promise<string> {
  return (await page.locator('[data-slot="aui_thread-viewport"]').first().textContent()) ?? ''
}

/**
 * Open the stored session and wait for its authoritative NEWEST-tail hydration
 * to land and hold.
 *
 * The transcript mounts a newest-tail window on purpose, so the newest seeded
 * turn is the exact anchor that proves real hydration of the resumed 741-row
 * conversation. "Hold" is observed rather than slept on: the anchor must be
 * present on two separate reads across a settle window, so a tail that is
 * still being replaced cannot be mistaken for a hydrated one.
 */
async function openSeededSession(page: Page): Promise<void> {
  const row = page.locator('[data-slot="sidebar"] button').filter({ hasText: SESSION_TITLE }).first()
  await row.waitFor({ state: 'visible', timeout: 120_000 })
  await row.click()

  await page.waitForFunction(
    expected => (document.querySelector('[data-slot="aui_thread-viewport"]')?.textContent ?? '').includes(expected),
    NEWEST_SEEDED_TEXT,
    { timeout: 120_000 },
  )

  await page.waitForTimeout(1_000)
  expect(await viewportText(page)).toContain(NEWEST_SEEDED_TEXT)
}

async function openNewSession(page: Page): Promise<void> {
  const button = page.locator('[data-slot="sidebar"] button').filter({ hasText: 'New session' }).first()
  await button.waitFor({ state: 'visible', timeout: 15_000 })
  await button.click()
  await page.waitForFunction(
    expected => !(document.querySelector('[data-slot="aui_thread-viewport"]')?.textContent ?? '').includes(expected),
    NEWEST_SEEDED_TEXT,
    { timeout: 30_000 },
  )
}

/** Exactly one mounted card, and at least one control the user can act on. */
async function expectOneAnswerableCard(page: Page): Promise<void> {
  const card = liveClarifyCard(page)
  await card.first().waitFor({ state: 'visible', timeout: 60_000 })
  await expect(card).toHaveCount(1)
  await expect(card.getByText(BLOCKING_CLARIFY_QUESTION)).toHaveCount(1)

  const choice = card.locator('button[data-choice]').first()
  await expect(choice).toBeEnabled()

  // The question is answerable exactly once across the whole transcript.
  await expect(page.getByText(BLOCKING_CLARIFY_QUESTION)).toHaveCount(1)
}

test.beforeAll(async () => {
  test.setTimeout(600_000)
  fixture = await setupSeededDesktop()
  await waitForAppReady(fixture, 180_000)
})

test.afterAll(async () => {
  await fixture?.cleanup()
  fixture = null
})

test.describe('clarify card durability across a long session', () => {
  test('one answerable card survives cold resume, tail hydration and a session switch', async () => {
    test.setTimeout(600_000)

    const page = fixture!.page

    // The persisted floor is read back from the store the desktop resumes
    // from, not inferred from the turn count.
    expect(fixture!.persistedMessageCount).toBeGreaterThanOrEqual(MIN_PERSISTED_MESSAGES)

    // Cold resume of the stored conversation: a new runtime session id for the
    // same durable history.
    await openSeededSession(page)

    // Raise the live blocking clarify. Nothing about it is persisted yet — the
    // tool has not returned.
    const composer = page.locator('[contenteditable="true"]').first()
    await composer.waitFor({ state: 'visible', timeout: 15_000 })
    await composer.click()
    await composer.type(BLOCKING_CLARIFY_TRIGGER, { delay: 5 })
    await page.keyboard.press('Enter')

    await expectOneAnswerableCard(page)

    // The attention state the field report showed, on the real control: one
    // session is asking for input, and it is asking exactly once.
    await expect(needsInputAttention(page)).toHaveCount(1)

    // Switch away and back. Re-activation rebuilds the transcript from the
    // authoritative newest tail, which legitimately lacks the running clarify
    // row — the exact transition that produced the field screenshot.
    await openNewSession(page)
    await expect(liveClarifyCard(page)).toHaveCount(0)

    // The card is not mounted here, but the request is still blocking, so the
    // sidebar must keep saying so.
    await expect(needsInputAttention(page)).toHaveCount(1)

    await openSeededSession(page)
    await expectOneAnswerableCard(page)
    await expect(needsInputAttention(page)).toHaveCount(1)

    // Answer through the card itself, against the original request.
    const card = liveClarifyCard(page)
    await card.getByRole('button', { name: /Yes/ }).first().click()

    const continueButton = liveClarifyForm(page).first().locator('button[type="submit"]')
    await expect(continueButton).toBeEnabled()
    await continueButton.click()

    // Settled exactly once: the live card is gone, the Q&A is on screen, and
    // no second live card reappeared behind it.
    const settled = page.locator('[data-clarify-settled]')
    await settled.first().waitFor({ state: 'visible', timeout: 60_000 })
    await expect(settled.getByText(BLOCKING_CLARIFY_QUESTION)).toHaveCount(1)
    await expect(liveClarifyCard(page)).toHaveCount(0)

    // Attention clears on that one settlement. Then the unique post-tool
    // assistant continuation must appear exactly once — seeded history must
    // not satisfy it.
    await expect(needsInputAttention(page)).toHaveCount(0, { timeout: 60_000 })
    await page.getByText(BLOCKING_CLARIFY_CONTINUATION).waitFor({ state: 'visible', timeout: 60_000 })
    await expect(page.getByText(BLOCKING_CLARIFY_CONTINUATION)).toHaveCount(1)
  })
})
