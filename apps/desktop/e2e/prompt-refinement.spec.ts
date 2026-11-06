/**
 * Prompt Refinement, end to end: the real Desktop app, a real gateway built
 * from this checkout, and the real `llm.oneshot` path to a mock OpenAI-
 * compatible model — all inside a throwaway sandbox (temp HOME, HERMES_HOME
 * and Electron userData; see fixtures.ts).
 *
 * The mock answers the refinement request (its input carries `<draft_prompt>`)
 * with a structured rewrite, and every agent turn with a fixed reply.
 * `receivedPrompts` is the witness of what the agent actually received:
 *  - Automatic sends a clean refinement with no click, and never the draft;
 *  - it stops (sends nothing) when the refinement would drop something, or
 *    asks for an action the prompt did not (a delete);
 *  - Cancel sends nothing; Approval-First opens the review on Enter and sends nothing
 *    until the user approves (or sends the original), and a failed refinement
 *    sends nothing whatever is pressed.
 *
 * Prerequisite: `pnpm run build` (dist/). Screenshots land in the test output dir.
 */

import { type MockBackendFixture, setupMockBackend, waitForAppReady } from './fixtures'
import { expect, test } from './test'

const AGENT_REPLY = 'Acknowledged: working on the refined task.'

const DRAFT_AUTO = 'fix the flaky upload test in src/upload.test.ts and do not change any other files'
const REFINED_AUTO = 'Fix the flaky upload test in src/upload.test.ts.\n\nConstraint: do not change any other files.'

// Held by the mock (see beforeAll) so the progress drawer can be seen and canceled.
const DRAFT_CANCEL = 'summarize the open issues about the release checklist for this week'

const DRAFT_MANUAL = 'rename the helper in src/utils/format.ts and update its callers'
const REFINED_MANUAL = 'Rename the helper in src/utils/format.ts and update its callers to match.'

const DRAFT_BLOCKED = 'migrate the legacy settings in config/legacy.yaml to the new schema'
const REFINED_BLOCKED = 'Migrate the legacy settings to the new schema.'

// The mock answers this one with JSON it cannot finish: the refinement fails.
const DRAFT_FAIL = 'outline the rollback steps for the payments migration'

// Read-only request; the mock's rewrite adds a delete the user never asked for.
const DRAFT_INTENT = 'explain how the retry logic in the upload client works'
const REFINED_INTENT = 'Explain how the retry logic in the upload client works, then delete the unused retry helpers.'

const answer = (prompt: string, why: string) =>
  JSON.stringify({
    prompt,
    professional: why,
    raw_material: 'The current source and its callers.',
    methods: 'Repository-wide search and the existing test suite.',
    verification: 'The full test suite passes before and after.'
  })

/** The refinement call's input wraps the draft; agent turns never do. */
function refineOrReply(prompt: string): string {
  if (!prompt.includes('<draft_prompt>')) {
    return AGENT_REPLY
  }

  if (prompt.includes('rollback steps for the payments migration')) {
    return '{"prompt": '
  }

  if (prompt.includes('retry logic in the upload client')) {
    return answer(REFINED_INTENT, 'Trace one failing request end to end.')
  }

  if (prompt.includes('config/legacy.yaml')) {
    return answer(REFINED_BLOCKED, 'Migrate with a reversible plan.')
  }

  if (prompt.includes('flaky upload test')) {
    return answer(REFINED_AUTO, 'Reproduce the flake before changing anything.')
  }

  if (prompt.includes('src/utils/format.ts')) {
    return answer(REFINED_MANUAL, 'Rename through every call site, not by hand.')
  }

  return 'Summarize this week’s open issues about the release checklist.'
}

let fixture: MockBackendFixture | null = null

/** Prompts that reached a model OTHER than the refinement call (agent turns, titles). */
const agentPrompts = () => fixture!.mock.receivedPrompts.filter(prompt => !prompt.includes('<draft_prompt>'))
const refinementPrompts = () => fixture!.mock.receivedPrompts.filter(prompt => prompt.includes('<draft_prompt>'))

const composer = () => fixture!.page.locator('[data-slot="composer-root"] [contenteditable]').first()
const drawer = () => fixture!.page.getByRole('region', { name: 'Prompt refinement' })

async function startNewChat() {
  const page = fixture!.page
  await page.evaluate(() => (window.location.hash = '#/'))
  await composer().waitFor({ state: 'visible', timeout: 30_000 })
}

async function clearComposer() {
  const page = fixture!.page
  await composer().click()
  await page.keyboard.press(process.platform === 'darwin' ? 'Meta+A' : 'Control+A')
  await page.keyboard.press('Backspace')
}

async function typeAndSend(text: string) {
  const page = fixture!.page
  await clearComposer()
  await composer().type(text)
  await page.keyboard.press('Enter')
}

async function setMode(label: 'Automatic' | 'Approval-First') {
  const page = fixture!.page
  await page.evaluate(() => (window.location.hash = '#/settings?tab=plugins&plugin=prompt-refinement'))
  await page.getByRole('button', { name: label, exact: true }).click()
}

test.beforeAll(async () => {
  // A cold source backend can take longer than one test's budget to boot.
  test.setTimeout(240_000)
  fixture = await setupMockBackend({
    mockServer: { replyForPrompt: refineOrReply, holdFirstCompletionContaining: 'release checklist' },
    // Bundled plugins are opt-in: enable the one that registers the `prompt_refinement`
    // slot, so the run covers the dedicated-slot path.
    extraConfig: 'plugins:\n  enabled:\n    - prompt-refinement\n'
  })
  await waitForAppReady(fixture!, 200_000)
})

test.afterAll(async () => {
  await fixture?.cleanup()
  fixture = null
})

test.describe.serial('prompt refinement', () => {
  test('ships Off: a send goes straight to the agent, no refinement call', async () => {
    await startNewChat()
    await typeAndSend('say hello to the release team')
    await fixture!.page.getByText(AGENT_REPLY).first().waitFor({ state: 'visible', timeout: 60_000 })

    expect(refinementPrompts()).toHaveLength(0)
    expect(agentPrompts().some(prompt => prompt.includes('say hello to the release team'))).toBe(true)
  })

  test('settings page switches to Automatic and steps the Depth slider', async () => {
    const page = fixture!.page
    await page.evaluate(() => (window.location.hash = '#/settings?tab=plugins&plugin=prompt-refinement'))
    await page.getByText('Refinement model', { exact: true }).waitFor({ state: 'visible', timeout: 30_000 })
    await page.screenshot({ path: test.info().outputPath('settings-off.png') })

    await page.getByRole('button', { name: 'Automatic', exact: true }).click()
    await expect(page.getByText(/Enter refines your prompt and sends it, no approval needed/)).toBeVisible()

    // Depth is chosen automatically by default; turning that off frees the slider.
    const auto = page.getByRole('switch', { name: 'Choose depth automatically' })
    const depth = page.getByRole('slider', { name: 'Depth' })
    await expect(auto).toHaveAttribute('aria-checked', 'true')
    await expect(depth).toBeDisabled()
    await auto.click()
    await expect(depth).toBeEnabled()

    // The Depth slider steps by keyboard like any range control, and says what each stop does.
    await expect(depth).toHaveAttribute('aria-valuetext', 'Deepen')
    await depth.focus()
    await page.keyboard.press('ArrowLeft')
    await expect(depth).toHaveAttribute('aria-valuetext', 'Clarify')
    await expect(page.getByText(/Spells out what your prompt implies/)).toBeVisible()
    await page.screenshot({ path: test.info().outputPath('settings-depth-clarify.png') })
    await page.keyboard.press('ArrowRight')
    await expect(depth).toHaveAttribute('aria-valuetext', 'Deepen')
    await auto.click()
    await expect(auto).toHaveAttribute('aria-checked', 'true')
    await page.screenshot({ path: test.info().outputPath('settings-auto.png') })
  })

  test('Automatic: the refined prompt is sent with no click, and the draft never reaches the agent', async () => {
    const page = fixture!.page
    await startNewChat()
    const before = agentPrompts().length

    await typeAndSend(DRAFT_AUTO)
    await page.getByText(AGENT_REPLY).first().waitFor({ state: 'visible', timeout: 60_000 })

    const sent = agentPrompts().slice(before)
    expect(refinementPrompts().some(prompt => prompt.includes(DRAFT_AUTO))).toBe(true)
    expect(sent.some(prompt => prompt.includes('Constraint: do not change any other files.'))).toBe(true)
    expect(sent.some(prompt => prompt.includes(DRAFT_AUTO))).toBe(false)
    await expect(drawer()).toHaveCount(0)
    await page.screenshot({ path: test.info().outputPath('auto-sent.png') })
  })

  test('Automatic: while refining, a slim drawer offers Cancel, which sends nothing and gives the draft back', async () => {
    const page = fixture!.page
    await startNewChat()
    const before = agentPrompts().length

    await typeAndSend(DRAFT_CANCEL)
    await expect(drawer().getByText('Refining your prompt before sending')).toBeVisible({ timeout: 30_000 })
    await page.screenshot({ path: test.info().outputPath('auto-refining.png') })

    await drawer().getByRole('button', { name: 'Cancel' }).click()
    await expect(drawer()).toHaveCount(0)
    await expect(composer()).toContainText(DRAFT_CANCEL)

    // Give a wrongly-sent turn time to show up before asserting it never did.
    await page.waitForTimeout(2_000)
    expect(agentPrompts().length).toBe(before)
  })

  test('Automatic: a refinement that drops a file path stops for review; Send original sends the draft', async () => {
    const page = fixture!.page
    await startNewChat()
    const before = agentPrompts().length

    await typeAndSend(DRAFT_BLOCKED)
    await expect(drawer().getByText(/Not sent automatically/)).toBeVisible({ timeout: 60_000 })
    await expect(drawer().getByText('File path: config/legacy.yaml')).toBeVisible()
    await expect(drawer().getByRole('button', { name: 'Approve and send' })).toBeDisabled()
    expect(agentPrompts().length).toBe(before)
    await page.screenshot({ path: test.info().outputPath('auto-stopped.png') })

    await drawer().getByRole('button', { name: 'Send original' }).click()
    await expect(drawer()).toHaveCount(0)
    await expect
      .poll(
        () =>
          agentPrompts()
            .slice(before)
            .some(prompt => prompt.includes(DRAFT_BLOCKED)),
        { timeout: 60_000 }
      )
      .toBe(true)
    expect(
      agentPrompts()
        .slice(before)
        .some(prompt => prompt.includes(REFINED_BLOCKED))
    ).toBe(false)
  })

  test('Automatic: a rewrite that adds a delete the prompt never asked for stops for review and is not sent', async () => {
    const page = fixture!.page
    await startNewChat()
    const before = agentPrompts().length

    await typeAndSend(DRAFT_INTENT)
    await expect(drawer().getByText(/Not sent automatically/)).toBeVisible({ timeout: 60_000 })
    await expect(drawer().getByText('Asks for more than your prompt did')).toBeVisible()
    await expect(drawer().getByText('Deletes or overwrites something: “delete”')).toBeVisible()
    await expect(drawer().getByRole('button', { name: 'Approve and send' })).toBeDisabled()
    await page.waitForTimeout(2_000)
    expect(agentPrompts().length).toBe(before)
    await page.screenshot({ path: test.info().outputPath('auto-intent-stopped.png') })

    await drawer().getByRole('button', { name: 'Cancel' }).click()
    await expect(drawer()).toHaveCount(0)
    await expect(composer()).toContainText(DRAFT_INTENT)
    expect(agentPrompts().length).toBe(before)
  })

  test('Approval-First: Enter opens the review; nothing is sent until ⌘/Ctrl+Enter approves, which sends the refined prompt', async () => {
    const page = fixture!.page
    await setMode('Approval-First')
    await expect(page.getByText(/Enter refines your prompt and shows it for review/)).toBeVisible()

    await startNewChat()
    // One pill, no separate refine button.
    await expect(page.getByRole('button', { name: /^Prompt refinement settings/ })).toContainText(
      'Refine: Approval-First'
    )
    await expect(page.getByRole('button', { name: 'Refine prompt' })).toHaveCount(0)
    const before = agentPrompts().length

    await typeAndSend(DRAFT_MANUAL)
    await expect(drawer().getByText(REFINED_MANUAL)).toBeVisible({ timeout: 60_000 })
    await expect(drawer().getByText('Everything from your prompt is kept')).toBeVisible()
    await drawer().getByRole('button', { name: 'What was added and why' }).click()
    await expect(drawer().getByText('Rename through every call site, not by hand.')).toBeVisible()
    await page.screenshot({ path: test.info().outputPath('manual-drawer.png') })

    // A clean refinement waits for the user in Approval-First: nothing has been sent.
    await page.waitForTimeout(1_500)
    expect(agentPrompts().length).toBe(before)

    // The keyboard shortcut, from wherever focus is inside the drawer.
    await page.keyboard.press(process.platform === 'darwin' ? 'Meta+Enter' : 'Control+Enter')
    await expect(drawer()).toHaveCount(0)
    await expect
      .poll(
        () =>
          agentPrompts()
            .slice(before)
            .some(prompt => prompt.includes(REFINED_MANUAL)),
        { timeout: 60_000 }
      )
      .toBe(true)
    expect(
      agentPrompts()
        .slice(before)
        .some(prompt => prompt.includes(DRAFT_MANUAL))
    ).toBe(false)
  })

  test('Approval-First: a failed refinement sends nothing — not on ⌘/Ctrl+Enter, not after Try again — and Cancel gives the draft back', async () => {
    const page = fixture!.page
    const approveChord = process.platform === 'darwin' ? 'Meta+Enter' : 'Control+Enter'
    await startNewChat()
    await expect(page.getByRole('button', { name: /^Prompt refinement settings/ })).toContainText(
      'Refine: Approval-First'
    )
    const before = agentPrompts().length
    const asked = refinementPrompts().length

    await typeAndSend(DRAFT_FAIL)
    await expect(drawer().getByText(/answered in a form that couldn’t be read/)).toBeVisible({ timeout: 60_000 })
    await expect(drawer().getByRole('button', { name: 'Approve and send' })).toBeDisabled()

    // The approve chord does nothing on a failed review.
    await page.keyboard.press(approveChord)
    await page.waitForTimeout(1_500)
    await expect(drawer()).toBeVisible()
    expect(agentPrompts().length).toBe(before)

    // Try again asks the model afresh; it fails again and still sends nothing.
    await drawer().getByRole('button', { name: 'Try again' }).click()
    await expect.poll(() => refinementPrompts().length - asked, { timeout: 60_000 }).toBe(2)
    await expect(drawer().getByText(/answered in a form that couldn’t be read/)).toBeVisible({ timeout: 60_000 })
    await expect(drawer().getByRole('button', { name: 'Approve and send' })).toBeDisabled()
    expect(agentPrompts().length).toBe(before)
    await page.screenshot({ path: test.info().outputPath('approval-first-failed.png') })

    await drawer().getByRole('button', { name: 'Cancel' }).click()
    await expect(drawer()).toHaveCount(0)
    await expect(composer()).toContainText(DRAFT_FAIL)
    await page.waitForTimeout(1_500)
    expect(agentPrompts().length).toBe(before)
    expect(
      agentPrompts()
        .slice(before)
        .some(prompt => prompt.includes(DRAFT_FAIL))
    ).toBe(false)
  })

  test('Chat context (opt-in) reaches the refiner; Send original sends as typed; the pill turns refinement off', async () => {
    const page = fixture!.page
    await page.evaluate(() => (window.location.hash = '#/settings?tab=plugins&plugin=prompt-refinement'))
    const contextSwitch = page.getByRole('switch', { name: 'Use recent chat context' })
    await expect(contextSwitch).toHaveAttribute('aria-checked', 'false')
    await contextSwitch.click()
    await expect(contextSwitch).toHaveAttribute('aria-checked', 'true')

    // Give the chat some history: in Approval-First, Send original sends it as typed.
    await startNewChat()
    const before = agentPrompts().length
    await typeAndSend('list the guides under docs/guides')
    await drawer().getByRole('button', { name: 'Send original' }).first().click()
    await expect
      .poll(
        () =>
          agentPrompts()
            .slice(before)
            .some(prompt => prompt.includes('list the guides under docs/guides')),
        { timeout: 60_000 }
      )
      .toBe(true)
    await page.getByText(AGENT_REPLY).first().waitFor({ state: 'visible', timeout: 60_000 })

    const asked = refinementPrompts().length
    await typeAndSend(DRAFT_MANUAL)
    await expect(drawer().getByText(REFINED_MANUAL)).toBeVisible({ timeout: 60_000 })

    const sent = refinementPrompts().slice(asked)
    expect(sent).toHaveLength(1)
    expect(sent[0]).toContain('<conversation>')
    expect(sent[0]).toContain('User: list the guides under docs/guides')
    const beforeCancel = agentPrompts().length
    await drawer().getByRole('button', { name: 'Cancel' }).click()
    await expect(drawer()).toHaveCount(0)
    await expect(composer()).toContainText(DRAFT_MANUAL)

    // Canceling an Approval-First review sends neither the draft nor the rewrite.
    await page.waitForTimeout(1_500)
    expect(
      agentPrompts()
        .slice(beforeCancel)
        .some(prompt => prompt.includes(DRAFT_MANUAL) || prompt.includes(REFINED_MANUAL))
    ).toBe(false)

    // The pill beside the model picker: one click to Off, and Enter sends as typed again.
    const pill = page.getByRole('button', { name: /^Prompt refinement settings/ })
    await pill.click()
    await page.screenshot({ path: test.info().outputPath('composer-pill-menu.png') })
    await page.getByRole('menuitemradio', { name: 'Off' }).click()
    await expect(pill).toContainText('Refine off')
  })
})
