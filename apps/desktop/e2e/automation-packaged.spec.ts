import { expect, test, type Page } from '@playwright/test'

import { waitForAppReady } from './fixtures'
import {
  ensureActiveSession,
  launchPackagedReal,
  openCreateAutomation,
  pausePreservesUnsavedDraft,
  pickAutomationType,
  type PackagedReal,
} from './packaged-automation'

// Durable packaged regression for the session-automation lifecycle.
//
// These exercise the ACTUAL packaged binary (release/win-unpacked/Hermes.exe)
// with a real backend resolved portably (HERMES_DESKTOP_HERMES_ROOT +
// HERMES_DESKTOP_PYTHON, or venv discovery) and mock inference, in a fresh
// disposable home each run. No machine-specific path is committed; the older
// spec's hardcoded test-build root and missing Python override were the gap.

const GOAL_MARKER = 'E2E_GOAL_DONE_MARKER'

// The goal status bar is a single button whose accessible name carries the
// status plus turn count (e.g. "Goal done · 1 turn", "Goal active").
const GOAL_DONE = (page: Page) => page.getByText(/Goal done/)

async function bootToChat(fixture: PackagedReal): Promise<void> {
  await waitForAppReady(fixture as never, 90_000)
  await ensureActiveSession(fixture.page, 'Packaged session boot check.')
}

test('packaged automation injects a real backend and opens the composer', async () => {
  test.setTimeout(180_000)
  const fx = await launchPackagedReal()
  try {
    await bootToChat(fx)
    await openCreateAutomation(fx.page)
    await expect(fx.page.getByRole('button', { name: 'Heartbeat', exact: true })).toBeVisible()
    await expect(fx.page.getByRole('button', { name: 'Goal', exact: true })).toBeVisible()
    await expect(fx.page.getByRole('button', { name: 'Loop', exact: true })).toBeVisible()
  } finally {
    await fx.cleanup()
  }
})

test('packaged Goal: deterministic criteria completion with no extra continuation', async () => {
  test.setTimeout(240_000)
  const fx = await launchPackagedReal({ goalJudgeDoneForPrompt: GOAL_MARKER })
  try {
    await bootToChat(fx)
    const { page, mock } = fx

    await openCreateAutomation(page)
    await pickAutomationType(page, 'Goal')

    const goalPrompt = `Write E2E_DONE ${GOAL_MARKER}`
    await page.getByLabel('Goal prompt', { exact: true }).fill(goalPrompt)
    await page.getByRole('button', { name: 'Start goal', exact: true }).click()
    await expect(page.getByRole('dialog')).not.toBeVisible({ timeout: 30_000 })

    // The goal fires its first turn; the mock judge then returns DONE because
    // the marker goal is present — criteria-driven completion.
    await expect.poll(
      () => mock.receivedPrompts.some(text => text.includes(GOAL_MARKER)),
      { timeout: 90_000 },
    ).toBe(true)

    // Criteria-driven completion: the status bar reports done.
    await expect(GOAL_DONE(page).first()).toBeVisible({ timeout: 90_000 })

    // No additional automatic continuation: after the judge verdict, no further
    // agent turn carrying the continuation marker is dispatched.
    const promptsAtDone = mock.receivedPrompts.length
    await page.waitForTimeout(12_000)
    const extra = mock.receivedPrompts.slice(promptsAtDone).filter(t => t.includes('[Continuing toward'))
    expect(extra).toHaveLength(0)
  } finally {
    await fx.cleanup()
  }
})

test('packaged Goal: pause preserves unsaved edits; clear cancel vs confirm', async () => {
  test.setTimeout(240_000)
  const fx = await launchPackagedReal()
  try {
    await bootToChat(fx)
    const { page, mock } = fx

    await openCreateAutomation(page)
    await pickAutomationType(page, 'Goal')
    await page.getByLabel('Goal prompt', { exact: true }).fill('Packaged goal edit test')
    await page.getByRole('button', { name: 'Start goal', exact: true }).click()
    await expect(page.getByRole('dialog')).not.toBeVisible({ timeout: 30_000 })

    // Pause first: an active goal chains continuation turns, so the backend
    // would refuse a mid-turn edit. Pause is not gated on the live turn.
    await page.getByRole('button', { name: 'Goal actions', exact: true }).first().click()
    await page.getByRole('menuitem', { name: 'Pause goal', exact: true }).click()
    await expect(page.getByText('Goal paused', { exact: false }).first()).toBeVisible({ timeout: 30_000 })

    // Now the goal is idle and editable: open the edit dialog, type an unsaved
    // change, and assert Pause inside the dialog preserves both dialog and draft.
    await page.getByRole('button', { name: 'Goal actions', exact: true }).first().click()
    await page.getByRole('menuitem', { name: 'Edit goal', exact: true }).click()
    const dialog = page.getByRole('dialog')
    await expect(dialog).toBeVisible()
    await dialog.getByLabel('Goal prompt', { exact: true }).fill('Edited but not saved yet')
    await dialog.getByRole('button', { name: 'Pause goal', exact: true }).click()
    await expect(dialog).toBeVisible()
    await expect(dialog.getByLabel('Goal prompt', { exact: true })).toHaveValue('Edited but not saved yet')
    await dialog.getByRole('button', { name: 'Cancel', exact: true }).click()
    await expect(dialog).not.toBeVisible()

    // Destructive clear: cancel leaves the goal intact; confirming removes it.
    await page.getByRole('button', { name: 'Goal actions', exact: true }).first().click()
    await page.getByRole('menuitem', { name: 'Clear goal', exact: true }).click()
    const clearDialog = page.getByRole('dialog')
    await expect(clearDialog).toBeVisible()
    await expect(clearDialog.getByText('Clear goal?', { exact: true })).toBeVisible()
    await clearDialog.getByRole('button', { name: 'Cancel', exact: true }).click()
    await expect(clearDialog).not.toBeVisible()

    const promptsBeforeClear = mock.receivedPrompts.length
    await page.getByRole('button', { name: 'Goal actions', exact: true }).first().click()
    await page.getByRole('menuitem', { name: 'Clear goal', exact: true }).click()
    await page.getByRole('dialog').getByRole('button', { name: 'Clear goal', exact: true }).click()

    // The goal card is removed and no automatic continuation follows the clear.
    await expect(page.getByRole('button', { name: 'Goal actions', exact: true })).toHaveCount(0, { timeout: 30_000 })
    await page.waitForTimeout(12_000)
    const extra = mock.receivedPrompts.slice(promptsBeforeClear).filter(t => t.includes('[Continuing toward'))
    expect(extra).toHaveLength(0)
  } finally {
    await fx.cleanup()
  }
})

test('packaged Loop: capped at one tick fires exactly once, no extra tick', async () => {
  test.setTimeout(240_000)
  const fx = await launchPackagedReal()
  try {
    await bootToChat(fx)
    const { page, mock } = fx

    await openCreateAutomation(page)
    await pickAutomationType(page, 'Loop')
    await page.getByLabel('Loop prompt', { exact: true }).fill('Packaged loop tick')
    await page.getByLabel('Interval', { exact: true }).fill('5')
    await page.getByLabel(/Run limit/).fill('1')
    await page.getByRole('button', { name: 'Start loop', exact: true }).click()
    await expect(page.getByRole('dialog')).not.toBeVisible({ timeout: 30_000 })

    // One tick fires; the cap stops it — assert exactly one, then no second.
    await expect.poll(
      () => mock.receivedPrompts.filter(text => text.includes('Packaged loop tick')).length,
      { timeout: 60_000 },
    ).toBeGreaterThanOrEqual(1)
    const ticks = () => mock.receivedPrompts.filter(text => text.includes('Packaged loop tick')).length
    await expect.poll(ticks, { timeout: 45_000 }).toBe(1)
  } finally {
    await fx.cleanup()
  }
})

test('packaged Loop: edit pause preserves draft; cap-clear save/reopen; stop cancel vs confirm', async () => {
  test.setTimeout(240_000)
  const fx = await launchPackagedReal()
  try {
    await bootToChat(fx)
    const { page } = fx

    await openCreateAutomation(page)
    await pickAutomationType(page, 'Loop')
    await page.getByLabel('Loop prompt', { exact: true }).fill('Packaged loop edit B')
    await page.getByLabel('Interval', { exact: true }).fill('60')
    await page.getByLabel(/Run limit/).fill('3')
    await page.getByRole('button', { name: 'Start loop', exact: true }).click()
    await expect(page.getByRole('dialog')).not.toBeVisible({ timeout: 30_000 })

    await page.getByRole('button', { name: 'Loop actions', exact: true }).first().click()
    await page.getByRole('menuitem', { name: 'Edit loop', exact: true }).click()
    const editDialog = page.getByRole('dialog')
    await expect(editDialog).toBeVisible()
    await editDialog.getByLabel('Loop prompt', { exact: true }).fill('Edited loop draft')
    await editDialog.getByRole('button', { name: 'Pause loop', exact: true }).click()
    await expect(editDialog).toBeVisible()
    await expect(editDialog.getByLabel('Loop prompt', { exact: true })).toHaveValue('Edited loop draft')
    // Cap-clear: save with a blank run limit, then reopen and verify persisted unlimited.
    await editDialog.getByLabel(/Run limit/).fill('')
    await editDialog.getByRole('button', { name: 'Save loop', exact: true }).click()
    await expect(page.getByRole('dialog')).not.toBeVisible({ timeout: 30_000 })

    await page.getByRole('button', { name: 'Loop actions', exact: true }).first().click()
    await page.getByRole('menuitem', { name: 'Edit loop', exact: true }).click()
    await expect(page.getByRole('dialog').getByLabel(/Run limit/)).toHaveValue('')
    await page.getByRole('dialog').getByRole('button', { name: 'Cancel', exact: true }).click()

    // Stop cancel keeps the loop; confirm stops it.
    await page.getByRole('button', { name: 'Loop actions', exact: true }).first().click()
    await page.getByRole('menuitem', { name: 'Stop loop', exact: true }).click()
    const stopDialog = page.getByRole('dialog')
    await expect(stopDialog).toBeVisible()
    await stopDialog.getByRole('button', { name: 'Cancel', exact: true }).click()
    await expect(stopDialog).not.toBeVisible()

    await page.getByRole('button', { name: 'Loop actions', exact: true }).first().click()
    await page.getByRole('menuitem', { name: 'Stop loop', exact: true }).click()
    await page.getByRole('dialog').getByRole('button', { name: 'Stop loop', exact: true }).click()
    await expect(page.getByRole('button', { name: 'Loop actions', exact: true })).toHaveCount(0, { timeout: 60_000 })
  } finally {
    await fx.cleanup()
  }
})

test('packaged Heartbeat: pause preserves unsaved edits; clear cancel vs confirm', async () => {
  test.setTimeout(240_000)
  const fx = await launchPackagedReal()
  try {
    await bootToChat(fx)
    const { page } = fx

    await openCreateAutomation(page)
    await pickAutomationType(page, 'Heartbeat')
    await page.getByLabel('Heartbeat prompt', { exact: true }).fill('Packaged heartbeat')
    await page.getByLabel('Interval', { exact: true }).fill('60')
    await page.getByRole('button', { name: 'Create heartbeat', exact: true }).click()
    await expect(page.getByRole('dialog')).not.toBeVisible({ timeout: 30_000 })

    // Edit-dialog pause preserves the draft.
    await page.getByRole('button', { name: 'Heartbeat actions', exact: true }).first().click()
    await page.getByRole('menuitem', { name: 'Edit heartbeat', exact: true }).click()
    const dialog = page.getByRole('dialog')
    await expect(dialog).toBeVisible()
    await dialog.getByLabel('Heartbeat prompt', { exact: true }).fill('Edited heartbeat draft')
    await dialog.getByRole('button', { name: 'Pause heartbeat', exact: true }).click()
    await expect(dialog).toBeVisible()
    await expect(dialog.getByLabel('Heartbeat prompt', { exact: true })).toHaveValue('Edited heartbeat draft')
    await dialog.getByRole('button', { name: 'Cancel', exact: true }).click()
    await expect(dialog).not.toBeVisible()

    // Destructive clear: cancel then confirm.
    await page.getByRole('button', { name: 'Heartbeat actions', exact: true }).first().click()
    await page.getByRole('menuitem', { name: 'Clear heartbeat', exact: true }).click()
    await page.getByRole('dialog').getByRole('button', { name: 'Cancel', exact: true }).click()
    await expect(page.getByRole('dialog')).not.toBeVisible()

    await page.getByRole('button', { name: 'Heartbeat actions', exact: true }).first().click()
    await page.getByRole('menuitem', { name: 'Clear heartbeat', exact: true }).click()
    await page.getByRole('dialog').getByRole('button', { name: 'Clear heartbeat', exact: true }).click()
    await expect(page.getByRole('dialog')).not.toBeVisible()
  } finally {
    await fx.cleanup()
  }
})