/**
 * RED regression: historical delegate cards must not borrow the one live child
 * left in the session store after message.start prunes prior terminal rows.
 */

import type { TestInfo } from '@playwright/test'

import { type MockBackendFixture, setupMockBackend, waitForAppReady } from './fixtures'
import {
  DELEGATE_CARD_COMPLETION_ACKS,
  DELEGATE_CARD_COMPLETION_MARKERS,
  DELEGATE_CARD_GOALS,
  DELEGATE_CARD_PARENT_DONE,
  DELEGATE_CARD_TRIGGERS,
  DELEGATE_CARD_TURNS,
  restartMockServer,
} from './mock-server'
import { expect, type Page, test } from './test'

const ACTIVE_SURFACE = '[data-composer-target]:not([data-pane-hidden] [data-composer-target])'
const C_TERMINAL_ACTIVITY = `Terminal("printf 'C-terminal-activity\\n'")`

function activeSurface(page: Page) {
  return page.locator(ACTIVE_SURFACE).last()
}

function transcript(page: Page) {
  return activeSurface(page).locator('[data-slot="aui_thread-viewport"]')
}

function delegateCards(page: Page) {
  return transcript(page).locator('[data-delegate-card]')
}

async function send(page: Page, text: string): Promise<void> {
  const composer = activeSurface(page).locator('[contenteditable="true"]').first()
  await composer.waitFor({ state: 'visible', timeout: 15_000 })
  await composer.click()
  await composer.fill(text)
  await page.keyboard.press('Enter')
}

async function awaitParentDone(page: Page, text: string): Promise<void> {
  await expect(transcript(page)).toContainText(text, { timeout: 30_000 })
  await expect(page.getByRole('button', { name: /^Session running\b/ })).toHaveCount(0)
}

async function assertParkedCard(page: Page, index: number, goal: string): Promise<void> {
  const card = delegateCards(page).nth(index)

  await expect(card).toContainText(goal)
  await expect(card).not.toContainText('Mock Model')
  await expect(card).not.toContainText(C_TERMINAL_ACTIVITY)
  await expect(card).not.toContainText(/\b\d+s\b/)
  await expect(card.getByRole('button', { name: goal, exact: true })).toBeDisabled()
  await expect(card.getByLabel('Running')).toHaveCount(0)
  await expect(card.getByLabel('Done')).toHaveCount(0)
  await expect(activeSurface(page).getByText('1 Subagent', { exact: true })).toHaveCount(0)
}

async function awaitCompletionAck(page: Page, key: keyof typeof DELEGATE_CARD_COMPLETION_ACKS): Promise<void> {
  await expect(transcript(page).getByText(DELEGATE_CARD_COMPLETION_ACKS[key], { exact: true })).toHaveCount(1, {
    timeout: 30_000,
  })
  await expect(page.getByRole('button', { name: /^Session running\b/ })).toHaveCount(0)
}

type DelegateKey = keyof typeof DELEGATE_CARD_GOALS

function branchNames(fixture: MockBackendFixture): string[] {
  return fixture.mock.receivedRequestSummaries.map(summary => summary.selectedBranch)
}

async function waitForPartialOrder(
  fixture: MockBackendFixture,
  expected: string[],
  precedences: Array<readonly [string, string]>,
  matches: (branch: string) => boolean,
): Promise<void> {
  const expectedResult = {
    branches: [...expected].sort(),
    counts: expected.map(branch => `${branch}:1`),
    precedences: precedences.map(([before, after]) => `${before}<${after}:true`),
  }

  await expect.poll(() => {
    const allBranches = branchNames(fixture)
    const matchingBranches = allBranches.filter(matches)

    return {
      branches: [...matchingBranches].sort(),
      counts: expected.map(branch => `${branch}:${matchingBranches.filter(candidate => candidate === branch).length}`),
      precedences: precedences.map(([before, after]) => (
        `${before}<${after}:${allBranches.indexOf(before) >= 0 && allBranches.indexOf(before) < allBranches.indexOf(after)}`
      )),
    }
  }, {
    message: [...expected, ...precedences.flat()].join(' '),
    timeout: 30_000,
  }).toEqual(expectedResult)
}

async function waitForAcceptedLifecycle(fixture: MockBackendFixture, key: Exclude<DelegateKey, 'c'>): Promise<void> {
  const parentCall = `delegate-${key}-parent-call`
  const parentDone = `delegate-${key}-parent-done`
  const child = `delegate-${key}-child`
  const completion = `delegate-${key}-completion`

  await waitForPartialOrder(
    fixture,
    [parentCall, parentDone, child, completion],
    [
      [parentCall, parentDone],
      [parentCall, child],
      [child, completion],
      [parentDone, completion],
    ],
    branch => branch.startsWith(`delegate-${key}-`),
  )

  const childSummary = fixture.mock.receivedRequestSummaries.find(summary => summary.selectedBranch === child)
  expect(childSummary).toEqual({ selectedBranch: child, toolResultIds: [] })
}

async function waitForCActiveLifecycle(fixture: MockBackendFixture): Promise<void> {
  const expected = [
    'delegate-c-parent-call',
    'delegate-c-terminal',
    'delegate-c-parent-done',
    'held-text',
  ]

  await waitForPartialOrder(
    fixture,
    expected,
    [
      ['delegate-c-parent-call', 'delegate-c-parent-done'],
      ['delegate-c-parent-call', 'delegate-c-terminal'],
      ['delegate-c-terminal', 'held-text'],
    ],
    branch => branch === 'held-text' || branch.startsWith('delegate-c-'),
  )
}

async function waitForCCompletionAccepted(fixture: MockBackendFixture): Promise<void> {
  const expected = [
    'delegate-c-parent-call',
    'delegate-c-terminal',
    'delegate-c-parent-done',
    'held-text',
    'delegate-c-completion',
  ]

  await waitForPartialOrder(
    fixture,
    expected,
    [
      ['delegate-c-parent-call', 'delegate-c-parent-done'],
      ['delegate-c-parent-call', 'delegate-c-terminal'],
      ['delegate-c-terminal', 'held-text'],
      ['held-text', 'delegate-c-completion'],
      ['delegate-c-parent-done', 'delegate-c-completion'],
    ],
    branch => branch === 'held-text' || branch.startsWith('delegate-c-'),
  )
}

async function waitForCChildHeld(fixture: MockBackendFixture): Promise<void> {
  const childSummaries = () => fixture.mock.receivedRequestSummaries.filter(
    summary => summary.selectedBranch === 'delegate-c-terminal' || summary.selectedBranch === 'held-text',
  )

  await expect.poll(() => childSummaries().length, {
    message: 'C child should issue its terminal request and then enter held text streaming',
    timeout: 30_000,
  }).toBe(2)

  await Promise.race([
    fixture.mock.waitForHeldCompletion(),
    new Promise<never>((_, reject) => {
      setTimeout(() => reject(new Error(
        `Timed out waiting for held C stream. Branches: ${branchNames(fixture).join(' -> ')}`,
      )), 30_000)
    }),
  ])

  const [first, second] = childSummaries()
  expect(first).toMatchObject({ selectedBranch: 'delegate-c-terminal' })
  expect(first!.toolResultIds).not.toContain('e2e-c-terminal')
  expect(second).toMatchObject({ selectedBranch: 'held-text' })
  expect(second!.toolResultIds).toContain('e2e-c-terminal')
}

async function assertHistoricalCard(card: ReturnType<ReturnType<typeof delegateCards>['nth']>, ownGoal: string): Promise<void> {
  await expect(card).toContainText(ownGoal)
  await expect(card).not.toContainText(DELEGATE_CARD_GOALS.c)
  await expect(card).not.toContainText('Mock Model')
  await expect(card).not.toContainText(C_TERMINAL_ACTIVITY)
  await expect(card).not.toContainText(/\b\d+s\b/)
  await expect(card.getByRole('button', { name: ownGoal, exact: true })).toBeDisabled()
  await expect(card.getByLabel('Running')).toHaveCount(0)
  await expect(card.getByLabel('Done')).toHaveCount(0)
}

async function assertLiveCCard(card: ReturnType<ReturnType<typeof delegateCards>['nth']>): Promise<void> {
  await expect(card).toContainText(DELEGATE_CARD_GOALS.c)
  await expect(card).toContainText('Mock Model')
  await expect(card).toContainText(C_TERMINAL_ACTIVITY)
  await expect(card.getByText(/^\d+s$/)).toHaveCount(1)
  await expect(card.getByRole('button', { name: DELEGATE_CARD_GOALS.c, exact: true })).toBeEnabled()
  await expect(card.getByLabel('Running')).toHaveCount(1)
}

function validateDelegateTurns(): void {
  const calls = DELEGATE_CARD_TURNS.map(turn => turn.toolCalls?.[0])
  const ids = calls.map(call => call?.id)
  expect(ids.every(id => typeof id === 'string' && id.length > 0)).toBe(true)
  expect(new Set(ids).size).toBe(ids.length)

  for (const call of calls) {
    expect(call?.name).toBe('delegate_task')
    const serialized = JSON.stringify(call?.args)
    const parsed = JSON.parse(serialized) as { tasks?: Array<{ goal?: unknown; context?: unknown }> }
    expect(Object.keys(parsed)).toEqual(['tasks'])
    expect(parsed.tasks).toHaveLength(1)
    expect(typeof parsed.tasks![0]!.goal).toBe('string')
    expect(typeof parsed.tasks![0]!.context).toBe('string')
  }
}

function validateCompletionAcknowledgements(): void {
  const acknowledgements = Object.values(DELEGATE_CARD_COMPLETION_ACKS)

  const forbidden = [
    '[ASYNC DELEGATION BATCH COMPLETE',
    'HOLD_DELEGATE_C_CHILD',
    ...Object.values(DELEGATE_CARD_TRIGGERS),
    ...Object.values(DELEGATE_CARD_COMPLETION_MARKERS),
    ...Object.values(DELEGATE_CARD_GOALS),
  ]

  expect(new Set(acknowledgements).size).toBe(acknowledgements.length)

  for (const acknowledgement of acknowledgements) {
    for (const text of forbidden) {
      expect(acknowledgement).not.toContain(text)
    }
  }
}

async function openFreshSession(page: Page): Promise<void> {
  await page.locator('[data-slot="sidebar"] button[aria-label="New session"]').first().click()

  await expect(transcript(page)).not.toContainText(DELEGATE_CARD_TRIGGERS.a)
}

async function openSavedSession(page: Page): Promise<void> {
  const row = page.locator('[data-slot="sidebar"] button').filter({ hasText: DELEGATE_CARD_TRIGGERS.a }).first()
  await row.waitFor({ state: 'visible', timeout: 30_000 })
  await row.click()
  await expect(transcript(page)).toContainText(DELEGATE_CARD_TRIGGERS.a, { timeout: 30_000 })
}

test.describe('delegate card identity', () => {
  test.setTimeout(300_000)

  let fixture: MockBackendFixture | null = null

  test.beforeEach(async () => {
    validateDelegateTurns()
    validateCompletionAcknowledgements()
    restartMockServer()
    fixture = await setupMockBackend({
      extraConfig: 'auxiliary:\n  title_generation:\n    enabled: false',
      mockServer: {
        delegateCardScenario: true,
        holdFirstCompletionContaining: DELEGATE_CARD_GOALS.c,
      },
    })
    expect(fixture.mock.receivedRequestSummaries).toEqual([])
    await waitForAppReady(fixture, 120_000)
  })

  test.afterEach(async () => {
    fixture?.mock.releaseHeldStream()
    await fixture?.cleanup()
    fixture = null
  })

  // Playwright requires object destructuring for its fixtures callback.
  // eslint-disable-next-line no-empty-pattern
  test('keeps historical A and B cards isolated from live C before, after, and across cold hydration', async ({}, testInfo: TestInfo) => {
    const { mock, page } = fixture!

    await send(page, DELEGATE_CARD_TRIGGERS.a)
    await waitForAcceptedLifecycle(fixture!, 'a')
    await awaitCompletionAck(page, 'a')
    await expect(delegateCards(page)).toHaveCount(1)
    await assertParkedCard(page, 0, DELEGATE_CARD_GOALS.a)

    await send(page, DELEGATE_CARD_TRIGGERS.b)
    await waitForAcceptedLifecycle(fixture!, 'b')
    await awaitCompletionAck(page, 'b')
    await expect(delegateCards(page)).toHaveCount(2)
    await assertParkedCard(page, 0, DELEGATE_CARD_GOALS.a)
    await assertParkedCard(page, 1, DELEGATE_CARD_GOALS.b)

    await send(page, DELEGATE_CARD_TRIGGERS.c)
    await awaitParentDone(page, DELEGATE_CARD_PARENT_DONE.c)
    await expect(delegateCards(page)).toHaveCount(3)
    await waitForCChildHeld(fixture!)
    await waitForCActiveLifecycle(fixture!)
    await expect(activeSurface(page).getByText('1 Subagent', { exact: true })).toHaveCount(1)

    const cards = delegateCards(page)
    const cardA = cards.nth(0)
    const cardB = cards.nth(1)
    const cardC = cards.nth(2)
    await page.screenshot({ path: testInfo.outputPath('delegate-card-identity-active.png') })

    await assertHistoricalCard(cardA, DELEGATE_CARD_GOALS.a)
    await assertHistoricalCard(cardB, DELEGATE_CARD_GOALS.b)
    await assertLiveCCard(cardC)

    mock.releaseHeldStream()
    await waitForCCompletionAccepted(fixture!)
    await awaitCompletionAck(page, 'c')
    await expect(activeSurface(page).getByText('1 Subagent', { exact: true })).toHaveCount(0)
    await expect(cards).toHaveCount(3)
    await assertParkedCard(page, 0, DELEGATE_CARD_GOALS.a)
    await assertParkedCard(page, 1, DELEGATE_CARD_GOALS.b)
    await assertParkedCard(page, 2, DELEGATE_CARD_GOALS.c)

    const savedRow = page.locator('[data-slot="sidebar"] button').filter({ hasText: DELEGATE_CARD_TRIGGERS.a }).first()
    await savedRow.waitFor({ state: 'visible', timeout: 30_000 })
    await page.reload()
    await waitForAppReady(fixture!, 120_000)
    await openFreshSession(page)
    await openSavedSession(page)

    await expect(delegateCards(page)).toHaveCount(3)
    await assertParkedCard(page, 0, DELEGATE_CARD_GOALS.a)
    await assertParkedCard(page, 1, DELEGATE_CARD_GOALS.b)
    await assertParkedCard(page, 2, DELEGATE_CARD_GOALS.c)
  })
})
