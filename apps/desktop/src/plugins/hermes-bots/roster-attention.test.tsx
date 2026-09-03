/**
 * The needs-you attention badge the roster shows for a Group Chat row is
 * derived from TWO independent stores: $groupNeedsYou (an @user mention)
 * and $groupClarify (a pending clarify/approval, via groupHasPendingClarify).
 * roster-pane.tsx subscribes to both and combines them with ||.
 *
 * This harness proves the REAL composition works end-to-end: real stores,
 * real useValue subscriptions, and the real groupHasPendingClarify helper —
 * not a re-implementation of the derive expression. Mutating a store here
 * must be enough to flip what the harness renders, with no manual refresh.
 */
import { useValue } from '@hermes/plugin-sdk'
import { act, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { $groupClarify, $groupNeedsYou } from './group-chat'
import { groupHasPendingClarify } from './group-turns'
import type { GroupPrompt } from './types'

const GROUP = 'Core'

function mockPrompt(overrides: Partial<GroupPrompt> = {}): GroupPrompt {
  return {
    at: Date.now(),
    choices: ['staging', 'prod'],
    group: GROUP,
    kind: 'clarify',
    member: 'research',
    memberKey: 'research',
    multiSelect: false,
    question: 'Which env?',
    questions: null,
    requestId: 'req-1',
    sessionId: null,
    ...overrides
  }
}

function AttentionHarness({ group }: { group: string }) {
  const mentions = useValue($groupNeedsYou)
  const clarifies = useValue($groupClarify)
  const needsYou = Boolean(mentions[group]) || groupHasPendingClarify(clarifies, group)

  return <div data-testid="attention">{String(needsYou)}</div>
}

describe('roster attention badge — real stores, real useValue, real helper', () => {
  afterEach(() => {
    $groupNeedsYou.set({})
    $groupClarify.set({})
  })

  it('reflects a clarify being added and then removed', () => {
    act(() => {
      render(<AttentionHarness group={GROUP} />)
    })

    expect(screen.getByTestId('attention').textContent).toBe('false')

    act(() => {
      $groupClarify.set({ 'Core::research': mockPrompt() })
    })

    expect(screen.getByTestId('attention').textContent).toBe('true')

    act(() => {
      $groupClarify.set({})
    })

    expect(screen.getByTestId('attention').textContent).toBe('false')
  })

  it('keeps attention true when clarify clears but the mention persists', () => {
    act(() => {
      render(<AttentionHarness group={GROUP} />)
    })

    act(() => {
      $groupNeedsYou.set({ [GROUP]: true })
      $groupClarify.set({ 'Core::research': mockPrompt() })
    })

    expect(screen.getByTestId('attention').textContent).toBe('true')

    // Clarify resolves — the unrelated mention must keep the badge lit.
    act(() => {
      $groupClarify.set({})
    })

    expect(screen.getByTestId('attention').textContent).toBe('true')
  })
})
