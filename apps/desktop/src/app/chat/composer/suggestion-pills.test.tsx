import { act, render } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { $composerSuggestionsBySession, type ComposerSuggestion, offerSuggestions } from '@/store/composer-suggestions'

import { SuggestionPills } from './suggestion-pills'

/**
 * The pill strip owns phase; the bus owns existence. These cover the seam
 * between them — what has to happen to a pill's phase and its in-flight work
 * when the bus withdraws it, which is the normal way a suggestion ends.
 */

interface Flow {
  resolve: () => void
  reject: (error: unknown) => void
  /** Latest value of the invoke context's `cancelled()` poll. */
  cancelled: () => boolean
}

function pill(id: string, provider = 'mcp'): { suggestion: ComposerSuggestion; flow: Flow } {
  const flow: Partial<Flow> = {}

  const suggestion: ComposerSuggestion = {
    doneLabel: `Added ${id}`,
    doneTip: 'done',
    id,
    invoke: context =>
      new Promise<void>((resolve, reject) => {
        flow.resolve = resolve
        flow.reject = reject
        flow.cancelled = context.cancelled
      }),
    label: `Add ${id}`,
    provider,
    tip: 'because',
    workingLabel: `Connecting ${id}…`,
    workingTip: 'cancel'
  }

  return { flow: flow as Flow, suggestion }
}

const labels = (container: HTMLElement) => [...container.querySelectorAll('button')].map(button => button.textContent)

const click = async (container: HTMLElement) => {
  await act(async () => {
    container.querySelector('button')!.click()
  })
}

afterEach(() => {
  $composerSuggestionsBySession.set({})
})

describe('SuggestionPills', () => {
  it('narrates idle → working → done through one invoke', async () => {
    const { flow, suggestion } = pill('github')
    const { container } = render(<SuggestionPills sessionId="s1" />)

    act(() => offerSuggestions('s1', 'mcp', [suggestion]))
    expect(labels(container)).toEqual(['Add github'])

    await click(container)
    expect(labels(container)).toEqual(['Connecting github…'])

    await act(async () => flow.resolve())
    expect(labels(container)).toEqual(['Added github'])
  })

  it('re-offering a withdrawn key starts from idle, not a stale done', async () => {
    const first = pill('github')
    const { container } = render(<SuggestionPills sessionId="s1" />)

    act(() => offerSuggestions('s1', 'mcp', [first.suggestion]))
    await click(container)
    await act(async () => first.flow.resolve())
    expect(labels(container)).toEqual(['Added github'])

    // The provider withdraws (draft lost the trigger) and offers again later.
    act(() => offerSuggestions('s1', 'mcp', []))
    act(() => offerSuggestions('s1', 'mcp', [pill('github').suggestion]))

    expect(labels(container)).toEqual(['Add github'])
  })

  it('a fresh offer is clickable again after an earlier one completed', async () => {
    const first = pill('github')
    const second = pill('github')
    const { container } = render(<SuggestionPills sessionId="s1" />)

    act(() => offerSuggestions('s1', 'mcp', [first.suggestion]))
    await click(container)
    await act(async () => first.flow.resolve())

    act(() => offerSuggestions('s1', 'mcp', []))
    act(() => offerSuggestions('s1', 'mcp', [second.suggestion]))
    await click(container)

    // A stale `done` swallows the click (only idle invokes) — this is the
    // user-visible half of the phase leak.
    expect(labels(container)).toEqual(['Connecting github…'])
  })

  it('cancels the in-flight action when the pill is withdrawn mid-flight', async () => {
    const { flow, suggestion } = pill('github')
    const { container } = render(<SuggestionPills sessionId="s1" />)

    act(() => offerSuggestions('s1', 'mcp', [suggestion]))
    await click(container)
    expect(flow.cancelled()).toBe(false)

    // The pill is the only cancel affordance; losing it must abort the work.
    act(() => offerSuggestions('s1', 'mcp', []))

    expect(labels(container)).toEqual([])
    expect(flow.cancelled()).toBe(true)
  })

  it('cancels in-flight work when the strip unmounts', async () => {
    const { flow, suggestion } = pill('github')
    const { container, unmount } = render(<SuggestionPills sessionId="s1" />)

    act(() => offerSuggestions('s1', 'mcp', [suggestion]))
    await click(container)

    unmount()

    expect(flow.cancelled()).toBe(true)
  })

  it('does not carry phase across a session switch on one mounted composer', async () => {
    const { flow, suggestion } = pill('github')
    const { container, rerender } = render(<SuggestionPills sessionId="s1" />)

    act(() => offerSuggestions('s1', 'mcp', [suggestion]))
    await click(container)
    await act(async () => flow.resolve())
    expect(labels(container)).toEqual(['Added github'])

    act(() => offerSuggestions('s2', 'mcp', [pill('github').suggestion]))
    rerender(<SuggestionPills sessionId="s2" />)

    expect(labels(container)).toEqual(['Add github'])
  })

  it('returns to idle when the provider rejects, so it can be retried', async () => {
    const { flow, suggestion } = pill('github')
    const { container } = render(<SuggestionPills sessionId="s1" />)

    act(() => offerSuggestions('s1', 'mcp', [suggestion]))
    await click(container)

    await act(async () => flow.reject(new Error('nope')))

    expect(labels(container)).toEqual(['Add github'])
  })

  it('a second click on a working pill requests cancel', async () => {
    const { flow, suggestion } = pill('github')
    const { container } = render(<SuggestionPills sessionId="s1" />)

    act(() => offerSuggestions('s1', 'mcp', [suggestion]))
    await click(container)
    await click(container)

    expect(flow.cancelled()).toBe(true)
    expect(labels(container)).toEqual(['Connecting github…'])
  })

  it('invokes each pill independently when two are offered', async () => {
    const a = pill('github')
    const b = pill('linear')
    const invoke = vi.spyOn(b.suggestion, 'invoke')
    const { container } = render(<SuggestionPills sessionId="s1" />)

    act(() => offerSuggestions('s1', 'mcp', [a.suggestion, b.suggestion]))
    await click(container)

    expect(labels(container)).toEqual(['Connecting github…', 'Add linear'])
    expect(invoke).not.toHaveBeenCalled()
  })
})
