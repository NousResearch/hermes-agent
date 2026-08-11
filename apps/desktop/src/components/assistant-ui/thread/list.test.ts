import { AssistantRuntimeProvider, type ThreadMessage, useExternalStoreRuntime } from '@assistant-ui/react'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { createElement, useCallback, useLayoutEffect, useState } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { PaneVisibleContext } from '@/components/pane-shell/pane-visibility'

import { Thread } from '.'

import {
  buildGroups,
  commitReceiptChain,
  firstVisibleGroupIndex,
  isThreadRenderComplete,
  LIVE_TAIL_MIN_GROUPS,
  LIVE_TAIL_PARTS,
  liveTailStart,
  type MessageGroup,
  partRequiresCommit,
  pruneCommitMap,
  RENDER_BUDGET,
  resolveThreadScrollTarget,
  shouldRestoreResumeAnchor,
  type ThreadCommitReceipt
} from './list'

class TestResizeObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
}

vi.stubGlobal('ResizeObserver', TestResizeObserver)
vi.stubGlobal('requestAnimationFrame', (callback: FrameRequestCallback) =>
  window.setTimeout(() => callback(performance.now()), 0)
)
vi.stubGlobal('cancelAnimationFrame', (id: number) => window.clearTimeout(id))
vi.stubGlobal('CSS', { escape: (str: string) => str })

Element.prototype.scrollTo = function scrollTo() {}
Element.prototype.animate = function animate() {
  return { cancel: () => {}, finished: Promise.resolve() } as unknown as Animation
}

afterEach(() => cleanup())

// Signature rows are `${index}:${id}:${role}:${weight}` (see the useAuiState
// selector in list.tsx).
const signature = (rows: [string, string, number][]) =>
  rows.map(([id, role, weight], index) => `${index}:${id}:${role}:${weight}`).join('\n')

describe('buildGroups', () => {
  it('returns no groups for an empty signature', () => {
    expect(buildGroups('')).toEqual([])
  })

  it('groups a user message with the assistant turn(s) that follow it', () => {
    const groups = buildGroups(
      signature([
        ['u1', 'user', 1],
        ['a1', 'assistant', 4],
        ['a2', 'assistant', 2],
        ['u2', 'user', 1],
        ['a3', 'assistant', 3]
      ])
    )

    expect(groups).toEqual([
      { id: 'u1', indices: [0, 1, 2], kind: 'turn', weight: 7 },
      { id: 'u2', indices: [3, 4], kind: 'turn', weight: 4 }
    ])
  })

  it('keeps leading non-user messages as standalone groups', () => {
    const groups = buildGroups(
      signature([
        ['s1', 'system', 1],
        ['a0', 'assistant', 2],
        ['u1', 'user', 1],
        ['a1', 'assistant', 5]
      ])
    )

    expect(groups).toEqual([
      { id: 's1', index: 0, kind: 'standalone', weight: 1 },
      { id: 'a0', index: 1, kind: 'standalone', weight: 2 },
      { id: 'u1', indices: [2, 3], kind: 'turn', weight: 6 }
    ])
  })

  it('defaults a missing/zero weight to 1', () => {
    const groups = buildGroups('0:a:assistant:0')

    expect(groups).toEqual([{ id: 'a', index: 0, kind: 'standalone', weight: 1 }])
  })
})

describe('resolveThreadScrollTarget', () => {
  const context = (scrollElement: Pick<HTMLElement, 'scrollTop'>) => ({
    contentElement: document.createElement('div'),
    scrollElement: scrollElement as HTMLElement
  })

  it('settles when the browser clamps the requested bottom within half a CSS pixel', () => {
    let actualScrollTop = 0
    let writes = 0

    const scrollElement = {
      get scrollTop() {
        return actualScrollTop
      },
      set scrollTop(value: number) {
        writes += 1
        actualScrollTop = value - 0.125
      }
    }

    const target = 899

    const requested = resolveThreadScrollTarget(target, context(scrollElement))
    scrollElement.scrollTop = requested
    const settled = resolveThreadScrollTarget(target, context(scrollElement))

    expect(requested).toBe(target)
    expect(actualScrollTop).toBe(898.875)
    expect(settled).toBe(actualScrollTop)
    expect(actualScrollTop < settled).toBe(false)
    expect(writes).toBe(1)
  })

  it('keeps following while more than half a CSS pixel remains', () => {
    const scrollElement = { scrollTop: 898.25 }

    expect(resolveThreadScrollTarget(899, context(scrollElement))).toBe(899)
  })

  it('re-arms after streaming content increases the target', () => {
    const scrollElement = { scrollTop: 898.875 }

    expect(resolveThreadScrollTarget(899, context(scrollElement))).toBe(898.875)
    expect(resolveThreadScrollTarget(999, context(scrollElement))).toBe(999)
  })
})

describe('shouldRestoreResumeAnchor', () => {
  it('restores only for a newer authority revision while still following the bottom', () => {
    expect(shouldRestoreResumeAnchor(2, 3, true)).toBe(true)
    expect(shouldRestoreResumeAnchor(2, 2, true)).toBe(false)
    expect(shouldRestoreResumeAnchor(3, 2, true)).toBe(false)
    expect(shouldRestoreResumeAnchor(2, 3, false)).toBe(false)
  })
})

describe('isThreadRenderComplete', () => {
  it('waits for first-paint backfill when earlier groups are budget-hidden', () => {
    expect(isThreadRenderComplete(3, 20)).toBe(false)
    expect(isThreadRenderComplete(0, 20)).toBe(true)
    expect(isThreadRenderComplete(3, RENDER_BUDGET)).toBe(true)
  })
})

describe('firstVisibleGroupIndex', () => {
  const group = (id: string, weight: number): MessageGroup => ({ id, index: 0, kind: 'standalone', weight })

  it('shows everything when total weight fits the budget', () => {
    const groups = [group('a', 10), group('b', 10), group('c', 10)]

    expect(firstVisibleGroupIndex(groups, 100)).toBe(0)
  })

  it('walks newest-first and hides everything before the turn that meets the budget', () => {
    const groups = [group('old', 50), group('mid', 30), group('new', 30)]

    // newest-first: 30 (new) < 60, +30 (mid) = 60 >= 60 → mid is the first
    // visible group, old is hidden.
    expect(firstVisibleGroupIndex(groups, 60)).toBe(1)
  })

  it('keeps whole turns intact — the turn that crosses the budget stays visible', () => {
    const groups = [group('old', 5), group('huge', 500)]

    expect(firstVisibleGroupIndex(groups, 60)).toBe(1)
  })

  it('returns groups.length for an empty list', () => {
    expect(firstVisibleGroupIndex([], 60)).toBe(0)
  })

  it('keeps a floor of turns visible however heavy they are', () => {
    // Without the floor a session of enormous turns puts "Show earlier" two
    // turns from the bottom, which reads as broken rather than as paging.
    const groups = Array.from({ length: 20 }, (_, i) => group(`g${i}`, 5_000))

    expect(firstVisibleGroupIndex(groups, 600, 8)).toBe(groups.length - 8)
  })

  it('does not force the floor to hide turns the budget already showed', () => {
    const groups = Array.from({ length: 20 }, (_, i) => group(`g${i}`, 1))

    expect(firstVisibleGroupIndex(groups, 600, 8)).toBe(0)
  })
})

describe('liveTailStart', () => {
  const group = (id: string, weight: number): MessageGroup => ({ id, index: 0, kind: 'standalone', weight })

  it('keeps the newest turns rendered until the parts budget is spent', () => {
    // 10 turns x 10 parts. A 40-part tail covers the newest 4-5 turns.
    const groups = Array.from({ length: 10 }, (_, i) => group(`g${i}`, 10))
    const start = liveTailStart(groups)

    expect(start).toBeGreaterThan(0)
    expect(start).toBeLessThan(groups.length)

    // Everything from `start` onward is the live tail...
    const tailParts = groups.slice(start).reduce((sum, g) => sum + g.weight, 0)
    expect(tailParts).toBeGreaterThan(LIVE_TAIL_PARTS)

    // ...and dropping its oldest member puts it back under budget, i.e. the
    // tail is minimal rather than sprawling.
    const withoutOldest = groups.slice(start + 1).reduce((sum, g) => sum + g.weight, 0)
    expect(withoutOldest).toBeLessThanOrEqual(LIVE_TAIL_PARTS)
  })

  it('virtualizes the old bulk of a long agent transcript', () => {
    // The regression this guards: heavy tool turns. A turn-count tail (6) left
    // NOTHING virtualized on transcripts like this, so every Radix overlay open
    // paid a whole-document style recalc.
    const groups = Array.from({ length: 40 }, (_, i) => group(`g${i}`, 120))

    // Only the min-group floor stays rendered; the other 38 turns skip.
    expect(liveTailStart(groups)).toBe(groups.length - LIVE_TAIL_MIN_GROUPS)
  })

  it('never virtualizes below the min-group floor, however heavy the turns', () => {
    const groups = Array.from({ length: 5 }, (_, i) => group(`g${i}`, 10_000))

    expect(liveTailStart(groups)).toBe(groups.length - LIVE_TAIL_MIN_GROUPS)
  })

  it('keeps every turn rendered when the whole transcript fits in the tail', () => {
    const groups = [group('a', 5), group('b', 5), group('c', 5)]

    expect(liveTailStart(groups)).toBe(0)
  })

  it('handles an empty transcript', () => {
    expect(liveTailStart([])).toBe(0)
  })

  it('honors a custom budget', () => {
    const groups = Array.from({ length: 10 }, (_, i) => group(`g${i}`, 1))

    // A 3-part budget would keep 4 turns, but the max-groups ceiling is not hit
    // here, so the parts budget wins.
    expect(liveTailStart(groups, 3)).toBe(6)
  })

  it('never renders more than the old turn-count tail did, on any shape', () => {
    // Guards the one way a parts budget can regress: a long transcript of tiny
    // turns, where walking back 40 parts reaches further than 6 turns would.
    const shapes = [
      Array.from({ length: 40 }, () => 4), // long chat, tiny turns
      Array.from({ length: 40 }, () => 1), // pathological: 1-part turns
      Array.from({ length: 12 }, () => 6),
      [80, 120, 60, 150, 90, 200, 70], // real agent tile
      [30, 45]
    ]

    for (const weights of shapes) {
      const groups = weights.map((weight, i) => group(`g${i}`, weight))
      const rendered = (start: number) => weights.slice(start).reduce((a, b) => a + b, 0)

      const oldStart = Math.max(0, groups.length - 6)

      expect(rendered(liveTailStart(groups))).toBeLessThanOrEqual(rendered(oldStart))
    }
  })
})

const receiptCreatedAt = new Date('2026-08-11T00:00:00.000Z')

function receiptAssistant(text: string): ThreadMessage {
  return {
    id: 'assistant-1',
    role: 'assistant',
    content: [{ type: 'text', text }],
    status: { type: 'complete', reason: 'stop' },
    createdAt: receiptCreatedAt,
    metadata: {
      unstable_state: null,
      unstable_annotations: [],
      unstable_data: [],
      steps: [],
      custom: {}
    }
  } as ThreadMessage
}

function receiptUser(text: string): ThreadMessage {
  return {
    id: 'user-1',
    role: 'user',
    content: [{ type: 'text', text }],
    attachments: [],
    createdAt: receiptCreatedAt,
    metadata: { custom: {} }
  } as ThreadMessage
}

function receiptHeadText(receipt: ThreadCommitReceipt): string {
  return (
    receipt.headMessage?.content
      .filter((part): part is { type: 'text'; text: string } => part.type === 'text')
      .map(part => part.text)
      .join('') ?? ''
  )
}

function LeafReceiptHarness({
  isRunning = false,
  message,
  messages,
  onCommitReceipt,
  revision
}: {
  isRunning?: boolean
  message?: ThreadMessage
  messages?: readonly ThreadMessage[]
  onCommitReceipt: (receipt: ThreadCommitReceipt) => void
  revision: number
}) {
  const runtime = useExternalStoreRuntime<ThreadMessage>({
    messages: messages ?? [message!],
    isRunning,
    onNew: async () => {}
  })

  return createElement(
    AssistantRuntimeProvider,
    { runtime },
    createElement(Thread, { onCommitReceipt, resumePublicationRevision: revision })
  )
}

function EmptyTranscriptRevealHarness({
  onCommitReceipt,
  visible
}: {
  onCommitReceipt: (receipt: ThreadCommitReceipt) => void
  visible: boolean
}) {
  const [revealReady, setRevealReady] = useState(visible)
  const runtime = useExternalStoreRuntime<ThreadMessage>({
    messages: [],
    isRunning: false,
    onNew: async () => {}
  })
  const handleCommitReceipt = useCallback(
    (receipt: ThreadCommitReceipt) => {
      onCommitReceipt(receipt)

      if (
        visible &&
        receipt.complete &&
        receipt.chainSignature === '' &&
        receipt.headMessage === null &&
        receipt.contentSignature === '[]'
      ) {
        setRevealReady(true)
      }
    },
    [onCommitReceipt, visible]
  )

  useLayoutEffect(() => {
    if (!visible) {
      setRevealReady(false)
    }
  }, [visible])

  return createElement(
    PaneVisibleContext.Provider,
    { value: visible },
    createElement(
      AssistantRuntimeProvider,
      { runtime },
      createElement(
        'div',
        { 'aria-hidden': !revealReady || undefined, 'data-testid': 'empty-chat-surface' },
        createElement(Thread, {
          onCommitReceipt: revealReady ? undefined : handleCommitReceipt,
          resumePublicationRevision: 17
        }),
        createElement('div', { 'data-testid': 'empty-chat-composer' })
      )
    )
  )
}

describe('Thread leaf commit receipt', () => {
  it('re-acknowledges an empty transcript after a hidden pane becomes visible', async () => {
    const receipts: ThreadCommitReceipt[] = []
    const onCommitReceipt = (receipt: ThreadCommitReceipt) => receipts.push(receipt)
    const { rerender } = render(
      createElement(EmptyTranscriptRevealHarness, {
        onCommitReceipt,
        visible: true
      })
    )
    const surface = screen.getByTestId('empty-chat-surface')

    expect(surface.getAttribute('aria-hidden')).toBeNull()
    expect(screen.getByTestId('empty-chat-composer')).toBeTruthy()

    rerender(
      createElement(EmptyTranscriptRevealHarness, {
        onCommitReceipt,
        visible: false
      })
    )
    await waitFor(() => expect(surface.getAttribute('aria-hidden')).toBe('true'))
    expect(receipts).toHaveLength(0)

    rerender(
      createElement(EmptyTranscriptRevealHarness, {
        onCommitReceipt,
        visible: true
      })
    )
    await waitFor(() => expect(surface.getAttribute('aria-hidden')).toBeNull())
    expect(screen.getByTestId('empty-chat-composer')).toBeTruthy()
    expect(receipts).toContainEqual({
      revision: 17,
      chainSignature: '',
      headMessage: null,
      contentSignature: '[]',
      complete: true
    })
  })

  it('signs every committed message so stale interior content cannot match an unchanged head', () => {
    const first = receiptAssistant('First answer')
    const head = { ...receiptAssistant('Head answer'), id: 'assistant-head' } as ThreadMessage
    const staleFirst = { ...first, content: [{ type: 'text', text: 'Stale first answer' }] } as ThreadMessage

    expect(commitReceiptChain([first, head]).contentSignature).not.toBe(
      commitReceiptChain([staleFirst, head]).contentSignature
    )
  })

  it('requires Text and Reasoning commits while excluding non-authoritative parts', () => {
    const reasoning = { type: 'reasoning', text: 'Thinking', status: { type: 'running' } }

    expect(partRequiresCommit({ type: 'text', text: 'Answer' })).toBe(true)
    expect(partRequiresCommit(reasoning)).toBe(true)
    expect(partRequiresCommit({ ...reasoning, text: '' })).toBe(true)
    expect(partRequiresCommit({ ...reasoning, status: { type: 'complete' } })).toBe(true)
    expect(partRequiresCommit({ type: 'tool-call' })).toBe(false)
  })

  it('prunes committed object graphs outside the current rendered chain', () => {
    const retained = new Map([
      ['old-session-message', { payload: 'large old graph' }],
      ['current-message', { payload: 'current graph' }]
    ])

    pruneCommitMap(retained, [{ id: 'current-message' }])

    expect([...retained.keys()]).toEqual(['current-message'])
  })

  it('publishes a matching receipt only after the matching message-part DOM has committed', async () => {
    const interim = receiptAssistant('Interim answer')
    const settled = receiptAssistant('Settled answer')
    const observations: { receipt: ThreadCommitReceipt; settledDomPresent: boolean }[] = []
    const onCommitReceipt = (receipt: ThreadCommitReceipt) => {
      observations.push({ receipt, settledDomPresent: screen.queryByText('Settled answer') !== null })
    }
    const { rerender } = render(createElement(LeafReceiptHarness, { message: interim, onCommitReceipt, revision: 1 }))

    await screen.findByText('Interim answer')
    rerender(createElement(LeafReceiptHarness, { message: settled, onCommitReceipt, revision: 7 }))

    await screen.findByText('Settled answer')
    await waitFor(() => {
      expect(
        observations.map(observation => ({
          revision: observation.receipt.revision,
          headText: receiptHeadText(observation.receipt),
          domPresent: observation.settledDomPresent,
          complete: observation.receipt.complete
        }))
      ).toContainEqual({ revision: 7, headText: 'Settled answer', domPresent: true, complete: true })
    })
  })

  it('does not wait for a separately rendered tool leaf before publishing structured text', async () => {
    const structured = {
      ...receiptAssistant('Structured settled answer'),
      content: [
        {
          type: 'tool-call',
          toolCallId: 'todo-1',
          toolName: 'todo',
          args: { todos: [] },
          argsText: '{"todos":[]}',
          result: { todos: [] }
        },
        { type: 'text', text: 'Structured settled answer' }
      ]
    } as ThreadMessage
    const observations: { receipt: ThreadCommitReceipt; settledDomPresent: boolean }[] = []
    const onCommitReceipt = (receipt: ThreadCommitReceipt) => {
      observations.push({ receipt, settledDomPresent: screen.queryByText('Structured settled answer') !== null })
    }

    render(
      createElement(LeafReceiptHarness, {
        messages: [receiptUser('Structured prompt'), structured],
        onCommitReceipt,
        revision: 8
      })
    )

    await screen.findByText('Structured settled answer')
    await waitFor(() => {
      expect(
        observations.map(observation => ({
          revision: observation.receipt.revision,
          headText: receiptHeadText(observation.receipt),
          domPresent: observation.settledDomPresent,
          complete: observation.receipt.complete
        }))
      ).toContainEqual({
        revision: 8,
        headText: 'Structured settled answer',
        domPresent: true,
        complete: true
      })
    })
  })

  it('acknowledges collapsed completed reasoning that has no mounted DOM leaf', async () => {
    const structured = {
      ...receiptAssistant('Answer after thinking'),
      content: [
        { type: 'reasoning', text: 'Completed hidden reasoning' },
        { type: 'text', text: 'Answer after thinking' }
      ]
    } as ThreadMessage
    const receipts: ThreadCommitReceipt[] = []

    render(
      createElement(LeafReceiptHarness, {
        message: structured,
        onCommitReceipt: receipt => receipts.push(receipt),
        revision: 9
      })
    )

    await screen.findByText('Answer after thinking')
    await waitFor(() => expect(receipts.some(receipt => receipt.revision === 9 && receipt.complete)).toBe(true))
  })

  it('publishes a user-open completed reasoning receipt only after the updated projected markdown DOM commits', async () => {
    const completedReasoning = (reasoning: string): ThreadMessage =>
      ({
        ...receiptAssistant('Live answer'),
        content: [
          { type: 'reasoning', text: reasoning },
          { type: 'text', text: 'Live answer' }
        ]
      }) as ThreadMessage
    const observations: { revision: number; settledDomPresent: boolean }[] = []
    const onCommitReceipt = (receipt: ThreadCommitReceipt) => {
      observations.push({
        revision: receipt.revision,
        settledDomPresent: screen.queryByText('Settled reasoning') !== null
      })
    }
    const { rerender } = render(
      createElement(LeafReceiptHarness, {
        message: completedReasoning('  Interim reasoning'),
        onCommitReceipt,
        revision: 10
      })
    )

    fireEvent.click(screen.getByRole('button', { name: /Thought/ }))
    await screen.findByText('Interim reasoning')
    rerender(
      createElement(LeafReceiptHarness, {
        message: completedReasoning('\n  Settled reasoning'),
        onCommitReceipt,
        revision: 11
      })
    )

    await screen.findByText('Settled reasoning')
    await waitFor(() => {
      const matching = observations.filter(observation => observation.revision === 11)

      expect(matching.length).toBeGreaterThan(0)
      expect(matching.every(observation => observation.settledDomPresent)).toBe(true)
      expect(matching[0]).toEqual({ revision: 11, settledDomPresent: true })
    })
  })
})
