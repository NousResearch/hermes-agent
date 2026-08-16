import { AssistantRuntimeProvider, type ThreadMessage, useExternalStoreRuntime } from '@assistant-ui/react'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { createElement, useCallback, useLayoutEffect, useState } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { PaneVisibleContext } from '@/components/pane-shell/pane-visibility'

import { Thread } from '.'
import { MESSAGE_PARTS_COMPONENTS } from './message-parts'

import {
  buildGroups,
  commitReceiptChain,
  firstVisibleGroupIndex,
  HIDDEN_TRANSCRIPT_RENDER_BUDGET,
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
  subscribeToThreadForeground,
  transcriptPaneBudget,
  type ThreadCommitReceipt,
  useThreadMessagePartRangeCommitCallback
} from './list'

const mutableToolComponents = MESSAGE_PARTS_COMPONENTS.tools as {
  Fallback: typeof MESSAGE_PARTS_COMPONENTS.tools.Fallback
}
const originalToolFallback = mutableToolComponents.Fallback

type ResizeObserverRecord = {
  callback: ResizeObserverCallback
  element: Element
  observer: TestResizeObserver
}

let resizeObserverRecords: ResizeObserverRecord[] = []

class TestResizeObserver {
  private readonly record: ResizeObserverRecord

  constructor(callback: ResizeObserverCallback) {
    this.record = { callback, element: document.body, observer: this }
  }

  observe(element: Element) {
    this.record.element = element
    resizeObserverRecords = resizeObserverRecords.filter(record => record !== this.record)
    resizeObserverRecords.push(this.record)
  }

  unobserve() {
    resizeObserverRecords = resizeObserverRecords.filter(record => record !== this.record)
  }

  disconnect() {
    resizeObserverRecords = resizeObserverRecords.filter(record => record !== this.record)
  }
}

/** Deliver a synthetic content resize to every observer watching `element`,
 *  the way the real ResizeObserver would after a layout change. The delivery
 *  stays inside act: the library's resize callback updates React state. */
function fireContentResize(element: Element, height: number) {
  act(() => {
    const entry = { contentRect: { height } } as ResizeObserverEntry

    for (const record of [...resizeObserverRecords]) {
      if (record.element === element) {
        record.callback([entry], record.observer)
      }
    }
  })
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

afterEach(() => {
  cleanup()
  mutableToolComponents.Fallback = originalToolFallback
  resizeObserverRecords = []
  vi.restoreAllMocks()
})

describe('subscribeToThreadForeground', () => {
  it('reanchors on focus when an active turn keeps document visibility pinned visible', () => {
    const reanchor = vi.fn()

    const raf = vi.spyOn(window, 'requestAnimationFrame').mockImplementation(callback => {
      callback(0)

      return 1
    })

    const unsubscribe = subscribeToThreadForeground(() => true, reanchor)

    window.dispatchEvent(new Event('focus'))

    expect(raf).toHaveBeenCalledOnce()
    expect(reanchor).toHaveBeenCalledOnce()
    unsubscribe()
  })

  it('leaves a scrolled-up reader in place when the window focuses', () => {
    const reanchor = vi.fn()
    const raf = vi.spyOn(window, 'requestAnimationFrame')
    const unsubscribe = subscribeToThreadForeground(() => false, reanchor)

    window.dispatchEvent(new Event('focus'))

    expect(raf).not.toHaveBeenCalled()
    expect(reanchor).not.toHaveBeenCalled()
    unsubscribe()
  })

  it('drops a queued reanchor when the reader scrolls away before the frame', () => {
    const frames: FrameRequestCallback[] = []
    let following = true
    const reanchor = vi.fn()

    vi.spyOn(window, 'requestAnimationFrame').mockImplementation(callback => {
      frames.push(callback)

      return 7
    })

    const unsubscribe = subscribeToThreadForeground(() => following, reanchor)

    window.dispatchEvent(new Event('focus'))
    following = false
    frames[0]?.(0)

    expect(reanchor).not.toHaveBeenCalled()
    unsubscribe()
  })

  it('cancels a queued reanchor when its thread unmounts', () => {
    const cancel = vi.spyOn(window, 'cancelAnimationFrame')
    const reanchor = vi.fn()

    vi.spyOn(window, 'requestAnimationFrame').mockReturnValue(9)

    const unsubscribe = subscribeToThreadForeground(() => true, reanchor)

    window.dispatchEvent(new Event('focus'))
    unsubscribe()

    expect(cancel).toHaveBeenCalledWith(9)
    expect(reanchor).not.toHaveBeenCalled()
  })
})

// Signature rows are `${index}:${id}:${role}:${weight}` (see the useAuiState
// selector in list.tsx).
const signature = (rows: [string, string, number][]) =>
  rows.map(([id, role, weight], index) => `${index}:${id}:${role}:${weight}`).join('\n')

describe('transcriptPaneBudget', () => {
  it('uses a fixed live-tail budget while hidden instead of charging every mounted transcript', () => {
    expect(transcriptPaneBudget(1, true)).toBe(HIDDEN_TRANSCRIPT_RENDER_BUDGET)
    expect(transcriptPaneBudget(4, true)).toBe(HIDDEN_TRANSCRIPT_RENDER_BUDGET)
    expect(transcriptPaneBudget(1, false)).toBeGreaterThan(HIDDEN_TRANSCRIPT_RENDER_BUDGET)
  })
})

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
      publicationIdentity: 'standalone',
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

  it('requires Text, Reasoning, and Tool commits', () => {
    const reasoning = { type: 'reasoning', text: 'Thinking', status: { type: 'running' } }

    expect(partRequiresCommit({ type: 'text', text: 'Answer' })).toBe(true)
    expect(partRequiresCommit(reasoning)).toBe(true)
    expect(partRequiresCommit({ ...reasoning, text: '' })).toBe(true)
    expect(partRequiresCommit({ ...reasoning, status: { type: 'complete' } })).toBe(true)
    expect(partRequiresCommit({ type: 'tool-call' })).toBe(true)
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

  it('does not complete an ordinary Tool until that Tool component acknowledges its committed DOM', async () => {
    mutableToolComponents.Fallback = function ControlledToolCommit() {
      const acknowledgeTool = useThreadMessagePartRangeCommitCallback(0, 0)

      return createElement('button', { onClick: () => acknowledgeTool?.(), type: 'button' }, 'Acknowledge Tool DOM')
    }

    const structured = {
      ...receiptAssistant('Answer after tool'),
      content: [
        {
          type: 'tool-call',
          toolCallId: 'terminal-1',
          toolName: 'terminal',
          args: { command: 'pwd' },
          argsText: '{"command":"pwd"}',
          result: { output: '/work' }
        },
        { type: 'text', text: 'Answer after tool' }
      ]
    } as ThreadMessage
    const receipts: ThreadCommitReceipt[] = []

    render(
      createElement(LeafReceiptHarness, {
        messages: [receiptUser('Run a command'), structured],
        onCommitReceipt: receipt => receipts.push(receipt),
        revision: 8
      })
    )

    await screen.findByText('Answer after tool')
    const acknowledgeButton = await screen.findByRole('button', { name: 'Acknowledge Tool DOM' })
    expect(receipts.some(receipt => receipt.revision === 8 && receipt.complete)).toBe(false)

    fireEvent.click(acknowledgeButton)
    await waitFor(() => expect(receipts.some(receipt => receipt.revision === 8 && receipt.complete)).toBe(true))
  })

  it.each([
    {
      label: 'ordinary',
      summary: 'Explored 2 files',
      tools: [
        {
          type: 'tool-call',
          toolCallId: 'read-file-collapsed-1',
          toolName: 'read_file',
          args: { path: 'alpha.ts' },
          argsText: '{"path":"alpha.ts"}',
          result: { content: 'alpha' }
        },
        {
          type: 'tool-call',
          toolCallId: 'search-files-collapsed-1',
          toolName: 'search_files',
          args: { pattern: 'alpha' },
          argsText: '{"pattern":"alpha"}',
          result: { matches: [] }
        }
      ]
    },
    {
      label: 'ordinary and deliberately silent',
      summary: 'Ran 1 command, used 1 tool',
      tools: [
        {
          type: 'tool-call',
          toolCallId: 'terminal-collapsed-1',
          toolName: 'terminal',
          args: { command: 'pwd' },
          argsText: '{"command":"pwd"}',
          result: { output: '/work' }
        },
        {
          type: 'tool-call',
          toolCallId: 'todo-collapsed-1',
          toolName: 'todo',
          args: { todos: [] },
          argsText: '{"todos":[]}',
          result: { todos: [] }
        }
      ]
    }
  ])('acknowledges a collapsed $label Tool run only after its summary DOM commits', async ({ summary, tools }) => {
    const structured = {
      ...receiptAssistant('Answer after collapsed tools'),
      content: [...tools, { type: 'text', text: 'Answer after collapsed tools' }]
    } as ThreadMessage
    const observations: {
      complete: boolean
      summaryDomPresent: boolean
      toolGroupDomPresent: boolean
    }[] = []
    const onCommitReceipt = (receipt: ThreadCommitReceipt) => {
      if (receipt.revision === 12) {
        observations.push({
          complete: receipt.complete,
          summaryDomPresent: screen.queryByText(summary) !== null,
          toolGroupDomPresent: document.querySelector('[data-tool-group]') !== null
        })
      }
    }

    render(
      createElement(LeafReceiptHarness, {
        messages: [receiptUser('Run collapsed tools'), structured],
        onCommitReceipt,
        revision: 12
      })
    )

    await screen.findByText(summary)
    expect(document.querySelector('[data-tool-group]')).toBeTruthy()
    expect(document.querySelector('[data-tool-row]')).toBeNull()
    await waitFor(() => {
      expect(observations).toContainEqual({
        complete: true,
        summaryDomPresent: true,
        toolGroupDomPresent: true
      })
    })
    expect(
      observations
        .filter(observation => observation.complete)
        .every(observation => observation.summaryDomPresent && observation.toolGroupDomPresent)
    ).toBe(true)
  })

  it.each(['todo', 'react_to_message'])(
    'acknowledges a deliberately null %s Tool from ChainToolFallback',
    async toolName => {
      const structured = {
        ...receiptAssistant('Structured settled answer'),
        content: [
          {
            type: 'tool-call',
            toolCallId: `${toolName}-1`,
            toolName,
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
    }
  )

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

describe('thread viewport shrink re-anchor', () => {
  // The packaged GUI failure: after the reveal settled at the bottom
  // (scrollHeight 7082 / clientHeight 730), a delayed content-visibility
  // re-measurement collapsed content ABOVE the viewport. The browser left
  // scrollTop ~32.67px short of the new true bottom (3659.33 vs 3691) while
  // the surface still reported following=true — and the rAF-based correction
  // painted exactly one visible frame at the gap before recovering.
  // use-stick-to-bottom's negative-resize branch only re-asserts the lock; it
  // trusts the browser's scrollTop clamp, so nothing else closes the gap.
  // The owner (this list) must re-anchor a strictly-following viewport to the
  // true bottom SYNCHRONOUSLY in the same ResizeObserver delivery that
  // observes the collapse — no rAF hop — and must leave an escaped reader
  // untouched. The composed content ref must also survive ordinary rerenders
  // (a streamed update re-renders this list) without detaching its observers
  // or resetting the height baseline.
  const VIEWPORT = 'aui_thread-viewport'
  const CONTENT = 'aui_thread-content'
  const SCROLL_HEIGHT = 7082
  const SHRUNK_SCROLL_HEIGHT = 4422
  const CLIENT_HEIGHT = 730
  const SHRUNK_BOTTOM = SHRUNK_SCROLL_HEIGHT - CLIENT_HEIGHT - 1

  function setScrollGeometry(viewport: HTMLElement, scrollHeight: number) {
    Object.defineProperty(viewport, 'scrollHeight', { value: scrollHeight, configurable: true, writable: true })
    Object.defineProperty(viewport, 'clientHeight', { value: CLIENT_HEIGHT, configurable: true, writable: true })
  }

  async function renderSettledThread(onCommitReceipt: (receipt: ThreadCommitReceipt) => void = vi.fn()) {
    const { rerender } = render(
      createElement(LeafReceiptHarness, {
        message: receiptAssistant('Settled answer'),
        // Revision 0 (a fresh session, not a resumed authority publication)
        // keeps the resume-anchor restore disarmed so its 250ms
        // MutationObserver cannot race the test's geometry install.
        onCommitReceipt,
        revision: 0
      })
    )

    const viewport = document.querySelector(`[data-slot="${VIEWPORT}"]`) as HTMLElement
    const content = document.querySelector(`[data-slot="${CONTENT}"]`) as HTMLElement

    expect(viewport).toBeTruthy()
    expect(content).toBeTruthy()

    // Track scrollTop writes so the test can wait the mount backfill ramp OUT
    // instead of racing it. Every ramp step commits and re-pins scrollTop
    // through the anchor/restore effect, and a stale anchor applied after the
    // test's geometry change would move an escaped reader. jsdom's scrollTop
    // is a plain data property; a closure-backed accessor preserves it.
    let backing = viewport.scrollTop as number
    let writeCount = 0
    let lastWriteAt = 0
    Object.defineProperty(viewport, 'scrollTop', {
      configurable: true,
      get: () => backing,
      set(value: number) {
        writeCount += 1
        lastWriteAt = Date.now()
        backing = value
      }
    })

    // Wait for the runtime to actually publish the message (assistant-ui
    // fills the thread asynchronously) so the mount settle loop has run
    // against the real content. The waits stay inside act — every timer/rAF
    // in the settle chain fires React updates.
    await screen.findByText('Settled answer')
    await act(async () => {
      await new Promise(resolve => setTimeout(resolve, 1000))
    })
    expect(viewport.dataset.following).toBe('true')

    // Wait until the render-budget backfill ramp has provably finished: it
    // must have written at least once, then gone quiet for a window several
    // times the inter-step gap (slow jsdom transition renders spread the
    // 20→600 steps ~150ms apart). Only then is no anchor/restore re-pin
    // pending that could clobber a later geometry change.
    await waitFor(() => expect(writeCount).toBeGreaterThan(0), { timeout: 5000 })
    await waitFor(() => expect(Date.now() - lastWriteAt).toBeGreaterThan(400), { timeout: 15000, interval: 100 })

    setScrollGeometry(viewport, SCROLL_HEIGHT)
    viewport.scrollTop = SCROLL_HEIGHT - CLIENT_HEIGHT

    return { content, rerender, viewport }
  }

  it('corrects a following viewport to the true bottom synchronously when content height collapses after reveal', async () => {
    const { content, viewport } = await renderSettledThread()

    fireContentResize(content, SCROLL_HEIGHT)
    // Let the library's own baseline-resize tick run against the settled
    // geometry (it writes nothing at the bottom) before the collapse lands.
    await act(async () => {
      await new Promise(resolve => setTimeout(resolve, 10))
    })

    // The delayed remeasurement collapses content above the viewport; the
    // position lands ~32.67px short of the new true bottom while still
    // following (packaged GUI: scrollTop 3659.33 vs 3691).
    setScrollGeometry(viewport, SHRUNK_SCROLL_HEIGHT)
    viewport.scrollTop = SHRUNK_SCROLL_HEIGHT - CLIENT_HEIGHT - 32.67
    fireContentResize(content, SHRUNK_SCROLL_HEIGHT)

    // No waitFor, no timer, no rAF: the ResizeObserver delivery that observes
    // the collapse must already have the true bottom painted. The library's
    // rAF scrollToBottom path is exactly one frame too late for this gate.
    expect(viewport.scrollTop).toBe(SHRUNK_BOTTOM)
    expect(viewport.dataset.following).toBe('true')
  })

  it('keeps the composed observer and its height baseline across an ordinary rerender', async () => {
    const onCommitReceipt = vi.fn()
    const { content, rerender, viewport } = await renderSettledThread(onCommitReceipt)

    fireContentResize(content, SCROLL_HEIGHT)
    await act(async () => {
      await new Promise(resolve => setTimeout(resolve, 10))
    })

    const contentObservers = () =>
      resizeObserverRecords.filter(record => record.element === content).map(record => record.observer)
    const before = contentObservers()

    // An ordinary same-session update (a streamed message landing) re-renders
    // this list. The composed content ref must keep the SAME observers — an
    // inline ref function is a new identity every render, so React detaches
    // (null) and reattaches (node) it, recreating both observers and
    // resetting the height baseline.
    rerender(
      createElement(LeafReceiptHarness, {
        message: receiptAssistant('Second settled answer'),
        onCommitReceipt,
        revision: 0
      })
    )
    await screen.findByText('Second settled answer')
    await act(async () => {
      await new Promise(resolve => setTimeout(resolve, 50))
    })

    expect(contentObservers()).toEqual(before)

    // The pre-rerender baseline must survive: this collapse still reads as a
    // shrink against it and corrects synchronously to the true bottom.
    setScrollGeometry(viewport, SHRUNK_SCROLL_HEIGHT)
    viewport.scrollTop = SHRUNK_SCROLL_HEIGHT - CLIENT_HEIGHT - 32.67
    fireContentResize(content, SHRUNK_SCROLL_HEIGHT)

    expect(viewport.scrollTop).toBe(SHRUNK_BOTTOM)
  })

  it('leaves an escaped reader untouched when content height collapses', async () => {
    const { content, viewport } = await renderSettledThread()

    fireContentResize(content, SCROLL_HEIGHT)
    await act(async () => {
      await new Promise(resolve => setTimeout(resolve, 10))
    })

    // Seed the library's last-scroll observation at the bottom, then scroll
    // up so the lock escapes (strict following turns off).
    viewport.scrollTop = SCROLL_HEIGHT - CLIENT_HEIGHT
    await act(async () => {
      viewport.dispatchEvent(new Event('scroll'))
      await new Promise(resolve => setTimeout(resolve, 5))
    })
    viewport.scrollTop = 3000
    await act(async () => {
      viewport.dispatchEvent(new Event('scroll'))
      await new Promise(resolve => setTimeout(resolve, 5))
    })
    await waitFor(() => expect(viewport.dataset.following).toBe('false'))

    // A collapse while reading earlier content must not move the position.
    setScrollGeometry(viewport, SHRUNK_SCROLL_HEIGHT)
    fireContentResize(content, SHRUNK_SCROLL_HEIGHT)
    await act(async () => {
      await new Promise(resolve => setTimeout(resolve, 10))
    })

    expect(viewport.scrollTop).toBe(3000)
  })
})
