// Streaming into a long transcript must not re-render the settled history
// (#126486, regression of #69120). A streamed delta replaces only the tail
// ChatMessage; everything upstream of the thread keeps identity for settled
// turns, so the only components allowed to render per chunk are the tail's.
//
// This drives the PRODUCTION transcript pipeline — ChatMessage[] ->
// useRuntimeMessageRepository -> useIncrementalExternalStoreRuntime -> <Thread />
// — and counts renders per message id at two levels: the message root
// (`useMessageReactions`, called once per root render) and the text part
// (`useMessagePartText`, read by every rendered text part).
import type * as assistantUiModule from '@assistant-ui/react'
import { AssistantRuntimeProvider, type ThreadMessage } from '@assistant-ui/react'
import { act, cleanup, render } from '@testing-library/react'
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import { useRuntimeMessageRepository } from '@/app/chat/runtime-repository'
import type * as messageReactionsModule from '@/components/assistant-ui/thread/use-message-reactions'
import type { ChatMessage } from '@/lib/chat-messages'
import { useIncrementalExternalStoreRuntime } from '@/lib/incremental-external-store-runtime'

import { stubThreadEnvironment, stubThreadViewportSize } from '../test-utils'

import { Thread } from '.'

const rootRenders = new Map<string, number>()
const textRenders = new Map<string, number>()

const bump = (map: Map<string, number>, id: string) => map.set(id, (map.get(id) ?? 0) + 1)

vi.mock('@/components/assistant-ui/thread/use-message-reactions', async importActual => {
  const actual = await importActual<typeof messageReactionsModule>()

  return {
    ...actual,
    // Called once per render by each message root (assistant and user).
    useMessageReactions: (messageId: string, role: 'assistant' | 'user') => {
      bump(rootRenders, messageId)

      return actual.useMessageReactions(messageId, role)
    }
  }
})

vi.mock('@assistant-ui/react', async importActual => {
  const actual = await importActual<typeof assistantUiModule>()

  // Every text-part consumer (the part component and the markdown surface
  // under it) reads its text through this hook, so a call is a text-part render.
  function useMessagePartText() {
    const id = actual.useAuiState(s => s.message.id)
    bump(textRenders, id)

    return actual.useMessagePartText()
  }

  return { ...actual, useMessagePartText }
})

stubThreadEnvironment()
stubThreadViewportSize()

// Markdown surfaces get the code plugin (all of shiki) from a one-shot async
// import (`useCodePlugin`), and every surface mounted before it lands
// re-renders once to swap it in. Cold, that import can take longer than the
// settle loop below, which put that one-time re-render inside the chunk window
// (0.1 per settled row). Load the module before the test so the hook's own
// import resolves while rows are still settling.
beforeAll(async () => {
  await import('@streamdown/code')
})

beforeEach(() => {
  rootRenders.clear()
  textRenders.clear()
})

afterEach(() => {
  cleanup()
})

const text = (id: string, role: ChatMessage['role'], body: string, pending = false): ChatMessage => ({
  id,
  role,
  parts: [{ type: 'text', text: body }],
  ...(pending ? { pending: true } : {})
})

function Harness({ messages, busy }: { busy: boolean; messages: ChatMessage[] }) {
  const messageRepository = useRuntimeMessageRepository(messages)

  const runtime = useIncrementalExternalStoreRuntime<ThreadMessage>({
    messageRepository,
    isRunning: busy,
    onNew: async () => {}
  })

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <Thread />
    </AssistantRuntimeProvider>
  )
}

// PROBE_TURNS=400 turns this into the long-transcript measurement from #126486
// (per-chunk wall time is printed); the default keeps the contract test cheap.
const PROBE = process.env.PROBE_TURNS !== undefined
const HISTORY_TURNS = Number(process.env.PROBE_TURNS ?? 3)
const CHUNKS = 10

function history(): ChatMessage[] {
  const out: ChatMessage[] = []

  for (let turn = 0; turn < HISTORY_TURNS; turn += 1) {
    out.push(text(`u${turn}`, 'user', `question ${turn}`))
    out.push({
      id: `a${turn}`,
      role: 'assistant',
      parts: [
        { type: 'reasoning', text: `thinking ${turn}` },
        {
          type: 'tool-call',
          toolCallId: `call-${turn}`,
          toolName: 'terminal',
          args: { command: 'ls' },
          argsText: '{"command":"ls"}',
          result: 'ok'
        },
        { type: 'text', text: `settled answer ${turn}` }
      ] as ChatMessage['parts']
    })
    // A tool-only follow-up turn: coalesced into its neighbour by the pipeline.
    out.push({
      id: `t${turn}`,
      role: 'assistant',
      parts: [
        {
          type: 'tool-call',
          toolCallId: `call-${turn}-b`,
          toolName: 'terminal',
          args: { command: 'pwd' },
          argsText: '{"command":"pwd"}',
          result: '/'
        }
      ] as ChatMessage['parts']
    })
  }

  return out
}

function sum(map: Map<string, number>, ids: readonly string[]) {
  return ids.reduce((total, id) => total + (map.get(id) ?? 0), 0)
}

describe('streaming into a settled transcript', () => {
  it('does not re-render settled messages per streamed chunk', async () => {
    const settled = history()
    const settledIds = settled.map(m => m.id)
    const settledAssistantIds = settledIds.filter(id => id.startsWith('a'))
    const prompt = text('u-live', 'user', 'live question')

    let body = 'tok0'
    const stream = () => [...settled, prompt, text('a-live', 'assistant', body, true)]

    const { findByText, rerender } = render(<Harness busy messages={stream()} />)

    await findByText('tok0')

    // Let the render-budget backfill finish mounting older rows, and any async
    // one-shot work (the code plugin swap) re-render them, so the window below
    // measures per-chunk RE-renders only. Settle on the render COUNT, not the
    // mounted-row count: a one-time re-render of a mounted row adds no row.
    const renderTotal = () => sum(rootRenders, [...rootRenders.keys()]) + sum(textRenders, [...textRenders.keys()])

    for (let settle = 0, last = -1; settle < 50 && renderTotal() !== last; settle += 1) {
      last = renderTotal()
      await act(async () => {
        await new Promise(resolve => setTimeout(resolve, 50))
      })
    }

    const mountedBefore = new Set(rootRenders.keys())

    // Sanity: the counters see the settled rows at all.
    expect(sum(textRenders, settledAssistantIds)).toBeGreaterThan(0)

    const rootBefore = sum(rootRenders, settledIds)
    const textBefore = sum(textRenders, settledIds)
    const liveTextBefore = textRenders.get('a-live') ?? 0

    const t0 = performance.now()

    for (let chunk = 1; chunk <= CHUNKS; chunk += 1) {
      body = `${body} tok${chunk}`
      await act(async () => {
        rerender(<Harness busy messages={stream()} />)
      })
    }

    const elapsed = performance.now() - t0
    await findByText(body)

    if (PROBE) {
      console.info(
        `[probe] turns=${HISTORY_TURNS} ms/chunk=${(elapsed / CHUNKS).toFixed(2)} mountedBefore=${mountedBefore.size} mountedAfter=${rootRenders.size}`
      )
    }

    const settledRootPerChunk = (sum(rootRenders, settledIds) - rootBefore) / CHUNKS
    const settledTextPerChunk = (sum(textRenders, settledIds) - textBefore) / CHUNKS
    const liveTextPerChunk = ((textRenders.get('a-live') ?? 0) - liveTextBefore) / CHUNKS

    // The live tail must actually be streaming, or the zero below is vacuous.
    expect(liveTextPerChunk).toBeGreaterThanOrEqual(1)
    expect(settledRootPerChunk).toBe(0)
    expect(settledTextPerChunk).toBe(0)
  })
})
