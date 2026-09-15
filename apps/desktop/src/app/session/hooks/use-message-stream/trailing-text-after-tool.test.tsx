import { act, cleanup } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { ClientSessionState } from '@/app/types'
import { chatMessageText } from '@/lib/chat-messages'
import { clearSessionTodos } from '@/store/todos'
import type { RpcEvent } from '@/types/hermes'

import { type MessageStreamHarness, renderMessageStream } from './test-harness'

// #105927: a turn ending with text AFTER a tool call
// (assistant(tool_calls) -> tool result -> assistant(text, stop)) must render
// the trailing text in the live view, not only after reload/hydration.
const SID = 'session-1'

let stream: MessageStreamHarness

function mountStream() {
  stream = renderMessageStream(SID)
}

function emit(type: RpcEvent['type'], payload: RpcEvent['payload'] = {}) {
  act(() => stream.handleEvent({ payload, session_id: SID, type }))
}

function getState(): ClientSessionState {
  return stream.state()
}

function assistantMessages(): string[] {
  return getState()
    .messages.filter(m => m.role === 'assistant' && !m.hidden)
    .map(m => chatMessageText(m))
}

function allLiveText(): string {
  return assistantMessages().join('\n')
}

const toolStart = () => emit('tool.start', { name: 'terminal', tool_id: 'tool-1', args: { command: 'ls' } })
const toolComplete = () =>
  emit('tool.complete', { name: 'terminal', tool_id: 'tool-1', args: { command: 'ls' }, result: 'file.txt' })

describe('trailing assistant text after a tool call (#105927)', () => {
  beforeEach(() => {
    clearSessionTodos(SID)
  })

  afterEach(() => {
    cleanup()
    clearSessionTodos(SID)
    vi.restoreAllMocks()
  })

  it('renders trailing text streamed as deltas after tool.complete', async () => {
    mountStream()
    await emit('message.start', {})

    toolStart()
    toolComplete()

    await emit('message.delta', { text: 'Here is the conclusion.' })
    await emit('message.complete', { text: 'Here is the conclusion.' })

    expect(allLiveText()).toContain('Here is the conclusion.')
  })

  it('renders trailing text that arrives only in the terminal frame (no deltas)', async () => {
    mountStream()
    await emit('message.start', {})

    toolStart()
    toolComplete()

    await emit('message.complete', { text: 'Here is the conclusion.' })

    expect(allLiveText()).toContain('Here is the conclusion.')
  })

  it('renders trailing text after an interim-sealed pre-tool narration', async () => {
    mountStream()
    await emit('message.start', {})

    await emit('message.delta', { text: 'Let me check.' })
    toolStart()
    toolComplete()
    await emit('message.interim', { text: 'Let me check.', already_streamed: true })

    await emit('message.delta', { text: 'Here is the conclusion.' })
    await emit('message.complete', { text: 'Here is the conclusion.' })

    expect(allLiveText()).toContain('Let me check.')
    expect(allLiveText()).toContain('Here is the conclusion.')
  })

  it('renders chunked trailing deltas after tool.complete', async () => {
    mountStream()
    await emit('message.start', {})

    toolStart()
    toolComplete()

    await emit('message.delta', { text: 'Here is ' })
    await emit('message.delta', { text: 'the conclusion.' })
    await emit('message.complete', { text: 'Here is the conclusion.' })

    expect(allLiveText()).toContain('Here is the conclusion.')
  })

  it('keeps streamed trailing text when the terminal frame restates only earlier narration (#105927)', async () => {
    mountStream()
    await emit('message.start', {})

    await emit('message.delta', { text: 'Let me check.' })
    toolStart()
    toolComplete()

    await emit('message.delta', { text: 'Here is the conclusion.' })
    // Terminal frame carries only the pre-tool narration while the streamed
    // tail already holds the conclusion: the live view must not delete the
    // newer trailing segment (reload rehydrates it from stored rows).
    await emit('message.complete', { text: 'Let me check.' })

    expect(allLiveText()).toContain('Here is the conclusion.')
  })

  it('replaces streamed text when the terminal frame rewrites it (authoritative final wins)', async () => {
    // Boundary of the #105927 fix: a terminal frame that says something NEW
    // (not restated on screen) still supersedes the streamed draft — e.g. a
    // rewrite after a retry. Only a terminal that restates on-screen text
    // keeps the richer streamed tail.
    mountStream()
    await emit('message.start', {})

    toolStart()
    toolComplete()

    await emit('message.delta', { text: 'draft conclusion' })
    await emit('message.complete', { text: 'Final rewritten conclusion.' })

    expect(allLiveText()).toContain('Final rewritten conclusion.')
  })
})
