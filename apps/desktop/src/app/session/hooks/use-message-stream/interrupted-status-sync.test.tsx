import type { GatewayEvent } from '@hermes/shared'
// #81114: a Stop/interrupt seals the assistant bubble, but background work
// keeps running and its completions still arrive. Those completions must
// retire the status-stack rows (subagent/delegate) that have no poll to
// cover them — otherwise spinners run forever and async results only appear
// after the next user message forces a re-hydrate.
import { act, cleanup } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { finalizeUserInterruptedMessages } from '@/app/session/hooks/use-prompt-actions/rewind'
import { $subagentsBySession } from '@/store/subagents'

import { type MessageStreamHarness, renderMessageStream } from './test-harness'

const SID = 'interrupted-status-sync-session'

let stream: MessageStreamHarness

function mountStream() {
  stream = renderMessageStream(SID)
}

const emit = (event: GatewayEvent) => act(() => stream.handleEvent(event))

// The state transform both Stop paths apply at the click.
const pressStop = () =>
  act(() => {
    const state = stream.state()
    stream.states.set(SID, {
      ...state,
      messages: finalizeUserInterruptedMessages(state.messages, state.streamId),
      busy: false,
      awaitingResponse: false,
      streamId: null,
      pendingBranchGroup: null,
      needsInput: false,
      interrupted: true,
      turnStartedAt: null,
      turnLive: false
    })
  })

const subagentsFor = (sid: string) => $subagentsBySession.get()[sid] ?? []

describe('interrupted turn still retires status rows (#81114)', () => {
  beforeEach(() => {
    $subagentsBySession.set({})
    mountStream()
  })

  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
  })

  it('a delegate_task tool.complete after Stop retires the fallback row', () => {
    emit({ payload: {}, session_id: SID, type: 'message.start' })
    emit({
      payload: { args: { goal: 'do it' }, name: 'delegate_task', tool_id: 't1' },
      session_id: SID,
      type: 'tool.start'
    })
    expect(subagentsFor(SID).length).toBe(1)

    pressStop()

    emit({
      payload: { name: 'delegate_task', result: { status: 'success', summary: 'done' }, tool_id: 't1' },
      session_id: SID,
      type: 'tool.complete'
    })

    expect(subagentsFor(SID)[0]?.status).toBe('completed')
  })

  it('late tool completions still do not grow the sealed bubble', () => {
    emit({ payload: {}, session_id: SID, type: 'message.start' })
    emit({
      payload: { args: { command: 'ls' }, name: 'terminal', tool_id: 'call-1' },
      session_id: SID,
      type: 'tool.start'
    })
    pressStop()
    const sealed = JSON.stringify(stream.state().messages)

    emit({
      payload: { name: 'terminal', result: 'file.txt', tool_id: 'call-1' },
      session_id: SID,
      type: 'tool.complete'
    })

    expect(JSON.stringify(stream.state().messages)).toBe(sealed)
  })
})
