import { atom } from 'nanostores'
import { afterEach, expect, it } from 'vitest'

import type { ChatMessage } from '@/lib/chat-messages'

import { clearClarifyRequest } from './clarify'
import { $connectionRequests, normalizeConnectionRequest, setConnectionRequest } from './connection-request'
import { clearAllSessionStates } from './session-states'
import { $sessionTranscriptViewGates, holdTranscriptView, transcriptMessagesForView } from './session-transcript-view'

afterEach(() => {
  clearAllSessionStates()
  clearClarifyRequest()
  $connectionRequests.set({})
})

it('keeps real deltas and tool results while withholding the unchanged cached prefix', () => {
  const baseline: ChatMessage[] = [
    {
      id: 'old',
      role: 'assistant',
      parts: [
        { type: 'text', text: 'old history' },
        { type: 'tool-call', toolName: 'terminal', toolCallId: 'tool', args: {}, argsText: '{}' }
      ]
    }
  ]

  const messages = atom(baseline)
  const view = transcriptMessagesForView(atom<string | null>('A'), messages)
  holdTranscriptView('A', Symbol('test'), baseline)
  expect(view.get()).toEqual([])
  messages.set([
    {
      ...baseline[0],
      parts: [
        { type: 'text', text: 'old history plus live delta' },
        { ...baseline[0].parts[1], result: 'live result' } as ChatMessage['parts'][number],
        { type: 'text', text: 'new response' }
      ]
    }
  ])
  expect(view.get().flatMap(message => message.parts)).toEqual([
    { type: 'text', text: ' plus live delta' },
    expect.objectContaining({ toolCallId: 'tool', result: 'live result' }),
    { type: 'text', text: 'new response' }
  ])
  expect(baseline[0].parts).toHaveLength(2)
})

it('an obsolete release cannot open a newer gate; session teardown clears it', () => {
  const owner = Symbol('test')
  const releaseOld = holdTranscriptView('A', owner, [])
  holdTranscriptView('A', owner, [])
  releaseOld()
  expect($sessionTranscriptViewGates.get()['A']).toBeDefined()
  clearAllSessionStates()
  expect($sessionTranscriptViewGates.get()).toEqual({})
})

it('preserves the upstream pending connection projection behind the history gate', () => {
  const messages = atom<ChatMessage[]>([])
  const view = transcriptMessagesForView(atom<string | null>('A'), messages)
  holdTranscriptView('A', Symbol('test'), [])
  setConnectionRequest(
    normalizeConnectionRequest(
      {
        op_id: 'op-A',
        tool_call_id: 'connection-tool',
        deadline_at: 123,
        timeout_seconds: 60,
        targets: [{ name: 'test-service', kind: 'mcp', action: 'install', state: 'pending' }]
      },
      'A'
    )!
  )
  expect(view.get().flatMap(message => message.parts)).toContainEqual(
    expect.objectContaining({ toolName: 'manage_connections', toolCallId: 'connection-tool' })
  )
  expect(messages.get()).toEqual([])
  $connectionRequests.set({})
  expect(view.get()).toEqual([])
})
