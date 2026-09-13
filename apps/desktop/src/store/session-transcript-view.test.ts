import { atom } from 'nanostores'
import { afterEach, expect, it } from 'vitest'

import type { ChatMessage } from '@/lib/chat-messages'
import { createClientSessionState } from '@/lib/chat-runtime'

import { clearClarifyRequest, setClarifyRequest } from './clarify'
import { clearAllSessionStates, dropSessionState, publishSessionState, releaseSessionTranscript } from './session-states'
import {
  $sessionTranscriptViewGates, clearTranscriptViewGates, holdTranscriptView, transcriptMessagesForView
} from './session-transcript-view'

const question = (id: string, requestId = 'R') => ({
  sessionId: id, requestId, question: 'Continue?', choices: ['Yes'], multiSelect: false
})

const raw: ChatMessage[] = [{ id: 'raw', role: 'assistant', parts: [{ type: 'text', text: 'UNVERIFIED' }] }]
afterEach(() => { clearAllSessionStates(); clearClarifyRequest() })

it('isolates attempt tokens and cache owners, including stale release after cleanup and rebind', () => {
  const owner = Symbol('first cache')
  const nextOwner = Symbol('second cache')
  const stale = holdTranscriptView('A', owner)
  const current = holdTranscriptView('A', nextOwner)
  holdTranscriptView('B', owner)
  stale()
  clearTranscriptViewGates(owner)
  expect(Object.keys($sessionTranscriptViewGates.get())).toEqual(['A'])
  current()
  expect($sessionTranscriptViewGates.get()).toEqual({})
  const rebound = holdTranscriptView('A', nextOwner)
  stale()
  current()
  expect($sessionTranscriptViewGates.get().A).toBeDefined()
  rebound()
})

it('projects only the selected runtime and preserves reference identity through unrelated updates', () => {
  const runtime = atom<string | null>('A')
  const messages = atom(raw)
  const view = transcriptMessagesForView(runtime, messages)
  const stop = view.listen(() => {})

  try {
    holdTranscriptView('A')
    setClarifyRequest(question('A'))
    const projected = view.get()
    expect(projected[0].id).toBe('pending-clarify:A:R')
    setClarifyRequest(question('B'))
    holdTranscriptView('B')
    messages.set([...raw])
    expect(view.get()).toBe(projected)
    setClarifyRequest(question('A', 'new'))
    expect(view.get()[0].id).toBe('pending-clarify:A:new')
    clearClarifyRequest('R', 'A')
    expect(view.get()[0].id).toBe('pending-clarify:A:new')
    clearClarifyRequest('new', 'A')
    expect(view.get()).toEqual([])
    runtime.set('B')
    expect(view.get()[0].id).toBe('pending-clarify:B:R')
    runtime.set(null)
    expect(view.get()).toBe(messages.get())
  } finally { stop() }
})

it.each(['drop', 'release', 'clear'] as const)('cleans gates when sessions %s without retaining a registry entry', operation => {
  publishSessionState('A', { ...createClientSessionState('stored-A'), messages: raw })
  const stale = holdTranscriptView('A')

  if (operation === 'drop') {dropSessionState('A')}

  if (operation === 'release') {releaseSessionTranscript('A')}

  if (operation === 'clear') {clearAllSessionStates()}
  expect($sessionTranscriptViewGates.get()).toEqual({})
  const next = holdTranscriptView('A')
  stale()
  expect($sessionTranscriptViewGates.get().A).toBeDefined()
  next()
})

it('retires an old binding gate after the replacement state is published', () => {
  const state = { ...createClientSessionState('old'), messages: raw, needsInput: true }
  publishSessionState('A', state)
  const release = holdTranscriptView('A')
  publishSessionState('A', { ...state, storedSessionId: 'new' })
  expect($sessionTranscriptViewGates.get()).toEqual({})
  holdTranscriptView('A')
  release()
  expect($sessionTranscriptViewGates.get().A).toBeDefined()
})
