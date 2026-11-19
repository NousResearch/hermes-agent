// @vitest-environment jsdom
import { act, cleanup, render, screen } from '@testing-library/react'
import { afterEach, expect, it } from 'vitest'

import type { ChatMessage } from '@/lib/chat-messages'
import { createClientSessionState } from '@/lib/chat-runtime'
import { $sessionStates } from '@/store/session-states'
import {
  $voiceLiveDrafts,
  appendVoiceLiveDraft,
  markVoiceLiveDraftClaimed,
  newVoiceLiveDraftRunId
} from '@/store/voice-live-draft'

import { VoiceLiveDraftRows } from './voice-live-draft'

afterEach(() => {
  cleanup()
  $voiceLiveDrafts.set({})
  $sessionStates.set({})
})

const timed = (startMs: number, endMs: number, eventId?: string) => ({ endMs, eventId, startMs, timingMissing: false })

const row = (role: ChatMessage['role'], text: string, voiceLive?: ChatMessage['voiceLive']): ChatMessage => ({
  id: `${role}-${text}-${voiceLive?.startMs ?? 'typed'}`,
  parts: [{ text, type: 'text' }],
  role,
  ...(voiceLive ? { voiceLive } : {})
})

function chat(messages: ChatMessage[], voiceClaimSubmissionId: null | string = null) {
  $sessionStates.set({ runtime: { ...createClientSessionState(), messages, storedSessionId: 'stored', voiceClaimSubmissionId } })
}

const mount = () => render(<VoiceLiveDraftRows sessionId="runtime" sessionKey="stored" />)

it('replaces the live words with their saved row exactly once, also after switching back', () => {
  chat([])
  appendVoiceLiveDraft('stored', newVoiceLiveDraftRunId(), 'assistant', 'Sure, checking.', timed(2000, 2400))
  const view = mount()
  expect(screen.getByText('Sure, checking.')).toBeTruthy()

  // The row is saved while another chat is on screen.
  view.unmount()
  chat([row('assistant', 'Sure, checking.', { endMs: 2400, startMs: 2000 })])
  mount()
  expect(screen.queryByText('Sure, checking.')).toBeNull()
  expect($voiceLiveDrafts.get().stored).toBeUndefined()
})

it('keeps the live words through Hermes replies and the optimistic bubble, which are not its saved row', () => {
  chat([row('assistant', 'Okay.')])
  const reply = newVoiceLiveDraftRunId()
  appendVoiceLiveDraft('stored', reply, 'assistant', 'Okay.', timed(3000, 3200))
  const earlier = newVoiceLiveDraftRunId()
  appendVoiceLiveDraft('stored', earlier, 'user', 'Ask Hermes', timed(2600, 2800, 'ev-earlier'))
  const spoken = newVoiceLiveDraftRunId()
  appendVoiceLiveDraft('stored', spoken, 'user', 'Ask Hermes', timed(3400, 3800, 'ev-claimed'))
  markVoiceLiveDraftClaimed('stored', 'ev-claimed', 'voice-live:["call","delegation"]')
  mount()
  expect(screen.getByText('Okay.')).toBeTruthy()
  expect(screen.getAllByText('Ask Hermes')).toHaveLength(2)

  // A typed message with the same words is not this speech.
  act(() => chat([row('assistant', 'Okay.'), row('user', 'Ask Hermes')]))
  expect(screen.getAllByText('Ask Hermes')).toHaveLength(2)

  // This window's pending voice claim shows the same words: hide, never retire.
  act(() => chat([row('assistant', 'Okay.'), row('user', 'Ask Hermes')], 'voice-live:["call","delegation"]'))
  // Only the claimed run hides; the earlier identical words stay.
  expect(screen.getAllByText('Ask Hermes')).toHaveLength(1)
  expect($voiceLiveDrafts.get().stored?.map(run => run.id)).toEqual([reply, earlier, spoken])

  // The bubble is withdrawn before the row lands: the live words come back.
  act(() => chat([row('assistant', 'Okay.')]))
  expect(screen.getAllByText('Ask Hermes')).toHaveLength(2)
})

it('hides a claimed run only while its own submission is pending', () => {
  chat([])
  const refused = newVoiceLiveDraftRunId()
  appendVoiceLiveDraft('stored', refused, 'user', 'Yes', timed(0, 200, 'ev-1'))
  markVoiceLiveDraftClaimed('stored', 'ev-1', 'voice-live:["call","d1"]')
  const retried = newVoiceLiveDraftRunId()
  appendVoiceLiveDraft('stored', retried, 'user', 'Yes', timed(900, 1100, 'ev-2'))
  markVoiceLiveDraftClaimed('stored', 'ev-2', 'voice-live:["call","d2"]')
  mount()

  // The first claim was refused; the second is pending with its own bubble.
  act(() => chat([row('user', 'Yes')], 'voice-live:["call","d2"]'))
  expect(screen.getAllByText('Yes')).toHaveLength(1)
})

it('retires each live run from its own saved row, before any display join', () => {
  appendVoiceLiveDraft('stored', newVoiceLiveDraftRunId(), 'user', ' them', timed(71200, 71400, 'ev-3'))
  chat([
    row('user', ' That was one of', { endMs: 71000, firstEventId: 'ev-1', startMs: 70400 }),
    row('assistant', ' Okay,', { endMs: 71200, firstEventId: 'ev-2', startMs: 71000 }),
    row('user', ' them', { endMs: 71400, firstEventId: 'ev-3', startMs: 71200 })
  ])
  mount()
  expect($voiceLiveDrafts.get().stored).toBeUndefined()
})
