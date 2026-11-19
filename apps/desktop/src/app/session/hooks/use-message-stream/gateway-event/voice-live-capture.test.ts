import type { GatewayEvent, VoiceLiveCapturePayload } from '@hermes/shared'
import { act, cleanup } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { getLatestSessionMessages } from '@/hermes'
import { createClientSessionState } from '@/lib/chat-runtime'
import { $notifications, clearNotifications } from '@/store/notifications'
import { _resetSessionOwnerHintsForTests, setSessionOwnerHint } from '@/store/session'
import { clearAllSessionStates, publishSessionState } from '@/store/session-states'

import { renderMessageStream } from '../test-harness'

vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal()),
  getLatestSessionMessages: vi.fn()
}))

const runtimeId = 'voice-runtime'
const storedId = 'voice-canonical'
const connectionId = 'voice-host'
const profile = 'voice-owner'

function capture(state: VoiceLiveCapturePayload['state'], envelope: Partial<GatewayEvent> = {}): GatewayEvent {
  return {
    type: 'voice.live.capture',
    session_id: runtimeId,
    connectionId,
    profile,
    payload: { call_id: 'call-1', state },
    ...envelope
  }
}

function renderVoiceSession(state = createClientSessionState(storedId)) {
  setSessionOwnerHint(storedId, { connectionId, profile, mode: 'remote' })

  return renderMessageStream('foreground-runtime', {
    states: new Map([[runtimeId, state]])
  })
}

afterEach(() => {
  cleanup()
  clearNotifications()
  clearAllSessionStates()
  _resetSessionOwnerHintsForTests()
  vi.resetAllMocks()
})

it.each(['updated', 'saved'] as const)('pulls the canonical transcript on %s for the background owner', async state => {
  vi.mocked(getLatestSessionMessages).mockResolvedValue({
    session_id: storedId,
    messages: [{ role: 'user', content: 'Words from the call', timestamp: 1 }]
  })
  const stream = renderVoiceSession()

  await act(() => stream.handleEvent(capture(state)))

  expect(stream.text(runtimeId)).toBe('Words from the call')
  expect(getLatestSessionMessages).toHaveBeenCalledWith(storedId, { connectionId, profile }, { passive: true })
  expect(stream.state().messages).toEqual([])
})

it('warns once for a fresh incomplete call, not for replay, a colliding owner, or a pending save', () => {
  const stream = renderVoiceSession()

  stream.handleEvent(capture('incomplete', { replayed: true }))
  stream.handleEvent(capture('incomplete', { connectionId: 'other-host' }))
  stream.handleEvent(capture('incomplete', { profile: 'other-profile' }))
  stream.handleEvent(capture('incomplete', { session_id: undefined }))
  stream.handleEvent(capture('recording'))
  stream.handleEvent(capture('pending'))
  expect($notifications.get()).toEqual([])
  expect(getLatestSessionMessages).not.toHaveBeenCalled()

  stream.handleEvent(capture('incomplete'))
  expect($notifications.get()).toMatchObject([
    { kind: 'warning', message: 'This live call transcript may be incomplete.' }
  ])

  clearNotifications()
  stream.handleEvent(capture('incomplete'))
  expect($notifications.get()).toEqual([])
})

it('rejects refreshes from another owner on a colliding session ID', () => {
  const stream = renderVoiceSession()
  stream.handleEvent(capture('updated', { connectionId: 'other-host' }))
  stream.handleEvent(capture('saved', { profile: 'other-profile' }))
  expect(getLatestSessionMessages).not.toHaveBeenCalled()
})

it('discards an in-flight capture read once a live turn takes ownership', async () => {
  const read = Promise.withResolvers<Awaited<ReturnType<typeof getLatestSessionMessages>>>()
  vi.mocked(getLatestSessionMessages).mockReturnValue(read.promise)
  const stream = renderVoiceSession()

  stream.handleEvent(capture('updated'))
  expect(getLatestSessionMessages).toHaveBeenCalledOnce()

  const live = { ...createClientSessionState(storedId), busy: true }
  publishSessionState(runtimeId, live)
  await act(() => read.resolve({
    session_id: storedId,
    messages: [{ role: 'user', content: 'Older capture snapshot', timestamp: 1 }]
  }))

  expect(stream.state(runtimeId).messages).toEqual([])
})

it('keeps an unacknowledged prompt and discards an older refresh superseded by a save', async () => {
  const older = Promise.withResolvers<Awaited<ReturnType<typeof getLatestSessionMessages>>>()
  const newer = Promise.withResolvers<Awaited<ReturnType<typeof getLatestSessionMessages>>>()
  vi.mocked(getLatestSessionMessages).mockReturnValueOnce(older.promise).mockReturnValueOnce(newer.promise)
  const prompt = { id: 'user-1758100000000-ab12cd', parts: [{ type: 'text' as const, text: 'Unsent prompt' }], role: 'user' as const }
  const stream = renderVoiceSession({ ...createClientSessionState(storedId), messages: [prompt] })

  stream.handleEvent(capture('updated'))
  stream.handleEvent(capture('saved'))
  await act(() => older.resolve({
    session_id: storedId,
    messages: [{ role: 'user', content: 'Earlier capture snapshot', timestamp: 1 }]
  }))
  expect(stream.state(runtimeId).messages).toEqual([prompt])

  await act(() => newer.resolve({
    session_id: storedId,
    messages: [{ role: 'user', content: 'Final capture snapshot', timestamp: 1 }]
  }))
  expect(stream.state(runtimeId).messages).toMatchObject([
    { parts: [{ type: 'text', text: 'Final capture snapshot' }] },
    prompt
  ])
})
