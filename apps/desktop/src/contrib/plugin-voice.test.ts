import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { VoiceLiveSession } from '@/lib/voice-live'
import { $voiceTimeline } from '@/lib/voice-timeline'
import { $voice, type VoiceState } from '@/store/voice-playback'

import { createPluginContext } from './plugin'

const IDLE: VoiceState = { endReason: null, messageId: null, source: null, status: 'idle' }
const SPEAKING: VoiceState = { ...IDLE, messageId: 'm1', source: 'read-aloud', status: 'speaking' }

function plugin() {
  const disposers: Array<() => void> = []
  const { voice } = createPluginContext('p', dispose => disposers.push(dispose))

  return { unload: () => disposers.forEach(dispose => dispose()), voice }
}

beforeEach(() => {
  $voice.set(IDLE)
  $voiceTimeline.set({ segments: [] })
})
afterEach(() => void vi.restoreAllMocks())

it('offers reads only, and stops calling a plugin back once it unloads', () => {
  const { unload, voice } = plugin()
  const seen = vi.fn()

  expect(Object.keys(voice.playback).sort()).toEqual(['get', 'listen', 'subscribe'])
  voice.playback.listen(seen)
  $voice.set(SPEAKING)
  expect(seen.mock.lastCall?.[0]).toEqual(SPEAKING)
  expect(voice.playback.get()).not.toHaveProperty('audioElement')

  unload()
  $voice.set(IDLE)
  expect(seen).toHaveBeenCalledTimes(1)
})

it('a throwing listener reaches neither playback nor other plugins', () => {
  vi.spyOn(console, 'warn').mockImplementation(() => undefined)
  const seen = vi.fn()

  plugin().voice.playback.listen(() => {
    throw new Error('plugin bug')
  })
  plugin().voice.playback.listen(seen)

  expect(() => $voice.set(SPEAKING)).not.toThrow()
  expect(seen).toHaveBeenCalledOnce()
})

it('a GPT-Live session shows its fragments, and closing it never ends a spoken reply', () => {
  vi.spyOn(performance, 'now').mockReturnValue(50_000)
  const session = new VoiceLiveSession({ onClosed: () => {}, onDelegation: () => {}, onError: () => {} })
  const server = session as unknown as { finish(reason: string, s: null): void; handleEvent(raw: string): void }
  const emit = (event: object) => server.handleEvent(JSON.stringify(event))

  emit({ type: 'session.started' })
  emit({ delta: 'Hello', end_ms: 1200, start_ms: 1000, type: 'session.output_transcript.delta' })
  emit({ delta: 'user', end_ms: 1200, start_ms: 1000, type: 'session.input_transcript.delta' })
  expect(plugin().voice.timeline.get().segments.map(s => [s.text, s.startMs, s.endMs])).toEqual([['Hello', 51_000, 51_200]])

  $voice.set(SPEAKING)
  server.finish('expired', null)
  expect($voice.get()).toEqual(SPEAKING)
})
