import { expect, it } from 'vitest'

import { toRuntimeMessage } from '@/lib/chat-runtime'
import type { SessionMessage } from '@/types/hermes'

import { shapeVoiceLiveMessages } from './voice-live-display'

import { chatMessageText, toChatMessages } from '.'

// A saved GPT-Live row keeps its call timing, so the chat can replace the live
// words of exactly that speaker run (and nothing else) when the row lands.
it('keeps the call timing of a saved voice row', () => {
  const rows: SessionMessage[] = [
    { role: 'user', content: ' Ask\tHermes ', display_metadata: { voice_live: { partial: false, start_ms: 800, end_ms: 1700, first_event_id: 'ev-1', call_key: 'c1' } }, timestamp: 1 },
    { role: 'assistant', content: 'Untimed.', display_metadata: '{"voice_live": {"partial": true, "start_ms": null, "end_ms": null}}', timestamp: 2 },
    { role: 'user', content: 'Typed.', timestamp: 3 }
  ] as SessionMessage[]

  const [spoken, untimed, typed] = toChatMessages(rows)
  expect(spoken.voiceLive).toEqual({ callKey: 'c1', endMs: 1700, firstEventId: 'ev-1', startMs: 800 })
  expect(untimed.voiceLive).toEqual({ endMs: null, startMs: null })
  expect(typed.voiceLive).toBeUndefined()
})

const voiceRow = (role: 'assistant' | 'user', content: string, start: number, end: number, id: number, call = 'c1'): SessionMessage =>
  ({
    content,
    display_metadata: { voice_live: { call_key: call, end_ms: end, first_event_id: `ev-${id}`, partial: false, start_ms: start } },
    role,
    row_id: id,
    timestamp: id
  }) as SessionMessage

const reply = (content: string, id: number) => ({ content, role: 'assistant', row_id: id, timestamp: id }) as SessionMessage
const shaped = (rows: SessionMessage[]) => shapeVoiceLiveMessages(toChatMessages(rows))

it('keeps one saved row per message; joining is a render-time view', () => {
  const rows = [voiceRow('user', ' That was one of', 70400, 71000, 1), voiceRow('assistant', ' Okay,', 71000, 71200, 2), voiceRow('user', ' them', 71200, 71400, 3)]
  expect(toChatMessages(rows).map(m => m.rowId)).toEqual([1, 2, 3])
})

// From a real call: an interruption splits each speaker's sentence across rows.
it('joins each speaker across crosstalk within one call', () => {
  const messages = shaped([
    voiceRow('user', ' That was one of', 70400, 71000, 1),
    voiceRow('assistant', ' Okay,', 71000, 71200, 2),
    voiceRow('user', ' them', 71200, 71400, 3),
    voiceRow('assistant', ' good. I could not confirm it.', 71200, 74600, 4),
    // A real pause: a new exchange, not crosstalk.
    voiceRow('user', ' It is okay.', 78800, 80000, 5)
  ])

  expect(messages.map(m => [m.role, chatMessageText(m).trim()])).toEqual([
    ['user', 'That was one of them'],
    ['assistant', 'Okay, good. I could not confirm it.'],
    ['user', 'It is okay.']
  ])
  expect(messages[0].rowId).toBe(1)
  expect(messages[1].voiceLiveRuns?.map(run => run.firstEventId)).toEqual(['ev-2', 'ev-4'])
})

it('never joins an ordinary exchange, across a Hermes reply, or across calls', () => {
  const messages = shaped([
    voiceRow('user', ' Everything cool', 4200, 5000, 1),
    voiceRow('assistant', ' Yeah.', 5000, 6800, 2),
    voiceRow('user', ' Testing my patch.', 9400, 11600, 3),
    reply('Hermes wrote this.', 4),
    voiceRow('user', ' Next', 11800, 12000, 5),
    voiceRow('assistant', ' Sure.', 12000, 12400, 6),
    // A second call in the same chat: its clock restarts, so the gap looks negative.
    voiceRow('user', ' New call', 1000, 1400, 7, 'c2')
  ])

  expect(messages.map(m => chatMessageText(m).trim())).toEqual([
    'Everything cool', 'Yeah.', 'Testing my patch.', 'Hermes wrote this.', 'Next', 'Sure.', 'New call'
  ])
})

it('marks the voice reading of a Hermes reply as spoken, only when it repeats the reply', () => {
  const messages = shaped([
    voiceRow('user', ' Tell me the time', 16000, 17400, 1),
    reply('It is 1:24 a.m. Eastern.', 2),
    voiceRow('assistant', ' Yeah, I will check. It is 1:24 a.m. Eastern.', 18000, 24200, 3),
    voiceRow('user', ' Thanks', 27000, 27400, 4),
    reply('Done, the report is saved.', 5),
    // Original speech after a reply: not a reading, never collapsed or hidden.
    voiceRow('assistant', ' One more thing before you go.', 30000, 31000, 6)
  ])

  expect(messages.map(m => m.spokenRelay ?? null)).toEqual([null, null, 'repeat', null, null, null])
})

it('never folds a voice reading into the tool-bearing Hermes reply before it', () => {
  const rows = [
    voiceRow('user', ' What Mac am I on?', 1000, 2000, 1),
    { content: '', role: 'assistant', row_id: 2, timestamp: 2, tool_calls: [{ id: 'c1', function: { name: 'terminal', arguments: '{}' }, type: 'function' }] },
    { content: 'macOS 27', role: 'tool', row_id: 3, timestamp: 3, tool_call_id: 'c1' },
    reply('You are on macOS 27.0.1.', 4),
    voiceRow('assistant', ' Sure, checking. You are on macOS 27.0.1.', 2500, 6000, 5)
  ] as SessionMessage[]

  const messages = toChatMessages(rows)
  const reading = messages.at(-1)!
  expect(reading.rowId).toBe(5)
  expect(reading.voiceLive?.firstEventId).toBe('ev-5')
  expect(shapeVoiceLiveMessages(messages).at(-1)?.spokenRelay).toBe('repeat')
})

it('keeps a spoken reading that changes the meaning visible as a difference', () => {
  const messages = shaped([
    voiceRow('user', ' Is the backup done?', 1000, 2000, 1),
    reply('The backup is complete.', 2),
    voiceRow('assistant', ' The backup is not complete.', 2500, 4000, 3)
  ])

  expect(messages.at(-1)?.spokenRelay).toBe('differs')
})

it('tells the renderer which messages are display joins', () => {
  const [joined] = shaped([
    voiceRow('user', ' That was one of', 70400, 71000, 1),
    voiceRow('assistant', ' Okay,', 71000, 71200, 2),
    voiceRow('user', ' them', 71200, 71400, 3)
  ])

  expect((toRuntimeMessage(joined).metadata.custom as Record<string, unknown>).voiceJoined).toBe(true)
})

it('never folds the next Hermes tool turn into a voice row', () => {
  const rows = [
    voiceRow('user', ' Check the disk', 1000, 2000, 1),
    voiceRow('assistant', ' Sure, checking now.', 2000, 3000, 2),
    { content: '', role: 'assistant', row_id: 3, timestamp: 3, tool_calls: [{ id: 'c1', function: { name: 'terminal', arguments: '{}' }, type: 'function' }] },
    { content: '40% used', role: 'tool', row_id: 4, timestamp: 4, tool_call_id: 'c1' },
    reply('The disk is 40% used.', 5)
  ] as SessionMessage[]

  const messages = toChatMessages(rows)
  const voice = messages.find(m => m.rowId === 2)!
  expect(chatMessageText(voice).trim()).toBe('Sure, checking now.')
  expect(voice.parts.some(part => part.type === 'tool-call')).toBe(false)
  expect(chatMessageText(messages.at(-1)!)).toContain('The disk is 40% used.')
})

it('keeps a reading visible when it drops a negation or changes a number', () => {
  const dropped = shaped([reply('The backup is not complete.', 1), voiceRow('assistant', ' The backup is complete.', 2500, 4000, 2)])
  const number = shaped([reply('It is 1:24 a.m. Eastern.', 1), voiceRow('assistant', ' Okay, it is 1:42 a.m. Eastern.', 2500, 4000, 2)])

  expect(dropped.at(-1)?.spokenRelay).toBe('differs')
  expect(number.at(-1)?.spokenRelay).toBe('differs')
  // A sign is part of the number, written or spoken.
  const sign = shaped([reply('The temperature is −5 degrees.', 1), voiceRow('assistant', ' The temperature is 5 degrees.', 2500, 4000, 2)])
  const spoken = shaped([reply('The temperature is 5 degrees.', 1), voiceRow('assistant', ' The temperature is minus 5 degrees.', 2500, 4000, 2)])
  expect(sign.at(-1)?.spokenRelay).toBe('differs')
  expect(spoken.at(-1)?.spokenRelay).toBe('differs')
})
