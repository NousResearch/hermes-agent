import { expect, it } from 'vitest'

import type { SessionMessage } from '@/types/hermes'

import { chatMessageText, toChatMessages } from '.'

// A saved GPT-Live row keeps its call timing and first provider event.
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

const voiceRow = (role: 'assistant' | 'user', content: string, start: number, end: number, id: number): SessionMessage =>
  ({
    content,
    display_metadata: { voice_live: { call_key: 'c1', end_ms: end, first_event_id: `ev-${id}`, partial: false, start_ms: start } },
    role,
    row_id: id,
    timestamp: id
  }) as SessionMessage

const reply = (content: string, id: number) => ({ content, role: 'assistant', row_id: id, timestamp: id }) as SessionMessage

const toolTurn = (id: number) => [
  { content: '', role: 'assistant', row_id: id, timestamp: id, tool_calls: [{ id: `c${id}`, function: { name: 'terminal', arguments: '{}' }, type: 'function' }] },
  { content: 'ok', role: 'tool', row_id: id + 1, timestamp: id + 1, tool_call_id: `c${id}` }
]

// A saved voice row is the voice model speaking, never part of a Hermes tool turn.
it('never folds a voice row into the tool-bearing Hermes reply before it', () => {
  const messages = toChatMessages([
    voiceRow('user', ' What Mac am I on?', 1000, 2000, 1),
    ...toolTurn(2),
    reply('You are on macOS 27.0.1.', 4),
    voiceRow('assistant', ' Sure, checking. You are on macOS 27.0.1.', 2500, 6000, 5)
  ] as SessionMessage[])

  const reading = messages.at(-1)!
  expect(reading.rowId).toBe(5)
  expect(reading.voiceLive?.firstEventId).toBe('ev-5')
})

it('never folds the next Hermes tool turn into a voice row', () => {
  const messages = toChatMessages([
    voiceRow('user', ' Check the disk', 1000, 2000, 1),
    voiceRow('assistant', ' Sure, checking now.', 2000, 3000, 2),
    ...toolTurn(3),
    reply('The disk is 40% used.', 5)
  ] as SessionMessage[])

  const voice = messages.find(m => m.rowId === 2)!
  expect(chatMessageText(voice).trim()).toBe('Sure, checking now.')
  expect(voice.parts.some(part => part.type === 'tool-call')).toBe(false)
  expect(chatMessageText(messages.at(-1)!)).toContain('The disk is 40% used.')
})
