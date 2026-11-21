import { renderHook } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import { graftRefreshedTailOntoBackfill } from '@/app/chat/transcript-backfill'
import { preserveLocalPendingTurnMessages } from '@/app/session/hooks/use-session-actions/utils'
import {
  type ChatMessage,
  preserveLocalAssistantErrors,
  preserveLocalSystemNotices,
  toChatMessages
} from '@/lib/chat-messages'
import type { SessionMessage } from '@/types/hermes'

import { useRuntimeMessageRepository } from './runtime-repository'

/**
 * The shapes a compaction generation pair takes on the way to the screen,
 * measured against a live store in #123985. The refresh read
 * (`include_compacted=true`) ships both generations, and
 * `mergeOverlappingTail` unions the window with the refreshed page by durable
 * row id, so whether the pair survives the merge depends on whether the window
 * and the page share one post-compaction row id — not on the row shape itself.
 *
 * Case 1 is the one neither client PR covers: the merge keeps both generations,
 * and only a logical-row key at the repository boundary collapses them.
 * Case 3 is a guard on that key; case 4 pins the uid rule both ways — same
 * uid merges two generations of one row, distinct uids stay two rows.
 */

const TIMESTAMP = 1_771_500_000

/** A hydrated chat row: distinct string ids, durable `rowId` per backend row,
 *  and the row's `messageUid` whenever the fixture models a backend row. */
const chat = (
  id: string,
  rowId: number,
  role: 'user' | 'assistant',
  text: string,
  timestamp: number | null = TIMESTAMP,
  messageUid?: string
): ChatMessage => ({
  id,
  role,
  parts: [{ type: 'text', text }],
  rowId,
  ...(messageUid !== undefined ? { messageUid } : {}),
  ...(timestamp === null ? {} : { timestamp })
})

/** The same row without a durable id (a live/streaming copy). */
const live = (
  id: string,
  role: 'user' | 'assistant',
  text: string,
  timestamp: number | null = TIMESTAMP
): ChatMessage => ({
  id,
  role,
  parts: [{ type: 'text', text }],
  ...(timestamp === null ? {} : { timestamp })
})

const rendered = (messages: ChatMessage[]): string[] =>
  messages.flatMap(message => message.parts.flatMap(part => (part.type === 'text' ? [part.text] : [])))

interface RuntimePart {
  type: string
  text?: string
}

const render = (messages: ChatMessage[]): string[] =>
  renderHook(() => useRuntimeMessageRepository(messages)).result.current.messages.flatMap(item =>
    (item.message.content as readonly RuntimePart[]).flatMap(part =>
      part.type === 'text' && part.text !== undefined ? [part.text] : []
    )
  )

/** The repository items themselves, for asserting which representative won. */
const repositoryItems = (messages: ChatMessage[]) =>
  renderHook(() => useRuntimeMessageRepository(messages)).result.current.messages

describe('compaction generations reaching the transcript', () => {
  it('renders one copy when the window shares a post-compaction row id with the refreshed page', () => {
    // The pre-compaction generation (row ids 100/101) is still in the live
    // window; the row persisted after the compaction (202) is on both sides, so
    // the merge unions by row id and both generations travel to the repository.
    const window = [
      chat('w-100', 100, 'user', 'Q'),
      chat('w-101', 101, 'assistant', 'A'),
      chat('w-202', 202, 'assistant', 'AFTER')
    ]

    const page = [
      chat('p-200', 200, 'user', 'Q'),
      chat('p-201', 201, 'assistant', 'A'),
      chat('p-202', 202, 'assistant', 'AFTER')
    ]

    const merged = graftRefreshedTailOntoBackfill(page, window)

    // The merge itself keeps the pair — that is what the repository key is for.
    expect(rendered(merged)).toEqual(['Q', 'A', 'Q', 'A', 'AFTER'])
    expect(render(merged)).toEqual(['Q', 'A', 'AFTER'])
  })

  it('renders one copy when the page shares no durable row id with the window', () => {
    // Same logical transcript, every id rebuilt and no post-compaction row in
    // the window: the page replaces the window, so nothing is left to dedupe.
    const window = [chat('w-100', 100, 'user', 'Q'), chat('w-101', 101, 'assistant', 'A')]
    const page = [chat('p-200', 200, 'user', 'Q'), chat('p-201', 201, 'assistant', 'A')]

    const merged = graftRefreshedTailOntoBackfill(page, window)

    expect(rendered(merged)).toEqual(['Q', 'A'])
    expect(render(merged)).toEqual(['Q', 'A'])
  })

  it('never collapses two rows on an absent timestamp', () => {
    // Review #123294 asked which path produces this input. None does today:
    // every durable write stamps a timestamp (hermes_state_messages.py
    // _insert_message_rows coerces against now_ts and writes it back), both
    // read projections pass it through (api_server._message_response,
    // chat-messages/hydration.ts), and desktop live rows stamp occurredAt
    // (use-message-stream). A compaction copy arrives with the commit time,
    // never with an absent timestamp. Expected: 2 rows. Neither row carries a
    // messageUid, so the pair takes the uid-less branch of the repository
    // rule — the `timestamp === undefined` guard is the whole defense. The
    // case stays as a tripwire — a key that substituted a placeholder for a
    // missing timestamp would merge two unrelated rows here.
    const messages = [live('a-1', 'assistant', 'same words'), live('a-2', 'assistant', 'same words', null)]

    expect(render(messages)).toEqual(['same words', 'same words'])
  })

  it('merges a commit-time-stamped compaction copy into its original by shared uid (#123985)', () => {
    // The producing path on latest main: the gateway replay keeps the durable
    // timestamp on user rows only (gateway/run.py _build_gateway_agent_history,
    // preserve_timestamp=(role == "user"); tool rows have it stripped), so the
    // carried tail is written back from timestamp-less dicts;
    // _carry_parent_timestamps cannot donate (the originals were rewound
    // before it runs — a documented no-op for a clean compaction), and
    // _insert_message_rows stamps the copy with the commit time. Original and
    // copy therefore arrive with DIFFERENT timestamps — and the SAME
    // message_uid: stamp_message_uid casts one uid per logical row and every
    // re-insert re-writes it (agent/message_metadata.py), so per latkinw-wq
    // (2026-11-17) the pair is two generations of ONE row and merges. The
    // copy is the generation the store still resolves, so it wins the rank
    // tie and its rowId rides the single rendered slot.
    const original = chat('a-orig', 101, 'assistant', 'the answer', TIMESTAMP, 'm101')
    const copy = chat('a-copy', 901, 'assistant', 'the answer', TIMESTAMP + 5_000, 'm101')

    expect(render([original, copy])).toEqual(['the answer'])

    // The newer generation won the election — future reads and actions
    // resolve its row id.
    const [item] = repositoryItems([original, copy])
    const custom = (item.message.metadata as { custom?: { rowId?: number } }).custom
    expect(custom?.rowId).toBe(901)
  })

  it('keeps two same-content rows with distinct uids apart (the #123985 shape, two rows)', () => {
    // The uid rule cuts both ways: distinct uids are distinct logical rows,
    // so the compaction shape with DIFFERENT uids stays two rows — the
    // "two identical replies at different times" semantics, now held by uid
    // identity rather than by the timestamp alone (these rows differ in
    // timestamp too, so the content key would hold them apart anyway; the
    // uid rule holds them apart even if the timestamps were to align).
    const original = chat('a-orig', 101, 'assistant', 'the answer', TIMESTAMP, 'm101')
    const twin = chat('a-twin', 901, 'assistant', 'the answer', TIMESTAMP + 5_000, 'm999')

    expect(render([original, twin])).toEqual(['the answer', 'the answer'])
  })

  it('keeps two identical replies at different timestamps apart', () => {
    // Measured shape: many groups share role/content and differ only in the
    // timestamp, so the key has to keep the timestamp.
    const messages = [
      chat('b-1', 1, 'assistant', 'same words', TIMESTAMP),
      chat('b-2', 2, 'assistant', 'same words', TIMESTAMP + 3_600)
    ]

    expect(render(messages)).toEqual(['same words', 'same words'])
  })
})

it('does not repeat the end of a rewritten tool turn cut by the refreshed page', () => {
  const call = (id: string) => ({
    id,
    type: 'function' as const,
    function: { name: 'terminal', arguments: '{"command":"echo"}' }
  })

  const original: SessionMessage[] = [
    { id: 100, message_uid: 'm100', role: 'user', content: 'older prompt', timestamp: TIMESTAMP },
    {
      id: 101,
      message_uid: 'm101',
      role: 'assistant',
      content: 'look one',
      timestamp: TIMESTAMP + 1,
      tool_calls: [call('c1')]
    },
    {
      id: 102,
      message_uid: 'm102',
      role: 'tool',
      content: 'ok',
      timestamp: TIMESTAMP + 2,
      tool_call_id: 'c1',
      tool_name: 'terminal'
    },
    {
      id: 103,
      message_uid: 'm103',
      role: 'assistant',
      content: 'look two',
      timestamp: TIMESTAMP + 3,
      tool_calls: [call('c2')]
    },
    {
      id: 104,
      message_uid: 'm104',
      role: 'tool',
      content: 'ok',
      timestamp: TIMESTAMP + 4,
      tool_call_id: 'c2',
      tool_name: 'terminal'
    },
    {
      id: 105,
      message_uid: 'm105',
      role: 'assistant',
      content: 'look three',
      timestamp: TIMESTAMP + 5,
      tool_calls: [call('c3')]
    },
    {
      id: 106,
      message_uid: 'm106',
      role: 'tool',
      content: 'ok',
      timestamp: TIMESTAMP + 6,
      tool_call_id: 'c3',
      tool_name: 'terminal'
    },
    { id: 107, message_uid: 'm107', role: 'assistant', content: 'older answer', timestamp: TIMESTAMP + 7 }
  ]

  const newer: SessionMessage[] = [
    { id: 300, message_uid: 'm300', role: 'user', content: 'newer prompt', timestamp: TIMESTAMP + 20 },
    { id: 301, message_uid: 'm301', role: 'assistant', content: 'newer answer', timestamp: TIMESTAMP + 21 }
  ]

  // The window holds the whole turn; the refreshed page starts at its second round,
  // under the ids the compaction gave the rows (100-107 became 200-207). The
  // rewrite keeps each row's message_uid, and the projections now ship it
  // (tui_gateway/session_history.py next to row_id; the REST transcript
  // whitelist), so the graft recognizes the renumbered rows by uid.
  const window = toChatMessages([...original, ...newer])

  const page = toChatMessages([
    ...original.slice(3).map((message, index) => ({ ...message, id: 203 + index })),
    ...newer
  ])

  const merged = preserveLocalSystemNotices(
    preserveLocalAssistantErrors(
      preserveLocalPendingTurnMessages(graftRefreshedTailOntoBackfill(page, window), window),
      window
    ),
    window
  )

  const visible = render(merged)

  for (const text of ['look one', 'look two', 'look three', 'older answer', 'newer prompt', 'newer answer']) {
    expect(
      visible.filter(item => item === text),
      text
    ).toHaveLength(1)
  }
})

it('keeps one copy when the page brings the whole turn the window holds cut', () => {
  const call = (id: string) => ({
    id,
    type: 'function' as const,
    function: { name: 'terminal', arguments: '{"command":"echo"}' }
  })

  const original: SessionMessage[] = [
    { id: 100, message_uid: 'm100', role: 'user', content: 'older prompt', timestamp: TIMESTAMP },
    {
      id: 101,
      message_uid: 'm101',
      role: 'assistant',
      content: 'look one',
      timestamp: TIMESTAMP + 1,
      tool_calls: [call('c1')]
    },
    {
      id: 102,
      message_uid: 'm102',
      role: 'tool',
      content: 'ok',
      timestamp: TIMESTAMP + 2,
      tool_call_id: 'c1',
      tool_name: 'terminal'
    },
    {
      id: 103,
      message_uid: 'm103',
      role: 'assistant',
      content: 'look two',
      timestamp: TIMESTAMP + 3,
      tool_calls: [call('c2')]
    },
    {
      id: 104,
      message_uid: 'm104',
      role: 'tool',
      content: 'ok',
      timestamp: TIMESTAMP + 4,
      tool_call_id: 'c2',
      tool_name: 'terminal'
    },
    {
      id: 105,
      message_uid: 'm105',
      role: 'assistant',
      content: 'look three',
      timestamp: TIMESTAMP + 5,
      tool_calls: [call('c3')]
    },
    {
      id: 106,
      message_uid: 'm106',
      role: 'tool',
      content: 'ok',
      timestamp: TIMESTAMP + 6,
      tool_call_id: 'c3',
      tool_name: 'terminal'
    },
    { id: 107, message_uid: 'm107', role: 'assistant', content: 'older answer', timestamp: TIMESTAMP + 7 }
  ]

  const newer: SessionMessage[] = [
    { id: 300, message_uid: 'm300', role: 'user', content: 'newer prompt', timestamp: TIMESTAMP + 20 },
    { id: 301, message_uid: 'm301', role: 'assistant', content: 'newer answer', timestamp: TIMESTAMP + 21 }
  ]

  // The window was itself grafted from a page that started inside the turn:
  // its fold lost the turn's first round ('look one'). The refreshed page
  // starts above the turn under the compaction's renumbered ids, stamped
  // with the commit-time timestamps #123985 documents, so its fold is the
  // whole turn in a later generation. The window fold's first uid is one the
  // page's fold carries: the page's whole copy takes the slot, the partial
  // window copy goes, and the lost first round returns.
  const window = toChatMessages([original[0], ...original.slice(3), ...newer])

  const page = toChatMessages([
    ...original
      .slice(1)
      .map((message, index) => ({ ...message, id: 203 + index, timestamp: TIMESTAMP + 5000 + index })),
    ...newer
  ])

  const merged = preserveLocalSystemNotices(
    preserveLocalAssistantErrors(
      preserveLocalPendingTurnMessages(graftRefreshedTailOntoBackfill(page, window), window),
      window
    ),
    window
  )

  const visible = render(merged)

  for (const text of ['look one', 'look two', 'look three', 'older answer', 'newer prompt', 'newer answer']) {
    expect(
      visible.filter(item => item === text),
      text
    ).toHaveLength(1)
  }
})

it('keeps the renumbered generation when both the window and the page hold the whole turn', () => {
  const call = (id: string) => ({
    id,
    type: 'function' as const,
    function: { name: 'terminal', arguments: '{"command":"echo"}' }
  })

  const original: SessionMessage[] = [
    { id: 100, message_uid: 'm100', role: 'user', content: 'older prompt', timestamp: TIMESTAMP },
    {
      id: 101,
      message_uid: 'm101',
      role: 'assistant',
      content: 'look one',
      timestamp: TIMESTAMP + 1,
      tool_calls: [call('c1')]
    },
    {
      id: 102,
      message_uid: 'm102',
      role: 'tool',
      content: 'ok',
      timestamp: TIMESTAMP + 2,
      tool_call_id: 'c1',
      tool_name: 'terminal'
    },
    {
      id: 103,
      message_uid: 'm103',
      role: 'assistant',
      content: 'look two',
      timestamp: TIMESTAMP + 3,
      tool_calls: [call('c2')]
    },
    {
      id: 104,
      message_uid: 'm104',
      role: 'tool',
      content: 'ok',
      timestamp: TIMESTAMP + 4,
      tool_call_id: 'c2',
      tool_name: 'terminal'
    },
    {
      id: 105,
      message_uid: 'm105',
      role: 'assistant',
      content: 'look three',
      timestamp: TIMESTAMP + 5,
      tool_calls: [call('c3')]
    },
    {
      id: 106,
      message_uid: 'm106',
      role: 'tool',
      content: 'ok',
      timestamp: TIMESTAMP + 6,
      tool_call_id: 'c3',
      tool_name: 'terminal'
    },
    { id: 107, message_uid: 'm107', role: 'assistant', content: 'older answer', timestamp: TIMESTAMP + 7 }
  ]

  const newer: SessionMessage[] = [
    { id: 300, message_uid: 'm300', role: 'user', content: 'newer prompt', timestamp: TIMESTAMP + 20 },
    { id: 301, message_uid: 'm301', role: 'assistant', content: 'newer answer', timestamp: TIMESTAMP + 21 }
  ]

  // The compaction dropped the window's oldest rows and renumbered the turn
  // it kept: the window holds ancient rows the page no longer carries plus
  // the whole turn under its pre-compaction ids (100-107), the page holds the
  // same whole turn under the renumbered ids (200-207), the same message_uid
  // on each row. The ancient rows keep the graft on the merge path — a page
  // that covered every window row would take over directly. There the
  // boundary cut used to drop the page's copy of the turn — its uids all
  // name rows a window fold carries — leaving the window's stale generation
  // on screen. The cut now requires the window fold to still carry a row the
  // page does not, so the page's renumbered copy takes the slots instead.
  const older: SessionMessage[] = [
    { id: 90, message_uid: 'm90', role: 'user', content: 'ancient prompt', timestamp: TIMESTAMP - 10 },
    { id: 91, message_uid: 'm91', role: 'assistant', content: 'ancient answer', timestamp: TIMESTAMP - 9 }
  ]

  const window = toChatMessages([...older, ...original, ...newer])

  const page = toChatMessages([...original.map((message, index) => ({ ...message, id: 200 + index })), ...newer])

  const merged = graftRefreshedTailOntoBackfill(page, window)

  const visible = render(merged)

  for (const text of [
    'ancient prompt',
    'ancient answer',
    'older prompt',
    'look one',
    'look two',
    'look three',
    'older answer',
    'newer prompt',
    'newer answer'
  ]) {
    expect(
      visible.filter(item => item === text),
      text
    ).toHaveLength(1)
  }

  const turnRows = merged.filter(message => message.rowId !== undefined && message.rowId > 91 && message.rowId < 300)

  // The renumbered generation won: the window's stale rows never render, and
  // future reads and actions resolve the ids the store still holds.
  expect(turnRows.map(message => message.rowId)).toEqual([200, 201])
})
