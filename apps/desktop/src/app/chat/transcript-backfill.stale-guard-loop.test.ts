import { describe, expect, it } from 'vitest'

import { graftRefreshedTailOntoBackfill } from '@/app/chat/transcript-backfill'
import { type ChatMessage, preserveLocalAssistantErrors, toChatMessages } from '@/lib/chat-messages'
import { messagesIfTranscriptBehind } from '@/lib/stale-transcript-guard'
import type { SessionMessage } from '@/types/hermes'

/**
 * REGRESSION: the pre-send stale-transcript guard never converges when the
 * newest display page opens on a page-local tool fold.
 *
 * `submit.ts` (use-prompt-actions) and `use-session-tile-delegate.ts` read the
 * cached transcript, drop the optimistic bubble, ask
 * `refreshIfTranscriptStale` whether this window is behind, and on a non-null
 * answer install the refreshed page and REFUSE the send. A user who retries
 * re-runs that loop with the installed transcript as `localMessages`, so the
 * guard has to reach a fixed point: a window that was just refreshed must not
 * be reported behind again.
 *
 * `LATEST_SESSION_MESSAGES_LIMIT` means the guard's authoritative read is only
 * the last page of a long session, and that page can begin mid tool batch.
 * `toChatMessages` flushes such a batch as a synthetic assistant message whose
 * id is not durable, so the page's first rendered row has no `rowId`. When the
 * window was hydrated from that same page, `graftRefreshedTailOntoBackfill`
 * used to read that unanchored prefix row as proof of earlier history and
 * re-prepended it to the refreshed page on every read. The refreshed window
 * came back one row longer each round, so `messagesIfTranscriptBehind` kept
 * answering "behind": send is refused on every attempt and the transcript
 * grows a duplicate per attempt.
 */

const OPTIMISTIC_ID = 'optimistic-1'

const optimistic = (): ChatMessage => ({
  id: OPTIMISTIC_ID,
  parts: [{ type: 'text', text: 'next' }],
  role: 'user'
})

const userRow = (rowId: number, text: string): SessionMessage =>
  ({ id: rowId, role: 'user', content: text, timestamp: 1_000 + rowId }) as SessionMessage

const assistantRow = (rowId: number, text: string): SessionMessage =>
  ({ id: rowId, role: 'assistant', content: text, timestamp: 1_000 + rowId }) as SessionMessage

/** A standalone tool row: `toChatMessages` folds it into a message with no rowId. */
const toolRow = (rowId: number, text: string): SessionMessage =>
  ({ id: rowId, role: 'tool', content: text, timestamp: 1_000 + rowId }) as SessionMessage

/**
 * The shape that reproduces it: the newest page opens on a TOOL ROW with no
 * assistant answer under it yet, so `toChatMessages` flushes it as its own
 * message with no durable rowId — exactly the leading fold a tool-heavy turn
 * puts at the top of the page.
 */
const pageOpeningOnToolRow = (): SessionMessage[] => [
  toolRow(100, 'reading files'),
  userRow(101, 'next question'),
  assistantRow(102, 'next answer'),
  userRow(103, 'final question'),
  assistantRow(104, 'final answer')
]

/** One submit attempt, mirroring the guard call + install both submit paths make. */
const attempt = (local: ChatMessage[], page: SessionMessage[]) => {
  const baseline = local.filter(message => message.id !== OPTIMISTIC_ID)
  const refreshed = messagesIfTranscriptBehind(baseline, toChatMessages(page))

  return {
    installed: refreshed ? preserveLocalAssistantErrors(refreshed, baseline) : baseline,
    refreshed
  }
}

/**
 * Retry the send until the guard lets it through. `rounds` counts the refusals
 * that preceded the allowed send — 0 means the first attempt went out.
 */
const retryUntilAllowed = (start: ChatMessage[], page: SessionMessage[], maxRounds = 4) => {
  let local = start
  const sizes: number[] = []

  for (let round = 0; round < maxRounds; round += 1) {
    const { refreshed, installed } = attempt(local, page)

    if (!refreshed) {
      return { rounds: round, settled: true, sizes }
    }

    sizes.push(installed.length)
    local = [...installed, optimistic()]
  }

  return { rounds: maxRounds, settled: false, sizes }
}

describe('stale transcript guard lets a retried send through', () => {
  it('is a fixed point when the window was hydrated from a page opening on a tool fold', () => {
    const page = pageOpeningOnToolRow()
    const local = toChatMessages(page)
    const remote = toChatMessages(page)

    // The page's leading fold is already in the window: nothing is missing, so
    // the graft must not manufacture an extra row.
    expect(local[0].rowId).toBeUndefined()
    expect(graftRefreshedTailOntoBackfill(remote, local)).toEqual(remote)
    expect(messagesIfTranscriptBehind(local, remote)).toBeNull()
  })

  it('sends on the first attempt when the window already holds the page', () => {
    const page = pageOpeningOnToolRow()

    expect(retryUntilAllowed([...toChatMessages(page), optimistic()], page)).toEqual({
      rounds: 0,
      settled: true,
      sizes: []
    })
  })

  it('refuses at most once from a genuinely behind window, then sends', () => {
    const page = pageOpeningOnToolRow()
    const behind = toChatMessages(page.slice(0, 4))

    expect(retryUntilAllowed([...behind, optimistic()], page)).toEqual({
      rounds: 1,
      settled: true,
      sizes: [toChatMessages(page).length]
    })
  })

  it('does not grow the window by one row on every retry', () => {
    const page = pageOpeningOnToolRow()
    const run = retryUntilAllowed([...toChatMessages(page), optimistic()], page)

    expect(run.settled).toBe(true)
    expect(run.sizes).toEqual([])
  })

  it('still keeps a backfilled prefix that is actually earlier than the page', () => {
    const page = pageOpeningOnToolRow()
    const olderPrefix = toChatMessages([userRow(1, 'ancient'), assistantRow(2, 'ancient reply')])
    const local = [...olderPrefix, ...toChatMessages(page)]

    const grafted = graftRefreshedTailOntoBackfill(toChatMessages(page), local)

    expect(grafted.length).toBe(local.length)
    expect(grafted[0]).toEqual(olderPrefix[0])
  })

  it('stays stable across repeated refreshes over a backfilled prefix', () => {
    const page = pageOpeningOnToolRow()
    const olderPrefix = toChatMessages([userRow(1, 'ancient'), assistantRow(2, 'ancient reply')])
    const local = [...olderPrefix, ...toChatMessages(page)]

    let grafted = local

    for (let round = 0; round < 5; round += 1) {
      grafted = graftRefreshedTailOntoBackfill(toChatMessages(page), grafted)

      expect(grafted.length).toBe(local.length)
      expect(grafted[0]).toEqual(olderPrefix[0])
    }

    expect(messagesIfTranscriptBehind(grafted, toChatMessages(page))).toBeNull()
  })

  it('still reads an unstored fold the page does not carry as earlier history', () => {
    const page = pageOpeningOnToolRow()

    // A fold the refreshed page has no copy of: it can only have come from an
    // older page, so it stays in front of the refreshed tail.
    const olderFold: ChatMessage = {
      id: '900-0-tools',
      parts: [{ completedAt: 900, type: 'tool-call', toolCallId: 'old-call', toolName: 'tool' }],
      role: 'assistant'
    }

    const local = [olderFold, ...toChatMessages(page)]

    const grafted = graftRefreshedTailOntoBackfill(toChatMessages(page), local)

    // The older fold survives in front of the page; the fold the page already
    // carries is not duplicated.
    expect(grafted.length).toBe(local.length)
    expect(grafted[0]).toEqual(olderFold)
  })
})
