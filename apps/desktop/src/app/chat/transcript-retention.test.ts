import { describe, expect, it } from 'vitest'

import type { ChatMessage } from '@/lib/chat-messages'
import { messageStoreWeight, RENDER_WEIGHT_CHARS } from '@/lib/render-weight'

import { boundRetainedTranscript, TRANSCRIPT_RETAIN_BUDGET } from './transcript-retention'

interface RowOptions {
  group?: string
  /** Backend rows this message folds (the hydration fold merges tool rows). */
  span?: number
  /** A row still in flight: no backend row yet, so it can never be released. */
  pendingRow?: boolean
  textUnits?: number
}

const row = (index: number, options: RowOptions = {}): ChatMessage => {
  const { group, pendingRow = false, span, textUnits = 1 } = options

  return {
    ...(group ? { branchGroupId: group } : {}),
    id: `m${index}`,
    parts: [{ type: 'text', text: 'x'.repeat(RENDER_WEIGHT_CHARS * textUnits) }],
    role: 'assistant',
    ...(pendingRow ? { pending: true } : { rowId: index + 1 }),
    ...(span ? { serverRowSpan: span } : {})
  }
}

const transcript = (count: number, each: (index: number) => ChatMessage = index => row(index)): ChatMessage[] =>
  Array.from({ length: count }, (_, index) => each(index))

/** Rows the slack buys behind the window, at the weight every `heavy` row carries. */
const slackRows = (sample: ChatMessage): number =>
  Math.ceil(TRANSCRIPT_RETAIN_BUDGET / messageStoreWeight(sample.parts))

const heavy = (index: number) => row(index, { textUnits: 100 })

/** The release path, or a failure naming the anchor that was expected to release. */
const released = (messages: readonly ChatMessage[], anchorId: null | string) => {
  const retention = boundRetainedTranscript(messages, anchorId)

  if (!retention.released) {
    throw new Error(`expected a release for anchor ${String(anchorId)}`)
  }

  return retention
}

const untouched = (messages: readonly ChatMessage[], anchorId: null | string) =>
  boundRetainedTranscript(messages, anchorId)

describe('boundRetainedTranscript', () => {
  it('releases the persisted rows older than the window and its slack', () => {
    const messages = transcript(60, heavy)
    const anchor = messages[40].id
    const keep = slackRows(messages[40])

    const retention = released(messages, anchor)

    expect(retention.releasedRows).toBe(40 - keep)
    expect(retention.messages[0].id).toBe(`m${40 - keep}`)
    expect(retention.messages).toHaveLength(messages.length - retention.releasedRows)
  })

  it('keeps a fixed amount of history behind the window, however long the session is', () => {
    const short = transcript(60, heavy)
    const long = transcript(600, heavy)
    const shortRetention = released(short, short[40].id)
    const longRetention = released(long, long[560].id)

    const behind = (retention: ReturnType<typeof released>, anchorId: string) =>
      retention.messages.findIndex(message => message.id === anchorId)

    // The window itself holds different amounts here; what must not grow with
    // the session is the history retained BEHIND it.
    expect(behind(longRetention, long[560].id)).toBe(behind(shortRetention, short[40].id))
    expect(behind(longRetention, long[560].id)).toBeLessThanOrEqual(slackRows(long[560]))

    // ...and the longer session releases proportionally more rows.
    expect(longRetention.releasedRows).toBeGreaterThan(shortRetention.releasedRows)
    expect(longRetention.messages.length).toBeLessThan(long.length / 2)
  })

  it('leaves the transcript whole when a released row is still in flight', () => {
    // Rows 27-29 are in flight: releasing the prefix around them would drop
    // content no fetch can restore.
    const messages = transcript(60, index =>
      index >= 27 && index <= 29 ? row(index, { pendingRow: true, textUnits: 100 }) : heavy(index)
    )

    expect(untouched(messages, messages[40].id)).toEqual({ released: false })
  })

  it('does not release a live correction or its zero-span folded-tool prefix', () => {
    const messages = transcript(60, heavy)
    const boundary = 40 - slackRows(messages[40])
    messages[boundary - 1] = { ...messages[boundary - 1], serverRowSpan: 0 }
    messages[boundary] = {
      id: 'user-correction',
      role: 'user',
      parts: [{ type: 'text', text: 'x'.repeat(RENDER_WEIGHT_CHARS * 100) }]
    }

    // A live redirect has no durable rowId but is not `pending: true`.
    // Cutting at the correction would also release the preceding virtual
    // assistant fragment, whose zero span could not rewind the older page.
    expect(untouched(messages, messages[40].id)).toEqual({ released: false })
  })

  it('does not split an assistant branch group', () => {
    // The slack boundary lands inside one three-message branch group; keeping
    // the whole group is what stops a branch being re-parented onto a fork
    // point that is no longer in the store.
    const messages = transcript(60, index =>
      row(index, { group: index >= 26 && index <= 28 ? 'g1' : undefined, textUnits: 100 })
    )

    const keep = slackRows(messages[40])

    const retention = released(messages, messages[40].id)

    expect(40 - keep).toBeGreaterThan(25)
    expect(40 - keep).toBeLessThanOrEqual(28)
    expect(retention.messages.filter(message => message.branchGroupId === 'g1')).toHaveLength(3)
    expect(retention.messages[0].id).toBe('m26')
  })

  it('releases nothing when nothing precedes the window', () => {
    const messages = transcript(30, heavy)

    for (const anchor of [null, messages[0].id]) {
      expect(untouched(messages, anchor)).toEqual({ released: false })
    }
  })

  it('releases nothing when a row older than the window is still in flight', () => {
    // Rows 0-19 were never persisted. Releasing them would drop content nothing
    // can fetch back, so the transcript is left whole rather than released in
    // part.
    const messages = transcript(30, index => row(index, { pendingRow: index < 20, textUnits: 100 }))

    expect(untouched(messages, messages[25].id)).toEqual({ released: false })
  })

  it('reports the released rows in backend rows, not messages', () => {
    // A tool-heavy turn folds its tool rows into one message: the older-page
    // offset the caller rewinds is counted in those backend rows.
    const messages = transcript(60, index => row(index, { span: 11, textUnits: 100 }))
    const keep = slackRows(messages[40])

    const retention = released(messages, messages[40].id)

    expect(retention.releasedRows).toBe(40 - keep)
    expect(retention.releasedServerRows).toBe((40 - keep) * 11)
  })

  it('does no work when there is nothing to release', () => {
    // A re-cut of an untouched transcript must stay cheap: the released:false
    // path must not walk the weights of a long array.
    let reads = 0

    const messages = transcript(20, index => {
      const base = heavy(index)

      return Object.defineProperty({ ...base }, 'parts', {
        enumerable: true,
        get: () => {
          reads += 1

          return base.parts
        }
      }) as ChatMessage
    })

    expect(untouched(messages, messages[0].id)).toEqual({ released: false })
    expect(reads).toBe(0)
  })
})
