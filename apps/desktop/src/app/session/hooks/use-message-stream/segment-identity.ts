/**
 * Wire identity of a turn's streamed reply (tui_gateway/stream_segments.py).
 *
 * A prompt turn stamps its message.start with `turn_id`, every message.delta
 * with `turn_id` + `segment` + `offset`, and every message.interim with the
 * `turn_id` + `segment` it seals. The same frame can reach the renderer twice
 * (a reconnect replay under new seqs, a superseded attempt re-streaming the
 * segment it already sent, a second socket); without an identity it can only
 * append, and the copy paints beside the original (#144609, #128158). The
 * cursor below remembers how far each turn's reply has been applied, so a
 * re-sent chunk or seal is an upsert of what is already there. Frames without
 * a stamp (older backends, non-prompt emitters) pass through unchanged.
 */

export interface SegmentStamp {
  turnId: string
  segment: number
  offset?: number
}

export interface SegmentCursor {
  turnId: string
  /** The segment deltas currently extend. */
  segment: number
  /** UTF-16 code units of `segment` already applied. */
  end: number
}

const nonNegativeInt = (value: unknown): null | number =>
  typeof value === 'number' && Number.isInteger(value) && value >= 0 ? value : null

/** The stamp a message.delta / message.interim payload carries, or null. */
export function segmentStamp(payload: unknown): null | SegmentStamp {
  const record = (payload ?? {}) as Record<string, unknown>
  const turnId = typeof record.turn_id === 'string' ? record.turn_id : ''
  const segment = nonNegativeInt(record.segment)

  if (!turnId || segment === null) {
    return null
  }

  const offset = nonNegativeInt(record.offset)

  return offset === null ? { turnId, segment } : { turnId, segment, offset }
}

/** The part of a stamped delta not applied yet ('' when it is a re-send). */
export function admitSegmentDelta(
  cursor: SegmentCursor | undefined,
  stamp: SegmentStamp,
  text: string
): { cursor: SegmentCursor; text: string } {
  const offset = stamp.offset ?? 0
  const end = offset + text.length

  if (!cursor || cursor.turnId !== stamp.turnId || stamp.segment > cursor.segment) {
    return { cursor: { turnId: stamp.turnId, segment: stamp.segment, end }, text }
  }

  if (stamp.segment < cursor.segment || end <= cursor.end) {
    return { cursor, text: '' }
  }

  // Overlap keeps only the unseen tail; a gap (frames lost) applies whole.
  return { cursor: { ...cursor, end }, text: text.slice(Math.max(0, cursor.end - offset)) }
}

/** Whether a stamped seal closes a segment not sealed yet, and the cursor after it. */
export function admitSegmentSeal(
  cursor: SegmentCursor | undefined,
  stamp: SegmentStamp
): { cursor: SegmentCursor; fresh: boolean } {
  const next = { turnId: stamp.turnId, segment: stamp.segment + 1, end: 0 }

  // The cursor already moved past this segment: an earlier copy of the seal did it.
  if (cursor?.turnId === stamp.turnId && stamp.segment < cursor.segment) {
    return { cursor, fresh: false }
  }

  return { cursor: next, fresh: true }
}
