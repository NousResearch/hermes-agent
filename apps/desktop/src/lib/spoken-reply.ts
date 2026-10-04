/**
 * Spoken-reply identity for Desktop auto-speak / Read Aloud.
 *
 * The live assistant row id (`assistant-stream-*`, `inflight-assistant-*`) is
 * not stable: hydrate rewrites that row under its durable backend id. Keying
 * "already spoken" on id alone then re-reads the same turn at the playback-idle
 * edge. A content fingerprint would swallow a later distinct turn that happens
 * to say the same thing ("Done.").
 *
 * Anchor on the user turn that owns the bubble, not the assistant-role
 * ordinal. Hydration rewrites the live id and folds tool segments into one
 * bubble, so the ordinal moves; the owning user turn does not. A later turn
 * has a new user row and stays unspoken. Text is not identity — two turns
 * that both say "Done." are different turns.
 */

export interface SpokenReplyAnchor {
  id: string
  ordinal: number
  /** User-turn index at mark time. Absent on anchors built before turn identity. */
  turnIndex?: number
  /** Read-aloud's progress through that turn: its assistant text already
   *  spoken, bubbles concatenated. Text survives the fold that merges the
   *  turn's bubbles into one row; a bubble id does not. */
  turnSpoken?: string
}

export interface SpokenReplyMessage {
  hidden?: boolean
  id: string
  pending?: boolean
  role: string
}

export interface UnspokenTurnPiece {
  /** The bubble the piece is read from. */
  id: string
  pending: boolean
  /** That bubble's text not yet spoken. */
  text: string
  /** The turn's text through the end of that bubble — what marking it records. */
  through: string
  /** Where the piece starts in the turn's text. */
  start: number
}

const NO_SESSION = '\0'

const lastSpokenBySession = new Map<string, SpokenReplyAnchor>()

export function isLiveTailReplyId(id: string): boolean {
  return id.startsWith('assistant-stream-') || id.startsWith('inflight-assistant-')
}

function sessionKey(sessionId: string | null | undefined): string {
  return sessionId ?? NO_SESSION
}

export function assistantReplyOrdinal(messages: readonly SpokenReplyMessage[], id: string): number {
  let ordinal = -1

  for (const message of messages) {
    if (message.role !== 'assistant' || message.hidden) {
      continue
    }

    ordinal += 1

    if (message.id === id) {
      return ordinal
    }
  }

  return -1
}

function lastVisibleAssistant(messages: readonly SpokenReplyMessage[]): SpokenReplyMessage | undefined {
  return messages.findLast(message => message.role === 'assistant' && !message.hidden)
}

/** Index of the user turn that owns `id`, or -1 when no user row precedes it.
 *  Hidden user rows count: a widget intent is a real turn boundary. Tool and
 *  assistant rows do not, so a fold that changes the assistant ordinal keeps
 *  this index. */
export function assistantTurnIndex(messages: readonly SpokenReplyMessage[], id: string): number {
  let turnIndex = -1

  for (const message of messages) {
    if (message.role === 'user') {
      turnIndex += 1
    }

    if (message.id === id) {
      return message.role === 'assistant' ? turnIndex : -1
    }
  }

  return -1
}

/** Stable across a live-id rewrite. Session-scoped so two chats' first turns
 *  do not share a speech claim. */
export function assistantTurnKey(
  sessionId: string | null | undefined,
  messages: readonly SpokenReplyMessage[],
  id: string
): string {
  return `${sessionId ?? ''}:${assistantTurnIndex(messages, id)}`
}

/** If a spoken live-tail row vanished and the same user turn now has a durable
 *  id, migrate the anchor — even when tool rows moved the assistant ordinal.
 *  Leave durable ids and later turns alone. */
export function absorbSpokenReplyRewrite(
  spoken: SpokenReplyAnchor | null,
  messages: readonly SpokenReplyMessage[]
): SpokenReplyAnchor | null {
  if (!spoken) {
    return null
  }

  if (assistantReplyOrdinal(messages, spoken.id) >= 0) {
    return spoken
  }

  if (!isLiveTailReplyId(spoken.id)) {
    return spoken
  }

  const last = lastVisibleAssistant(messages)

  if (!last) {
    return spoken
  }

  const ordinal = assistantReplyOrdinal(messages, last.id)
  const turnIndex = assistantTurnIndex(messages, last.id)
  const sameTurn = spoken.turnIndex !== undefined && spoken.turnIndex >= 0 && turnIndex === spoken.turnIndex

  // Turn identity wins over the assistant ordinal. A missing turnIndex is a
  // legacy anchor: keep the old same-slot check so those still migrate.
  if (spoken.turnIndex !== undefined && spoken.turnIndex >= 0) {
    if (!sameTurn) {
      return spoken
    }

    return { ...spoken, id: last.id, ordinal }
  }

  if (ordinal !== spoken.ordinal) {
    return spoken
  }

  return { id: last.id, ordinal }
}

export function spokenReplyOf(sessionId: string | null | undefined): SpokenReplyAnchor | null {
  return lastSpokenBySession.get(sessionKey(sessionId)) ?? null
}

function markSpokenReply(sessionId: string | null | undefined, anchor: SpokenReplyAnchor): void {
  lastSpokenBySession.set(sessionKey(sessionId), anchor)
}

export function markAssistantIdSpoken(
  sessionId: string | null | undefined,
  messages: readonly SpokenReplyMessage[],
  id: string
): void {
  const ordinal = assistantReplyOrdinal(messages, id)

  if (ordinal < 0) {
    return
  }

  markSpokenReply(sessionId, { id, ordinal, turnIndex: assistantTurnIndex(messages, id) })
}

/**
 * Carry the spoken anchor when a chat gets a real session id (null → created)
 * mid voice-conversation. Do not copy across two real sessions — that would
 * leak "already spoken" into a different transcript. The null-session entry is
 * moved, not copied: left behind, it would mark the NEXT new chat's first reply
 * as already spoken.
 */
export function adoptSpokenReplySession(
  fromSessionId: string | null | undefined,
  toSessionId: string | null | undefined
): void {
  const fromKey = sessionKey(fromSessionId)
  const toKey = sessionKey(toSessionId)

  if (fromKey !== NO_SESSION || toKey === NO_SESSION) {
    return
  }

  const from = lastSpokenBySession.get(fromKey)

  if (!from) {
    return
  }

  // Dropped even when not adopted below: the anchor belongs to this chat.
  lastSpokenBySession.delete(fromKey)

  if (!lastSpokenBySession.has(toKey)) {
    lastSpokenBySession.set(toKey, from)
  }
}

/** Current spoken anchor, migrated in place when the live row was rewritten. */
export function resolveSpokenReply(
  sessionId: string | null | undefined,
  messages: readonly SpokenReplyMessage[]
): SpokenReplyAnchor | null {
  const current = spokenReplyOf(sessionId)
  const next = absorbSpokenReplyRewrite(current, messages)

  if (next && next.id !== current?.id) {
    markSpokenReply(sessionId, next)
  }

  return next
}

/** Index in `text` just past `prefix`, ignoring whitespace; -1 when `text`
 *  does not start with it. Stored history may space a turn differently. */
function indexPastPrefix(text: string, prefix: string): number {
  let index = 0

  for (const char of prefix) {
    if (/\s/.test(char)) {
      continue
    }

    while (index < text.length && /\s/.test(text[index] ?? '')) {
      index += 1
    }

    if (text[index] !== char) {
      return -1
    }

    index += 1
  }

  return index
}

/**
 * Read-aloud's next piece of the current turn: the first bubble after what was
 * already spoken, in order. A tool turn streams narration bubbles and then the
 * answer, and hydration folds them into one row once the turn ends. Reading
 * only the last bubble skipped a narration sealed behind a newer one, and
 * migrating the anchor onto the folded row marked the unread answer spoken.
 * Progress is the turn's spoken text, so it survives the fold. Text that no
 * longer lines up counts as spoken: never read a turn twice.
 */
export function nextUnspokenTurnPiece<M extends SpokenReplyMessage>(
  sessionId: string | null | undefined,
  messages: readonly M[],
  textOf: (message: M) => string
): UnspokenTurnPiece | null {
  const lastUser = messages.findLastIndex(message => message.role === 'user')
  const turnIndex = messages.slice(0, lastUser + 1).filter(message => message.role === 'user').length - 1
  const bubbles = messages.slice(lastUser + 1).filter(message => message.role === 'assistant' && !message.hidden)
  const texts = bubbles.map(textOf)
  const turnText = texts.join('')
  const spoken = resolveSpokenReply(sessionId, messages)
  let offset = 0

  if (spoken?.turnSpoken !== undefined && spoken.turnIndex === turnIndex) {
    const past = indexPastPrefix(turnText, spoken.turnSpoken)

    offset = past < 0 ? turnText.length : past
  } else if (spoken) {
    // A whole-bubble mark (opening a chat, conversation mode, the Read Aloud
    // button) covers the turn through that bubble.
    const at = bubbles.findIndex(message => message.id === spoken.id)

    if (at >= 0) {
      offset = texts.slice(0, at + 1).join('').length
    } else if (spoken.turnIndex === turnIndex) {
      offset = turnText.length
    }
  }

  let start = 0

  for (const [index, bubble] of bubbles.entries()) {
    const text = texts[index] ?? ''
    const end = start + text.length
    const rest = end > offset ? text.slice(Math.max(0, offset - start)).trim() : ''

    if (rest) {
      return {
        id: bubble.id,
        pending: Boolean(bubble.pending),
        start: Math.max(offset, start),
        text: rest,
        through: turnText.slice(0, end)
      }
    }

    start = end
  }

  return null
}

/** Record read-aloud's progress: the turn is spoken through `through`. */
export function markTurnSpokenThrough(
  sessionId: string | null | undefined,
  messages: readonly SpokenReplyMessage[],
  id: string,
  through: string
): void {
  const ordinal = assistantReplyOrdinal(messages, id)

  if (ordinal < 0) {
    return
  }

  markSpokenReply(sessionId, { id, ordinal, turnIndex: assistantTurnIndex(messages, id), turnSpoken: through })
}

export function clearSpokenRepliesForTests(): void {
  lastSpokenBySession.clear()
}

/** A play that never started must not consume the turn. Only clears the anchor
 *  still pointing at the turn we marked — a newer turn's mark stays. With
 *  `previous`, the mark is rolled back to it, so the pieces of the turn already
 *  spoken stay spoken. */
export function releaseUnplayedSpokenReply(
  sessionId: string | null | undefined,
  marked: SpokenReplyAnchor | null,
  previous: SpokenReplyAnchor | null = null
): void {
  if (!marked) {
    return
  }

  const current = spokenReplyOf(sessionId)

  if (!current) {
    return
  }

  const sameId = current.id === marked.id
  const sameTurn = marked.turnIndex !== undefined && marked.turnIndex >= 0 && current.turnIndex === marked.turnIndex

  if (sameId || sameTurn) {
    if (previous) {
      markSpokenReply(sessionId, previous)
    } else {
      lastSpokenBySession.delete(sessionKey(sessionId))
    }
  }
}
