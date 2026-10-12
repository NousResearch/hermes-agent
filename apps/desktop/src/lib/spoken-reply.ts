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
}

export interface SpokenReplyMessage {
  hidden?: boolean
  id: string
  role: string
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

    return { id: last.id, ordinal, turnIndex: spoken.turnIndex }
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

export function clearSpokenRepliesForTests(): void {
  lastSpokenBySession.clear()
}

/** A play that never started must not consume the turn. Only clears the anchor
 *  still pointing at the turn we marked — a newer turn's mark stays. */
export function releaseUnplayedSpokenReply(
  sessionId: string | null | undefined,
  marked: SpokenReplyAnchor | null
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
    lastSpokenBySession.delete(sessionKey(sessionId))
  }
}
