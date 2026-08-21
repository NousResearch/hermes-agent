/**
 * Spoken-reply identity for Desktop auto-speak / Read Aloud.
 *
 * The live assistant row id (`assistant-stream-*`, `inflight-assistant-*`) is
 * not stable: hydrate rewrites that row under its durable backend id. Keying
 * "already spoken" on id alone then re-reads the same turn at the playback-idle
 * edge. A content fingerprint would swallow a later distinct turn that happens
 * to say the same thing ("Done.").
 *
 * Anchor on the assistant-role ordinal (nth visible assistant bubble). The
 * rewrite keeps that slot; a new turn appends and the ordinal moves.
 */

export interface SpokenReplyAnchor {
  id: string
  ordinal: number
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

/** If a spoken live-tail row vanished and the same assistant slot now has a
 *  durable id, migrate the anchor. Leave durable ids and later turns alone. */
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

  markSpokenReply(sessionId, { id, ordinal })
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
