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
const MAX_SPOKEN_DURABLE_IDS_PER_SESSION = 64

const lastSpokenBySession = new Map<string, SpokenReplyAnchor>()
const spokenDurableIdsBySession = new Map<string, Set<string>>()

export function isLiveTailReplyId(id: string): boolean {
  return id.startsWith('assistant-stream-') || id.startsWith('inflight-assistant-')
}

function sessionKey(sessionId: string | null | undefined): string {
  return sessionId ?? NO_SESSION
}

function rememberSpokenDurableId(sessionId: string | null | undefined, id: string): void {
  if (isLiveTailReplyId(id)) {
    return
  }

  const key = sessionKey(sessionId)
  const ids = spokenDurableIdsBySession.get(key) ?? new Set<string>()

  // Refresh repeated notifications in the bounded insertion-ordered history.
  ids.delete(id)
  ids.add(id)

  while (ids.size > MAX_SPOKEN_DURABLE_IDS_PER_SESSION) {
    const oldest = ids.values().next().value

    if (oldest === undefined) {
      break
    }

    ids.delete(oldest)
  }

  spokenDurableIdsBySession.set(key, ids)
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
  rememberSpokenDurableId(sessionId, anchor.id)
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

  // When this mounted composer has an older snapshot, answer from the bounded
  // durable history without regressing the session's forward frontier.
  const durableIds = spokenDurableIdsBySession.get(sessionKey(sessionId))
  let latestDurable: SpokenReplyAnchor | null = null

  if (durableIds) {
    for (let index = messages.length - 1; index >= 0; index -= 1) {
      const message = messages[index]

      if (message.role !== 'assistant' || message.hidden || !durableIds.has(message.id)) {
        continue
      }

      latestDurable = { id: message.id, ordinal: assistantReplyOrdinal(messages, message.id) }

      break
    }
  }

  if (!next) {
    return latestDurable
  }

  const nextOrdinal = assistantReplyOrdinal(messages, next.id)

  if (nextOrdinal < 0) {
    return latestDurable ?? next
  }

  // Live ids remain governed by the existing ordinal rewrite frontier. For
  // durable rows, the newest consumed id present in this snapshot wins.
  if (isLiveTailReplyId(next.id) || !latestDurable || nextOrdinal >= latestDurable.ordinal) {
    return next
  }

  return latestDurable
}

export function clearSpokenRepliesForTests(): void {
  lastSpokenBySession.clear()
  spokenDurableIdsBySession.clear()
}
