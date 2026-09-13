/**
 * Spoken-reply identity for Desktop auto-speak / Read Aloud.
 *
 * The live assistant row id (`assistant-stream-*`, `inflight-assistant-*`) is
 * not stable: hydrate rewrites that row under its durable backend id. Keying
 * "already spoken" on id alone then re-reads the same turn at the playback-idle
 * edge. A content fingerprint would swallow a later distinct turn that happens
 * to say the same thing ("Done.").
 *
 * Prefer durable row ids, then the owning user turn. Global ordinals are only
 * a legacy fallback: refreshing/backfilling history can merge older tool
 * bubbles or move the transcript window without creating a new reply.
 */

export interface SpokenReplyAnchor {
  id: string
  ordinal: number
  rowId?: number
  text?: string
  user?: SpeechUserAnchor
  turnOrdinal?: number
}

export interface SpokenReplyMessage {
  hidden?: boolean
  id: string
  role: string
  pending?: boolean
  rowId?: number
  parts?: readonly { type: string; text?: string }[]
  attachmentRefs?: readonly string[]
}

interface SpeechUserAnchor {
  id: string
  rowId?: number
  text: string
  attachments: string
}

const normalizedText = (text: string) => text.replace(/\s+/g, ' ').trim()

const textParts = (message: SpokenReplyMessage) =>
  message.parts?.filter(part => part.type === 'text').map(part => part.text ?? '') ?? []

const messageText = (message: SpokenReplyMessage) => normalizedText(textParts(message).join(''))

function userAnchor(message: SpokenReplyMessage): SpeechUserAnchor {
  return {
    id: message.id,
    rowId: message.rowId,
    text: messageText(message),
    attachments: (message.attachmentRefs ?? []).join('\n')
  }
}

export function latestSpeechUser(messages: readonly SpokenReplyMessage[]): SpeechUserAnchor | undefined {
  const user = messages.findLast(message => message.role === 'user' && !message.hidden)

  return user ? userAnchor(user) : undefined
}

/** Hydrating an optimistic prompt is not new input. A new optimistic id or a
 * different durable row IS new input, even when its words are identical. */
export function sameSpeechUser(
  previous: SpeechUserAnchor | undefined,
  current: SpeechUserAnchor | undefined,
  messages?: readonly SpokenReplyMessage[]
): boolean {
  if (!previous || !current) {
    return previous === current
  }

  if (previous.id === current.id) {
    return true
  }

  if (previous.rowId !== undefined && current.rowId !== undefined) {
    return previous.rowId === current.rowId
  }

  return (
    previous.id.startsWith('user-') &&
    !messages?.some(message => message.id === previous.id) &&
    current.rowId !== undefined &&
    previous.text === current.text &&
    previous.attachments === current.attachments
  )
}

export function anchorSpeechReply(messages: readonly SpokenReplyMessage[], id: string): SpokenReplyAnchor {
  const index = messages.findIndex(message => message.id === id)
  const message = messages[index]
  const anchor: SpokenReplyAnchor = { id, ordinal: assistantReplyOrdinal(messages, id) }

  if (!message?.parts) {
    return anchor
  }

  const preceding = messages.slice(0, index)
  const userIndex = preceding.findLastIndex(message => message.role === 'user' && !message.hidden)

  return {
    ...anchor,
    rowId: message.rowId,
    text: messageText(message),
    ...(userIndex >= 0
      ? {
          user: userAnchor(preceding[userIndex]),
          turnOrdinal: preceding.slice(userIndex + 1).filter(message => message.role === 'assistant' && !message.hidden)
            .length
        }
      : {})
  }
}

/** Same words, different display whitespace (e.g. Markdown hard breaks) are
 * still an append-only speech source. Return only the not-yet-sent raw suffix. */
export function speechSourceDelta(previous: string, next: string): string | null {
  if (next.startsWith(previous)) {
    return next.slice(previous.length)
  }

  const prefix = normalizedText(previous)

  if (!prefix) {
    return next
  }

  if (!normalizedText(next).startsWith(prefix)) {
    return null
  }

  let length = 0

  for (const match of next.matchAll(/\s+|\S/g)) {
    if (length === 0 && /^\s+$/.test(match[0])) {
      continue
    }

    length += 1

    if (length === prefix.length) {
      return next.slice(match.index + match[0].length)
    }
  }

  return ''
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

/** Preserve the consumed prefix when REST folds narration/tool/final rows into
 * one bubble. A later text part can move behind earlier narration, but new
 * suffix text must remain unspoken. Match only at text-part boundaries. */
function mergedSpeechPrefix(spoken: SpokenReplyAnchor, message: SpokenReplyMessage): string | null {
  if (spoken.text === undefined) {
    return null
  }

  const prefix = normalizedText(spoken.text)
  const parts = textParts(message)

  for (let start = 0; start < parts.length; start += 1) {
    if (normalizedText(parts.slice(start).join('')).startsWith(prefix)) {
      return normalizedText(parts.slice(0, start).join('') + prefix)
    }
  }

  return null
}

/** Follow renderer/durable identity changes within the owning user turn,
 * without advancing how much text has actually been consumed. */
export function absorbSpokenReplyRewrite(
  spoken: SpokenReplyAnchor | null,
  messages: readonly SpokenReplyMessage[]
): SpokenReplyAnchor | null {
  if (!spoken) {
    return null
  }

  const direct = messages.find(
    message =>
      message.role === 'assistant' &&
      !message.hidden &&
      (message.id === spoken.id || (spoken.rowId !== undefined && message.rowId === spoken.rowId))
  )

  if (direct) {
    return direct.parts
      ? { ...anchorSpeechReply(messages, direct.id), text: mergedSpeechPrefix(spoken, direct) ?? spoken.text }
      : spoken
  }

  if (spoken.user) {
    const userIndex = messages.findLastIndex(
      message =>
        message.role === 'user' && !message.hidden && sameSpeechUser(spoken.user, userAnchor(message), messages)
    )

    if (userIndex < 0) {
      return spoken
    }

    const nextUser = messages.findIndex(
      (message, index) => index > userIndex && message.role === 'user' && !message.hidden
    )

    const replies = messages
      .slice(userIndex + 1, nextUser < 0 ? undefined : nextUser)
      .filter(message => message.role === 'assistant' && !message.hidden)

    const match =
      replies.find(message => mergedSpeechPrefix(spoken, message) !== null) ?? replies[spoken.turnOrdinal ?? -1]

    return match
      ? { ...anchorSpeechReply(messages, match.id), text: mergedSpeechPrefix(spoken, match) ?? spoken.text }
      : spoken
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

  return last.parts
    ? { ...anchorSpeechReply(messages, last.id), text: mergedSpeechPrefix(spoken, last) ?? spoken.text }
    : { id: last.id, ordinal }
}

export function spokenReplyOf(sessionId: string | null | undefined): SpokenReplyAnchor | null {
  return lastSpokenBySession.get(sessionKey(sessionId)) ?? null
}

export function markSpokenReply(sessionId: string | null | undefined, anchor: SpokenReplyAnchor): void {
  lastSpokenBySession.set(sessionKey(sessionId), {
    ...anchor,
    ...(anchor.text !== undefined ? { text: normalizedText(anchor.text) } : {})
  })
}

/** Latest reply's unconsumed suffix, including a final answer newly merged
 * into a bubble whose narration has already played. */
export function pendingSpeechReply(sessionId: string | null | undefined, messages: readonly SpokenReplyMessage[]) {
  const last = lastVisibleAssistant(messages)

  if (!last) {
    return null
  }

  const spoken = resolveSpokenReply(sessionId, messages)
  const raw = textParts(last).join('').trim()
  const sameReply = spoken?.id === last.id
  const spokenText = sameReply ? spoken.text : ''
  const text = sameReply ? (spokenText === undefined ? '' : (speechSourceDelta(spokenText, raw) ?? '')) : raw

  return text.trim() ? { id: last.id, pending: Boolean(last.pending), text, spokenText } : null
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

  markSpokenReply(sessionId, anchorSpeechReply(messages, id))
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

  if (next && next !== current) {
    markSpokenReply(sessionId, next)
  }

  return next
}

export function clearSpokenRepliesForTests(): void {
  lastSpokenBySession.clear()
}
