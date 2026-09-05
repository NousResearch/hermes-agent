import type { GroupChat, GroupMember, GroupMessage, GroupMessageAuthor, HostedMessageIdentity } from './types'

export const GROUP_CHAT_SYNC_TEXT_CHARS = 1200

export function groupChatSyncSequence(entry: Pick<GroupMessage, 'seq'> | null | undefined) {
  const seq = Number(entry?.seq)

  return Number.isSafeInteger(seq) && seq > 0 ? seq : null
}

/** ID aliases and sequence are corroborating fields, not interchangeable winners. */
export function groupMessageAnchors(entry: GroupMessage) {
  const ids = [entry?.eventId, entry?.id].filter(Boolean).map(id => `event:${String(id)}`)
  const seq = groupChatSyncSequence(entry)

  return [...new Set(ids), ...(seq !== null ? [`seq:${seq}`] : [])]
}

function explicitThread(entry: GroupMessage) {
  return entry.thread && !/^legacy(?:-\d+)?$/.test(entry.thread) ? entry.thread : undefined
}

/** Missing legacy fields can be filled; contradictory fields cannot lend an actor. */
export function compatibleGroupMessageCopies(left: GroupMessage, right: GroupMessage) {
  const leftAnchors = groupMessageAnchors(left)
  const rightAnchors = groupMessageAnchors(right)
  const ids = new Set([...leftAnchors, ...rightAnchors].filter(key => key.startsWith('event:')))
  const sequences = new Set([...leftAnchors, ...rightAnchors].filter(key => key.startsWith('seq:')))

  if (ids.size > 1 || sequences.size > 1) {
    return false
  }

  if (left.roomId && right.roomId && left.roomId !== right.roomId) {
    return false
  }

  if ((left.roomId || right.roomId) && left.from?.kind !== right.from?.kind) {
    return false
  }

  const hosted =
    ((left.roomId || right.roomId) && sequences.size > 0) ||
    left.from?.hostedIdentity ||
    right.from?.hostedIdentity ||
    left.from?.hostedIdentityEvidence ||
    right.from?.hostedIdentityEvidence

  const sharedAnchor = leftAnchors.some(key => rightAnchors.includes(key))

  if (!sharedAnchor && (hosted || (leftAnchors.length && rightAnchors.length))) {
    return false
  }

  if (!hosted) {
    return true
  }

  if (left.from?.kind !== right.from?.kind) {
    return false
  }

  const leftText = String(left.text || '')
  const rightText = String(right.text || '')
  const shorter = leftText.length < rightText.length ? leftText : rightText
  const longer = leftText.length < rightText.length ? rightText : leftText
  const shorterEntry = leftText.length < rightText.length ? left : right
  const longerEntry = leftText.length < rightText.length ? right : left

  // Compact projections omit sequence. A replay at the character limit is
  // still complete and cannot borrow extra text from an unsequenced mirror.
  const compactPrefix =
    groupChatSyncSequence(shorterEntry) === null &&
    groupChatSyncSequence(longerEntry) !== null &&
    shorter.length === GROUP_CHAT_SYNC_TEXT_CHARS &&
    longer.startsWith(shorter)

  if (shorter !== longer && !compactPrefix) {
    return false
  }

  if (explicitThread(left) && explicitThread(right) && explicitThread(left) !== explicitThread(right)) {
    return false
  }

  // Compact mirrors omit attachments. Two supplied attachment sets must still
  // identify the same payload; a stored attachment's opaque ID is its identity.
  const attachmentKeys = (entry: GroupMessage) =>
    entry.images?.map(image =>
      image.attachmentId
        ? ['stored', image.attachmentId]
        : [image.kind, image.name, image.mime, image.size, image.data, image.uploadId]
    )

  return (
    !left.images?.length ||
    !right.images?.length ||
    JSON.stringify(attachmentKeys(left)) === JSON.stringify(attachmentKeys(right))
  )
}

function identityText(value: unknown): string | undefined {
  return typeof value === 'string' && value.trim() && value.length <= 128 ? value.trim() : undefined
}

export function normalizeHostedMessageIdentity(value: unknown): HostedMessageIdentity | undefined {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return undefined
  }

  const identity = value as Record<string, unknown>
  const roomId = identityText(identity.roomId)
  const memberId = identityText(identity.memberId)
  const profile = identityText(identity.profile)

  if (!roomId || !memberId || (identity.profile != null && !profile)) {
    return undefined
  }

  return { roomId, memberId, ...(profile ? { profile } : {}) }
}

function authorEvidence(author: GroupMessageAuthor, seq?: number) {
  const evidence = author?.hostedIdentityEvidence
  const conflict = author?.kind === 'member' && (evidence === 'mirror-conflict' || evidence === 'replay-conflict')

  const identity =
    author?.kind === 'member' && !conflict ? normalizeHostedMessageIdentity(author.hostedIdentity) : undefined

  let rank = 0

  if (author?.kind === 'member') {
    if (conflict) {
      rank = evidence === 'replay-conflict' ? 2 : 1
    } else if (identity) {
      rank = evidence === 'mirror' || groupChatSyncSequence({ seq }) === null ? 1 : 2
    }
  }

  return { identity, conflict, rank }
}

export function compactGroupMessageAuthor(author: GroupMessageAuthor): GroupMessageAuthor {
  const kind = author?.kind === 'member' ? 'member' : 'user'
  const { identity, conflict, rank } = authorEvidence(author)

  const evidence = conflict
    ? rank === 2
      ? 'replay-conflict'
      : 'mirror-conflict'
    : identity && author.hostedIdentityEvidence === 'mirror'
      ? 'mirror'
      : undefined

  return {
    kind,
    name: String(author?.name || (kind === 'member' ? 'Bot' : 'You')).slice(0, 128),
    ...(author?.source ? { source: String(author.source).slice(0, 128) } : {}),
    ...(identity ? { hostedIdentity: identity } : {}),
    ...(evidence ? { hostedIdentityEvidence: evidence } : {})
  }
}

/** Hosted authors reach this merge only through event ID/sequence matching. */
export function mergeGroupMessageAuthor(prior: GroupMessage, incoming: GroupMessage): GroupMessageAuthor {
  const left = authorEvidence(prior.from, prior.seq)
  const right = authorEvidence(incoming.from, incoming.seq)

  if (!compatibleGroupMessageCopies(prior, incoming) || (!left.rank && !right.rank)) {
    return incoming.from
  }

  let best = left.rank > right.rank ? left : right

  if (left.rank === right.rank) {
    const same =
      left.identity &&
      right.identity &&
      left.identity.roomId === right.identity.roomId &&
      left.identity.memberId === right.identity.memberId &&
      (!left.identity.profile || !right.identity.profile || left.identity.profile === right.identity.profile)

    best =
      same && !left.conflict && !right.conflict
        ? { ...best, identity: { ...left.identity!, ...right.identity! } }
        : { rank: best.rank, conflict: true, identity: undefined }
  }

  const { hostedIdentity: _identity, hostedIdentityEvidence: _evidence, ...author } = incoming.from

  let evidence: GroupMessageAuthor['hostedIdentityEvidence']

  if (best.conflict) {
    evidence = best.rank === 2 ? 'replay-conflict' : 'mirror-conflict'
  } else if (best.rank === 1 && (groupChatSyncSequence(prior) !== null || groupChatSyncSequence(incoming) !== null)) {
    // Learning a sequence from an identity-less copy must not promote a weak actor.
    evidence = 'mirror'
  }

  return {
    ...author,
    ...(best.identity ? { hostedIdentity: best.identity } : {}),
    ...(evidence ? { hostedIdentityEvidence: evidence } : {})
  }
}

/** Merge a corroborated copy as one event, keeping the full hosted payload. */
export function mergeGroupMessageCopies(prior: GroupMessage, incoming: GroupMessage): GroupMessage {
  const from = mergeGroupMessageAuthor(prior, incoming)

  const hosted =
    prior.from?.hostedIdentity ||
    incoming.from?.hostedIdentity ||
    prior.from?.hostedIdentityEvidence ||
    incoming.from?.hostedIdentityEvidence

  const seq = groupChatSyncSequence(prior) ?? groupChatSyncSequence(incoming)
  const priorText = String(prior.text || '')
  const incomingText = String(incoming.text || '')

  return {
    ...prior,
    ...incoming,
    from,
    ...(hosted
      ? {
          text: priorText.length > incomingText.length ? priorText : incomingText,
          thread: explicitThread(incoming) || explicitThread(prior) || incoming.thread || prior.thread
        }
      : {}),
    ...(prior.images && !incoming.images ? { images: prior.images } : {}),
    ...(prior.id && !incoming.id ? { id: prior.id } : {}),
    ...(prior.eventId && !incoming.eventId ? { eventId: prior.eventId } : {}),
    ...(seq !== null ? { seq } : {})
  }
}

/** Null means classic. A hosted but unbound author keeps its saved label and
 * no handle; profile names, display names, and device labels are not IDs. */
export function hostedMessageSpeaker(author: GroupMessageAuthor, room: GroupChat, members: GroupMember[]) {
  if (author?.kind !== 'member' || (!room.hosted && !author.hostedIdentity && !author.hostedIdentityEvidence)) {
    return null
  }

  const { identity } = authorEvidence(author)

  const candidates =
    identity && identity.roomId === room.roomId
      ? members.filter(
          member =>
            member.hostedIdentity?.roomId === identity.roomId && member.hostedIdentity.memberId === identity.memberId
        )
      : []

  const candidate = candidates.length === 1 ? candidates[0] : null

  const member =
    candidate && (!identity?.profile || candidate.hostedIdentity?.profile === identity.profile) ? candidate : null

  return {
    member,
    profile: member?.hostedIdentity?.profile || null,
    display: member?.display_name?.trim() || member?.title?.trim() || author.name || 'Bot',
    handle: member?.handle?.trim() || null
  }
}
