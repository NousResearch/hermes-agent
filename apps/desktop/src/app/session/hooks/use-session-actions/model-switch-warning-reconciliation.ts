import { type ChatMessage, chatMessageText } from '@/lib/chat-messages'

function turnSignature(messages: ChatMessage[], user: ChatMessage): string {
  const userIndex = messages.indexOf(user)
  const nextUser = messages.findIndex((row, index) => index > userIndex && row.role === 'user')
  const end = nextUser >= 0 ? nextUser : messages.length
  const assistant = messages.find((row, index) => index > userIndex && index < end && row.role === 'assistant')
  return `${user.id}\u0000${chatMessageText(user)}\u0000${assistant ? chatMessageText(assistant) : ''}`
}

function correspondingUserIndex(owner: ChatMessage, previous: ChatMessage[], next: ChatMessage[]): number {
  const previousUsers = previous.filter(row => row.role === 'user')
  const nextUsers = next.filter(row => row.role === 'user')

  if (owner.rowId !== undefined) {
    const rowMatches = nextUsers.filter(candidate => candidate.rowId === owner.rowId)
    if (rowMatches.length === 1) {
      return next.indexOf(rowMatches[0])
    }
  }

  const signature = turnSignature(previous, owner)
  const previousSignatureCount = previousUsers.filter(
    candidate => turnSignature(previous, candidate) === signature
  ).length
  const signatureMatches = nextUsers.filter(candidate => turnSignature(next, candidate) === signature)
  if (signature && previousSignatureCount === 1 && signatureMatches.length === 1) {
    return next.indexOf(signatureMatches[0])
  }

  const idMatches = nextUsers.filter(candidate => candidate.id === owner.id)
  if (idMatches.length === 1 && previousUsers.filter(candidate => candidate.id === owner.id).length === 1) {
    return next.indexOf(idMatches[0])
  }

  const ownerText = chatMessageText(owner)
  const textMatches = nextUsers.filter(candidate => chatMessageText(candidate) === ownerText)
  if (
    ownerText &&
    textMatches.length === 1 &&
    previousUsers.filter(candidate => chatMessageText(candidate) === ownerText).length === 1
  ) {
    return next.indexOf(textMatches[0])
  }

  return -1
}

/** A durable answer still identifies the turn after history truncates its user row. */
function correspondingAssistantIndex(
  assistant: ChatMessage | undefined,
  previous: ChatMessage[],
  next: ChatMessage[]
): number {
  if (!assistant || assistant.rowId === undefined) {
    return -1
  }
  const matches = next.filter(row => row.role === 'assistant' && row.rowId === assistant.rowId)
  return matches.length === 1 &&
    previous.filter(row => row.role === 'assistant' && row.rowId === assistant.rowId).length === 1
    ? next.indexOf(matches[0])
    : -1
}

/** Preserve each local warning occurrence until a stored system row represents it. */
export function preserveLocalModelSwitchWarnings(next: ChatMessage[], previous: ChatMessage[]): ChatMessage[] {
  const represented = next.filter(message => message.role === 'system')
  // Durable rows already carried by the previous view cannot represent a later local occurrence.
  // A local warning id remains available so its own persisted replacement can claim it once below.
  const claimed = new Set(
    represented.filter(row =>
      previous.some(old => old.role === 'system' && old.id === row.id && !old.id.startsWith('model-switch-warning-'))
    )
  )
  const matched = new Map<number, number>()
  const missing: { index: number; message: ChatMessage }[] = []

  previous.forEach((message, index) => {
    if (
      message.role !== 'system' ||
      !message.id.startsWith('model-switch-warning-') ||
      !chatMessageText(message).startsWith('warning: ')
    ) {
      return
    }

    const owner = previous.slice(0, index).findLast(row => row.role === 'user')
    const ownerIndex = owner ? correspondingUserIndex(owner, previous, next) : -1
    const followingUser = previous.findIndex((row, i) => i > index && row.role === 'user')
    const turnEnd = followingUser >= 0 ? followingUser : previous.length
    const followingAssistant = previous.find((row, i) => i > index && i < turnEnd && row.role === 'assistant')
    const precedingAssistant = previous
      .slice(owner ? previous.indexOf(owner) + 1 : 0, index)
      .findLast(row => row.role === 'assistant')
    const followingAnchor = correspondingAssistantIndex(followingAssistant, previous, next)
    const precedingAnchor = correspondingAssistantIndex(precedingAssistant, previous, next)
    const sameTurn = (row: ChatMessage) => {
      if (!owner && followingAnchor < 0 && precedingAnchor < 0) {
        return true
      }
      const storedOwner = next.slice(0, next.indexOf(row)).findLast(candidate => candidate.role === 'user')
      if (storedOwner && ownerIndex >= 0 && next.indexOf(storedOwner) === ownerIndex) {
        return true
      }
      if (ownerIndex >= 0) {
        return false
      }
      const rowIndex = next.indexOf(row)
      if (followingAnchor >= 0) {
        const nextUser = next.findIndex((candidate, i) => i > followingAnchor && candidate.role === 'user')
        return (
          (rowIndex < followingAnchor &&
            next.slice(rowIndex + 1, followingAnchor).every(candidate => candidate.role === 'system')) ||
          (rowIndex >= followingAnchor && (nextUser < 0 || rowIndex < nextUser))
        )
      }
      const nextUser = next.findIndex((candidate, i) => i > precedingAnchor && candidate.role === 'user')
      return precedingAnchor >= 0 && rowIndex > precedingAnchor && (nextUser < 0 || rowIndex < nextUser)
    }
    const match =
      represented.find(row => !claimed.has(row) && sameTurn(row) && row.id === message.id) ??
      represented.find(row => !claimed.has(row) && sameTurn(row) && chatMessageText(row) === chatMessageText(message))
    if (match) {
      claimed.add(match)
      matched.set(index, next.indexOf(match))
    } else {
      missing.push({ index, message })
    }
  })

  if (!missing.length) {
    return next
  }

  const slots = new Map<number, ChatMessage[]>()
  for (const { index, message } of missing) {
    const followingUser = previous.findIndex((row, i) => i > index && row.role === 'user')
    const followingWarning = [...matched].find(([i]) => i > index && (followingUser < 0 || i < followingUser))
    let position = followingWarning?.[1]

    if (position === undefined) {
      const user = previous
        .slice(0, index)
        .reverse()
        .find(row => row.role === 'user')
      if (user) {
        const anchor = correspondingUserIndex(user, previous, next)
        if (anchor >= 0) {
          const nextUser = next.findIndex((row, i) => i > anchor && row.role === 'user')
          const end = nextUser >= 0 ? nextUser : next.length
          const assistant = next.findIndex((row, i) => i > anchor && i < end && row.role === 'assistant')
          position = assistant >= 0 ? assistant : end
        }
      }
    }

    if (position === undefined) {
      const owner = previous.slice(0, index).findLast(row => row.role === 'user')
      const nextUser = previous.findIndex((row, i) => i > index && row.role === 'user')
      const following = previous.find(
        (row, i) => i > index && (nextUser < 0 || i < nextUser) && row.role === 'assistant'
      )
      const followingIndex = correspondingAssistantIndex(following, previous, next)
      if (followingIndex >= 0) {
        position = followingIndex
      } else {
        const preceding = previous
          .slice(owner ? previous.indexOf(owner) + 1 : 0, index)
          .findLast(row => row.role === 'assistant')
        const precedingIndex = correspondingAssistantIndex(preceding, previous, next)
        if (precedingIndex >= 0) {
          position = precedingIndex + 1
        }
      }
    }

    const slot = position ?? next.length
    slots.set(slot, [...(slots.get(slot) ?? []), message])
  }

  return next.flatMap((message, index) => [...(slots.get(index) ?? []), message]).concat(slots.get(next.length) ?? [])
}
