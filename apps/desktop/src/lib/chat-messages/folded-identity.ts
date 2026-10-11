import { withUniqueToolCallIds } from './tool-parts'
import type { ChatMessage, ChatMessagePart } from './types'

/** Membership, not an interval: database ids can have arbitrarily large gaps. */
export function representedRowIds(message: ChatMessage): number[] {
  return [
    ...new Set([
      ...(message.sourceRowIds ?? []),
      ...(message.rowId === undefined ? [] : [message.rowId]),
      ...message.parts.flatMap(part => (part.sourceRowId === undefined ? [] : [part.sourceRowId]))
    ])
  ]
}

function shallowEqual(left: object, right: object): boolean {
  const keys = new Set([...Object.keys(left), ...Object.keys(right)])

  return [...keys].every(key => (left as Record<string, unknown>)[key] === (right as Record<string, unknown>)[key])
}

function unchangedRepresentation(previous: ChatMessage, merged: ChatMessage): boolean {
  const { parts: oldParts, sourceRowIds: oldRows = [], ...oldFields } = previous
  const { parts, sourceRowIds: rows = [], ...fields } = merged

  return (
    shallowEqual(oldFields, fields) &&
    oldRows.length === rows.length &&
    oldRows.every((row, index) => row === rows[index]) &&
    oldParts.length === parts.length &&
    oldParts.every((part, index) => shallowEqual(part, parts[index]))
  )
}

/** Recover old hydrated reasoning provenance only from a co-located call or
 * the cache's own durable first row. Prose/timestamps alone are not identity.
 * Unanchored live reasoning is deliberately retained. */
function legacyReasoningRow(message: ChatMessage, part: ChatMessagePart, messages: ChatMessage[]): number | undefined {
  if (part.type !== 'reasoning' || part.sourceRowId !== undefined || part.timestamp === undefined) {
    return undefined
  }

  const index = message.parts.indexOf(part)
  const next = message.parts[index + 1]
  const candidates = new Set<number>()

  if (next?.type === 'tool-call' && next.timestamp === part.timestamp) {
    for (const candidate of messages.flatMap(item => item.parts)) {
      if (
        candidate.type === 'tool-call' &&
        !candidate.unpairedStoredToolResult &&
        candidate.toolCallId === next.toolCallId &&
        candidate.timestamp === part.timestamp &&
        candidate.sourceRowId !== undefined
      ) {
        if (
          messages.some(item =>
            item.parts.some(
              reasoning =>
                reasoning.type === 'reasoning' &&
                reasoning.sourceRowId === candidate.sourceRowId &&
                reasoning.timestamp === part.timestamp &&
                reasoning.text === part.text
            )
          )
        ) {
          candidates.add(candidate.sourceRowId)
        }
      }
    }
  }

  // Pre-change hydration used this exact id shape and retained the first row.
  if (
    index === 0 &&
    /^\d+(?:\.\d+)?-\d+-assistant$/.test(message.id) &&
    message.timestamp === part.timestamp &&
    message.rowId !== undefined
  ) {
    candidates.add(message.rowId)
  }

  return candidates.size === 1 ? candidates.values().next().value : undefined
}

function mergeRepresentations(messages: ChatMessage[]): ChatMessage {
  const latest = messages[messages.length - 1]
  const rows = [...new Set(messages.flatMap(representedRowIds))].sort((a, b) => a - b)
  const first = messages.reduce((a, b) => ((a.rowId ?? Infinity) <= (b.rowId ?? Infinity) ? a : b))
  const parts: ChatMessagePart[] = []
  const byOccurrence = new Map<string, number>()
  const byTool = new Map<string, number>()
  const rowOrder = new Map<number, number>()

  for (const message of messages) {
    const ordinals = new Map<string, number>()
    const hasSourceText = message.parts.some(part => part.type === 'text' && part.sourceRowId !== undefined)

    for (const part of message.parts) {
      // Older cached live completions have a durable reply id but no per-part
      // source metadata. That id identifies their text, not their tool rounds.
      const row =
        part.sourceRowId ??
        legacyReasoningRow(message, part, messages) ??
        (part.type === 'text' && !hasSourceText ? message.rowId : undefined)

      const group = row === undefined ? undefined : `${row}:${part.type}`
      const ordinal = group === undefined ? 0 : (ordinals.get(group) ?? 0)

      if (group !== undefined) {
        ordinals.set(group, ordinal + 1)
      }

      const key = group === undefined ? undefined : `${group}:${ordinal}`
      const resultKey = part.resultRowId === undefined ? undefined : `result:${part.resultRowId}`
      const toolIndex = part.type === 'tool-call' && part.toolCallId ? byTool.get(part.toolCallId) : undefined
      const candidate = toolIndex === undefined ? undefined : parts[toolIndex]
      // A reused provider call id is not an occurrence identity. Bridge an
      // unanchored legacy call only when its name, arguments and timestamp
      // independently agree with the stored call.

      const matchingLegacyTool =
        candidate?.type === 'tool-call' &&
        part.type === 'tool-call' &&
        (row === undefined || candidate.sourceRowId === undefined) &&
        candidate.toolName === part.toolName &&
        candidate.timestamp !== undefined &&
        candidate.timestamp === part.timestamp &&
        JSON.stringify(candidate.args ?? {}) === JSON.stringify(part.args ?? {})

      const index =
        (key === undefined ? undefined : byOccurrence.get(key)) ??
        (resultKey === undefined ? undefined : byOccurrence.get(resultKey)) ??
        (matchingLegacyTool ? toolIndex : undefined)

      const value = row === undefined ? part : { ...part, sourceRowId: row }
      const target = index ?? parts.length

      if (index === undefined) {
        parts.push(value)
      } else {
        const existing = parts[index]
        // A page beginning at the result cannot describe the original call.
        // Keep its name/arguments/source while accepting the fresh result.
        parts[index] =
          value.unpairedStoredToolResult && !existing.unpairedStoredToolResult
            ? ({
                ...value,
                ...existing,
                unpairedStoredToolResult: existing.unpairedStoredToolResult,
                result: 'result' in value ? value.result : undefined
              } as ChatMessagePart)
            : ({ ...existing, ...value, unpairedStoredToolResult: value.unpairedStoredToolResult } as ChatMessagePart)
      }

      if (row !== undefined && !part.unpairedStoredToolResult) {
        rowOrder.set(target, message.parts.indexOf(part))
      }

      if (key !== undefined) {
        byOccurrence.set(key, target)
      }

      if (resultKey !== undefined) {
        byOccurrence.set(resultKey, target)
      }

      if (part.type === 'tool-call' && part.toolCallId) {
        byTool.set(part.toolCallId, target)
      }
    }
  }

  // A result-only page's insertion order is not the call row's segment order.
  const order = new Map(parts.map((part, index) => [part, rowOrder.get(index) ?? index]))
  parts.sort((a, b) => (a.sourceRowId ?? Infinity) - (b.sourceRowId ?? Infinity) || order.get(a)! - order.get(b)!)

  const merged = withUniqueToolCallIds([
    {
      ...first,
      ...latest,
      id: first.id,
      rowId: first.rowId,
      timestamp: first.timestamp,
      durableComplete: messages.some(message => message.durableComplete) || latest.durableComplete,
      sourceRowIds: rows,
      serverRowSpan: Math.max(rows.length, ...messages.map(message => message.serverRowSpan ?? 1)),
      parts
    }
  ])[0]

  return messages.find(message => unchangedRepresentation(message, merged)) ?? merged
}

/** Reconcile both sides BEFORE choosing a seam, including transitive stale
 * cache copies. Only overlapping durable assistant occurrences can coalesce;
 * equal prose and row-count arithmetic are never identity evidence. */
export function reconcileFoldedMessages(
  previous: ChatMessage[],
  incoming: ChatMessage[]
): [ChatMessage[], ChatMessage[]] {
  const groups: ChatMessage[][] = []
  const byRow = new Map<number, ChatMessage[]>()

  for (const message of [...previous, ...incoming]) {
    if (message.role !== 'assistant') {
      continue
    }

    const rows = representedRowIds(message)
    const matches = new Set(rows.flatMap(row => (byRow.has(row) ? [byRow.get(row)!] : [])))
    const group = matches.values().next().value ?? []

    if (!matches.size) {
      groups.push(group)
    }

    for (const match of matches) {
      if (match === group) {
        continue
      }

      group.push(...match)
      groups.splice(groups.indexOf(match), 1)

      for (const item of match) {
        for (const row of representedRowIds(item)) {
          byRow.set(row, group)
        }
      }
    }

    if (!group.includes(message)) {
      group.push(message)
    }

    for (const row of rows) {
      byRow.set(row, group)
    }
  }

  const replacements = new Map<ChatMessage, ChatMessage>()

  for (const group of groups) {
    if (
      group.length < 2 ||
      !group.some(message => (message.serverRowSpan ?? 1) > 1 || representedRowIds(message).length > 1)
    ) {
      continue
    }

    const merged = mergeRepresentations(group)

    for (const message of group) {
      replacements.set(message, merged)
    }
  }

  const replace = (messages: ChatMessage[]) => {
    if (!messages.some(message => replacements.has(message))) {
      return messages
    }

    const seen = new Set<ChatMessage>()

    const replaced = messages.flatMap(message => {
      const replacement = replacements.get(message) ?? message

      if (seen.has(replacement)) {
        return []
      }

      seen.add(replacement)

      return [replacement]
    })

    return replaced.length === messages.length && replaced.every((message, index) => message === messages[index])
      ? messages
      : replaced
  }

  return [replace(previous), replace(incoming)]
}
