import type { TranscriptDocument, TranscriptTurn } from './transcript-document'

type Call = NonNullable<TranscriptTurn['tool_calls']>[number]
interface Occurrence { turn: number; call: number; value: Call }

// Match the producer's coalesce_tool_call_id: call_id is authoritative for
// Responses items, and the part before | is the provider's pairing key.
function callId(call: Call): string | undefined {
  const raw = call.call_id?.trim() || call.id?.trim()

  return raw ? raw.split('|', 1)[0].trim() || raw : undefined
}

function add<T>(map: Map<string, T[]>, key: string | undefined, value: T): void {
  if (!key) {return}
  const entries = map.get(key)

  if (entries) {entries.push(value)}
  else {map.set(key, [value])}
}

function hasCompetingResult(match: Occurrence, m: TranscriptTurn, idOnlyCounts: Map<string, number>, resultsByUid: Map<string, number[]>): boolean {
  const id = callId(match.value)
  const uid = match.value.uid
  const otherById = id ? (idOnlyCounts.get(id) ?? 0) > (!m.tool_call_uid && m.tool_call_id === id ? 1 : 0) : false
  const otherByUid = uid ? (resultsByUid.get(uid)?.length ?? 0) > (m.tool_call_uid === uid ? 1 : 0) : false

  return otherById || otherByUid
}

function associate(doc: TranscriptDocument): { labels: Array<string | undefined>; owners: Array<number | undefined> } {
  const callsByUid = new Map<string, Occurrence[]>()
  const callsById = new Map<string, Occurrence[]>()
  const resultsByUid = new Map<string, number[]>()
  const resultsById = new Map<string, number[]>()
  const idOnlyCounts = new Map<string, number>()
  const labels: Array<string | undefined> = doc.messages.map(() => undefined)
  const owners: Array<number | undefined> = doc.messages.map(() => undefined)
  doc.messages.forEach((m, turn) => {
    if (m.role === 'assistant' && m.tool_calls?.length) {
      labels[turn] = `Assistant turn ${turn + 1}`
      m.tool_calls.forEach((value, call) => {
        const occurrence = { turn, call, value }
        add(callsByUid, value.uid, occurrence)
        add(callsById, callId(value), occurrence)
      })
    }

    if (m.role === 'tool') {
      add(resultsByUid, m.tool_call_uid, turn)
      add(resultsById, m.tool_call_id, turn)

      if (m.tool_call_id && !m.tool_call_uid) {idOnlyCounts.set(m.tool_call_id, (idOnlyCounts.get(m.tool_call_id) ?? 0) + 1)}
    }
  })
  doc.messages.forEach((m, turn) => {
    if (m.role !== 'tool') {return}
    let match: Occurrence | undefined

    if (m.tool_call_uid) {
      const candidates = callsByUid.get(m.tool_call_uid)

      if (candidates?.length === 1 && resultsByUid.get(m.tool_call_uid)?.length === 1) {match = candidates[0]}
    } else if (m.tool_call_id) {
      const candidates = callsById.get(m.tool_call_id)

      if (candidates?.length === 1 && resultsById.get(m.tool_call_id)?.length === 1) {match = candidates[0]}
    }

    // Conflicting recorded identities, forward references and ambiguity never associate.
    if (match && (match.turn >= turn || (m.tool_call_id && m.tool_call_id !== callId(match.value)) ||
      (match.value.uid && callsByUid.get(match.value.uid)?.length !== 1))) {match = undefined}

    if (match && hasCompetingResult(match, m, idOnlyCounts, resultsByUid)) {match = undefined}

    owners[turn] = match?.turn
    labels[turn] = match ? `Result for assistant turn ${match.turn + 1}, call ${match.call + 1}` : 'Unassociated tool result (original position)'
  })

  return { labels, owners }
}

/** Labels and contiguous activity groups share one association pass; stored order is unchanged. */
export function transcriptLayout(doc: TranscriptDocument): {
  associations: Array<string | undefined>
  groups: Array<{ index: number; results: number[] }>
} {
  const { labels, owners } = associate(doc)
  const groups: Array<{ index: number; results: number[] }> = []

  for (let index = 0; index < doc.messages.length; index++) {
    const results: number[] = []

    if (doc.messages[index].role === 'assistant') {
      while (index + results.length + 1 < doc.messages.length) {
        const next = index + results.length + 1

        if (owners[next] !== index) {break}
        results.push(next)
      }
    }

    groups.push({ index, results })
    index += results.length
  }

  return { associations: labels, groups }
}
