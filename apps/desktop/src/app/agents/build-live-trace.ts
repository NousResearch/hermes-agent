import type { LiveTurn } from '@/store/live-turn'
import type { SubagentProgress } from '@/store/subagents'
import type { TraceDoc, TraceSpan, TraceSpanStatus } from '@/store/trace'

const TERMINAL_SUB = new Set(['completed', 'failed', 'interrupted'])

/**
 * Stitch the live turn + subagent stream into a TraceDoc (client time, epoch
 * seconds) in the SAME shape the DB trace produces, so the waterfall renders the
 * in-flight turn with no separate code path. LLM spans are the gaps between tool
 * calls; subagents nest under the `delegate_task` tool span that spawned them.
 *
 * `live=false` finalizes the snapshot (running → ok) so a settled turn stops
 * pulsing. Returns null when there's nothing in flight to draw.
 */
export function buildLiveTrace(
  turn: LiveTurn | undefined,
  subs: SubagentProgress[],
  nowMs: number,
  rootLabel?: string,
  live = true
): null | TraceDoc {
  const tools = turn?.tools ?? []

  // Robust to opening/reloading mid-turn (when message.start was never captured
  // in this renderer): build from whatever live data exists — the captured turn,
  // its tools, or streamed subagents.
  if (!turn?.busy && tools.length === 0 && subs.length === 0) {
    return null
  }

  const nowSec = nowMs / 1000

  const startCandidates = [turn?.turnStart, ...subs.map(s => s.startedAt)].filter(
    (n): n is number => typeof n === 'number'
  )

  const startSec = (startCandidates.length ? Math.min(...startCandidates) : nowMs) / 1000
  const rootId = 'live:root'

  const spans: TraceSpan[] = [
    {
      id: rootId,
      parentId: null,
      name: rootLabel || 'Current turn',
      kind: 'AGENT',
      start: startSec,
      end: nowSec,
      duration: Math.max(0, nowSec - startSec),
      status: live ? 'running' : 'ok',
      sessionId: null,
      attributes: {}
    }
  ]

  const toolSpanIds = new Set<string>()
  const sortedTools = [...tools].sort((a, b) => a.start - b.start)
  let prev = startSec
  let llmIdx = 0

  const pushLlm = (start: number, end: number, running: boolean, output?: string) => {
    if (end - start <= 0.05) {
      return
    }

    const text = output?.trim()

    spans.push({
      id: `live:llm:${llmIdx++}`,
      parentId: rootId,
      name: 'llm',
      kind: 'LLM',
      start,
      end,
      duration: end - start,
      status: running ? 'running' : 'ok',
      sessionId: null,
      attributes: text ? { 'output.value': text } : {}
    })
  }

  for (const t of sortedTools) {
    const ts = t.start / 1000
    const te = (t.end ?? nowMs) / 1000
    pushLlm(prev, ts, false)
    const spanId = `live:tool:${t.id}`
    toolSpanIds.add(spanId)
    spans.push({
      id: spanId,
      parentId: rootId,
      name: t.name,
      kind: 'TOOL',
      start: ts,
      end: Math.max(te, ts),
      duration: Math.max(0, te - ts),
      status: t.status,
      sessionId: null,
      attributes: { 'tool.name': t.name }
    })
    prev = Math.max(prev, te)
  }

  // Trailing llm = the model's response after the last tool. Show it ONLY when
  // there were tools (it's a distinct segment, and it's what streams/grows during
  // "reporting back"). A pure no-tool turn skips it — the root already is the
  // response, so a lone "llm" child would just duplicate it.
  if (sortedTools.length > 0) {
    pushLlm(prev, nowSec, live, turn?.replyText)
  } else if (turn?.replyText.trim()) {
    // No-tool turn: the root IS the response, so don't add a redundant llm row —
    // instead hang the streamed reply on the root so selecting it shows the text.
    spans[0].attributes = { ...spans[0].attributes, 'output.value': turn.replyText.trim() }
  }

  // Nest each subagent under: another subagent (parentId), else the delegate_task
  // tool span that spawned it. Native subagents don't carry the tool id, so match
  // by the nearest delegate span that started at/before the subagent.
  const subIds = new Set(subs.map(s => s.id))

  const delegateSpans = sortedTools
    .filter(t => t.name === 'delegate_task')
    .map(t => ({ id: `live:tool:${t.id}`, startMs: t.start }))

  for (const s of subs) {
    const start = s.startedAt / 1000
    const terminal = TERMINAL_SUB.has(s.status)
    const end = terminal && s.durationSeconds ? start + s.durationSeconds : nowSec

    const status: TraceSpanStatus =
      s.status === 'failed' || s.status === 'interrupted' ? 'error' : terminal ? 'ok' : 'running'

    let parentId = rootId

    if (s.parentId && subIds.has(s.parentId)) {
      parentId = `live:sub:${s.parentId}`
    } else {
      const idMatch = /^delegate-tool:(.+):\d+$/.exec(s.id)

      if (idMatch && toolSpanIds.has(`live:tool:${idMatch[1]}`)) {
        parentId = `live:tool:${idMatch[1]}`
      } else {
        let best: null | { id: string; startMs: number } = null

        for (const d of delegateSpans) {
          if (d.startMs <= s.startedAt + 1000 && (!best || d.startMs > best.startMs)) {
            best = d
          }
        }

        if (best) {
          parentId = best.id
        }
      }
    }

    spans.push({
      id: `live:sub:${s.id}`,
      parentId,
      name: s.goal || 'subagent',
      kind: 'AGENT',
      start,
      end: Math.max(end, start),
      duration: Math.max(0, end - start),
      status,
      sessionId: s.sessionId ?? null,
      attributes: {
        'llm.model_name': s.model,
        'llm.token_count.completion': s.outputTokens,
        'llm.token_count.prompt': s.inputTokens
      }
    })
  }

  // Finalized snapshot (turn settled): no span should keep a 'running' status,
  // or its bar pulses forever.
  if (!live) {
    for (const sp of spans) {
      if (sp.status === 'running') {
        sp.status = 'ok'
      }
    }
  }

  return {
    traceId: 'live',
    rootSessionId: 'live',
    rootSpanId: rootId,
    start: startSec,
    end: nowSec,
    duration: Math.max(0, nowSec - startSec),
    metadata: { live: true },
    spans
  }
}
