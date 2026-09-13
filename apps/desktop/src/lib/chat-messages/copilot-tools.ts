import { assistantTextPart } from './parts'
import type { ChatMessagePart } from './types'

/**
 * Cursor / Copilot bridge often embeds tool activity as plain transcript chrome
 * (`⚙ name — {args}` / `✓ name — {result}`) instead of structured tool-call parts.
 * Lift those into DisclosureRow-backed tool-call parts; leave narration as text.
 *
 * The Cursor OpenAI proxy also emits chrome without a payload (`⚙ shell`,
 * `✓ read`) and MCP labels with a server suffix (`⚙ CallMcpTool · user-git`).
 * Both must match or those rows stay as plain text beside collapsed ones.
 */
const COPILOT_TOOL_MARK_RE =
  /(?<mark>[⚙✓✗✕❌])\uFE0F?[ \t]+(?<name>[A-Za-z][\w./:@-]{0,80})(?:[ \t]+·[ \t]+[A-Za-z][\w./:@-]{0,80})?(?:[ \t]+[—–-][ \t]*)?/g

function parseCopilotToolPayload(raw: string): unknown {
  const trimmed = raw.trim()

  if (!trimmed) {
    return ''
  }

  try {
    return JSON.parse(trimmed)
  } catch {
    /* Copilot sometimes emits Python-ish single quotes */
  }

  try {
    return JSON.parse(trimmed.replace(/'/g, '"'))
  } catch {
    return trimmed
  }
}

function isLikelyToolName(name: string): boolean {
  if (!name || name.length < 2) {
    return false
  }

  if (/^(https?|www|http)$/i.test(name)) {
    return false
  }

  return /[_-]/.test(name) || /^[a-z][a-z0-9]*$/i.test(name)
}

/** True when a line is finished tool chrome (safe to promote mid-stream). */
function isCompleteCopilotToolLine(line: string): boolean {
  const hits = findCopilotActivityHits(line)

  if (!hits.length) {
    return false
  }

  if (line.slice(0, hits[0].index).trim()) {
    return false
  }

  for (const hit of hits) {
    const payload = hit.payload

    if (!payload) {
      continue
    }

    const opens = (payload.match(/\{/g) || []).length
    const closes = (payload.match(/\}/g) || []).length

    if (opens !== closes) {
      return false
    }
  }

  return true
}

function serializeCopilotToolPart(part: Extract<ChatMessagePart, { type: 'tool-call' }>): string {
  const name = part.toolName.replace(/_/g, '-')
  const argsText =
    typeof part.argsText === 'string' && part.argsText.trim()
      ? part.argsText.trim()
      : JSON.stringify(part.args ?? {})
  let out = `⚙ ${name} — ${argsText}\n`

  if (part.result !== undefined) {
    const mark = part.isError ? '✗' : '✓'
    const resultText = typeof part.result === 'string' ? part.result : JSON.stringify(part.result)
    out += `${mark} ${name} — ${resultText}\n`
  }

  return out
}

type CopilotActivityHit = {
  index: number
  end: number
  mark: string
  name: string
  payload: string
}

function findCopilotActivityHits(line: string): CopilotActivityHit[] {
  const hits: CopilotActivityHit[] = []
  COPILOT_TOOL_MARK_RE.lastIndex = 0

  for (const match of line.matchAll(COPILOT_TOOL_MARK_RE)) {
    const mark = match.groups?.mark || ''
    const name = match.groups?.name || ''

    if (!mark || !isLikelyToolName(name) || match.index === undefined) {
      continue
    }

    hits.push({ index: match.index, end: match.index + match[0].length, mark, name, payload: '' })
  }

  for (let i = 0; i < hits.length; i += 1) {
    const start = hits[i].end
    const stop = i + 1 < hits.length ? hits[i + 1].index : line.length
    hits[i].payload = line.slice(start, stop).trim()
  }

  return hits
}

function attachOrPushCopilotTool(
  parts: ChatMessagePart[],
  opts: {
    mark: string
    name: string
    payload: unknown
    toolSeq: { n: number }
  }
): void {
  const toolName = opts.name.replace(/-/g, '_')
  const isError = opts.mark === '✗' || opts.mark === '✕' || opts.mark === '❌'
  const isResult = opts.mark === '✓' || isError

  if (!isResult) {
    const args =
      typeof opts.payload === 'object' && opts.payload && !Array.isArray(opts.payload)
        ? (() => {
            const record = opts.payload as Record<string, unknown>

            if (typeof record.preview === 'string' || typeof record.context === 'string') {
              return record
            }

            // Tool fallback headers read `preview`/`context` — keep a short
            // string so Cursor JSON args still expand under DisclosureRow.
            return { ...record, preview: JSON.stringify(record) }
          })()
        : { preview: String(opts.payload) }

    parts.push({
      type: 'tool-call',
      toolCallId: `copilot-tool:${toolName}:${opts.toolSeq.n++}`,
      toolName,
      args: args as never,
      argsText: typeof opts.payload === 'string' ? opts.payload : JSON.stringify(opts.payload)
    })

    return
  }

  for (let i = parts.length - 1; i >= 0; i -= 1) {
    const part = parts[i]

    if (part.type !== 'tool-call') {
      continue
    }

    if (part.toolName !== toolName) {
      continue
    }

    if (part.result !== undefined) {
      continue
    }

    parts[i] = {
      ...part,
      result: opts.payload,
      isError
    } as ChatMessagePart

    return
  }

  parts.push({
    type: 'tool-call',
    toolCallId: `copilot-tool:${toolName}:${opts.toolSeq.n++}`,
    toolName,
    args: {} as never,
    argsText: '',
    result: opts.payload,
    isError
  })
}

/** Parse Cursor/Copilot tool chrome out of assistant text into tool-call parts. */
export function extractCopilotToolActivity(
  text: string,
  opts?: { holdIncompleteLastLine?: boolean; toolSeq?: { n: number } }
): ChatMessagePart[] {
  if (!text || !/[⚙✓✗✕❌]/.test(text)) {
    return text ? [assistantTextPart(text)] : []
  }

  const holdIncomplete = Boolean(opts?.holdIncompleteLastLine) && !text.endsWith('\n')
  const rawLines = text.split('\n')
  let incomplete = ''
  let lines = rawLines

  if (holdIncomplete && rawLines.length) {
    const last = rawLines[rawLines.length - 1] ?? ''

    // Promote finished chrome immediately (`⚙ shell` / `✓ read — ok`). Only
    // hold mid-stream fragments (partial JSON args, half-typed mark lines).
    if (last && !isCompleteCopilotToolLine(last)) {
      incomplete = last
      lines = rawLines.slice(0, -1)
    }
  }

  const parts: ChatMessagePart[] = []
  const textBuf: string[] = []
  // Caller may pass a shared counter so multi-segment rebalance stays unique
  // (assistant-ui useResources keys on toolCallId and throws on duplicates).
  const toolSeq = opts?.toolSeq ?? { n: 0 }

  const flushText = () => {
    if (!textBuf.length) {
      return
    }

    const chunk = textBuf.join('\n').replace(/^\n+/, '').replace(/\n+$/, '')
    textBuf.length = 0

    if (chunk.trim()) {
      parts.push(assistantTextPart(chunk))
    }
  }

  for (const line of lines) {
    const hits = findCopilotActivityHits(line)

    if (!hits.length) {
      textBuf.push(line)
      continue
    }

    const lead = line.slice(0, hits[0].index)

    if (lead.trim()) {
      textBuf.push(lead.trimEnd())
    }

    flushText()

    for (const hit of hits) {
      attachOrPushCopilotTool(parts, {
        mark: hit.mark,
        name: hit.name,
        payload: parseCopilotToolPayload(hit.payload),
        toolSeq
      })
    }
  }

  flushText()

  if (incomplete) {
    const last = parts[parts.length - 1]

    if (last?.type === 'text') {
      parts[parts.length - 1] = assistantTextPart(`${last.text}\n${incomplete}`)
    } else {
      parts.push(assistantTextPart(incomplete))
    }
  }

  return parts.length ? parts : text ? [assistantTextPart(text)] : []
}

/**
 * Rewrite text / prior copilot-tool parts that contain Cursor tool chrome into
 * structured tool-call parts. Safe to call on every stream tick.
 */
export function rebalanceCopilotToolText(
  parts: ChatMessagePart[],
  opts?: { holdIncompleteLastLine?: boolean }
): ChatMessagePart[] {
  const hasActivity = parts.some(
    part =>
      (part.type === 'text' && /[⚙✓✗✕❌]/.test(part.text)) ||
      (part.type === 'tool-call' && String(part.toolCallId || '').startsWith('copilot-tool:'))
  )

  if (!hasActivity) {
    return parts
  }

  type Segment = { kind: 'other'; part: ChatMessagePart } | { kind: 'activity'; transcript: string }
  const segments: Segment[] = []
  let activityBuf = ''

  const flushActivity = () => {
    if (!activityBuf) {
      return
    }

    segments.push({ kind: 'activity', transcript: activityBuf })
    activityBuf = ''
  }

  for (const part of parts) {
    if (part.type === 'text') {
      activityBuf += part.text
      continue
    }

    if (part.type === 'tool-call' && String(part.toolCallId || '').startsWith('copilot-tool:')) {
      activityBuf += serializeCopilotToolPart(part)
      continue
    }

    flushActivity()
    segments.push({ kind: 'other', part })
  }

  flushActivity()

  const next: ChatMessagePart[] = []
  const toolSeq = { n: 0 }

  for (let i = 0; i < segments.length; i += 1) {
    const seg = segments[i]

    if (seg.kind === 'other') {
      next.push(seg.part)
      continue
    }

    const isLastActivity = !segments.slice(i + 1).some(s => s.kind === 'activity')
    next.push(
      ...extractCopilotToolActivity(seg.transcript, {
        holdIncompleteLastLine: Boolean(opts?.holdIncompleteLastLine) && isLastActivity,
        toolSeq
      })
    )
  }

  return withUniquePartToolCallIds(next)
}

/** Ensure tool-call parts in a single message have unique toolCallIds. */
function withUniquePartToolCallIds(parts: ChatMessagePart[]): ChatMessagePart[] {
  const seen = new Set<string>()
  let changed = false

  const next = parts.map((part, index) => {
    if (part.type !== 'tool-call') {
      return part
    }

    const id = part.toolCallId || `part-tool-${index}`

    if (!seen.has(id)) {
      seen.add(id)

      if (part.toolCallId) {
        return part
      }

      changed = true

      return { ...part, toolCallId: id } as ChatMessagePart
    }

    changed = true
    const uniqueId = `${id}#${index}`
    seen.add(uniqueId)

    return { ...part, toolCallId: uniqueId } as ChatMessagePart
  })

  return changed ? next : parts
}

export function isCopilotToolPart(part: ChatMessagePart): boolean {
  return part.type === 'tool-call' && String(part.toolCallId || '').startsWith('copilot-tool:')
}
