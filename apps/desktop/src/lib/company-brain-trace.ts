export interface CompanyBrainCorrection {
  after: string
  before?: string
}

export interface CompanyBrainTrace {
  changedAfterCorrection?: CompanyBrainCorrection[]
  conflictWinner?: string
  sources: string[]
}

export type CompanyBrainTraceField = 'source' | 'conflictWinner' | 'changedAfterCorrection'

const TRACE_HEADING_RE = /^#{2,6}\s*Company Brain trace\s*$/im
const TRACE_BLOCK_RE = /(?:^|\n)#{2,6}\s*Company Brain trace\s*\n(?<body>[\s\S]*?)(?=\n#{1,6}\s|\n?$)/i
const BULLET_RE = /^\s*[-*]\s*(?<label>Source|Sources|Conflict winner|Changed after correction)\s*:\s*(?<value>.+?)\s*$/i

function cleanLine(value: string): string {
  return value.replace(/\s+/g, ' ').trim()
}

function unique(values: string[]): string[] {
  return [...new Set(values.map(cleanLine).filter(Boolean))]
}

function parseCorrection(value: string): CompanyBrainCorrection | null {
  const cleaned = cleanLine(value)
  if (!cleaned) return null

  const arrow = cleaned.match(/^(?<before>.+?)\s*(?:->|→)\s*(?<after>.+)$/)
  if (arrow?.groups?.after) {
    return {
      after: cleanLine(arrow.groups.after),
      before: cleanLine(arrow.groups.before)
    }
  }

  return { after: cleaned }
}

export function parseCompanyBrainTrace(markdown: string): CompanyBrainTrace | null {
  if (!TRACE_HEADING_RE.test(markdown)) {
    return null
  }

  const body = markdown.match(TRACE_BLOCK_RE)?.groups?.body ?? ''
  const sources: string[] = []
  const corrections: CompanyBrainCorrection[] = []
  let conflictWinner = ''

  for (const line of body.split('\n')) {
    const match = line.match(BULLET_RE)
    if (!match?.groups) continue

    const label = match.groups.label.toLowerCase()
    const value = cleanLine(match.groups.value)

    if (!value) continue

    if (label === 'source' || label === 'sources') {
      sources.push(...value.split(/\s*;\s*/))
    } else if (label === 'conflict winner') {
      conflictWinner = value
    } else if (label === 'changed after correction') {
      const correction = parseCorrection(value)
      if (correction) corrections.push(correction)
    }
  }

  return {
    changedAfterCorrection: corrections,
    conflictWinner: conflictWinner || undefined,
    sources: unique(sources)
  }
}

export function missingCompanyBrainTraceFields(trace: CompanyBrainTrace | null): CompanyBrainTraceField[] {
  if (!trace) return ['source', 'conflictWinner', 'changedAfterCorrection']

  const missing: CompanyBrainTraceField[] = []
  if (trace.sources.length === 0) missing.push('source')
  if (!trace.conflictWinner) missing.push('conflictWinner')
  if (!trace.changedAfterCorrection || trace.changedAfterCorrection.length === 0) missing.push('changedAfterCorrection')

  return missing
}

export function formatCompanyBrainTrace(trace: CompanyBrainTrace): string {
  const lines = ['### Company Brain trace']

  for (const source of unique(trace.sources)) {
    lines.push(`- Source: ${source}`)
  }

  lines.push(`- Conflict winner: ${cleanLine(trace.conflictWinner || 'none')}`)

  const corrections = trace.changedAfterCorrection?.length ? trace.changedAfterCorrection : [{ after: 'none' }]
  for (const correction of corrections) {
    const value = correction.before ? `${correction.before} → ${correction.after}` : correction.after
    lines.push(`- Changed after correction: ${cleanLine(value)}`)
  }

  return `${lines.join('\n')}\n`
}

export function answerHasCompleteCompanyBrainTrace(markdown: string): boolean {
  return missingCompanyBrainTraceFields(parseCompanyBrainTrace(markdown)).length === 0
}
