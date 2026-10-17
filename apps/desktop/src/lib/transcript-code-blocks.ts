import type { ReactNode } from 'react'

export const TRANSCRIPT_CODE_BLOCK_AREA = 'transcript.codeBlocks'

export interface TranscriptCodeBlockOrigin {
  connectionKey: string
  profile: string
  storedSessionId: string
  messageId: string
  partIndex: number
  blockIndex: number
}

export interface TranscriptCodeBlockProps {
  /** Untrusted literal source. The plugin validates its own language/schema. */
  code: string
  language: string
  readOnly: boolean
  busy: boolean
  origin: Readonly<TranscriptCodeBlockOrigin>
  /** Return this for unsupported payloads. Unclaimed/unfinished fences keep it automatically. */
  fallback: ReactNode
  /** Only call from an explicit user gesture. Resolves after composer admission. */
  submitInteraction: (prompt: string) => Promise<boolean>
}

export interface TranscriptCodeBlockContribution {
  /** Exact fence language, e.g. `my-form`. First enabled registration wins. */
  language: string
  render: (props: TranscriptCodeBlockProps) => ReactNode
}

export interface SourceFence {
  language: string
  code: string
  closed: boolean
  index: number
  startLine: number
}

/** Inspect literal top-level fences: Markdown repair must not close a live program. */
export function sourceFences(source: string): SourceFence[] {
  const fences: SourceFence[] = []
  let open: { marker: string; language: string; startLine: number; body: string[] } | null = null

  for (const [lineIndex, line] of source.replace(/\r\n/g, '\n').split('\n').entries()) {
    if (!open) {
      const match = /^ {0,3}(`{3,}|~{3,})([^\r\n]*)$/.exec(line)

      if (match && !(match[1][0] === '`' && match[2].includes('`'))) {
        open = { marker: match[1], language: match[2].trim(), startLine: lineIndex + 1, body: [] }
      }

      continue
    }

    const closer = /^ {0,3}(`{3,}|~{3,})\s*$/.exec(line)

    if (closer && closer[1][0] === open.marker[0] && closer[1].length >= open.marker.length) {
      fences.push({ ...open, code: open.body.join('\n'), closed: true, index: fences.length })
      open = null
    } else {
      open.body.push(line)
    }
  }

  if (open) {
    fences.push({ ...open, code: open.body.join('\n'), closed: false, index: fences.length })
  }

  return fences
}
