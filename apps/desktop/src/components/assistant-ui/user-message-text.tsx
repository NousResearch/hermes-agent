import type { FC } from 'react'
import { Fragment, useMemo } from 'react'

import { DirectiveContent } from '@/components/assistant-ui/directive-text'
import { hasRTLCharacters } from '@/lib/markdown-code'
import { cn } from '@/lib/utils'

// User messages should render the bare-minimum of markdown: backtick `code`
// spans and ``` fenced blocks. We deliberately don't pull in the full
// assistant Markdown pipeline (Streamdown + KaTeX + syntax highlighter)
// because user input rarely contains structured docs and the heavy pipeline
// adds a lot of runtime cost per bubble.
//
// Directive chips (`@file:`, `@image:`, ...) still resolve via DirectiveContent
// inside the plain-text segments.

interface FenceSegment {
  kind: 'fence'
  code: string
  lang: string | null
}

interface InlineSegment {
  kind: 'inline'
  text: string
}

interface InlineCodeSegment {
  kind: 'inline-code'
  code: string
}

interface InlineTextSegment {
  kind: 'inline-text'
  text: string
}

type TopSegment = FenceSegment | InlineSegment
type InlineNode = InlineCodeSegment | InlineTextSegment

const FENCE_RE = /```([^\n`]*)\n([\s\S]*?)```/g

// Greedy backtick run length so ``code with `backticks` inside`` works.
const INLINE_CODE_RE = /(`+)([^`\n][\s\S]*?)\1/g

function splitFences(text: string): TopSegment[] {
  const segments: TopSegment[] = []
  let cursor = 0

  for (const match of text.matchAll(FENCE_RE)) {
    const start = match.index ?? 0

    if (start > cursor) {
      segments.push({ kind: 'inline', text: text.slice(cursor, start) })
    }

    segments.push({
      kind: 'fence',
      lang: (match[1] || '').trim() || null,
      code: match[2] ?? ''
    })
    cursor = start + match[0].length
  }

  if (cursor < text.length) {
    segments.push({ kind: 'inline', text: text.slice(cursor) })
  }

  return segments
}

function splitInlineCode(text: string): InlineNode[] {
  const nodes: InlineNode[] = []
  let cursor = 0

  for (const match of text.matchAll(INLINE_CODE_RE)) {
    const start = match.index ?? 0

    if (start > cursor) {
      nodes.push({ kind: 'inline-text', text: text.slice(cursor, start) })
    }

    nodes.push({ kind: 'inline-code', code: match[2] })
    cursor = start + match[0].length
  }

  if (cursor < text.length) {
    nodes.push({ kind: 'inline-text', text: text.slice(cursor) })
  }

  return nodes
}

interface UserMessageTextProps {
  text: string
  className?: string
}

export const UserMessageText: FC<UserMessageTextProps> = ({ className, text }) => {
  const top = useMemo(() => splitFences(text), [text])

  return (
    <span className={cn('block', className)} data-slot="aui_user-message-text">
      {top.map((segment, segmentIndex) => {
        if (segment.kind === 'fence') {
          const hasRtl = hasRTLCharacters(segment.code)

          return (
            <pre
              className={cn(
                'my-1.5 max-w-full overflow-x-auto rounded-md border border-border/45 bg-[color-mix(in_srgb,currentColor_5%,transparent)] px-2.5 py-2 font-mono text-[0.86em] leading-snug',
                hasRtl && 'whitespace-pre-wrap wrap-anywhere'
              )}
              data-bidi-plaintext={hasRtl ? '' : undefined}
              data-slot="aui_user-fence"
              key={`fence-${segmentIndex}`}
            >
              <code className={cn('block', hasRtl ? 'whitespace-pre-wrap' : 'whitespace-pre')}>{segment.code}</code>
            </pre>
          )
        }

        return (
          <Fragment key={`inline-${segmentIndex}`}>
            <InlineSegmentView text={segment.text} />
          </Fragment>
        )
      })}
    </span>
  )
}

const InlineSegmentView: FC<{ text: string }> = ({ text }) => {
  const nodes = useMemo(() => splitInlineCode(text), [text])

  return (
    // styles.css bidi hook (#44150); whitespace-pre-line makes each line its own
    // UAX#9 paragraph so it resolves direction independently.
    <span className="wrap-anywhere block whitespace-pre-line" data-slot="aui_user-inline-text">
      {nodes.map((node, nodeIndex) =>
        node.kind === 'inline-code' ? (
          <code
            className="mx-px rounded bg-[color-mix(in_srgb,currentColor_8%,transparent)] px-1 py-px font-mono text-[0.92em]"
            data-slot="aui_user-inline-code"
            key={`code-${nodeIndex}`}
          >
            {node.code}
          </code>
        ) : (
          // Pass plain-text bits through DirectiveContent so @file:/@url: chips
          // still render. DirectiveContent already preserves whitespace.
          <Fragment key={`text-${nodeIndex}`}>
            <DirectiveContent text={node.text} />
          </Fragment>
        )
      )}
    </span>
  )
}
