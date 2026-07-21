'use client'

import type { SyntaxHighlighterProps } from '@assistant-ui/react-streamdown'
import { type CSSProperties, type FC, useEffect, useMemo, useRef, useState } from 'react'

import {
  CodeCard,
  CodeCardBody,
  CodeCardHeader,
  CodeCardIcon,
  CodeCardSubtitle,
  CodeCardTitle
} from '@/components/chat/code-card'
import { ExpandableBlock } from '@/components/chat/expandable-block'
import { CopyButton } from '@/components/ui/copy-button'
import { useI18n } from '@/i18n'
import { codiconForLanguage, isLikelyProseCodeBlock, sanitizeLanguageTag } from '@/lib/markdown-code'

import type { ShikiWorkerToken } from './shiki-worker'
import { startShikiHighlight } from './shiki-worker-client'

/**
 * Streamdown's code adapter renders header + body as inline siblings, so we
 * own the wrapping `<CodeCard>` here and neutralize the upstream
 * `data-streamdown="code-block"` chrome from styles.css. Anything that wants
 * a card-shaped code surface should compose `CodeCard*` directly.
 *
 * Shiki runs in a shared, lazily-created worker so all bundled languages remain
 * available without putting grammar work on the renderer thread. Theme switches
 * follow the document `color-scheme` via `defaultColor="light-dark()"`.
 */
interface HermesSyntaxHighlighterProps extends SyntaxHighlighterProps {
  defer?: boolean
}

// `github-dark-dimmed` is GitHub's lower-contrast dark palette — the vivid
// `github-dark-default` tokens read harsh at our small code size. Shared by the
// inline diff renderer too (see diff-lines.tsx) so code + diffs match.
export const SHIKI_THEME = { dark: 'github-dark-dimmed', light: 'github-light-default' } as const

/**
 * `github-light-default` colors comments `#6e7781` (~4.2:1 against the code
 * card background) — borderline unreadable at our 11px code size, and worst of
 * all for shell snippets where a single `#` turns the rest of the line into one
 * long comment span. Remap light-mode comments to GitHub's darker muted gray
 * (`#57606a`, ~6.4:1). Dark mode (`#8b949e`, ~6.1:1) already reads fine, so we
 * leave it untouched. Keyed per theme name so the bump only applies in light.
 */
const SHIKI_COLOR_REPLACEMENTS: Record<string, Record<string, string>> = {
  'github-light-default': { '#6e7781': '#57606a' }
}

const MAX_HIGHLIGHT_CHARS = 150_000
const MAX_HIGHLIGHT_LINES = 3_000
const CHUNK_LINES = 200
const EST_LINE_PX = 16

function useNearViewportHighlight(enabled: boolean) {
  const ref = useRef<HTMLDivElement | null>(null)
  const [nearViewport, setNearViewport] = useState(() => typeof IntersectionObserver === 'undefined')

  useEffect(() => {
    const target = ref.current

    if (!enabled || nearViewport || !target) {
      return
    }

    if (typeof IntersectionObserver === 'undefined') {
      setNearViewport(true)

      return
    }

    const observer = new IntersectionObserver(
      entries => {
        if (entries.some(entry => entry.isIntersecting)) {
          setNearViewport(true)
          observer.disconnect()
        }
      },
      { rootMargin: '300px 0px' }
    )

    observer.observe(target)

    return () => observer.disconnect()
  }, [enabled, nearViewport])

  return { nearViewport, ref }
}

function useWorkerHighlight(code: string, enabled: boolean, language: string) {
  const [tokens, setTokens] = useState<ShikiWorkerToken[][] | null>(null)

  useEffect(() => {
    setTokens(null)

    if (!enabled) {
      return
    }

    const job = startShikiHighlight(code, language)
    let active = true

    void job.promise
      .then(result => {
        if (active) {
          setTokens(result)
        }
      })
      .catch(() => {
        // Plain code is the intentional fallback when a grammar/worker fails.
      })

    return () => {
      active = false
      job.dispose()
    }
  }, [code, enabled, language])

  return tokens
}

function tokenStyle(style: Record<string, string> | undefined): CSSProperties | undefined {
  if (!style) {
    return undefined
  }

  return Object.fromEntries(
    Object.entries(style).map(([property, value]) => {
      // Dual-theme Shiki tokens put the light color in both `color:
      // light-dark(...)` and `--shiki-light`. Never rewrite `--shiki-dark`:
      // color replacements are intentionally theme-scoped.
      const adjusted =
        property === 'color' || property === '--shiki-light'
          ? Object.entries(SHIKI_COLOR_REPLACEMENTS['github-light-default']).reduce(
              (next, [from, to]) => next.replaceAll(from.toLowerCase(), to),
              value.toLowerCase()
            )
          : value

      return [property, adjusted]
    })
  ) as CSSProperties
}

function HighlightedCode({ tokens }: { tokens: ShikiWorkerToken[][] }) {
  return (
    <code className="block whitespace-pre">
      {tokens.map((line, lineIndex) => (
        <span key={lineIndex}>
          {line.map((token, tokenIndex) => (
            <span key={tokenIndex} style={tokenStyle(token.htmlStyle)}>
              {token.content}
            </span>
          ))}
          {lineIndex < tokens.length - 1 ? '\n' : null}
        </span>
      ))}
    </code>
  )
}

export function exceedsHighlightBudget(code: string): boolean {
  if (code.length > MAX_HIGHLIGHT_CHARS) {
    return true
  }

  let lines = 1
  let idx = code.indexOf('\n')

  while (idx !== -1) {
    if ((lines += 1) > MAX_HIGHLIGHT_LINES) {
      return true
    }

    idx = code.indexOf('\n', idx + 1)
  }

  return false
}

interface CodeChunk {
  text: string
  lines: number
}

export function chunkByLines(code: string, perChunk: number): CodeChunk[] {
  const lines = code.split('\n')

  if (lines.length <= perChunk) {
    return [{ text: code, lines: lines.length }]
  }

  const chunks: CodeChunk[] = []

  for (let i = 0; i < lines.length; i += perChunk) {
    const slice = lines.slice(i, i + perChunk)
    chunks.push({ text: slice.join('\n'), lines: slice.length })
  }

  return chunks
}

const PlainCode: FC<{ code: string }> = ({ code }) => {
  const chunks = useMemo(() => chunkByLines(code, CHUNK_LINES), [code])

  if (chunks.length === 1) {
    return <code className="block whitespace-pre">{code}</code>
  }

  return (
    <>
      {chunks.map((chunk, index) => (
        <code
          className="block whitespace-pre [content-visibility:auto]"
          key={index}
          style={{ containIntrinsicSize: `auto ${chunk.lines * EST_LINE_PX}px` }}
        >
          {chunk.text}
        </code>
      ))}
    </>
  )
}

export const SyntaxHighlighter: FC<HermesSyntaxHighlighterProps> = ({
  components: { Pre },
  language,
  code,
  defer = false
}) => {
  const { t } = useI18n()
  const trimmed = (code ?? '').replace(/^\n+/, '').trimEnd()
  const overBudget = exceedsHighlightBudget(trimmed)
  const { nearViewport, ref } = useNearViewportHighlight(!defer && !overBudget && Boolean(trimmed.trim()))
  const highlighted = useWorkerHighlight(trimmed, !defer && !overBudget && nearViewport, language || 'text')

  // Streaming may hand us empty/incomplete fences — render nothing rather
  // than a transient empty card.
  if (!trimmed.trim()) {
    return null
  }

  if (isLikelyProseCodeBlock(language, trimmed)) {
    return <div className="aui-prose-fence whitespace-pre-wrap wrap-anywhere text-foreground">{trimmed}</div>
  }

  const cleanLanguage = sanitizeLanguageTag(language || '')
  const label = cleanLanguage && cleanLanguage !== 'unknown' ? cleanLanguage : ''

  return (
    <CodeCard data-streaming={defer ? 'true' : undefined} ref={ref}>
      <CodeCardHeader>
        <CodeCardTitle>
          <CodeCardIcon name={codiconForLanguage(label)} />
          {t.assistant.tool.code}
          {label && <CodeCardSubtitle> · {label}</CodeCardSubtitle>}
        </CodeCardTitle>
        <CopyButton
          appearance="inline"
          className="-my-1 -mr-1 h-5 px-1 opacity-55 hover:opacity-100"
          iconClassName="size-2.5"
          label={t.assistant.tool.copyCode}
          showLabel={false}
          text={trimmed}
        />
      </CodeCardHeader>
      <CodeCardBody>
        <ExpandableBlock>
          <Pre className="aui-shiki m-0 overflow-hidden bg-transparent p-0">
            {highlighted ? <HighlightedCode tokens={highlighted} /> : <PlainCode code={trimmed} />}
          </Pre>
        </ExpandableBlock>
      </CodeCardBody>
    </CodeCard>
  )
}
