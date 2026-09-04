'use client'

import type { SyntaxHighlighterProps } from '@assistant-ui/react-streamdown'
import { type FC, lazy, Suspense, useMemo } from 'react'

import { CodeCard, CodeCardBody } from '@/components/chat/code-card'
import { ExpandableBlock } from '@/components/chat/expandable-block'
// Theme constants live in shiki-config (dependency-free) so the lazy shiki
// chunk can import them without pulling this module into the shiki bundle.
import { SHIKI_COLOR_REPLACEMENTS } from '@/components/chat/shiki-config'
import { CopyButton } from '@/components/ui/copy-button'
import { useI18n } from '@/i18n'
import { isLikelyProseCodeBlock } from '@/lib/markdown-code'

import type { CachedShikiBlockProps } from './shiki-block'
export { SHIKI_COLOR_REPLACEMENTS, SHIKI_THEME } from '@/components/chat/shiki-config'

/**
 * Streamdown's code adapter renders header + body as inline siblings, so we
 * own the wrapping `<CodeCard>` here and neutralize the upstream
 * `data-streamdown="code-block"` chrome from styles.css. The card is
 * background-only — no header row, no language label — so a fence reads as a
 * tinted slab of the reply; copy is a hover-reveal control in the corner.
 *
 * The heavy lifting lives in the lazy `shiki-block` chunk (full bundle so all
 * `bundledLanguages` work; theme switches follow the document `color-scheme`
 * via `defaultColor="light-dark()"`), and its output is cached by content so
 * warm-session switches never re-tokenize unchanged blocks (#95595).
 */
interface HermesSyntaxHighlighterProps extends SyntaxHighlighterProps {
  defer?: boolean
}

const MAX_HIGHLIGHT_CHARS = 150_000
const MAX_HIGHLIGHT_LINES = 3_000
const CHUNK_LINES = 200
const EST_LINE_PX = 16

// shiki (and through it the multi-MB grammar/theme/wasm bundle) is the
// heaviest dependency in the renderer. `shiki-block.tsx` is its only static
// importer, so this lazy() is the single seam that keeps shiki out of the
// entry chunk — it loads on the first highlighted code block, not at boot.
// The lazy module is cache-aware (#95595): unchanged blocks paint from a
// content-keyed cache instead of re-tokenizing on every mount.
const ShikiBlock = lazy(() => import('./shiki-block'))

/** Suspends on first use and renders the code as plain preformatted text
 *  until the shiki chunk arrives. Highlighted output is cached by
 *  (theme, language, code), so revisits never re-tokenize (#95595). */
export const LazyShiki: FC<CachedShikiBlockProps> = ({ language, code, theme, colorReplacements }) => (
  <Suspense fallback={<PlainCode code={code} />}>
    <ShikiBlock code={code} colorReplacements={colorReplacements} language={language} theme={theme} />
  </Suspense>
)

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

  // Streaming may hand us empty/incomplete fences — render nothing rather
  // than a transient empty card.
  if (!trimmed.trim()) {
    return null
  }

  if (isLikelyProseCodeBlock(language, trimmed)) {
    return <div className="aui-prose-fence whitespace-pre-wrap wrap-anywhere text-foreground">{trimmed}</div>
  }

  const plain = defer || exceedsHighlightBudget(trimmed)

  return (
    <CodeCard data-streaming={defer ? 'true' : undefined}>
      <CopyButton
        appearance="inline"
        className="absolute right-1.5 top-1.5 z-10 h-5 gap-0 rounded-md px-1 opacity-0 transition-opacity group-hover/code:opacity-100 focus-visible:opacity-100"
        iconClassName="size-2.5"
        label={t.assistant.tool.copyCode}
        showLabel={false}
        text={trimmed}
      />
      <CodeCardBody className="[&_pre]:px-3 [&_pre]:py-2.5">
        <ExpandableBlock>
          <Pre className="aui-shiki m-0 overflow-hidden bg-transparent p-0">
            {plain ? (
              <PlainCode code={trimmed} />
            ) : (
              <LazyShiki code={trimmed} colorReplacements={SHIKI_COLOR_REPLACEMENTS} language={language || 'text'} />
            )}
          </Pre>
        </ExpandableBlock>
      </CodeCardBody>
    </CodeCard>
  )
}
