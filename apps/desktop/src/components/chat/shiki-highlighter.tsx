'use client'

import type { SyntaxHighlighterProps } from '@assistant-ui/react-streamdown'
import { type ComponentProps, type FC, lazy, Suspense, useMemo } from 'react'
import type ShikiHighlighter from 'react-shiki'

import { CodeCard, CodeCardBody } from '@/components/chat/code-card'
import { ExpandableBlock } from '@/components/chat/expandable-block'
import { CopyButton } from '@/components/ui/copy-button'
import { useI18n } from '@/i18n'
import { isLikelyProseCodeBlock } from '@/lib/markdown-code'

/**
 * Streamdown's code adapter renders header + body as inline siblings, so we
 * own the wrapping `<CodeCard>` here and neutralize the upstream
 * `data-streamdown="code-block"` chrome from styles.css. The card is
 * background-only — no header row, no language label — so a fence reads as a
 * tinted slab of the reply. Copy stays visible in the corner (not hover-only)
 * so Windows / touch / no-hover surfaces can still find it.
 *
 * `react-shiki` full bundle so all `bundledLanguages` work; theme switches
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

// react-shiki (and through it the multi-MB shiki grammar/theme bundle) is the
// heaviest dependency in the renderer. `shiki-block.tsx` is its only static
// importer, so this lazy() is the single seam that keeps shiki out of the
// entry chunk — it loads on the first highlighted code block, not at boot.
const ShikiBlock = lazy(() => import('./shiki-block'))

/** Drop-in ShikiHighlighter that suspends on first use and renders the code
 *  as plain preformatted text until the shiki chunk arrives. */
export const LazyShiki: FC<ComponentProps<typeof ShikiHighlighter>> = props => (
  <Suspense fallback={<PlainCode code={String(props.children ?? '')} />}>
    <ShikiBlock {...props} />
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
        className="absolute right-1.5 top-1.5 z-10 h-5 gap-0 rounded-md bg-(--ui-bg-editor)/90 px-1"
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
              <LazyShiki
                addDefaultStyles={false}
                as="div"
                colorReplacements={SHIKI_COLOR_REPLACEMENTS}
                defaultColor="light-dark()"
                delay={120}
                language={language || 'text'}
                showLanguage={false}
                theme={SHIKI_THEME}
              >
                {trimmed}
              </LazyShiki>
            )}
          </Pre>
        </ExpandableBlock>
      </CodeCardBody>
    </CodeCard>
  )
}
