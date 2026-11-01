'use client'

/**
 * The Shiki-highlighted compact diff body loads lazily. Cached HTML reuses the
 * same bounded cache as ordinary code fences, so revisiting an unchanged chat
 * does not tokenize its completed edits again.
 */
import { useEffect, useMemo } from 'react'
import { useShikiHighlighter } from 'react-shiki'

import { DiffBody, type DiffLine, diffLineTransformer } from '@/components/chat/diff-lines'
import { SHIKI_HIGHLIGHT_SCOPE, SHIKI_THEME } from '@/components/chat/shiki-config'
import { highlightCache, highlightCacheKey } from '@/components/chat/shiki-highlight-cache'

interface DiffHtmlProps {
  html: string
}

interface SyntaxDiffProps {
  language: string
  lines: DiffLine[]
}

interface UncachedSyntaxDiffProps extends SyntaxDiffProps {
  cacheKey: string
  code: string
}

function DiffHtml({ html }: DiffHtmlProps) {
  return <div dangerouslySetInnerHTML={{ __html: html }} />
}

function UncachedSyntaxDiff({ cacheKey, code, language, lines }: UncachedSyntaxDiffProps) {
  const transformers = useMemo(() => [diffLineTransformer(lines.map(line => line.kind))], [lines])

  const highlighted = useShikiHighlighter(code, language, SHIKI_THEME, {
    outputFormat: 'html',
    defaultColor: 'light-dark()',
    transformers
  })

  useEffect(() => {
    if (typeof highlighted === 'string') {
      highlightCache.set(cacheKey, highlighted)
    }
  }, [cacheKey, highlighted])

  return typeof highlighted === 'string' ? <DiffHtml html={highlighted} /> : <DiffBody lines={lines} />
}

export default function SyntaxDiff({ language, lines }: SyntaxDiffProps) {
  const code = useMemo(() => lines.map(line => line.text).join('\n'), [lines])

  const cacheKey = useMemo(
    () => highlightCacheKey(`${SHIKI_HIGHLIGHT_SCOPE}:diff:${lines.map(line => line.kind).join(',')}`, language, code),
    [code, language, lines]
  )

  const cached = highlightCache.get(cacheKey)

  return cached !== undefined ? (
    <DiffHtml html={cached} />
  ) : (
    // The hook retains its previous output until an update settles. A new key
    // starts with the plain fallback rather than caching that output for new text.
    <UncachedSyntaxDiff cacheKey={cacheKey} code={code} key={cacheKey} language={language} lines={lines} />
  )
}
