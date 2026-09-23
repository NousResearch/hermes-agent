'use client'

/**
 * The Shiki-highlighted compact diff body, split out of diff-lines.tsx so the
 * `react-shiki` static import (and the multi-MB shiki chunk behind it) loads
 * lazily on first use instead of on the cold-start path. diff-lines.tsx
 * reaches this through `React.lazy` with a plain `DiffBody` fallback.
 */
import type { ReactNode } from 'react'
import { useMemo } from 'react'
import { useShikiHighlighter } from 'react-shiki'

import { DiffBody, type DiffLine, diffLineTransformer } from '@/components/chat/diff-lines'
import { SHIKI_THEME } from '@/components/chat/shiki-highlighter'

export default function SyntaxDiff({ language, lines }: { language: string; lines: DiffLine[] }) {
  const code = useMemo(() => lines.map(line => line.text).join('\n'), [lines])
  const transformers = useMemo(() => [diffLineTransformer(lines.map(line => line.kind))], [lines])

  const highlighted = useShikiHighlighter(code, language, SHIKI_THEME, {
    defaultColor: 'light-dark()',
    transformers
  })

  // Until Shiki resolves, show the plain colored diff so there's no flash.
  return (highlighted as ReactNode) ?? <DiffBody lines={lines} />
}
