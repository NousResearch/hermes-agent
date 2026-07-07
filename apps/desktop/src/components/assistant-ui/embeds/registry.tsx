'use client'

import { type ComponentType, lazy, type LazyExoticComponent, type ReactNode, Suspense } from 'react'

import { parseHermesUiArtifact } from '@/lib/hermes-ui-artifacts'

import { RichBoundary } from './rich-boundary'
import type { RichFenceProps } from './types'

// Root renderer for fenced code blocks: a language → lazy-renderer table. Each
// renderer is its own split chunk (mermaid pulls in the mermaid lib, svg pulls
// in DOMPurify), loaded only when a block of that language actually appears.
const LAZY_FENCE: Record<string, LazyExoticComponent<ComponentType<RichFenceProps>>> = {
  'hermes-ui': lazy(() => import('./hermes-ui-artifact')),
  mermaid: lazy(() => import('./mermaid-embed')),
  svg: lazy(() => import('./svg-embed'))
}

export const RICH_FENCE_LANGUAGES: ReadonlySet<string> = new Set(Object.keys(LAZY_FENCE))

interface RichCodeBlockProps extends RichFenceProps {
  /** Rendered for unhandled languages, while the chunk loads, and on failure
   *  (typically the normal syntax-highlighted code block). */
  fallback: ReactNode
  language?: string
}

export function RichCodeBlock({ code, fallback, language, streaming }: RichCodeBlockProps) {
  const normalizedLanguage = language?.toLowerCase()
  const Renderer = normalizedLanguage ? LAZY_FENCE[normalizedLanguage] : undefined

  if (!Renderer) {
    return <>{fallback}</>
  }

  if (normalizedLanguage === 'hermes-ui' && !parseHermesUiArtifact(code).ok) {
    return <>{fallback}</>
  }

  return (
    <RichBoundary fallback={fallback} resetKey={code}>
      <Suspense fallback={fallback}>
        <Renderer code={code} streaming={streaming} />
      </Suspense>
    </RichBoundary>
  )
}
