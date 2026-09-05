import { normalizeMathDelimiters } from '@assistant-ui/react-streamdown'
import { useEffect, useMemo, useState } from 'react'

import type { createMemoizedMathPlugin } from '@/lib/katex-memo'

export type MathPlugin = ReturnType<typeof createMemoizedMathPlugin>

interface MathPluginModule {
  createMemoizedMathPlugin: typeof createMemoizedMathPlugin
}

type MathPluginImporter = () => Promise<MathPluginModule>

export interface LazyMathPluginLoader {
  load: (markdown: string) => Promise<MathPlugin | undefined>
  peek: (markdown: string) => MathPlugin | undefined
}

const MATH_FENCE_RE =
  /(?:^|\r?\n)[ \t]{0,3}(?:>[ \t]*)*(?:(?:[-+*]|\d+[.)])[ \t]+)?[ \t]*(?:`{3,}|~{3,})[ \t]*(?:latex|math|tex)(?=[\s,{]|$)/i

const NORMALIZED_MATH_MARKER_RE = /\[(?:\/?)(?:inline|math)\]|\\(?:\(|\[|begin\{)/i

function normalizeForMathDetection(markdown: string): string {
  return normalizeMathDelimiters(markdown)
}

/** Cheap admission check used before the KaTeX module is requested. */
export function hasRenderableMath(markdown: string): boolean {
  // False positives only load a deferred chunk; false negatives expose raw
  // delimiters and break rendering. Admit any dollar/custom-LaTeX marker, even
  // inside code, and let remark-math decide whether it is actual math.
  const normalized = normalizeForMathDetection(markdown)

  return normalized.includes('$') || NORMALIZED_MATH_MARKER_RE.test(normalized) || MATH_FENCE_RE.test(normalized)
}

export function createLazyMathPluginLoader(
  importer: MathPluginImporter = () => import('@/lib/katex-memo')
): LazyMathPluginLoader {
  let completed: MathPlugin | undefined
  let pending: Promise<MathPlugin> | undefined
  let failed: unknown
  let hasFailed = false

  const peek = (markdown: string) => (hasRenderableMath(markdown) ? completed : undefined)

  return {
    load(markdown) {
      if (!hasRenderableMath(markdown)) {
        return Promise.resolve(undefined)
      }

      if (completed) {
        return Promise.resolve(completed)
      }

      if (hasFailed) {
        return Promise.reject(failed)
      }

      if (pending) {
        return pending
      }

      const operation = importer()
        .then(module => {
          completed = module.createMemoizedMathPlugin({ singleDollarTextMath: true })
          pending = undefined

          return completed
        })
        .catch(error => {
          // A failed dynamic import is a poisoned module URL in Chromium. Do
          // not claim that retrying the same specifier can recover it; remember
          // the failure and let the caller keep its existing fallback.
          failed = error
          hasFailed = true
          pending = undefined
          throw error
        })

      pending = operation

      return operation
    },
    peek
  }
}

export const lazyMathPluginLoader = createLazyMathPluginLoader()

export interface LazyMathPluginState {
  failed: boolean
  loading: boolean
  plugin: MathPlugin | undefined
}

export function useLazyMathPluginState(
  markdown: string,
  loader: LazyMathPluginLoader = lazyMathPluginLoader
): LazyMathPluginState {
  const normalizedMarkdown = useMemo(() => normalizeForMathDetection(markdown), [markdown])
  const eligible = useMemo(() => hasRenderableMath(normalizedMarkdown), [normalizedMarkdown])
  const [plugin, setPlugin] = useState<MathPlugin | undefined>(() => loader.peek(normalizedMarkdown))
  const [loading, setLoading] = useState(() => eligible && !loader.peek(normalizedMarkdown))
  const [failed, setFailed] = useState(false)

  useEffect(() => {
    if (!eligible) {
      setLoading(false)
      setFailed(false)
      setPlugin(undefined)

      return undefined
    }

    let active = true
    const cached = loader.peek(normalizedMarkdown)

    if (cached) {
      setPlugin(cached)
      setLoading(false)
      setFailed(false)

      return undefined
    }

    setLoading(true)
    setFailed(false)

    void loader
      .load(normalizedMarkdown)
      .then(loaded => {
        if (active) {
          setPlugin(loaded)
          setLoading(false)
        }
      })
      .catch(() => {
        if (active) {
          setLoading(false)
          setFailed(true)
        }
      })

    return () => {
      active = false
    }
  }, [eligible, loader, normalizedMarkdown])

  return {
    failed: eligible && failed,
    loading: eligible && (loading || (!plugin && !failed)),
    plugin: eligible ? plugin : undefined
  }
}

export function useLazyMathPlugin(
  markdown: string,
  loader: LazyMathPluginLoader = lazyMathPluginLoader
): MathPlugin | undefined {
  return useLazyMathPluginState(markdown, loader).plugin
}
