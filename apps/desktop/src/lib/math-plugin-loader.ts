/**
 * LAZY KaTeX MATH PLUGIN — light entry-graph module.
 *
 * `katex` (~250–350KB + fonts) must never sit on the boot path, but two
 * surfaces need its memoized rehype wrapper (`@/lib/katex-memo`): the chat
 * transcript (`assistant-ui/markdown-text`) and the static file preview
 * (`app/chat/right-rail/preview-file`). Both previously instantiated the
 * plugin at module scope, which pulled all of katex into the entry chunk.
 *
 * This module carries NO katex import itself: it dynamic-imports katex-memo
 * on first markdown mount (same grace as the `@streamdown/code` plugin in
 * markdown-text) and shares the one instance — both call sites configure it
 * identically (`singleDollarTextMath: true`). Until it lands, `$x$` renders
 * as source text and upgrades in place; the remark plugin object is shared
 * exactly like the old module-scope instances were.
 */

import { useEffect, useState } from 'react'

import type { createMemoizedMathPlugin } from '@/lib/katex-memo'

export type MathPlugin = ReturnType<typeof createMemoizedMathPlugin>

let mathPluginCache: MathPlugin | null = null
let mathPluginPromise: Promise<MathPlugin> | null = null

export function loadMathPlugin(): Promise<MathPlugin> {
  if (mathPluginCache) {
    return Promise.resolve(mathPluginCache)
  }

  mathPluginPromise ??= import('@/lib/katex-memo').then(plugin => {
    mathPluginCache = plugin.createMemoizedMathPlugin({ singleDollarTextMath: true })

    return mathPluginCache
  })

  return mathPluginPromise
}

export function useMathPlugin(): MathPlugin | null {
  const [plugin, setPlugin] = useState(mathPluginCache)

  useEffect(() => {
    if (plugin) {
      return
    }

    let cancelled = false

    void loadMathPlugin().then(loaded => {
      if (!cancelled) {
        setPlugin(loaded)
      }
    })

    return () => {
      cancelled = true
    }
  }, [plugin])

  return plugin
}
