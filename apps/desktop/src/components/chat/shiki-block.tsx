'use client'

/**
 * The ONLY static importer of shiki (and through it the multi-MB shiki
 * language/theme/wasm bundle). Every consumer reaches this module through
 * `React.lazy(() => import('./shiki-block'))` — see `LazyShiki` in
 * shiki-highlighter.tsx — so the shiki chunk stays entirely off the
 * cold-start path and loads on the first highlighted code block instead.
 *
 * Do NOT import this module statically from anything the entry graph
 * reaches, or the chunk moves back into boot.
 *
 * Unlike the previous pass-through of `react-shiki`'s component, this module
 * is cache-aware: highlighted output is stored in a module-level LRU cache
 * keyed by (theme scope, language, code), so a REMOUNT of an unchanged code
 * block (the warm-session-switch path of #95595) paints the cached HTML
 * synchronously and never re-tokenizes. Only cache misses run shiki, and
 * misses are debounced so a streaming block settles before the heavy work
 * starts.
 */
import { useEffect, useMemo, useState } from 'react'
import { bundledLanguages, getSingletonHighlighter } from 'shiki'
import type { BundledLanguage, BundledTheme, Highlighter } from 'shiki'
import { createOnigurumaEngine } from 'shiki/engine/oniguruma'

import {
  SHIKI_HIGHLIGHT_SCOPE,
  SHIKI_THEME
} from '@/components/chat/shiki-config'
import { highlightCache, highlightCacheKey } from '@/components/chat/shiki-highlight-cache'

/** Same debounce react-shiki's `delay` used to throttle highlight work with. */
const HIGHLIGHT_DELAY_MS = 120

// Stable identity for "no color replacements" so the memo/effect deps below
// never churn on renders that don't pass the prop.
const NO_COLOR_REPLACEMENTS: Record<string, Record<string, string>> = {}

export interface CachedShikiBlockProps {
  language: string
  code: string
  /** Theme override; defaults to the shared SHIKI_THEME. */
  theme?: { dark: string; light: string }
  /** Color replacements; defaults to none (the chat passes its own). */
  colorReplacements?: Record<string, Record<string, string>>
}

function isLoadableLanguage(language: string): boolean {
  return language === 'text' || language in bundledLanguages
}

/**
 * Cache scope for one theme configuration. Part of the cache key: a block
 * highlighted under a different theme is a different render.
 */
function highlightScope(
  theme: { dark: string; light: string },
  colorReplacements: Record<string, Record<string, string>>
): string {
  return `${SHIKI_HIGHLIGHT_SCOPE}:${theme.dark}:${theme.light}:${JSON.stringify(colorReplacements)}`
}

let highlighterPromise: Promise<Highlighter> | null = null
let loadedThemes = new Set<string>([SHIKI_THEME.dark, SHIKI_THEME.light])

/**
 * Lazily-created shiki singleton, mirroring react-shiki's full bundle: only
 * the languages actually seen are loaded into it (the singleton is created
 * with the first block's language, later ones are `loadLanguage`d on demand;
 * override themes are `loadTheme`d the same way). Unknown languages are left
 * unloaded and fall through to shiki's plain-text handling, as before.
 */
async function highlightToHtml(
  language: string,
  code: string,
  theme: { dark: string; light: string },
  colorReplacements: Record<string, Record<string, string>>
): Promise<string> {
  if (!highlighterPromise) {
    highlighterPromise = getSingletonHighlighter({
      // Only bundled languages are ever passed here (isLoadableLanguage
      // guards both call sites), so the cast is safe.
      langs: isLoadableLanguage(language) ? [language as BundledLanguage] : [],
      themes: [SHIKI_THEME.dark, SHIKI_THEME.light],
      engine: createOnigurumaEngine(import('shiki/wasm'))
    })
  }

  const highlighter = await highlighterPromise

  if (isLoadableLanguage(language) && !highlighter.getLoadedLanguages().includes(language)) {
    await highlighter.loadLanguage(language as BundledLanguage)
  }

  const missingThemes = [theme.dark, theme.light].filter(name => !loadedThemes.has(name))

  if (missingThemes.length > 0) {
    await highlighter.loadTheme(...(missingThemes as BundledTheme[]))
    missingThemes.forEach(name => loadedThemes.add(name))
  }

  return highlighter.codeToHtml(code, {
    lang: language,
    themes: { dark: theme.dark, light: theme.light },
    defaultColor: 'light-dark()',
    colorReplacements
  })
}

function escapeHtml(text: string): string {
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
}

/** Never let a highlight failure blank a block — degrade to escaped plain text. */
function plainTextHtml(code: string): string {
  return `<pre class="shiki" style="background-color:transparent;margin:0"><code>${escapeHtml(code)}</code></pre>`
}

export default function CachedShikiBlock({
  language,
  code,
  theme,
  colorReplacements
}: CachedShikiBlockProps) {
  const themeConfig = theme ?? SHIKI_THEME
  const replacements = colorReplacements ?? NO_COLOR_REPLACEMENTS

  const cacheKey = useMemo(
    () => highlightCacheKey(highlightScope(themeConfig, replacements), language, code),
    [language, code, replacements, themeConfig]
  )

  const [html, setHtml] = useState<string | null>(() => highlightCache.get(cacheKey) ?? null)

  useEffect(() => {
    let cancelled = false

    // Cache hit — no highlighter work at all. This is the warm-switch path:
    // the previous visit already rendered this block, so paint it again.
    const cached = highlightCache.get(cacheKey)

    if (cached !== undefined) {
      setHtml(cached)

      return
    }

    const timer = window.setTimeout(() => {
      highlightToHtml(language, code, themeConfig, replacements)
        .then(result => {
          if (cancelled) {
            return
          }

          highlightCache.set(cacheKey, result)
          setHtml(result)
        })
        .catch(error => {
          if (cancelled) {
            return
          }

          console.error('shiki highlight failed; rendering plain code', error)
          setHtml(plainTextHtml(code))
        })
    }, HIGHLIGHT_DELAY_MS)

    return () => {
      cancelled = true
      window.clearTimeout(timer)
    }
  }, [cacheKey, code, language, replacements, themeConfig])

  if (html === null) {
    // Nothing to paint yet (miss, debounce pending). Matches react-shiki's
    // own empty render while the highlight is in flight.
    return null
  }

  return <div className="rs-root not-prose" dangerouslySetInnerHTML={{ __html: html }} data-testid="shiki-container" />
}
