import { useEffect, useState } from 'react'
import { bundledLanguages, codeToHtml } from 'shiki'

// `github-dark-dimmed` is GitHub's lower-contrast dark palette — the vivid
// `github-dark-default` tokens read harsh at our small code size. Shared by the
// inline diff renderer too (see diff-lines.tsx) so code + diffs match.
// (Canonical home moved here from shiki-highlighter.tsx so the cache module
// doesn't import the component module — re-exported there for back-compat.)
export const SHIKI_THEME = { dark: 'github-dark-dimmed', light: 'github-light-default' } as const

/**
 * `github-light-default` colors comments `#6e7781` (~4.2:1 against the code
 * card background) — borderline unreadable at our 11px code size, and worst of
 * all for shell snippets where a single `#` turns the rest of the line into one
 * long comment span. Remap light-mode comments to GitHub's darker muted gray
 * (`#57606a`, ~6.4:1). Dark mode (`#8b949e`, ~6.1:1) already reads fine, so we
 * leave it untouched. Keyed per theme name so the bump only applies in light.
 */
export const SHIKI_COLOR_REPLACEMENTS: Record<string, Record<string, string>> = {
  'github-light-default': { '#6e7781': '#57606a' }
}

// Highlighted-HTML LRU — the "no plain->color pop" layer (Ace 2026-07-15: the
// chat pane still flashed occasionally after the switch-paint fixes; traced to
// code cards re-highlighting from scratch on every remount — the SWR wholesale
// replace and the idle budget raise both remount cards, and each one rendered
// plain first, then swapped to highlighted ~120ms later, IN the viewport).
//
// Keyed on language + code. A hit renders the highlighted HTML SYNCHRONOUSLY in
// the same commit as the mount — no swap, no flash. Only the first-ever render
// of a given fence pays the async tokenize (plain-first once, then cached for
// the app run). Shiki output is trusted static HTML from our own tokenizer over
// already-escaped code — the standard shiki consumption pattern.
export const MAX_HTML_ENTRIES = 400

const htmlCache = new Map<string, string>()

function cacheGet(key: string): string | undefined {
  const value = htmlCache.get(key)

  if (value !== undefined) {
    // refresh LRU position
    htmlCache.delete(key)
    htmlCache.set(key, value)
  }

  return value
}

function cacheSet(key: string, value: string): void {
  htmlCache.delete(key)
  htmlCache.set(key, value)

  while (htmlCache.size > MAX_HTML_ENTRIES) {
    const oldest = htmlCache.keys().next().value

    if (oldest === undefined) {
      break
    }

    htmlCache.delete(oldest)
  }
}

/** Test hook. */
export function clearShikiHtmlCache(): void {
  htmlCache.clear()
}

/** Test introspection. */
export function shikiHtmlCacheSize(): number {
  return htmlCache.size
}

/**
 * Test-only: seed the cache directly to exercise the LRU boundary without
 * driving the async hook N times. Not used by the render path.
 */
export function seedShikiHtmlCacheForTest(language: string, code: string, html: string): void {
  cacheSet(`${resolveLanguage(language)}\u0000${code}`, html)
}

function resolveLanguage(language: string | undefined): string {
  const lang = (language || '').toLowerCase()

  return lang && lang in bundledLanguages ? lang : 'text'
}

export interface CachedShikiHtml {
  /** Highlighted HTML when available (cache hit = synchronously on first render). */
  html: string | null
}

/**
 * Cache-first Shiki highlight. Returns highlighted HTML synchronously when the
 * (language, code) pair has been highlighted before this app run; otherwise
 * kicks off the async tokenize and returns null until it lands (caller shows
 * plain code for that one first render). Failures stay null — plain code is
 * the fail-open state.
 */
export function useCachedShikiHtml(code: string, language: string | undefined): CachedShikiHtml {
  const lang = resolveLanguage(language)
  const key = `${lang}\u0000${code}`
  const [html, setHtml] = useState<string | null>(() => cacheGet(key) ?? null)

  useEffect(() => {
    const cached = cacheGet(key)

    if (cached !== undefined) {
      setHtml(current => (current === cached ? current : cached))

      return
    }

    setHtml(null)

    let alive = true

    codeToHtml(code, {
      colorReplacements: SHIKI_COLOR_REPLACEMENTS,
      defaultColor: 'light-dark()',
      lang,
      themes: SHIKI_THEME
    })
      .then(rendered => {
        cacheSet(key, rendered)

        if (alive) {
          setHtml(rendered)
        }
      })
      .catch(() => {
        // fail-open: plain code keeps rendering
      })

    return () => {
      alive = false
    }
  }, [key, code, lang])

  return { html }
}
