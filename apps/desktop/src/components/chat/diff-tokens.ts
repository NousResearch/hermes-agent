import * as React from 'react'
import type { BundledLanguage, ThemedToken } from 'shiki'

import { SHIKI_THEME } from './shiki-highlighter'

// shiki FontStyle is a bitmask: Italic=1, Bold=2, Underline=4.
export function tokenStyle({ bgColor, color, fontStyle = 0 }: ThemedToken): React.CSSProperties | undefined {
  if (!color && !bgColor && !fontStyle) {
    return undefined
  }

  return {
    backgroundColor: bgColor,
    color,
    fontStyle: fontStyle & 1 ? 'italic' : undefined,
    fontWeight: fontStyle & 2 ? 700 : undefined,
    textDecorationLine: fontStyle & 4 ? 'underline' : undefined
  }
}

function useThemeName() {
  const current = () => (document.documentElement.classList.contains('dark') ? SHIKI_THEME.dark : SHIKI_THEME.light)
  const [theme, setTheme] = React.useState(current)

  React.useEffect(() => {
    const observer = new MutationObserver(() => setTheme(current()))

    observer.observe(document.documentElement, { attributeFilter: ['class'], attributes: true })

    return () => observer.disconnect()
  }, [])

  return theme
}

export function useDiffTokens(code: string, language: string | null) {
  const theme = useThemeName()
  const [tokens, setTokens] = React.useState<ThemedToken[][] | null>(null)

  React.useEffect(() => {
    let cancelled = false

    setTokens(null)

    if (!language) {
      return
    }

    // Dynamic import so the multi-MB shiki chunk stays off the cold-start
    // path — this effect only runs once a highlightable diff is on screen.
    void import('shiki')
      .then(({ codeToTokens }) => codeToTokens(code, { lang: language as BundledLanguage, theme }))
      .then(result => {
        if (!cancelled) {
          setTokens(result.tokens)
        }
      })
      .catch(() => {
        if (!cancelled) {
          setTokens([])
        }
      })

    return () => {
      cancelled = true
    }
  }, [code, language, theme])

  return tokens
}
