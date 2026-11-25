// Shiki theme/color constants shared by the chat code-block renderer
// (shiki-highlighter.tsx) and the lazy shiki chunk (shiki-block.tsx). Kept in
// their own dependency-free module so the lazy chunk can import them without
// pulling the main chat module (or react-shiki) into the shiki bundle.

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
export const SHIKI_COLOR_REPLACEMENTS: Record<string, Record<string, string>> = {
  'github-light-default': { '#6e7781': '#57606a' }
}

/**
 * Cache-key scope for the content-addressed highlight cache. Bumping this
 * invalidates every cached highlight at once — bump it whenever the rendering
 * options (themes, color replacements) change, because keys are NOT allowed to
 * silently produce a different DOM than the one they were computed with.
 */
export const SHIKI_HIGHLIGHT_SCOPE = `hermes-shiki-v1:${JSON.stringify({
  dark: SHIKI_THEME.dark,
  light: SHIKI_THEME.light,
  colorReplacements: SHIKI_COLOR_REPLACEMENTS
})}`
