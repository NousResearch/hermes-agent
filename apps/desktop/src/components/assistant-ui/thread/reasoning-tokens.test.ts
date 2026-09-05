// @vitest-environment node
import { readFileSync } from 'node:fs'
import { dirname } from 'node:path'
import { fileURLToPath } from 'node:url'

import { compile } from '@tailwindcss/node'
import { describe, expect, it } from 'vitest'

const SRC = dirname(fileURLToPath(import.meta.url))

// Reasoning tokens live in src/styles.css — three levels up from this file.
const STYLES = `${SRC}/../../../styles.css`

// The exact classes ReasoningTextPart hands to its markdown container
// (message-parts.tsx) — the contract is that these compile to token-driven
// declarations, not to the old hard-coded text-xs/tertiary utilities.
const REASONING_CONTAINER_CLASSES = [
  'text-[length:var(--conversation-reasoning-font-size)]',
  'leading-(--conversation-reasoning-line-height)',
  'text-(--conversation-reasoning-color)'
]

// The reasoning body's presentation is a theming contract (#99793): the
// component consumes tokens, tokens carry the stock defaults, and nothing in
// the chain re-hard-codes the old `text-xs leading-snug text-muted-foreground/85`
// utilities. Compile-level test (same pattern as hover-variant.test.ts):
// jsdom cannot color-mix, and the contract that matters is what the theme
// stylesheet emits and what the component class string references.
describe('reasoning typography tokens (#99793)', () => {
  async function compiled(): Promise<string> {
    const { build } = await compile(`@import "${STYLES}";\n`, { base: SRC, onDependency() {} })

    return build(REASONING_CONTAINER_CLASSES)
  }

  it('exposes reasoning size, line-height, and color tokens with stock defaults', async () => {
    const css = await compiled()

    expect(css).toContain('--conversation-reasoning-font-size: 0.75rem')
    expect(css).toContain('--conversation-reasoning-line-height: 1.375')
    // Same effective ink as the old tertiary × /85 chain: 0.54 × 0.85 = 0.459.
    expect(css).toContain('color-mix(in srgb, var(--ui-base) 45.9%, transparent)')
  })

  it('compiles the reasoning container classes from the tokens', async () => {
    const css = await compiled()

    // Utilities emit in @layer utilities, before the component rules —
    // assert against the whole sheet, not a slice.
    expect(css).toContain('font-size: var(--conversation-reasoning-font-size)')
    expect(css).toContain('line-height: var(--conversation-reasoning-line-height)')
    expect(css).toContain('color: var(--conversation-reasoning-color)')
  })

  it('gates the open-disclosure opacity behind a token with the stock fade as default', async () => {
    const css = await compiled()

    expect(css).toContain('--conversation-reasoning-open-opacity, 0.67')
    // The rule targets an open disclosure specifically — the body must be a
    // direct child, so collapsed rows keep the stock fade.
    expect(css).toContain("[data-slot='aui_thinking-disclosure']:has(> [data-slot='aui_thinking-body'])")
  })

  it('keeps the component free of the old hard-coded utilities', () => {
    const parts = readFileSync(`${SRC}/message-parts.tsx`, 'utf8')

    expect(parts).not.toContain('text-muted-foreground/85')
    expect(parts).toContain('--conversation-reasoning-font-size')
  })
})
