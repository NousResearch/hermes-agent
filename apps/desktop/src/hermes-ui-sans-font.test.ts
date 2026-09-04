// @vitest-environment node
import { readFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

import { describe, expect, it } from 'vitest'

const SRC = dirname(fileURLToPath(import.meta.url))

// The actual rendering bug (#102773 — Electron 40.10.2 font-matching failure
// on Fedora Atomic / KDE Plasma / Wayland) can't be asserted here: font
// rasterization is platform/compositor-specific and this suite runs in
// jsdom, not a real Chromium renderer. What we *can* assert is the contract
// the fix depends on: the bundled family is declared via @font-face and
// sits first in the sans stack that --dt-font-sans resolves to, in both the
// CSS default and the JS theme-preset override.

describe('bundled Hermes UI Sans font', () => {
  it('is declared via @font-face in styles.css for both weights', () => {
    const css = readFileSync(join(SRC, 'styles.css'), 'utf-8')

    expect(css).toMatch(/@font-face\s*{\s*font-family:\s*'Hermes UI Sans';[^}]*font-weight:\s*400/s)
    expect(css).toMatch(/@font-face\s*{\s*font-family:\s*'Hermes UI Sans';[^}]*font-weight:\s*700/s)
    expect(css).toContain("url('./fonts/NotoSans-Regular.woff2')")
    expect(css).toContain("url('./fonts/NotoSans-Bold.woff2')")
  })

  it('is first in the --dt-font-sans stack in styles.css', () => {
    const css = readFileSync(join(SRC, 'styles.css'), 'utf-8')
    const match = css.match(/--dt-font-sans:\s*\n\s*([^;]+);/)

    expect(match).not.toBeNull()
    expect(match![1].split(',')[0].trim()).toBe("'Hermes UI Sans'")
  })

  it('does not touch --dt-font-kbd (native UI face is intentional there)', () => {
    const css = readFileSync(join(SRC, 'styles.css'), 'utf-8')
    const match = css.match(/--dt-font-kbd:\s*([^;]+);/)

    expect(match).not.toBeNull()
    expect(match![1]).not.toContain('Hermes UI Sans')
  })

  it('is first in SYSTEM_SANS in themes/presets.ts', () => {
    const presets = readFileSync(join(SRC, 'themes', 'presets.ts'), 'utf-8')
    const match = presets.match(/const SYSTEM_SANS =\s*\n\s*'([^']+)'/)

    expect(match).not.toBeNull()
    expect(match![1].split(',')[0].trim()).toBe('"Hermes UI Sans"')
  })
})
