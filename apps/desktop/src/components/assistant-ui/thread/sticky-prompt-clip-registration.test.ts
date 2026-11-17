import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'

import { describe, expect, it } from 'vitest'

// `use-sticky-prompt-clip.ts` writes this value once per scroll frame onto
// elements that are whole turns (markdown, code, tool output). If the property
// stops being registered as a typed, NON-inherited property, every write goes
// back to invalidating the element's entire subtree — a style recalculation per
// frame on a long thread that also charges to whichever geometry read forces the
// flush. Measured on a real scroll before the fix: 195 UpdateLayoutTree events,
// 4,646ms total, 189ms worst.
const STYLES = readFileSync(resolve(process.cwd(), 'src/styles.css'), 'utf8')

const block = STYLES.match(/@property\s+--sticky-prompt-clip\s*\{[^}]*\}/)?.[0] ?? ''

describe('--sticky-prompt-clip registration', () => {
  it('is registered at all', () => {
    expect(block).not.toBe('')
  })

  it('declares a typed length syntax so the write is a cheap property update', () => {
    expect(block).toMatch(/syntax:\s*'<length>'/)
  })

  it('is non-inherited, so a write cannot invalidate every descendant', () => {
    expect(block).toMatch(/inherits:\s*false/)
  })

  it('carries an initial value, so an unset property still means "no clip"', () => {
    expect(block).toMatch(/initial-value:\s*0px/)
  })

  it('is the property the clip rule consumes', () => {
    expect(STYLES).toContain('clip-path: inset(var(--sticky-prompt-clip) 0 0)')
  })
})
