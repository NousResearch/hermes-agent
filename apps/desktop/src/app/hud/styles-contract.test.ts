import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'

import { describe, expect, it } from 'vitest'

const css = readFileSync(resolve(process.cwd(), 'src/styles.css'), 'utf8')

function stripComments(source: string): string {
  return source.replace(/\/\*[\s\S]*?\*\//g, '')
}

function normalizeSelector(selector: string): string {
  return selector.trim().replace(/\s+/g, ' ')
}

/** Last-write-wins map of selector → concatenated declaration bodies. */
function ruleBodies(source: string): Map<string, string> {
  const bodies = new Map<string, string>()
  const re = /([^{}]+)\{([^{}]*)\}/g
  let match: RegExpExecArray | null

  while ((match = re.exec(stripComments(source)))) {
    const body = match[2].trim()

    for (const selector of match[1].split(',').map(normalizeSelector)) {
      if (!selector) {
        continue
      }

      const previous = bodies.get(selector)

      bodies.set(selector, previous ? `${previous}; ${body}` : body)
    }
  }

  return bodies
}

function declarations(body: string | undefined): Record<string, string> {
  const out: Record<string, string> = {}

  if (!body) {
    return out
  }

  for (const part of body.split(';')) {
    const index = part.indexOf(':')

    if (index < 0) {
      continue
    }

    const property = part.slice(0, index).trim()
    const value = part.slice(index + 1).trim()

    if (property) {
      out[property] = value
    }
  }

  return out
}

const rules = ruleBodies(css)
const of = (selector: string) => declarations(rules.get(normalizeSelector(selector)))

describe('HUD sheet CSS contract', () => {
  it('shows the sheet while a turn is recent, not only while the composer is focused', () => {
    // The bug: composer-bounds rose on data-hud-recent, but the glass stayed
    // at opacity 0 until :focus. White overlay ink then sat on the desktop.
    // Games are excluded — they pin data-hud-recent for the whole overlay
    // session, and a sheet that followed recent would cover the action.
    const recent = of('[data-hud-shell][data-hud-recent]:not([data-hud-game]) [data-hud-glass]')
    const gameRecent = of('[data-hud-shell][data-hud-game][data-hud-recent] [data-hud-glass]')

    expect(recent.opacity).toBe('1')
    expect(recent.translate).toBe('none')
    expect(gameRecent.opacity ?? '').not.toBe('1')
  })

  it('paints the resting sheet as opaque theme paper, not a translucent scrim', () => {
    const glass = of('[data-hud-shell] [data-hud-glass]')

    expect(glass.background).toBe('var(--ui-bg-elevated)')
    expect(glass.background).not.toMatch(/\/\s*0\.\d+/)
  })

  it('keeps focusing from punching a hole back through to the desktop', () => {
    const focused = of(
      "[data-hud-shell]:not([data-hud-game]):has([data-slot='composer-rich-input']:focus) [data-hud-glass]"
    )
    const recent = of('[data-hud-shell][data-hud-recent]:not([data-hud-game]) [data-hud-glass]')

    expect(focused.background).toBe('var(--ui-bg-elevated)')
    expect(recent.background).toBe('var(--ui-bg-elevated)')
  })

  it('fills the window below the bar instead of an inset, content-sized band', () => {
    // Inset + --hud-band-height left gutters and a hole under the last line
    // where the desktop (and whatever was behind the HUD) showed through.
    const glass = of('[data-hud-shell] [data-hud-glass]')
    const topEdge = of("[data-hud-shell][data-hud-edge='top'] [data-hud-glass]")

    expect(glass.top).toBe('0')
    expect(glass.right).toBe('0')
    expect(glass.left).toBe('0')
    expect(glass.height).toBe('auto')
    expect(glass['border-radius']).toBe('0')
    expect(glass.bottom).toBe('var(--hud-bar-height, var(--composer-fallback-height))')
    expect(topEdge.bottom).toBe('0')
    expect(css).not.toMatch(/\[data-hud-glass\][^{]*\{[^}]*--hud-band-height/)
  })

  it('uses theme ink on the card and keeps overlay ink for game overlay only', () => {
    const bounds = of("[data-hud-shell] [data-slot='composer-bounds']")
    const gameBounds = of("[data-hud-shell][data-hud-game] [data-slot='composer-bounds']")
    const descendants = of("[data-hud-shell] [data-slot='composer-bounds'] *")
    const gameDescendants = of("[data-hud-shell][data-hud-game] [data-slot='composer-bounds'] *")

    expect(bounds.color).toBe('var(--ui-text-primary)')
    expect(gameBounds.color).toBe('var(--hud-overlay-ink)')
    expect(gameBounds['--hud-overlay-ink']).toBe('#fff')
    expect(descendants.color ?? '').not.toContain('!important')
    expect(gameDescendants.color).toContain('!important')
  })

  it('keeps a translucent scrim over a game, including Glass-on and focus', () => {
    // The (0,3,0) game-only rule lost to later opaque-paper rules
    // (focus :has is (0,4,0); glass-on resting is (0,4,0); glass-on +
    // recent/focus is higher). Overlay ink is #fff, so those losses put
    // white text on --ui-bg-elevated.
    const scrim = /rgb\(12 14 18\s*\/\s*0\.\d+\)/
    const gameStates = [
      '[data-hud-shell][data-hud-game] [data-hud-glass]',
      '[data-hud-shell][data-hud-game][data-hud-recent] [data-hud-glass]',
      "[data-hud-shell][data-hud-game]:has([data-slot='composer-rich-input']:focus) [data-hud-glass]",
      ':root[data-hermes-glass-on] [data-hud-shell][data-hud-game] [data-hud-glass]',
      ':root[data-hermes-glass-on] [data-hud-shell][data-hud-game][data-hud-recent] [data-hud-glass]',
      ":root[data-hermes-glass-on] [data-hud-shell][data-hud-game]:has([data-slot='composer-rich-input']:focus) [data-hud-glass]"
    ]

    for (const selector of gameStates) {
      expect(of(selector).background, selector).toMatch(scrim)
    }
  })
})
