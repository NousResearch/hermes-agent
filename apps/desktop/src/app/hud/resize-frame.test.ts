// @vitest-environment node
import { readFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

import { describe, expect, it } from 'vitest'

const SRC = dirname(fileURLToPath(import.meta.url))
const css = readFileSync(join(SRC, '../../styles.css'), 'utf8')

/** The declaration block of the first rule whose selector list contains
 *  `selector` — enough machine for these one-property rules without a CSS
 *  parser. */
function ruleBody(selector: string): string {
  const start = css.indexOf(selector)
  expect(start, `styles.css must still carry the rule: ${selector}`).toBeGreaterThanOrEqual(0)

  const open = css.indexOf('{', start)
  const close = css.indexOf('}', open)

  return css.slice(open + 1, close)
}

describe('the HUD resize frame and click-through', () => {
  // The bug this replaces (#108793): the frame was unconditionally
  // hit-testable, so a collapsed HUD kept an invisible ring of window —
  // nothing painted, every click and drag eaten — while the rest of the
  // rectangle fell through to the app underneath.
  it('keeps the resize frame out of the hit test while the HUD is idle', () => {
    const body = ruleBody('[data-hud-shell] [data-hud-resize] {')

    expect(body).toContain('pointer-events: none')
    expect(body).not.toContain('pointer-events: auto')
  })

  // Engagement is the same gate the band answers: the caret in the composer,
  // or a held band that must take clicks without focus. Solid-input hosts
  // (X11) never give the window away, so their frame stays live regardless.
  it('re-arms the frame exactly where the HUD is being used', () => {
    const body = ruleBody('[data-hud-shell]:has([data-slot=\'composer-rich-input\']:focus) [data-hud-resize]')

    expect(body).toContain('pointer-events: auto')
  })

  it('keeps the frame live on solid-input hosts where nothing falls through anyway', () => {
    const body = ruleBody('[data-hud-shell][data-hud-input=\'solid\'] [data-hud-resize]')

    expect(body).toContain('pointer-events: auto')
  })
})
