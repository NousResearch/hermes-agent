// @vitest-environment jsdom
/**
 * The peek is scoped with :has() so only the settings overlay that owns the
 * translucency row pays for an opacity transition. jsdom implements :has() in
 * matches(), so the scoping claim is checkable directly: the marked overlay
 * matches the peek selector and an unmarked one does not.
 */
import { describe, expect, it } from 'vitest'

const PEEK_SELECTOR = '[data-overlay-surface]:has([data-translucency-peek-scope])'

describe('peek scoping', () => {
  it('matches only the overlay that armed the interaction', () => {
    document.body.innerHTML = `
      <div data-overlay-surface id="settings">
        <div data-translucency-peek-scope></div>
      </div>
      <div data-overlay-surface id="command-center">
        <div></div>
      </div>
    `

    const settings = document.getElementById('settings')
    const commandCenter = document.getElementById('command-center')

    expect(settings?.matches(PEEK_SELECTOR)).toBe(true)
    // The regression this guards: an unscoped rule gave EVERY overlay in the
    // app a 420ms opacity transition for the life of the process.
    expect(commandCenter?.matches(PEEK_SELECTOR)).toBe(false)
  })
})
