import { afterEach, describe, expect, it } from 'vitest'

import { chatSurfaceRoot, clearSurfaceVar, COMPOSER_HEIGHT_VAR, setSurfaceVar } from './surface-vars'

/**
 * Regression: the thread's bottom gap going randomly huge and staying huge.
 *
 * `--thread-last-message-clearance` is `composer + status-stack + 2rem`, and
 * both inputs are published by JS onto the owning `[data-chat-surface]`. The
 * helpers used to fall back to `document.documentElement` when they couldn't
 * find a surface — which is exactly the state a publisher is in once React has
 * detached it. `:root` is where every surface's DEFAULTS live, so one such
 * write becomes a global floor under every thread's clearance, inherited by
 * every tab, until reload. Observed in the wild as
 * `calc(56px + 176px + 2rem)`: a 176px status stack that had been gone for
 * minutes, on a session that never had one.
 */
describe('surface measured-height vars', () => {
  afterEach(() => {
    document.documentElement.style.removeProperty(COMPOSER_HEIGHT_VAR)
    document.body.replaceChildren()
  })

  function surface() {
    const root = document.createElement('div')
    root.setAttribute('data-chat-surface', '')

    const publisher = document.createElement('div')
    root.append(publisher)
    document.body.append(root)

    return { publisher, root }
  }

  it('publishes onto the owning surface, not the document root', () => {
    const { publisher, root } = surface()

    setSurfaceVar(publisher, COMPOSER_HEIGHT_VAR, '176px')

    expect(root.style.getPropertyValue(COMPOSER_HEIGHT_VAR)).toBe('176px')
    expect(document.documentElement.style.getPropertyValue(COMPOSER_HEIGHT_VAR)).toBe('')
  })

  it('never poisons the document root from a detached publisher', () => {
    const { publisher } = surface()

    // How the real leak happens: React removes the stack/composer div when it
    // collapses, orphaning it from its surface, and a pending measurement
    // publishes anyway. `closest()` from an orphan finds nothing.
    publisher.remove()
    setSurfaceVar(publisher, COMPOSER_HEIGHT_VAR, '176px')

    expect(document.documentElement.style.getPropertyValue(COMPOSER_HEIGHT_VAR)).toBe('')
  })

  it('never poisons the document root from a publisher outside any surface', () => {
    const orphan = document.createElement('div')
    document.body.append(orphan)

    setSurfaceVar(orphan, COMPOSER_HEIGHT_VAR, '176px')

    expect(document.documentElement.style.getPropertyValue(COMPOSER_HEIGHT_VAR)).toBe('')
  })

  it('reports no owner for an orphaned or unowned node', () => {
    const { publisher, root } = surface()
    expect(chatSurfaceRoot(publisher)).toBe(root)

    // A subtree detached whole still resolves — the surface is still an
    // ancestor. Only losing the surface from the chain orphans the publisher.
    root.remove()
    expect(chatSurfaceRoot(publisher)).toBe(root)

    publisher.remove()
    expect(chatSurfaceRoot(publisher)).toBeNull()
    expect(chatSurfaceRoot(null)).toBeNull()
  })

  it('clears from the captured surface and tolerates a missing one', () => {
    const { publisher, root } = surface()
    setSurfaceVar(publisher, COMPOSER_HEIGHT_VAR, '176px')

    clearSurfaceVar(root, COMPOSER_HEIGHT_VAR)
    expect(root.style.getPropertyValue(COMPOSER_HEIGHT_VAR)).toBe('')

    expect(() => clearSurfaceVar(null, COMPOSER_HEIGHT_VAR)).not.toThrow()
    expect(document.documentElement.style.getPropertyValue(COMPOSER_HEIGHT_VAR)).toBe('')
  })
})
