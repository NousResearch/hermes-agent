import { beforeEach, describe, expect, it } from 'vitest'

import { hudIgnoresMouse } from './click-through'

/** The HUD's real shape: a React mount wrapping the shell, the bar inside it,
 *  and a portalled overlay as a sibling of the mount under <body>. */
function hud() {
  document.body.innerHTML = ''

  const mount = document.createElement('div')
  mount.id = 'root'

  const shell = document.createElement('div')
  shell.setAttribute('data-hud-shell', '')

  const bar = document.createElement('input')

  const overlay = document.createElement('div')
  overlay.setAttribute('role', 'dialog')

  shell.append(bar)
  mount.append(shell)
  document.body.append(mount, overlay)

  return { bar, mount, overlay, shell }
}

describe('hudIgnoresMouse', () => {
  beforeEach(() => {
    document.body.innerHTML = ''
  })

  it('hands the mouse back over the scaffolding the HUD hangs in', () => {
    const { mount, shell } = hud()

    for (const around of [mount, document.body, document.documentElement, shell]) {
      expect(hudIgnoresMouse(shell, around, null, true)).toBe(true)
    }
  })

  it('keeps the window solid over the HUD and over portalled overlays', () => {
    const { bar, overlay, shell } = hud()

    expect(hudIgnoresMouse(shell, bar, null, true)).toBe(false)
    expect(hudIgnoresMouse(shell, overlay, null, true)).toBe(false)
  })

  it('does not pin the window just because the composer holds the caret', () => {
    const { bar, mount, shell } = hud()

    expect(hudIgnoresMouse(shell, mount, bar, true)).toBe(true)
  })

  it('pins the window while a portalled overlay holds focus, so an outside click can dismiss it', () => {
    const { mount, overlay, shell } = hud()

    expect(hudIgnoresMouse(shell, mount, overlay, true)).toBe(false)
  })

  it('lets a stale focus go once the window is no longer the active one', () => {
    const { mount, overlay, shell } = hud()

    expect(hudIgnoresMouse(shell, mount, overlay, false)).toBe(true)
  })

  it('treats an unfocused document body as nothing holding focus', () => {
    const { mount, shell } = hud()

    expect(hudIgnoresMouse(shell, mount, document.body, true)).toBe(true)
  })

  it('stays solid while a gesture owns the window, even once the hit test goes empty', () => {
    const { mount, shell } = hud()
    const handle = document.createElement('div')
    handle.setAttribute('data-hud-grabbing', '')
    shell.append(handle)

    // The corner resize grows the window out from under the cursor, so the hit
    // test reports the scaffolding — handing the mouse away mid-gesture.
    expect(hudIgnoresMouse(shell, mount, null, true)).toBe(false)
    expect(hudIgnoresMouse(shell, null, null, true)).toBe(false)
  })
})
