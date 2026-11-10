// @vitest-environment jsdom
import { afterEach, describe, expect, it } from 'vitest'

import { registerFloatingComposer } from './floating-target'

/** One pane: its composer host plus the chat surface the transcript lives in.
 * Inline media players delivered by the agent render inside that surface. */
function mountPane(id: string) {
  const host = document.createElement('div')
  host.dataset.composerOwner = id
  const editor = document.createElement('div')
  editor.dataset.slot = 'composer-rich-input'
  editor.tabIndex = -1
  host.appendChild(editor)
  document.body.appendChild(host)

  const surface = document.createElement('div')
  surface.dataset.chatSurface = ''
  surface.dataset.composerSurfaceId = id
  document.body.appendChild(surface)

  return { editor, surface }
}

/** A native media element with controls. When its overflow menu opens,
 * Chromium focuses content in the user-agent shadow root and the focus event
 * outside that root targets the AUDIO/VIDEO host, which is what we simulate. */
function mountPlayer(surface: Element, tag: 'audio' | 'video', controls = true) {
  const player = document.createElement(tag)
  player.controls = controls
  player.tabIndex = 0
  surface.appendChild(player)

  return player
}

function movePointerOver(target: Element, x: number) {
  target.dispatchEvent(new PointerEvent('pointermove', { bubbles: true, buttons: 0, clientX: x, clientY: 44 }))
}

/** Opening the ⋮ menu (Download, Playback speed) of an inline player moved
 * focus to the AUDIO/VIDEO host; the focus-follow handed it to the composer
 * and Chromium closed the menu the instant it opened (#135225). */
describe('floating composer focus-follow vs native media controls', () => {
  const unregister: Array<() => void> = []

  afterEach(() => {
    unregister.splice(0).forEach(fn => fn())
    document.body.innerHTML = ''
  })

  it.each(['audio', 'video'] as const)('keeps focus on a <%s controls> menu on focusin and on pointermove', tag => {
    const { editor, surface } = mountPane('surface-1')
    const player = mountPlayer(surface, tag)
    unregister.push(registerFloatingComposer('surface-1', { groupId: 'g1', target: 'main' }))

    player.focus()
    expect(document.activeElement).toBe(player)

    movePointerOver(player, 43)
    expect(document.activeElement).toBe(player)
    expect(document.activeElement).not.toBe(editor)
  })

  it('still redirects focus that lands on a media element without native controls', () => {
    const { editor, surface } = mountPane('surface-1')
    const player = mountPlayer(surface, 'video', false)
    unregister.push(registerFloatingComposer('surface-1', { groupId: 'g1', target: 'main' }))

    player.focus()
    expect(document.activeElement).toBe(editor)
  })
})
