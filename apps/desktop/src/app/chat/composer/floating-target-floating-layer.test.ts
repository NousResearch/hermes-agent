// @vitest-environment jsdom
import { afterEach, describe, expect, it } from 'vitest'

import { registerFloatingComposer } from './floating-target'

/** The owner's composer host, a chat surface with a plain control, and a
 * body-portaled Radix popper layer holding a control — the shapes the
 * window-level focus-follow has to tell apart. */
function mount() {
  const host = document.createElement('div')
  host.dataset.composerOwner = 'surface-1'
  const editor = document.createElement('div')
  editor.dataset.slot = 'composer-rich-input'
  editor.tabIndex = -1
  host.appendChild(editor)
  document.body.appendChild(host)

  const surface = document.createElement('div')
  surface.dataset.chatSurface = ''
  surface.dataset.composerSurfaceId = 'surface-1'
  const plainButton = document.createElement('button')
  surface.appendChild(plainButton)
  document.body.appendChild(surface)

  const layer = document.createElement('div')
  layer.setAttribute('data-radix-popper-content-wrapper', '')
  const layerButton = document.createElement('button')
  layer.appendChild(layerButton)
  document.body.appendChild(layer)

  return { editor, layerButton, plainButton, surface }
}

/** Button-up movement over the chat surface — every hover of the transcript. */
function movePointerOver(target: Element, x: number) {
  target.dispatchEvent(new PointerEvent('pointermove', { bubbles: true, buttons: 0, clientX: x, clientY: 40 }))
}

/** Radix moves focus into a popover/menu when it opens and dismisses it as soon
 * as focus leaves. The focus-follow used to pull that focus back into the pane
 * composer on the next pointermove, so the message reaction picker closed
 * before the pointer could reach it. */
describe('floating composer focus-follow vs an open floating layer', () => {
  let unregister: (() => void) | undefined

  afterEach(() => {
    unregister?.()
    unregister = undefined
    document.body.innerHTML = ''
  })

  it('leaves focus inside a floating layer, but still follows the pointer otherwise', () => {
    const { editor, layerButton, plainButton, surface } = mount()
    unregister = registerFloatingComposer('surface-1', { groupId: 'g1', target: 'main' })

    layerButton.focus()
    movePointerOver(surface, 10)
    expect(document.activeElement).toBe(layerButton)

    // Control: focus on an ordinary control in the surface is not "owned" —
    // hovering the transcript hands the caret to the composer as before.
    plainButton.focus()
    movePointerOver(surface, 20)
    expect(document.activeElement).toBe(editor)
  })
})
