import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { group, split } from '@/components/pane-shell/tree/model'
import { $activeTreeGroup, $layoutTree, trackActiveTreeGroup } from '@/components/pane-shell/tree/store'
import { $composerPopout, $composerPopoutGesturesEnabled } from '@/store/composer-popout'
import { $selectedStoredSessionId, setSessions } from '@/store/session'
import { $focusedStoredSessionId } from '@/store/session-focus'
import { $focusedWorkspaceCwd } from '@/store/session-states'

import { $floatingComposerOwner } from './floating-state'
import { pinFloatingComposerCapture, registerFloatingComposer } from './floating-target'
import { getActiveComposer } from './focus'

const KEY = 'hermes.desktop.focusFollowsMouse'
const disposers: Array<() => void> = []
let x = 0

// Exercise the same storage adoption used by machine preferences/other windows.
function preference(enabled: boolean) {
  window.localStorage.setItem(KEY, String(enabled))
  window.dispatchEvent(new StorageEvent('storage', { key: KEY, newValue: String(enabled) }))
}

function surface(id: string) {
  const root = document.createElement('section')
  root.dataset.treeGroup = id
  root.dataset.chatSurface = ''
  root.dataset.composerSurfaceId = id
  const text = document.createElement('p')
  text.textContent = `transcript ${id}`
  const host = document.createElement('div')
  host.dataset.composerOwner = id
  host.dataset.treeGroup = id
  const editor = document.createElement('div')
  editor.dataset.slot = 'composer-rich-input'
  editor.contentEditable = 'true'
  editor.tabIndex = 0
  editor.textContent = `draft ${id}`
  host.append(editor)
  const input = document.createElement('input')
  const button = document.createElement('button')
  button.textContent = 'Action'
  root.append(text, host, input, button)
  document.body.append(root)
  disposers.push(registerFloatingComposer(id, { groupId: id, target: `tile:${id}` }))

  return { text, host, editor, input, button }
}

function move(target: Element, buttons = 0, pointerType = 'mouse') {
  target.dispatchEvent(
    new PointerEvent('pointermove', {
      bubbles: true,
      clientX: ++x,
      clientY: 50,
      buttons,
      pointerType
    })
  )
}

function select(editor: HTMLElement) {
  editor.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true, buttons: 1, pointerType: 'mouse' }))
  editor.focus()
  editor.dispatchEvent(new PointerEvent('pointerup', { bubbles: true, pointerType: 'mouse' }))
}

beforeEach(() => {
  preference(false)
  $selectedStoredSessionId.set('A')
  setSessions(['A', 'B'].map(id => ({ id, cwd: `/repo-${id}` })) as never)
  $floatingComposerOwner.set(null)
  $activeTreeGroup.set(null)
  $layoutTree.set(split('row', [group(['session-tile:A'], { id: 'A' }), group(['session-tile:B'], { id: 'B' })]))
  $composerPopoutGesturesEnabled.set(true)
  $composerPopout.set({ poppedOut: false, position: { bottom: 24, right: 24 } })
  disposers.push(trackActiveTreeGroup())
})

afterEach(async () => {
  disposers
    .splice(0)
    .reverse()
    .forEach(stop => stop())
  await Promise.resolve()
  window.getSelection()?.removeAllRanges()
  document.body.replaceChildren()
  preference(false)
})

describe.each([false, true])('configurable mouse focus (floating=%s)', floating => {
  function setup() {
    const a = surface('A')
    const b = surface('B')
    $composerPopout.set({ poppedOut: floating, position: { bottom: 24, right: 24 } })
    select(a.editor)

    return { a, b }
  }

  it('takes effect live: OFF preserves A, ON selects B, then OFF preserves B', () => {
    const { a, b } = setup()
    move(b.text)
    expect(getActiveComposer()).toBe('tile:A')
    preference(true)
    // The setting change itself must not move focus.
    expect(document.activeElement).toBe(a.editor)
    move(b.text)
    expect(getActiveComposer()).toBe('tile:B')
    expect(document.activeElement).toBe(b.editor)
    expect($focusedStoredSessionId.get()).toBe('B')
    expect($focusedWorkspaceCwd.get()).toBe('/repo-B')
    preference(false)
    move(a.text)
    expect(getActiveComposer()).toBe('tile:B')
    expect($focusedWorkspaceCwd.get()).toBe('/repo-B')
    expect(a.editor.textContent).toBe('draft A')
    expect(b.editor.textContent).toBe('draft B')
    select(a.editor)
    expect(getActiveComposer()).toBe('tile:A')
  })

  it('ignores stationary pointerover and drag/touch/pen movement when enabled', () => {
    const { a, b } = setup()
    preference(true)
    b.text.dispatchEvent(new PointerEvent('pointerover', { bubbles: true, pointerType: 'mouse' }))
    move(b.text, 1)
    move(b.text, 0, 'touch')
    move(b.text, 0, 'pen')
    expect(getActiveComposer()).toBe('tile:A')
    expect(document.activeElement).toBe(a.editor)
    move(b.text)
    expect(getActiveComposer()).toBe('tile:B')
  })

  it('keeps a clicked input and a transcript selection intact while hover is enabled', () => {
    const { a, b } = setup()
    preference(true)
    select(a.input)
    a.input.value = 'answer'
    move(b.text)
    expect(document.activeElement).toBe(a.input)
    expect(a.input.value).toBe('answer')
    expect(getActiveComposer()).toBe('tile:A')
    select(a.editor)
    const range = document.createRange()
    range.selectNodeContents(a.text)
    window.getSelection()!.removeAllRanges()
    window.getSelection()!.addRange(range)
    move(b.text)
    expect(window.getSelection()!.toString()).toBe('transcript A')
    expect(getActiveComposer()).toBe('tile:A')
  })

  it('preserves controls, menus and inline edits while the pointer crosses another pane', () => {
    const { a, b } = setup()
    preference(true)
    const media = document.createElement('video')
    media.controls = true
    media.tabIndex = 0
    a.text.after(media)
    const layer = document.createElement('div')
    layer.dataset.radixPopperContentWrapper = ''
    const menuButton = document.createElement('button')
    layer.append(menuButton)
    document.body.append(layer)
    const inline = document.createElement('div')
    inline.dataset.slot = 'aui_edit-composer-root'
    const editInput = document.createElement('textarea')
    inline.append(editInput)
    a.text.after(inline)

    for (const control of [a.button, media, menuButton, editInput]) {
      select(control)
      move(b.text)
      expect(document.activeElement).toBe(control)
      expect(getActiveComposer()).toBe('tile:A')
    }
  })

  it('does not switch during IME composition, and follows the next movement after commit', () => {
    const { a, b } = setup()
    preference(true)
    a.editor.dispatchEvent(new CompositionEvent('compositionstart', { bubbles: true, data: '' }))
    move(b.text)
    expect(getActiveComposer()).toBe('tile:A')
    expect(document.activeElement).toBe(a.editor)
    a.editor.dispatchEvent(new CompositionEvent('compositionend', { bubbles: true, data: '中文' }))
    move(b.text)
    expect(getActiveComposer()).toBe('tile:B')
  })

  it.each(['blur', 'ordinary key'])('recovers mouse focus after %s when compositionend was missed', recovery => {
    const { a, b } = setup()
    preference(true)
    a.editor.dispatchEvent(new CompositionEvent('compositionstart', { bubbles: true }))
    move(b.text)
    expect(getActiveComposer()).toBe('tile:A')

    if (recovery === 'blur') {
      a.editor.blur()
      select(a.editor)
    } else {
      a.editor.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, key: 'a', isComposing: false }))
    }

    move(b.text)
    expect(getActiveComposer()).toBe('tile:B')
    expect(document.activeElement).toBe(b.editor)
    expect($focusedWorkspaceCwd.get()).toBe('/repo-B')
  })

  it('keeps genuine composition protected through composing keydown and keyup', () => {
    const { a, b } = setup()
    preference(true)
    a.editor.dispatchEvent(new CompositionEvent('compositionstart', { bubbles: true }))
    a.editor.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, key: 'Process', isComposing: true }))
    a.editor.dispatchEvent(new KeyboardEvent('keyup', { bubbles: true, key: 'Process', isComposing: false }))
    move(b.text)
    expect(getActiveComposer()).toBe('tile:A')
    expect(document.activeElement).toBe(a.editor)
    a.editor.dispatchEvent(new CompositionEvent('compositionend', { bubbles: true }))
    move(b.text)
    expect(getActiveComposer()).toBe('tile:B')
  })

  it('pins capture only while floating and resumes hover after release', () => {
    const { b } = setup()
    preference(true)
    const release = pinFloatingComposerCapture('A')
    disposers.push(release)
    move(b.text)
    expect(getActiveComposer()).toBe(floating ? 'tile:A' : 'tile:B')
    release()
    move(b.text)
    expect(getActiveComposer()).toBe('tile:B')
  })
})
