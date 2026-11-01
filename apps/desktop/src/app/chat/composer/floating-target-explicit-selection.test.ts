import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { group, split } from '@/components/pane-shell/tree/model'
import { $activeTreeGroup, $layoutTree, trackActiveTreeGroup } from '@/components/pane-shell/tree/store'
import { $composerPopout, $composerPopoutGesturesEnabled } from '@/store/composer-popout'
import { $currentCwd, $selectedStoredSessionId, $workspaceCwdOwner, setSessions } from '@/store/session'
import { $focusedStoredSessionId } from '@/store/session-focus'
import { $focusedWorkspaceCwd } from '@/store/session-states'

import { $floatingComposerOwner } from './floating-state'
import { registerFloatingComposer } from './floating-target'
import { getActiveComposer, onComposerFocusRequest, requestComposerFocus } from './focus'

const disposers: Array<() => void> = []

// Only session metadata and DOM are synthetic. The composer, layout tracker,
// focused-session derivation and the CWD store consumed by Files are real.

function pane(id: string, groupId: string) {
  const surface = document.createElement('section')
  surface.dataset.treeGroup = groupId
  surface.dataset.chatSurface = ''
  surface.dataset.composerSurfaceId = id
  surface.dataset.composerTarget = `tile:${id}`
  const text = document.createElement('p')
  text.textContent = `transcript ${id}`
  const button = document.createElement('button')
  button.textContent = `Action ${id}`
  const host = document.createElement('div')
  host.dataset.treeGroup = groupId
  host.dataset.composerOwner = id
  const editor = document.createElement('div')
  editor.dataset.slot = 'composer-rich-input'
  editor.tabIndex = 0
  editor.contentEditable = 'true'
  editor.textContent = `draft ${id}`
  host.append(editor)
  surface.append(text, button, host)
  document.body.append(surface)
  disposers.push(registerFloatingComposer(id, { groupId, target: `tile:${id}` }))

  return { surface, text, button, host, editor }
}

function pointer(target: Element, type: string, x: number, buttons = 0) {
  target.dispatchEvent(
    new PointerEvent(type, { bubbles: true, pointerType: 'mouse', clientX: x, clientY: 50, buttons })
  )
}

function clickEditor(editor: HTMLElement) {
  pointer(editor, 'pointerdown', 1, 1)
  editor.focus()
  pointer(editor, 'pointerup', 1)
}

beforeEach(() => {
  $selectedStoredSessionId.set('A')
  $workspaceCwdOwner.set('A')
  $currentCwd.set('/repo-A')
  setSessions([
    { id: 'A', cwd: '/repo-A' },
    { id: 'B', cwd: '/repo-B' }
  ] as never)
  $floatingComposerOwner.set(null)
  $activeTreeGroup.set(null)
  $layoutTree.set(
    split('row', [
      group(['files'], { id: 'files' }),
      group(['workspace'], { id: 'ga' }),
      group(['session-tile:B'], { id: 'gb' })
    ])
  )
  $composerPopoutGesturesEnabled.set(true)
  $composerPopout.set({ poppedOut: false, position: { bottom: 24, right: 24 } })
  disposers.push(trackActiveTreeGroup())
})

afterEach(async () => {
  disposers
    .splice(0)
    .reverse()
    .forEach(dispose => dispose())
  await Promise.resolve()
  document.body.replaceChildren()
  window.getSelection()?.removeAllRanges()
  $floatingComposerOwner.set(null)
  $activeTreeGroup.set(null)
  vi.useRealTimers()
})

describe.each([false, true])('explicit composer selection (floating=%s)', floating => {
  it('keeps the selected session, draft and input recipient while crossing another transcript to Files', () => {
    const a = pane('A', 'ga')
    const b = pane('B', 'gb')
    $composerPopout.set({ poppedOut: floating, position: { bottom: 24, right: 24 } })
    clickEditor(b.editor)
    const files = document.createElement('aside')
    files.dataset.treeGroup = 'files'
    document.body.append(files)

    pointer(a.text, 'pointerover', 200)
    pointer(a.text, 'pointermove', 201)
    pointer(files, 'pointerover', 202)
    pointer(files, 'pointermove', 203)

    expect(document.activeElement).toBe(b.editor)
    expect($floatingComposerOwner.get()?.id).toBe('B')
    expect(getActiveComposer()).toBe('tile:B')
    expect($focusedStoredSessionId.get()).toBe('B')
    expect($focusedWorkspaceCwd.get()).toBe('/repo-B')
    expect(a.editor.textContent).toBe('draft A')
    expect(b.editor.textContent).toBe('draft B')
  })

  it('selects another transcript on a primary click and routes the next typing request there', () => {
    vi.useFakeTimers()
    const a = pane('A', 'ga')
    const b = pane('B', 'gb')
    clickEditor(b.editor)
    $composerPopout.set({ poppedOut: floating, position: { bottom: 24, right: 24 } })
    pointer(a.text, 'pointerdown', 30, 1)
    pointer(a.text, 'pointerup', 30)
    expect(getActiveComposer()).toBe('tile:A')
    expect($focusedStoredSessionId.get()).toBe('A')
    const received: string[] = []
    disposers.push(onComposerFocusRequest(({ target }) => received.push(target)))
    requestComposerFocus('active', { typeChar: 'x' })
    expect(received).toEqual(['tile:A'])
    expect(a.editor.textContent).toBe('draft A')
  })

  it('accepts keyboard Tab focus without changing the target again when the mouse moves', () => {
    const a = pane('A', 'ga')
    const b = pane('B', 'gb')
    clickEditor(a.editor)
    $composerPopout.set({ poppedOut: floating, position: { bottom: 24, right: 24 } })
    document.body.dispatchEvent(new KeyboardEvent('keydown', { key: 'Tab', bubbles: true }))
    b.button.focus()
    document.body.dispatchEvent(new KeyboardEvent('keyup', { key: 'Tab', bubbles: true }))
    pointer(a.text, 'pointermove', 40)
    expect(document.activeElement).toBe(b.button)
    expect(getActiveComposer()).toBe('tile:B')
    expect($focusedStoredSessionId.get()).toBe('B')
  })
})
