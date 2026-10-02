// @vitest-environment jsdom
// Right-click clipboard verbs where the gesture actually lands: the composer.
// The custom menu is the ONLY menu the window shows (AppContextMenu listens in
// the capture phase and stops the gesture), so a dead verb here means the mouse
// cannot copy or paste at all — ⌘C/⌘V go through the native menu roles instead.
//
// jsdom cannot reproduce the contenteditable half of this gesture (it leaves
// `isContentEditable` undefined, so target resolution finds no editable), so
// these cases drive the menu with the target the gesture produces and pin the
// renderer half: every verb acts on THAT field, never through main.
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { onComposerInsertRequest } from '@/app/chat/composer/focus'
import { I18nProvider } from '@/i18n'

import { AppContextMenu } from './app-context-menu'
import { $contextMenu, openDomContextMenu } from './store'
import type { ContextMenuDomTarget } from './target'

const desktopWindow = window as unknown as { hermesDesktop?: Window['hermesDesktop'] }

function installBridge(partial: Partial<Window['hermesDesktop']> = {}) {
  desktopWindow.hermesDesktop = {
    readClipboard: vi.fn().mockResolvedValue(''),
    writeClipboard: vi.fn().mockResolvedValue(true),
    ...partial
  } as unknown as Window['hermesDesktop']
}

function mountMenu() {
  return render(
    <MemoryRouter>
      <I18nProvider configClient={null} initialLocale="en">
        <AppContextMenu />
      </I18nProvider>
    </MemoryRouter>
  )
}

/** A composer field: the paste path keys its composer routing off the edit
 *  composer root (`composerRoot` in app-context-menu.tsx). */
function attachComposer(text = 'hello composer') {
  const root = window.document.createElement('div')

  root.dataset.slot = 'aui_edit-composer-root'

  const editor = window.document.createElement('div')

  editor.contentEditable = 'true'
  editor.textContent = text
  root.appendChild(editor)
  window.document.body.appendChild(root)

  return editor
}

function targetFor(editor: HTMLElement, overrides: Partial<ContextMenuDomTarget> = {}): ContextMenuDomTarget {
  return {
    dialogPortalContainer: null,
    editable: editor,
    imageUrl: '',
    linkUrl: '',
    onImage: false,
    selectionText: window.getSelection()?.toString() ?? '',
    ...overrides
  }
}

function selectIn(editor: HTMLElement, start: number, end: number) {
  const text = editor.firstChild!

  const range = window.document.createRange()

  range.setStart(text, start)
  range.setEnd(text, end)

  const selection = window.getSelection()

  selection?.removeAllRanges()
  selection?.addRange(range)
}

/** The store opens the menu through a nanostores update, so the row appears a
 *  tick after `openDomContextMenu` returns. */
const menuItem = async (label: string) =>
  (await screen.findByText(label)).closest('[data-slot="dropdown-menu-item"]') as HTMLElement

afterEach(() => {
  $contextMenu.set(null)
  cleanup()
  vi.restoreAllMocks()
  window.document.body.innerHTML = ''
  delete desktopWindow.hermesDesktop
})

describe('right-click on the composer', () => {
  it('pastes the clipboard into the composer through its insert bus', async () => {
    installBridge({ readClipboard: vi.fn().mockResolvedValue('pasted text') } as never)

    const inserts: Array<{ mode: string; target: string; text: string }> = []

    const unsubscribe = onComposerInsertRequest(detail =>
      inserts.push({ mode: detail.mode, target: detail.target, text: detail.text })
    )

    mountMenu()
    const editor = attachComposer()

    openDomContextMenu(40, 40, targetFor(editor, { selectionText: '' }))

    const paste = await menuItem('Paste')

    // Paste is never gated on a clipboard probe: the probe can report empty on a
    // platform where the read itself succeeds (#91553).
    expect(paste.getAttribute('data-disabled')).toBeNull()

    fireEvent.click(paste)

    await waitFor(() => expect(inserts).toEqual([{ mode: 'inline', target: 'main', text: 'pasted text' }]))

    unsubscribe()
  })

  it('copies the composer selection through the clipboard bridge', async () => {
    const writeClipboard = vi.fn().mockResolvedValue(true)

    installBridge({ writeClipboard } as never)

    mountMenu()
    const editor = attachComposer('pick this text')

    selectIn(editor, 5, 9) // "this"
    openDomContextMenu(40, 40, targetFor(editor))

    const copy = await menuItem('Copy')

    expect(copy.getAttribute('data-disabled')).toBeNull()

    fireEvent.click(copy)

    await waitFor(() => expect(writeClipboard).toHaveBeenCalledWith('this'))
    expect(editor.textContent).toBe('pick this text')
  })

  it('cuts the composer selection through the clipboard bridge', async () => {
    const writeClipboard = vi.fn().mockResolvedValue(true)

    installBridge({ writeClipboard } as never)

    mountMenu()
    const editor = attachComposer('pick this text')

    selectIn(editor, 5, 9) // "this"
    openDomContextMenu(40, 40, targetFor(editor))

    fireEvent.click(await menuItem('Cut'))

    await waitFor(() => expect(writeClipboard).toHaveBeenCalledWith('this'))
    // The DOM removal itself is deliberately NOT asserted here: jsdom ships no
    // `Selection.deleteFromDocument`, so the field cannot change in this
    // environment (Chromium deletes the range; undo coverage for that mutation
    // is covered by the composer's own undo-cut-drag suite).
  })
})
