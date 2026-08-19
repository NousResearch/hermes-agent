import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { registerTerminalContextMenu } from '@/app/right-sidebar/terminal/terminal-context-menu'
import { formatCombo } from '@/lib/keybinds/combo'
import { $previewTabs, closeRightRail } from '@/store/preview'
import { $connection } from '@/store/session'

import { AppContextMenu } from './app-context-menu'
import {
  $contextMenu,
  augmentSpellcheck,
  type GuestMenuHandle,
  type GuestMenuParams,
  openGuestContextMenu
} from './store'
import { resolveDomTarget } from './target'

const desktopWindow = window as unknown as { hermesDesktop?: Window['hermesDesktop'] }

function installBridge(partial: Partial<Window['hermesDesktop']> = {}) {
  desktopWindow.hermesDesktop = {
    openExternal: vi.fn().mockResolvedValue(undefined),
    writeClipboard: vi.fn().mockResolvedValue(undefined),
    ...partial
  } as unknown as Window['hermesDesktop']
}

function mountMenu() {
  return render(
    <MemoryRouter>
      <AppContextMenu />
    </MemoryRouter>
  )
}

function attach(html: string): HTMLElement {
  const host = document.createElement('div')

  host.innerHTML = html
  document.body.appendChild(host)

  return host
}

afterEach(() => {
  $contextMenu.set(null)
  $connection.set(null)
  closeRightRail()
  cleanup()
  vi.restoreAllMocks()
  document.body.innerHTML = ''
  delete desktopWindow.hermesDesktop
})

describe('resolveDomTarget', () => {
  it('resolves an anchor and its href', () => {
    const host = attach('<a href="https://example.com/x">link</a>')

    expect(resolveDomTarget(host.querySelector('a')).linkUrl).toBe('https://example.com/x')
  })

  it('ignores hash-only placeholder anchors', () => {
    const host = attach('<a href="#">stub</a>')

    expect(resolveDomTarget(host.querySelector('a')).linkUrl).toBe('')
  })

  it('resolves an image, and the anchor wrapping it', () => {
    const host = attach('<a href="https://example.com/page"><img src="https://example.com/pic.png"></a>')
    const target = resolveDomTarget(host.querySelector('img'))

    expect(target.onImage).toBe(true)
    expect(target.imageUrl).toContain('pic.png')
    expect(target.linkUrl).toBe('https://example.com/page')
  })

  it('resolves editables, skipping disabled and readonly fields', () => {
    const host = attach('<textarea></textarea><input readonly><div contenteditable="true"><span>x</span></div>')

    expect(resolveDomTarget(host.querySelector('textarea')).editable).toBeTruthy()
    expect(resolveDomTarget(host.querySelector('input')).editable).toBeNull()
  })
})

describe('AppContextMenu', () => {
  it('opens the link menu on a chat link right-click', async () => {
    installBridge()
    mountMenu()
    const host = attach('<a href="https://example.com/docs">Docs</a>')

    fireEvent.contextMenu(host.querySelector('a')!)

    expect(await screen.findByText('Open in in-app browser')).toBeTruthy()
    expect(screen.getByText('Open in external browser')).toBeTruthy()
    expect(screen.getByText('Copy URL')).toBeTruthy()
    expect(screen.queryByText('Copy resolved URL')).toBeNull()
  })

  it('opens the in-app browser from the link menu', async () => {
    installBridge()
    mountMenu()
    const host = attach('<a href="https://example.com/docs">Docs</a>')

    fireEvent.contextMenu(host.querySelector('a')!)
    fireEvent.click(await screen.findByText('Open in in-app browser'))

    await waitFor(() => expect($previewTabs.get().at(-1)?.target.url).toBe('https://example.com/docs'))
  })

  it('offers the resolved copy only for loopback links on a remote gateway', async () => {
    $connection.set({ mode: 'remote' } as never)
    const reachPreviewUrl = vi.fn(async () => 'http://127.0.0.1:45173/')
    const writeClipboard = vi.fn().mockResolvedValue(undefined)

    installBridge({
      reachPreviewUrl: reachPreviewUrl as unknown as Window['hermesDesktop']['reachPreviewUrl'],
      writeClipboard: writeClipboard as unknown as Window['hermesDesktop']['writeClipboard']
    })
    mountMenu()
    const host = attach('<a href="http://localhost:5173/">Dev</a>')

    fireEvent.contextMenu(host.querySelector('a')!)
    fireEvent.click(await screen.findByText('Copy resolved URL'))

    await waitFor(() => expect(writeClipboard).toHaveBeenCalledWith('http://127.0.0.1:45173/'))
  })

  it('opens the image menu with copy, address, and save', async () => {
    installBridge()
    mountMenu()
    const host = attach('<img src="https://example.com/pic.png" alt="pic">')

    fireEvent.contextMenu(host.querySelector('img')!)

    expect(await screen.findByText('Copy image')).toBeTruthy()
    expect(screen.getByText('Copy image address')).toBeTruthy()
    expect(screen.getByText('Save image as…')).toBeTruthy()
  })

  it('opens the edit menu in an editable and augments it with spellcheck', async () => {
    installBridge()
    mountMenu()
    const host = attach('<textarea></textarea>')

    fireEvent.contextMenu(host.querySelector('textarea')!)

    expect(await screen.findByText('Select all')).toBeTruthy()
    expect(screen.getByText('Paste')).toBeTruthy()
    expect(screen.queryByText('Add to dictionary')).toBeNull()

    // The main-process forward lands after the menu opened.
    augmentSpellcheck({ misspelledWord: 'teh', suggestions: ['the', 'ten'] })

    expect(await screen.findByText('Add to dictionary')).toBeTruthy()
    expect(screen.getByText('the')).toBeTruthy()
  })

  it('shows edit verbs without icons and with faded accelerators', async () => {
    installBridge()
    mountMenu()
    const host = attach('<textarea></textarea>')

    fireEvent.contextMenu(host.querySelector('textarea')!)

    // formatCombo picks ⌘/Ctrl from the host running the test, exactly like
    // the menu itself — so the assertion is platform-honest, not hardcoded.
    const pasteItem = (await screen.findByText('Paste')).closest('[data-slot="dropdown-menu-item"]')!

    expect(pasteItem.querySelector('[data-slot="dropdown-menu-shortcut"]')?.textContent).toBe(formatCombo('mod+v'))
    expect(pasteItem.querySelector('.codicon')).toBeNull()

    const selectAllItem = screen.getByText('Select all').closest('[data-slot="dropdown-menu-item"]')!

    expect(selectAllItem.querySelector('[data-slot="dropdown-menu-shortcut"]')?.textContent).toBe(formatCombo('mod+a'))
    expect(selectAllItem.querySelector('.codicon')).toBeNull()
  })

  it('runs edit verbs after the menu closed, with focus back on the editable', async () => {
    const contextMenuEdit = vi.fn().mockResolvedValue(undefined)

    installBridge({ contextMenuEdit: contextMenuEdit as unknown as Window['hermesDesktop']['contextMenuEdit'] })
    mountMenu()
    const host = attach('<textarea>some draft text</textarea>')
    const textarea = host.querySelector('textarea')!

    // Cut/copy act on the selection, so give the field one.
    textarea.setSelectionRange(0, 4)
    fireEvent.contextMenu(textarea)
    fireEvent.click(await screen.findByText('Copy'))

    // The verb waits a frame so the radix focus trap unmounts first —
    // dispatching while the trap holds focus sent the command to `body`.
    expect(contextMenuEdit).not.toHaveBeenCalled()

    await waitFor(() => expect(contextMenuEdit).toHaveBeenCalledWith('copy'))
    expect($contextMenu.get()).toBeNull()
    expect(document.activeElement).toBe(textarea)
  })

  it('grays out cut and copy when the field has text but no selection', async () => {
    installBridge()
    mountMenu()
    const host = attach('<textarea>plenty of text, none selected</textarea>')
    const textarea = host.querySelector('textarea')!

    textarea.setSelectionRange(0, 0)
    fireEvent.contextMenu(textarea)

    const item = (label: string) => screen.getByText(label).closest('[data-slot="dropdown-menu-item"]') as HTMLElement

    await screen.findByText('Select all')

    expect(item('Cut').getAttribute('data-disabled')).not.toBeNull()
    expect(item('Copy').getAttribute('data-disabled')).not.toBeNull()
    // Content is there, so select all stays live.
    expect(item('Select all').getAttribute('data-disabled')).toBeNull()
  })

  it('enables cut and copy when the field has a selection', async () => {
    installBridge()
    mountMenu()
    const host = attach('<textarea>pick some of this</textarea>')
    const textarea = host.querySelector('textarea')!

    textarea.setSelectionRange(0, 4)
    fireEvent.contextMenu(textarea)

    const item = (label: string) => screen.getByText(label).closest('[data-slot="dropdown-menu-item"]') as HTMLElement

    await screen.findByText('Select all')

    expect(item('Cut').getAttribute('data-disabled')).toBeNull()
    expect(item('Copy').getAttribute('data-disabled')).toBeNull()
  })

  it('select all stays inside the field and never reaches main', async () => {
    const contextMenuEdit = vi.fn().mockResolvedValue(undefined)

    installBridge({ contextMenuEdit: contextMenuEdit as unknown as Window['hermesDesktop']['contextMenuEdit'] })
    mountMenu()
    const host = attach('<textarea>alpha beta gamma</textarea>')
    const textarea = host.querySelector('textarea')!

    fireEvent.contextMenu(textarea)
    fireEvent.click(await screen.findByText('Select all'))

    // Renderer-side selection scoped to the field: main's selectAll acts on
    // the focused FRAME and selected the whole transcript when focus
    // slipped (the edit composer re-parents focus on blur).
    await waitFor(() => {
      expect(textarea.selectionStart).toBe(0)
      expect(textarea.selectionEnd).toBe('alpha beta gamma'.length)
    })
    expect(contextMenuEdit).not.toHaveBeenCalled()
    expect(document.activeElement).toBe(textarea)
  })

  it('splits select all into its own section under the edit verbs', async () => {
    installBridge()
    mountMenu()
    const host = attach('<textarea>text</textarea>')

    fireEvent.contextMenu(host.querySelector('textarea')!)

    const selectAllItem = (await screen.findByText('Select all')).closest('[data-slot="dropdown-menu-item"]')!
    const pasteItem = screen.getByText('Paste').closest('[data-slot="dropdown-menu-item"]')!

    // Sections render as sibling `.contents` wrappers with the separator
    // inside the later one — different wrappers = different sections.
    expect(pasteItem.parentElement).not.toBe(selectAllItem.parentElement)
    expect(selectAllItem.parentElement?.querySelector('[data-slot="dropdown-menu-separator"]')).not.toBeNull()
  })

  it('grays out cut, copy, and select all in an empty field', async () => {
    installBridge()
    mountMenu()
    const host = attach('<textarea></textarea>')

    fireEvent.contextMenu(host.querySelector('textarea')!)

    const item = (label: string) => screen.getByText(label).closest('[data-slot="dropdown-menu-item"]') as HTMLElement

    await screen.findByText('Select all')

    expect(item('Cut').getAttribute('data-disabled')).not.toBeNull()
    expect(item('Copy').getAttribute('data-disabled')).not.toBeNull()
    expect(item('Select all').getAttribute('data-disabled')).not.toBeNull()
  })

  it('grays out paste until the clipboard reports text', async () => {
    const readClipboard = vi.fn().mockResolvedValue('clip content')

    installBridge({ readClipboard: readClipboard as unknown as Window['hermesDesktop']['readClipboard'] })
    mountMenu()
    const host = attach('<textarea>text</textarea>')

    fireEvent.contextMenu(host.querySelector('textarea')!)

    // The clipboard read is async — the item enables when it lands.
    const pasteItem = (await screen.findByText('Paste')).closest('[data-slot="dropdown-menu-item"]')!

    await waitFor(() => expect(pasteItem.getAttribute('data-disabled')).toBeNull())
  })

  it('keeps paste grayed out on an empty clipboard', async () => {
    const readClipboard = vi.fn().mockResolvedValue('')

    installBridge({ readClipboard: readClipboard as unknown as Window['hermesDesktop']['readClipboard'] })
    mountMenu()
    const host = attach('<textarea>text</textarea>')

    fireEvent.contextMenu(host.querySelector('textarea')!)

    const pasteItem = (await screen.findByText('Paste')).closest('[data-slot="dropdown-menu-item"]')!

    await waitFor(() => expect(readClipboard).toHaveBeenCalled())
    expect(pasteItem.getAttribute('data-disabled')).not.toBeNull()
  })

  it('offers the window verbs on bare chrome', async () => {
    installBridge()
    mountMenu()
    const host = attach('<div><p>plain chrome</p></div>')

    fireEvent.contextMenu(host.querySelector('p')!)

    expect(await screen.findByText('Settings')).toBeTruthy()
  })

  it('skips plain right-clicks inside a skip-marked surface, but not links in it', async () => {
    installBridge()
    mountMenu()

    const host = attach(
      '<div data-context-menu-skip=""><p>bubble text</p><a href="https://example.com/in">In-bubble</a></div>'
    )

    fireEvent.contextMenu(host.querySelector('p')!)
    expect($contextMenu.get()).toBeNull()

    fireEvent.contextMenu(host.querySelector('a')!)
    expect(await screen.findByText('Copy URL')).toBeTruthy()
  })

  it('leaves surfaces with their own radix menu alone', () => {
    installBridge()
    mountMenu()
    const host = attach('<div data-slot="context-menu-trigger"><span>session row</span></div>')

    fireEvent.contextMenu(host.querySelector('span')!)

    expect($contextMenu.get()).toBeNull()
  })

  it('shows the terminal menu through a registered handle', async () => {
    installBridge()
    mountMenu()
    const host = attach('<div data-terminal=""><canvas></canvas></div>')
    const paste = vi.fn()

    const unregister = registerTerminalContextMenu(host.querySelector('[data-terminal]')!, {
      getSelection: () => 'picked text',
      paste,
      selectAll: vi.fn()
    })

    fireEvent.contextMenu(host.querySelector('canvas')!)

    expect(await screen.findByText('Copy')).toBeTruthy()
    expect(screen.getByText('Paste')).toBeTruthy()
    expect(screen.getByText('Select all')).toBeTruthy()
    unregister()
  })

  it('hides paste on the read-only agent terminal', async () => {
    installBridge()
    mountMenu()
    const host = attach('<div data-terminal=""><canvas></canvas></div>')

    const unregister = registerTerminalContextMenu(host.querySelector('[data-terminal]')!, {
      getSelection: () => '',
      paste: null,
      selectAll: vi.fn()
    })

    fireEvent.contextMenu(host.querySelector('canvas')!)

    expect(await screen.findByText('Select all')).toBeTruthy()
    expect(screen.queryByText('Paste')).toBeNull()
    unregister()
  })
})

describe('AppContextMenu guest (in-app browser)', () => {
  const guestHandle = (overrides: Partial<GuestMenuHandle> = {}): GuestMenuHandle => ({
    addToDictionary: vi.fn(),
    copyImage: vi.fn(),
    editCommand: vi.fn(),
    inspectElement: vi.fn(),
    replaceMisspelling: vi.fn(),
    ...overrides
  })

  const guestParams = (overrides: Partial<GuestMenuParams> = {}): GuestMenuParams => ({
    dictionarySuggestions: [],
    editFlags: { canCopy: true, canCut: true, canPaste: true, canSelectAll: true },
    hasImageContents: false,
    isEditable: false,
    linkURL: '',
    misspelledWord: '',
    selectionText: '',
    srcURL: '',
    ...overrides
  })

  it('shows text tools and inspect on a bare page right-click', async () => {
    installBridge()
    mountMenu()
    const guest = guestHandle()

    openGuestContextMenu(10, 10, guestParams(), guest)

    expect(await screen.findByText('Select all')).toBeTruthy()
    expect(screen.getByText('Inspect element')).toBeTruthy()
    // The page verbs live on the browser bar only now.
    expect(screen.queryByText('Copy page URL')).toBeNull()
    expect(screen.queryByText('Open in browser')).toBeNull()
    expect(screen.queryByText('Show preview console')).toBeNull()
  })

  it('draws a line between select all and inspect element', async () => {
    installBridge()
    mountMenu()

    openGuestContextMenu(10, 10, guestParams(), guestHandle())

    const selectAllItem = (await screen.findByText('Select all')).closest('[data-slot="dropdown-menu-item"]')!
    const inspectItem = screen.getByText('Inspect element').closest('[data-slot="dropdown-menu-item"]')!

    expect(selectAllItem.parentElement).not.toBe(inspectItem.parentElement)
    expect(inspectItem.parentElement?.querySelector('[data-slot="dropdown-menu-separator"]')).not.toBeNull()
  })

  it('runs inspect element against the handle', async () => {
    installBridge()
    mountMenu()
    const guest = guestHandle()

    openGuestContextMenu(10, 10, guestParams(), guest)
    fireEvent.click(await screen.findByText('Inspect element'))

    expect(guest.inspectElement).toHaveBeenCalled()
  })

  it('adds a link section above the tools for guest links', async () => {
    installBridge()
    mountMenu()

    openGuestContextMenu(10, 10, guestParams({ linkURL: 'https://example.com/deep' }), guestHandle())

    expect(await screen.findByText('Open in in-app browser')).toBeTruthy()
    expect(screen.getByText('Copy URL')).toBeTruthy()
    // Inspect element rides every guest menu; the bar-only verbs do not.
    expect(screen.getByText('Inspect element')).toBeTruthy()
    expect(screen.queryByText('Copy page URL')).toBeNull()
  })

  it('keeps inspect element in guest editable menus', async () => {
    installBridge()
    mountMenu()

    openGuestContextMenu(10, 10, guestParams({ isEditable: true }), guestHandle())

    expect(await screen.findByText('Paste')).toBeTruthy()
    expect(screen.getByText('Inspect element')).toBeTruthy()
    expect(screen.queryByText('Copy page URL')).toBeNull()
    expect(screen.queryByText('Open in browser')).toBeNull()
    expect(screen.queryByText('Show preview console')).toBeNull()
  })

  it('grays out guest edit verbs from Chromium editFlags', async () => {
    installBridge()
    mountMenu()

    // An empty input: Chromium reports nothing to cut/copy/select, paste ok.
    openGuestContextMenu(
      10,
      10,
      guestParams({
        editFlags: { canCopy: false, canCut: false, canPaste: true, canSelectAll: false },
        isEditable: true
      }),
      guestHandle()
    )

    const item = (label: string) => screen.getByText(label).closest('[data-slot="dropdown-menu-item"]') as HTMLElement

    await screen.findByText('Select all')

    expect(item('Cut').getAttribute('data-disabled')).not.toBeNull()
    expect(item('Copy').getAttribute('data-disabled')).not.toBeNull()
    expect(item('Select all').getAttribute('data-disabled')).not.toBeNull()
    expect(item('Paste').getAttribute('data-disabled')).toBeNull()
  })

  it('splits guest select all into its own section', async () => {
    installBridge()
    mountMenu()

    openGuestContextMenu(10, 10, guestParams({ isEditable: true }), guestHandle())

    const selectAllItem = (await screen.findByText('Select all')).closest('[data-slot="dropdown-menu-item"]')!
    const pasteItem = screen.getByText('Paste').closest('[data-slot="dropdown-menu-item"]')!

    expect(pasteItem.parentElement).not.toBe(selectAllItem.parentElement)
    expect(selectAllItem.parentElement?.querySelector('[data-slot="dropdown-menu-separator"]')).not.toBeNull()
  })

  it('dispatches guest edit verbs a frame after the menu closes', async () => {
    installBridge()
    mountMenu()
    const guest = guestHandle()

    openGuestContextMenu(10, 10, guestParams(), guest)
    fireEvent.click(await screen.findByText('Select all'))

    // Deferred past the radix unmount so the webview focus() is not stolen
    // back — dispatching with host focus selected the address bar + chat.
    expect(guest.editCommand).not.toHaveBeenCalled()

    await waitFor(() => expect(guest.editCommand).toHaveBeenCalledWith('selectAll'))
    expect($contextMenu.get()).toBeNull()
  })

  it('adds an image section with copy and save for guest images', async () => {
    installBridge()
    mountMenu()

    openGuestContextMenu(
      10,
      10,
      guestParams({ hasImageContents: true, srcURL: 'https://example.com/pic.png' }),
      guestHandle()
    )

    expect(await screen.findByText('Copy image')).toBeTruthy()
    expect(screen.getByText('Save image as…')).toBeTruthy()
  })

  it('shows spell suggestions immediately for guest editables', async () => {
    installBridge()
    mountMenu()
    const guest = guestHandle()

    openGuestContextMenu(
      10,
      10,
      guestParams({ dictionarySuggestions: ['the'], isEditable: true, misspelledWord: 'teh' }),
      guest
    )

    fireEvent.click(await screen.findByText('the'))

    expect(guest.replaceMisspelling).toHaveBeenCalledWith('the')
    expect(screen.queryByText('Add to dictionary')).toBeNull()
  })
})
