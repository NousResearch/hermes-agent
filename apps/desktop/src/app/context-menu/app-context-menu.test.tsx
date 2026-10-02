import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { registerTerminalContextMenu } from '@/app/right-sidebar/terminal/terminal-context-menu'
import { DirectiveContent } from '@/components/assistant-ui/directive-text'
import { ContextMenu, ContextMenuTrigger, HERMES_CONTEXT_MENU_TRIGGER_ATTR } from '@/components/ui/context-menu'
import { Dialog, DialogContent, DialogTitle } from '@/components/ui/dialog'
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

  it('resolves the enclosing dialog as the menu portal container', () => {
    const host = attach('<div data-slot="dialog-content"><a href="https://example.com">link</a></div>')
    const dialog = host.firstElementChild

    expect(resolveDomTarget(host.querySelector('a')).dialogPortalContainer).toBe(dialog)
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

  // The url chip used to be a `<button>`: `resolveDomTarget` only knows
  // `a[href]`, so the right-click fell through to the shell fallback menu.
  it('offers the link verbs on a message url chip right-click', async () => {
    installBridge()
    mountMenu()
    // The coordinator binds to window in the capture phase, so a second
    // render alongside the menu is fine — same as a real transcript.
    render(<DirectiveContent text="@url:`https://example.com/pr/1`" />)
    const chip = document.querySelector('[data-slot="aui_directive-chip"]')!

    fireEvent.contextMenu(chip)

    expect(await screen.findByText('Open in in-app browser')).toBeTruthy()
    expect(screen.getByText('Open in external browser')).toBeTruthy()
    expect(screen.getByText('Copy URL')).toBeTruthy()
  })

  it('opens the in-app browser from the link menu', async () => {
    installBridge()
    mountMenu()
    const host = attach('<a href="https://example.com/docs">Docs</a>')

    fireEvent.contextMenu(host.querySelector('a')!)
    fireEvent.click(await screen.findByText('Open in in-app browser'))

    await waitFor(() => expect($previewTabs.get().at(-1)?.target.url).toBe('https://example.com/docs'))
  })

  it('skips Open in in-app browser on the HUD — that window has no browser pane', async () => {
    const originalLocation = window.location

    Object.defineProperty(window, 'location', {
      configurable: true,
      value: { ...originalLocation, search: '?win=hud' }
    })

    try {
      installBridge()
      mountMenu()
      const host = attach('<a href="https://accounts.google.com/o/oauth2/auth">Sign in</a>')

      fireEvent.contextMenu(host.querySelector('a')!)

      expect(await screen.findByText('Open in external browser')).toBeTruthy()
      expect(screen.queryByText('Open in in-app browser')).toBeNull()
    } finally {
      Object.defineProperty(window, 'location', { configurable: true, value: originalLocation })
    }
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

  it('copies the field selection through the clipboard bridge after the menu closed', async () => {
    const writeClipboard = vi.fn().mockResolvedValue(true)

    installBridge({ writeClipboard: writeClipboard as unknown as Window['hermesDesktop']['writeClipboard'] })
    mountMenu()
    const host = attach('<textarea>some draft text</textarea>')
    const textarea = host.querySelector('textarea')!

    // Cut/copy act on the selection, so give the field one.
    textarea.setSelectionRange(0, 4)
    fireEvent.contextMenu(textarea)
    fireEvent.click(await screen.findByText('Copy'))

    // The verb waits a frame so the radix focus trap unmounts first — running
    // while the trap held focus aimed main's edit command at the menu, not the
    // field, which is why the item looked dead.
    expect(writeClipboard).not.toHaveBeenCalled()

    await waitFor(() => expect(writeClipboard).toHaveBeenCalledWith('some'))
    expect($contextMenu.get()).toBeNull()
    expect(document.activeElement).toBe(textarea)
  })

  it('keeps a modal textarea paste menu inside its dialog and pastes into the field', async () => {
    installBridge({ readClipboard: vi.fn().mockResolvedValue('clipboard payload') })
    render(
      <MemoryRouter>
        <AppContextMenu />
        <Dialog open>
          <DialogContent>
            <DialogTitle>Modal editor</DialogTitle>
            <textarea aria-label="modal textarea" />
          </DialogContent>
        </Dialog>
      </MemoryRouter>
    )
    const textarea = screen.getByLabelText('modal textarea') as HTMLTextAreaElement

    fireEvent.contextMenu(textarea)

    const paste = (await screen.findByText('Paste')).closest('[data-slot="dropdown-menu-item"]') as HTMLElement

    await waitFor(() => expect(paste.getAttribute('data-disabled')).toBeNull())

    const menu = paste.closest('[data-slot="dropdown-menu-content"]')
    const dialog = screen.getByRole('dialog')

    expect(dialog.contains(menu)).toBe(true)

    fireEvent.click(paste)

    await waitFor(() => expect(textarea.value).toBe('clipboard payload'))
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

  it('select all stays inside the field and never reaches the clipboard', async () => {
    const writeClipboard = vi.fn().mockResolvedValue(true)

    installBridge({ writeClipboard: writeClipboard as unknown as Window['hermesDesktop']['writeClipboard'] })
    mountMenu()
    const host = attach('<textarea>alpha beta gamma</textarea>')
    const textarea = host.querySelector('textarea')!

    fireEvent.contextMenu(textarea)
    fireEvent.click(await screen.findByText('Select all'))

    // Renderer-side selection scoped to the field: a main-side selectAll acts on
    // the focused FRAME and selected the whole transcript when focus slipped
    // (the edit composer re-parents focus on blur).
    await waitFor(() => {
      expect(textarea.selectionStart).toBe(0)
      expect(textarea.selectionEnd).toBe('alpha beta gamma'.length)
    })
    expect(writeClipboard).not.toHaveBeenCalled()
    expect(document.activeElement).toBe(textarea)
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

  it('routes a paste in the edit composer through the composer insert API', async () => {
    // The composer is a custom contenteditable with its own sanitize/chip
    // pipeline, so the menu must not poke the DOM: it hands the text to the
    // composer's insert API, the same one the preview pane inserts through.
    installBridge({ readClipboard: vi.fn().mockResolvedValue('from clipboard') })
    mountMenu()
    attach(`<div data-slot="aui_edit-composer-root"><div contenteditable="true">draft</div></div>`)

    const editor = document.querySelector('[contenteditable]') as HTMLElement

    // jsdom does not implement the property the resolver checks.
    Object.defineProperty(editor, 'isContentEditable', { value: true })

    const events: Array<{ text?: string; mode?: string }> = []
    const listener = (event: Event) => events.push((event as CustomEvent).detail)

    window.addEventListener('hermes:composer-insert', listener)
    try {
      fireEvent.contextMenu(editor)
      fireEvent.click(await screen.findByText('Paste'))
      await waitFor(() => expect(events.at(-1)?.text).toBe('from clipboard'))
      expect(events.at(-1)?.mode).toBe('inline')
    } finally {
      window.removeEventListener('hermes:composer-insert', listener)
    }
  })

  it('keeps paste clickable even when the clipboard probe reports empty', async () => {
    // The paste verb reads the system clipboard through main
    // (`hermes:readClipboard`, the same bridge the terminal paste uses). A probe
    // that comes back empty must not gray the item out; pasting on a truly empty
    // clipboard is a harmless no-op, so the item fails open.
    const readClipboard = vi.fn().mockResolvedValue('')

    installBridge({ readClipboard: readClipboard as unknown as Window['hermesDesktop']['readClipboard'] })
    mountMenu()
    const host = attach('<textarea>text</textarea>')

    fireEvent.contextMenu(host.querySelector('textarea')!)

    const pasteItem = (await screen.findByText('Paste')).closest('[data-slot="dropdown-menu-item"]')!

    expect(pasteItem.getAttribute('data-disabled')).toBeNull()
  })

  it('keeps paste clickable when the bridge has no clipboard read at all', async () => {
    installBridge()
    mountMenu()
    const host = attach('<textarea>text</textarea>')

    fireEvent.contextMenu(host.querySelector('textarea')!)

    const pasteItem = (await screen.findByText('Paste')).closest('[data-slot="dropdown-menu-item"]')!

    expect(pasteItem.getAttribute('data-disabled')).toBeNull()
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
      reload: vi.fn(),
      selectAll: vi.fn(),
      wordErase: null
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
      reload: vi.fn(),
      selectAll: vi.fn(),
      wordErase: null
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
    expect(screen.getByText('Inspect element')).toBeTruthy()
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

describe('ContextMenuTrigger asChild', () => {
  it('keeps the coordinator marker when the child overwrites data-slot', () => {
    render(
      <ContextMenu>
        <ContextMenuTrigger asChild>
          <footer data-slot="statusbar">bar</footer>
        </ContextMenuTrigger>
      </ContextMenu>
    )

    const footer = screen.getByText('bar')

    expect(footer.getAttribute('data-slot')).toBe('statusbar')
    expect(footer.hasAttribute(HERMES_CONTEXT_MENU_TRIGGER_ATTR)).toBe(true)
  })
})

// #127313: the pane-body zone menu swallowed right-clicks on transcript text, so
// the only Copy affordance a mouse user had disappeared. A live text selection
// belongs to the app menu even though the target sits inside a zone trigger.
it('offers Copy for text selected inside a component-owned zone trigger', async () => {
  installBridge()
  mountMenu()

  const host = attach('<div data-slot="context-menu-trigger"><p>selected reply text</p></div>')
  const paragraph = host.querySelector('p')!
  const range = document.createRange()

  range.selectNodeContents(paragraph.firstChild!)

  const selection = window.getSelection()!

  selection.removeAllRanges()
  selection.addRange(range)

  fireEvent.contextMenu(paragraph, { clientX: 30, clientY: 30 })
  const copy = (await screen.findByText('Copy')).closest('[data-slot="dropdown-menu-item"]') as HTMLElement

  expect(copy.getAttribute('data-disabled')).toBeNull()

  fireEvent.click(copy)

  await waitFor(() => expect(desktopWindow.hermesDesktop!.writeClipboard).toHaveBeenCalledWith('selected reply text'))
})
