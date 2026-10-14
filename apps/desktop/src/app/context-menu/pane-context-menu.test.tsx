import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, expect, it, vi } from 'vitest'

import { ContextMenu, ContextMenuContent, ContextMenuItem, ContextMenuTrigger } from '@/components/ui/context-menu'

import { AppContextMenu } from './app-context-menu'
import { $contextMenu } from './store'

const desktopDescriptor = Object.getOwnPropertyDescriptor(window, 'hermesDesktop')
const editableDescriptor = Object.getOwnPropertyDescriptor(HTMLElement.prototype, 'isContentEditable')

function setup() {
  // jsdom lacks the browser's isContentEditable property.
  Object.defineProperty(HTMLElement.prototype, 'isContentEditable', {
    configurable: true,
    get() {
      const host = this.closest('[contenteditable]')

      return !!host && host.getAttribute('contenteditable') !== 'false'
    }
  })
  Object.defineProperty(window, 'hermesDesktop', {
    configurable: true,
    value: { writeClipboard: vi.fn().mockResolvedValue(undefined) }
  })
  render(
    <MemoryRouter>
      <AppContextMenu />
      <ContextMenu>
        <ContextMenuTrigger asChild>
          <div data-zone-body="test">
            <textarea data-testid="empty" />
            <input data-testid="input" defaultValue="draft text" />
            <div contentEditable data-testid="editable">
              <span data-testid="editable-child">draft</span>
            </div>
            <p data-testid="selected">transcript words</p>
            <p data-testid="bare">empty space</p>
            <a data-testid="link" href="https://example.com">
              link
            </a>
            <img data-testid="image" src="https://example.com/pic.png" />
            <ContextMenu>
              <ContextMenuTrigger asChild>
                <div data-testid="nested">
                  <a data-testid="nested-link" href="https://example.com/row">
                    row
                  </a>
                </div>
              </ContextMenuTrigger>
              <ContextMenuContent>
                <ContextMenuItem>Row action</ContextMenuItem>
              </ContextMenuContent>
            </ContextMenu>
          </div>
        </ContextMenuTrigger>
        <ContextMenuContent>
          <ContextMenuItem>Zone action</ContextMenuItem>
        </ContextMenuContent>
      </ContextMenu>
    </MemoryRouter>
  )
}

function select() {
  const range = document.createRange()
  range.selectNodeContents(screen.getByTestId('selected'))
  window.getSelection()!.removeAllRanges()
  window.getSelection()!.addRange(range)
}

function reset() {
  $contextMenu.set(null)
  cleanup()
  window.getSelection()?.removeAllRanges()
  vi.restoreAllMocks()

  if (desktopDescriptor) {
    Object.defineProperty(window, 'hermesDesktop', desktopDescriptor)
  } else {
    Reflect.deleteProperty(window, 'hermesDesktop')
  }

  if (editableDescriptor) {
    Object.defineProperty(HTMLElement.prototype, 'isContentEditable', editableDescriptor)
  } else {
    Reflect.deleteProperty(HTMLElement.prototype, 'isContentEditable')
  }
}

afterEach(reset)

it('the pane fallback serves clicked content without taking explicit or unrelated menus', async () => {
  const cases = [
    { id: 'empty', label: 'Paste', app: true },
    { id: 'input', label: 'Paste', app: true },
    { id: 'editable-child', label: 'Paste', app: true },
    { id: 'link', label: 'Copy URL', app: true },
    { id: 'image', label: 'Copy image', app: true },
    { id: 'selected', label: 'Copy', app: true, selection: true, copy: true },
    { id: 'bare', label: 'Zone action', app: false },
    { id: 'bare', label: 'Zone action', app: false, selection: true },
    { id: 'nested-link', label: 'Row action', app: false },
    { id: 'nested', label: 'Row action', app: false, selection: true }
  ]

  for (const { id, label, app, selection, copy } of cases) {
    setup()

    if (selection) {
      select()
    }

    fireEvent.contextMenu(screen.getByTestId(id))
    await screen.findByRole('menu')
    const item = screen.queryByText(label)
    const context = `${id}, selected=${Boolean(selection)} must offer ${label}`

    // Soft assertions let every ownership case run even on an unfixed tree.
    expect.soft(item, context).not.toBeNull()
    expect.soft($contextMenu.get() !== null, context).toBe(app)

    if (app) {
      expect.soft(screen.queryByText('Zone action'), context).toBeNull()
    }

    if (copy && item) {
      fireEvent.click(item)
      expect.soft(window.hermesDesktop.writeClipboard, context).toHaveBeenCalledWith('transcript words')
    }

    reset()
  }
})
