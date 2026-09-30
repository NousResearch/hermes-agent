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

afterEach(() => {
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
})
it.each([
  ['empty', 'Paste'],
  ['input', 'Paste'],
  ['editable-child', 'Paste'],
  ['link', 'Copy URL'],
  ['image', 'Copy image']
])('content %s gets %s', async (id, label) => {
  setup()
  fireEvent.contextMenu(screen.getByTestId(id))
  expect(await screen.findByText(label)).toBeTruthy()
  expect(screen.queryByText('Zone action')).toBeNull()
})
it('selection offers Copy and copies actual text', async () => {
  setup()
  select()
  fireEvent.contextMenu(screen.getByTestId('selected'))
  fireEvent.click(await screen.findByText('Copy'))
  expect(window.hermesDesktop!.writeClipboard).toHaveBeenCalledWith('transcript words')
})
it('bare pane keeps zone menu', async () => {
  setup()
  fireEvent.contextMenu(screen.getByTestId('bare'))
  expect(await screen.findByText('Zone action')).toBeTruthy()
  expect($contextMenu.get()).toBeNull()
})
it('stale selection does not steal bare pane menu', async () => {
  setup()
  select()
  fireEvent.contextMenu(screen.getByTestId('bare'))
  expect(await screen.findByText('Zone action')).toBeTruthy()
  expect($contextMenu.get()).toBeNull()
})
it('nested explicit menu keeps links', async () => {
  setup()
  fireEvent.contextMenu(screen.getByTestId('nested-link'))
  expect(await screen.findByText('Row action')).toBeTruthy()
  expect($contextMenu.get()).toBeNull()
})
it('stale selection does not steal nested menu', async () => {
  setup()
  select()
  fireEvent.contextMenu(screen.getByTestId('nested'))
  expect(await screen.findByText('Row action')).toBeTruthy()
  expect($contextMenu.get()).toBeNull()
})
