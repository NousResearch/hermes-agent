import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { registry } from '@/contrib/registry'

import { AppContextMenu } from './app-context-menu'
import { CONTEXT_MENU_AREA, type ContextMenuContribution, wordUnderCaret } from './contrib'
import { $contextMenu, closeContextMenu, openDomContextMenu } from './store'
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

  return host.firstElementChild as HTMLElement
}

/** Open the app's DOM context menu over an element, as a right-click would. */
function openMenuOver(element: HTMLElement) {
  const target = resolveDomTarget(element)

  expect(target).not.toBeNull()

  openDomContextMenu(10, 10, target!)
}

afterEach(() => {
  cleanup()
  closeContextMenu()
  registry.registerMany([])
})

describe('context-menu contribution area', () => {
  it('renders a plugin row in the app\'s own menu', async () => {
    installBridge()

    const onSelect = vi.fn()

    registry.register({
      area: CONTEXT_MENU_AREA,
      data: {
        rows: () => [{ icon: 'edit', label: 'Plugin suggestion', onSelect }]
      } satisfies ContextMenuContribution,
      id: 'test-plugin'
    })

    const host = attach('<textarea>text</textarea>')

    mountMenu()
    openMenuOver(host)

    expect(await screen.findByText('Plugin suggestion')).toBeTruthy()
  })

  it('runs the row\'s onSelect when it is picked', async () => {
    installBridge()

    const onSelect = vi.fn()

    registry.register({
      area: CONTEXT_MENU_AREA,
      data: { rows: () => [{ label: 'Plugin suggestion', onSelect }] } satisfies ContextMenuContribution,
      id: 'test-plugin'
    })

    const host = attach('<textarea>text</textarea>')

    mountMenu()
    openMenuOver(host)

    fireEvent.click(await screen.findByText('Plugin suggestion'))

    await waitFor(() => {
      expect(onSelect).toHaveBeenCalledTimes(1)
    })
  })

  it('shows the app\'s own rows alongside the plugin\'s', async () => {
    installBridge()

    registry.register({
      area: CONTEXT_MENU_AREA,
      data: { rows: () => [{ label: 'Plugin suggestion', onSelect: vi.fn() }] } satisfies ContextMenuContribution,
      id: 'test-plugin'
    })

    const host = attach('<textarea>text</textarea>')

    mountMenu()
    openMenuOver(host)

    // The app's own edit verbs are still there.
    expect(await screen.findByText('Copy')).toBeTruthy()
    expect(await screen.findByText('Paste')).toBeTruthy()
    expect(await screen.findByText('Plugin suggestion')).toBeTruthy()
  })

  it('contributes nothing when the plugin returns no rows', async () => {
    installBridge()

    registry.register({
      area: CONTEXT_MENU_AREA,
      data: { rows: () => [] } satisfies ContextMenuContribution,
      id: 'test-plugin'
    })

    const host = attach('<textarea>text</textarea>')

    mountMenu()
    openMenuOver(host)

    expect(await screen.findByText('Copy')).toBeTruthy()
    expect(screen.queryByText('Plugin suggestion')).toBeNull()
  })

  it('survives a plugin whose rows() throws', async () => {
    installBridge()

    registry.register({
      area: CONTEXT_MENU_AREA,
      data: {
        rows: () => {
          throw new Error('plugin bug')
        }
      } satisfies ContextMenuContribution,
      id: 'test-plugin'
    })

    const host = attach('<textarea>text</textarea>')

    mountMenu()
    openMenuOver(host)

    // The menu still opens with the app's own rows.
    expect(await screen.findByText('Copy')).toBeTruthy()
  })

  it('tells the plugin whether the click was on an editable, and what word it was on', async () => {
    installBridge()

    const seen: unknown[] = []

    registry.register({
      area: CONTEXT_MENU_AREA,
      data: {
        rows: context => {
          seen.push(context)

          return []
        }
      } satisfies ContextMenuContribution,
      id: 'test-plugin'
    })

    const host = attach('<textarea>text</textarea>')

    mountMenu()
    openMenuOver(host)

    await waitFor(() => {
      expect(seen.length).toBeGreaterThan(0)
    })

    expect((seen[0] as { editable: boolean }).editable).toBe(true)
  })
})

describe('wordUnderCaret', () => {
  afterEach(() => {
    document.getSelection()?.removeAllRanges()
  })

  it('finds the word under the caret', () => {
    const host = attach('<div contenteditable="true">hello world</div>')
    const textNode = host.firstChild as Text

    const range = document.createRange()

    range.setStart(textNode, 3)
    range.collapse(true)

    const selection = document.getSelection()

    selection?.removeAllRanges()
    selection?.addRange(range)

    expect(wordUnderCaret(host)).toBe('hello')
  })

  it('walks back to the nearest word when the caret sits on a boundary', () => {
    // A caret at the very end of 'hello' is a BOUNDARY, not a word -- the
    // function steps back to the word it belongs to, so the plugin is told
    // what the user is pointing at rather than nothing.
    const host = attach('<div contenteditable="true">hello</div>')
    const textNode = host.firstChild as Text

    const range = document.createRange()

    range.setStart(textNode, 5)
    range.collapse(true)

    const selection = document.getSelection()

    selection?.removeAllRanges()
    selection?.addRange(range)

    expect(wordUnderCaret(host)).toBe('hello')
  })

  it('returns null when there is no word at all', () => {
    const host = attach('<div contenteditable="true">   </div>')
    const textNode = host.firstChild as Text

    const range = document.createRange()

    range.setStart(textNode, 1)
    range.collapse(true)

    const selection = document.getSelection()

    selection?.removeAllRanges()
    selection?.addRange(range)

    expect(wordUnderCaret(host)).toBeNull()
  })

  it('returns null with no editable', () => {
    expect(wordUnderCaret(null)).toBeNull()
  })
})

describe('the menu store', () => {
  it('starts closed', () => {
    expect($contextMenu.get()).toBeNull()
  })
})

describe('a plugin row runs against the field, not the menu', () => {
  it('closes the menu and refocuses the field before running the action', async () => {
    installBridge()

    const editable = attach('<textarea>some text</textarea>') as HTMLTextAreaElement
    const seen: { active: string | null; menuOpen: boolean }[] = []

    registry.register({
      area: CONTEXT_MENU_AREA,
      data: {
        rows: () => [
          {
            label: 'Plugin action',
            onSelect: () => {
              seen.push({
                active: document.activeElement?.tagName ?? null,
                menuOpen: document.querySelector('[role="menu"]') !== null
              })
            }
          }
        ]
      } satisfies ContextMenuContribution,
      id: 'test-plugin'
    })

    mountMenu()
    openMenuOver(editable)

    const row = await screen.findByText('Plugin action')

    // Focus the field the way a right-click would leave it, then pick the row.
    editable.focus()
    fireEvent.click(row)

    await waitFor(() => {
      expect(seen.length).toBe(1)
    })

    // The action must not run with the menu still mounted: Radix fires
    // `onSelect` before it closes the content, and the content is a focus
    // trap, so a handler that ran directly would see the menu rather than
    // the field.
    expect(seen[0].menuOpen).toBe(false)
    expect(seen[0].active).toBe('TEXTAREA')
  })

  it('puts a top row above the app\'s own rows', async () => {
    installBridge()

    registry.register({
      area: CONTEXT_MENU_AREA,
      data: { rows: () => [{ label: 'Plugin suggestion', onSelect: vi.fn() }] } satisfies ContextMenuContribution,
      id: 'test-plugin'
    })

    const host = attach('<textarea>text</textarea>')

    mountMenu()
    openMenuOver(host)

    const plugin = await screen.findByText('Plugin suggestion')
    const copy = await screen.findByText('Copy')

    // DOM order is menu order: the plugin's row comes first.
    expect(plugin.compareDocumentPosition(copy) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()
  })
})
