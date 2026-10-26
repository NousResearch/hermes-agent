/**
 * Arrow-key navigation and highlight for the composer context menu.
 *
 * Radix highlights a row by FOCUSING it: `MenuItemImpl` sets `data-highlighted`
 * from its own focus state, and the row's styles paint on `:focus`. It steps
 * with a `RovingFocusGroup` built from the rows it rendered, so "can I arrow
 * onto this row" and "does this row highlight" are the same question — is the
 * row the active element, marked highlighted?
 *
 * The menu is assembled from SECTIONS, and a section can be inserted while the
 * menu is already open: Chromium's spell-check facts for an editable arrive
 * from the main process a tick AFTER the menu opens (`augmentSpellcheck`). A
 * row has to keep its identity across that insert, or the highlight the user
 * was looking at disappears and the arrow keys restart from the top of the
 * menu.
 */

import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { COMPOSER_AREAS, type ComposerContextMenuItem } from '@/app/chat/composer/contrib'
import { RICH_INPUT_SLOT } from '@/app/chat/composer/rich-editor'
import { createPluginContext } from '@/contrib/plugin'

import { AppContextMenu } from './app-context-menu'
import { $contextMenu, augmentSpellcheck } from './store'

const disposers: Array<() => void> = []

function plugin(id: string, items: () => ComposerContextMenuItem[]) {
  const ctx = createPluginContext(id, dispose => disposers.push(dispose))

  ctx.register({ area: COMPOSER_AREAS.contextMenu, data: { items }, id: 'menu' })

  return ctx
}

function composer(text: string): HTMLElement {
  const editor = document.createElement('div')

  editor.setAttribute('contenteditable', 'true')
  editor.setAttribute('role', 'textbox')
  editor.dataset.slot = RICH_INPUT_SLOT
  editor.textContent = text
  // jsdom lacks the browser's isContentEditable.
  Object.defineProperty(editor, 'isContentEditable', { configurable: true, value: true })
  document.body.appendChild(editor)

  return editor
}

/** A selection, so Cut and Copy are enabled and every row is focusable. */
function select(editor: HTMLElement, start: number, end: number) {
  const range = document.createRange()

  range.setStart(editor.firstChild!, start)
  range.setEnd(editor.firstChild!, end)
  window.getSelection()!.removeAllRanges()
  window.getSelection()!.addRange(range)
}

function rows(): HTMLElement[] {
  return Array.from(document.querySelectorAll<HTMLElement>('[role="menuitem"]'))
}

/** The row's label — its first `<span>`, never the accelerator beside it. */
function rowLabel(row: Element): string {
  return (row.querySelector('span')?.textContent ?? row.textContent ?? '').trim()
}

function labels(): string[] {
  return rows().map(rowLabel)
}

/** The rows Radix has marked highlighted, in menu order. */
function highlighted(): string[] {
  return rows()
    .filter(row => row.hasAttribute('data-highlighted'))
    .map(rowLabel)
}

const activeLabel = () => rowLabel(document.activeElement as Element)

/** Radix defers the focus move inside its keydown handler to a timer. */
async function flush() {
  await act(async () => {
    await new Promise(resolve => setTimeout(resolve, 0))
  })
}

/** The real flow: the key goes to whatever row has focus. */
async function press(key: 'ArrowDown' | 'ArrowUp') {
  fireEvent.keyDown(document.activeElement as HTMLElement, { key })
  await flush()
}

async function openMenu(editor: HTMLElement) {
  select(editor, 0, 3)
  fireEvent.contextMenu(editor)
  expect(await screen.findByText('Paste')).toBeTruthy()
}

beforeEach(() => {
  render(
    <MemoryRouter>
      <AppContextMenu />
    </MemoryRouter>
  )
})

afterEach(() => {
  disposers.splice(0).forEach(dispose => dispose())
  $contextMenu.set(null)
  cleanup()
  document.body.innerHTML = ''
  window.getSelection()?.removeAllRanges()
})

describe('composer context menu: keyboard navigation and highlight', () => {
  it('steps one highlight across the plugin rows and the app’s own rows', async () => {
    plugin('alpha', () => [
      { label: 'freshen', onSelect: vi.fn() },
      { label: 'freshet', onSelect: vi.fn() }
    ])

    const editor = composer('teh quick fox')

    await openMenu(editor)

    // The plugin's rows are real menu items in the app's own list, ahead of the
    // edit verbs — one section, not a separate navigable group.
    expect(labels()).toEqual(['freshen', 'freshet', 'Cut', 'Copy', 'Paste', 'Select all'])

    // Radix starts the menu on its surface, not on a row.
    expect(highlighted()).toEqual([])

    act(() => rows()[0].focus())
    expect(highlighted()).toEqual(['freshen'])

    for (const expected of ['freshet', 'Cut', 'Copy', 'Paste', 'Select all']) {
      await press('ArrowDown')
      expect(activeLabel()).toBe(expected)
      // Highlighted IS focused: exactly one row, and it is the focused one.
      expect(highlighted()).toEqual([expected])
    }

    for (const expected of ['Paste', 'Copy', 'Cut', 'freshet', 'freshen']) {
      await press('ArrowUp')
      expect(activeLabel()).toBe(expected)
      expect(highlighted()).toEqual([expected])
    }
  })

  it('still steps the app’s own rows without any plugin rows', async () => {
    const editor = composer('teh quick fox')

    await openMenu(editor)

    expect(labels()).toEqual(['Cut', 'Copy', 'Paste', 'Select all'])

    act(() => rows()[0].focus())
    expect(highlighted()).toEqual(['Cut'])

    for (const expected of ['Copy', 'Paste', 'Select all']) {
      await press('ArrowDown')
      expect(activeLabel()).toBe(expected)
      expect(highlighted()).toEqual([expected])
    }

    await press('ArrowUp')
    expect(activeLabel()).toBe('Paste')
    expect(highlighted()).toEqual(['Paste'])
  })

  it('keeps the row the user is on when the spell-check rows arrive after open', async () => {
    plugin('alpha', () => [
      { label: 'freshen', onSelect: vi.fn() },
      { label: 'freshet', onSelect: vi.fn() }
    ])

    const editor = composer('teh quick fox')

    await openMenu(editor)
    expect(labels()).toEqual(['freshen', 'freshet', 'Cut', 'Copy', 'Paste', 'Select all'])

    // The user is on the plugin's second row when main reports the spell-check
    // facts for this right-click.
    const held = rows()[1]

    act(() => held.focus())
    expect(highlighted()).toEqual(['freshet'])

    await act(async () => {
      augmentSpellcheck({ misspelledWord: 'teh', suggestions: ['then', 'ten'] })
    })
    await flush()

    // The app's spell-check section is inserted ABOVE the plugin's rows, so
    // every row below it moves down the menu.
    expect(labels()).toEqual(['then', 'ten', 'Add to dictionary', 'freshen', 'freshet', 'Cut', 'Copy', 'Paste', 'Select all'])

    // Moving a row must not re-create it: the row the user is on keeps the
    // highlight, keeps focus, and keeps its element.
    expect(highlighted()).toEqual(['freshet'])
    expect(document.activeElement).toBe(held)
    expect(held.isConnected).toBe(true)

    // And the arrow keys continue from that row, not from the top of the menu.
    await press('ArrowDown')
    expect(activeLabel()).toBe('Cut')
    expect(highlighted()).toEqual(['Cut'])
  })

  it('keeps the row the user is on when the spell-check rows arrive late, with no plugin', async () => {
    const editor = composer('teh quick fox')

    await openMenu(editor)

    const held = rows()[2]

    act(() => held.focus())
    expect(rowLabel(held)).toBe('Paste')
    expect(highlighted()).toEqual(['Paste'])

    await act(async () => {
      augmentSpellcheck({ misspelledWord: 'teh', suggestions: ['then', 'ten'] })
    })
    await flush()

    expect(labels()).toEqual(['then', 'ten', 'Add to dictionary', 'Cut', 'Copy', 'Paste', 'Select all'])
    expect(document.activeElement).toBe(held)
    expect(highlighted()).toEqual(['Paste'])
  })
})
