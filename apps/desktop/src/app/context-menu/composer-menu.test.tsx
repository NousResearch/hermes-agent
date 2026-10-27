import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { createRef, type RefObject } from 'react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import {
  COMPOSER_AREAS,
  type ComposerContextMenuAction,
  type ComposerContextMenuContext,
  type ComposerContextMenuItem
} from '@/app/chat/composer/contrib'
import { useComposerUndo } from '@/app/chat/composer/hooks/use-composer-undo'
import { insertComposerContentsAtCaret, RICH_INPUT_SLOT } from '@/app/chat/composer/rich-editor'
import { createPluginContext } from '@/contrib/plugin'

import { AppContextMenu } from './app-context-menu'
import { $contextMenu } from './store'

// The plugin side goes through the real scoped context, so provenance stamping
// and unload teardown are the production ones.
const disposers: Array<() => void> = []

function plugin(id: string, items: (context: ComposerContextMenuContext) => ComposerContextMenuItem[]) {
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

function select(editor: HTMLElement, start: number, end = start) {
  const range = document.createRange()

  range.setStart(editor.firstChild!, start)
  range.setEnd(editor.firstChild!, end)
  window.getSelection()!.removeAllRanges()
  window.getSelection()!.addRange(range)
}

/** The composer's real undo hook, mounted on `editor` the way the composer does. */
function mountUndo(editor: HTMLElement) {
  const editorRef = createRef<HTMLDivElement>() as RefObject<HTMLDivElement | null>
  const api: { current: ReturnType<typeof useComposerUndo> | null } = { current: null }

  editorRef.current = editor as HTMLDivElement

  const Harness = () => {
    api.current = useComposerUndo({ editorRef, syncDraftFromEditor: () => editor.textContent ?? '' })

    return null
  }

  render(<Harness />)

  return api
}

function menuLabels(): string[] {
  return screen.queryAllByRole('menuitem').map(item => item.textContent ?? '')
}

beforeEach(() => {
  // jsdom has no editing commands; this stands in for Chromium's insertText on
  // the live selection, which is all the app asks of it.
  document.execCommand = vi.fn((command: string, _ui?: boolean, value?: string) => {
    const range = window.getSelection()?.getRangeAt(0)

    if (command !== 'insertText' || !range) {
      return false
    }

    range.deleteContents()
    range.insertNode(document.createTextNode(value ?? ''))
    range.startContainer.parentElement?.normalize()

    return true
  }) as typeof document.execCommand
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

describe('composer context-menu contribution area', () => {
  it('offers plugin items on the composer and never on any other editable', async () => {
    const items = vi.fn(() => [{ label: 'Plugin row', onSelect: vi.fn() }])

    plugin('alpha', items)

    const settingsField = document.createElement('input')

    document.body.appendChild(settingsField)
    fireEvent.contextMenu(settingsField)

    expect(await screen.findByText('Paste')).toBeTruthy()
    expect(screen.queryByText('Plugin row')).toBeNull()
    expect(items).not.toHaveBeenCalled()

    act(() => $contextMenu.set(null))

    const editor = composer('teh quick fox')

    select(editor, 1)
    fireEvent.contextMenu(editor)

    expect(await screen.findByText('Plugin row')).toBeTruthy()
  })

  it('hands a provider only the word under the caret and the composer selection', async () => {
    const seen: ComposerContextMenuContext[] = []

    plugin('alpha', context => {
      seen.push(context)

      return []
    })

    const editor = composer('teh quick fox')

    select(editor, 1)
    fireEvent.contextMenu(editor)
    await screen.findByText('Paste')

    expect(seen[0]).toEqual({ selection: '', word: 'teh' })
    expect(Object.keys(seen[0]).sort()).toEqual(['selection', 'word'])

    act(() => $contextMenu.set(null))
    select(editor, 4, 9)
    fireEvent.contextMenu(editor)
    await screen.findByText('Paste')

    expect(seen.at(-1)).toEqual({ selection: 'quick', word: 'quick' })
  })

  it('reads and replaces a whole word that a paste split across text nodes', async () => {
    let action: ComposerContextMenuAction | null = null

    plugin('alpha', () => [{ label: 'Fix', onSelect: received => void (action = received) }])

    const editor = composer('Her')

    // The composer's real paste path appends the pasted text as its own node.
    select(editor, 3)
    insertComposerContentsAtCaret(editor, 'mse rocks')
    expect(editor.childNodes.length).toBeGreaterThan(1)

    const pasted = [...editor.childNodes].find(node => node.textContent === 'mse rocks')!
    const range = document.createRange()

    range.setStart(pasted, 1)
    range.collapse(true)
    window.getSelection()!.removeAllRanges()
    window.getSelection()!.addRange(range)
    fireEvent.contextMenu(editor)
    fireEvent.click(await screen.findByText('Fix'))
    await waitFor(() => expect(action).not.toBeNull())

    expect(action!.word).toBe('Hermse')
    expect(action!.replaceWord('Hermes')).toBe(true)
    expect(editor.textContent).toBe('Hermes rocks')
  })

  it('keeps astral-plane letters inside the word', async () => {
    const seen: ComposerContextMenuContext[] = []

    plugin('alpha', context => {
      seen.push(context)

      return []
    })

    const editor = composer('hi 𠮷田 there')

    select(editor, 6)
    fireEvent.contextMenu(editor)
    await screen.findByText('Paste')

    expect(seen[0]?.word).toBe('𠮷田')
  })

  it('merges plugins in a deterministic order, caps each one, and isolates a throwing provider', async () => {
    plugin('zeta', () => Array.from({ length: 8 }, (_, index) => ({ label: `zeta ${index + 1}`, onSelect: vi.fn() })))
    plugin('broken', () => {
      throw new Error('plugin bug')
    })
    plugin('alpha', () => [
      { label: '  Fix\u0000   this\n ', onSelect: vi.fn() },
      { label: 42 as unknown as string, onSelect: vi.fn() },
      { label: 'No callback' } as unknown as ComposerContextMenuItem
    ])

    const editor = composer('teh quick fox')

    select(editor, 1)
    fireEvent.contextMenu(editor)
    await screen.findByText('Paste')

    const labels = menuLabels()
    const pluginLabels = labels.slice(0, labels.indexOf(labels.find(label => label.startsWith('Cut'))!))

    expect(pluginLabels).toEqual(['Fix this', 'zeta 1', 'zeta 2', 'zeta 3', 'zeta 4', 'zeta 5', 'zeta 6'])
  })

  it('runs an item after the menu closed and lets it replace only the right-clicked word, once', async () => {
    let action: ComposerContextMenuAction | null = null
    let menuOpenDuringSelect: boolean | null = null

    plugin('alpha', context => [
      {
        label: `Replace ${context.word}`,
        onSelect: received => {
          menuOpenDuringSelect = $contextMenu.get() !== null
          action = received
        }
      }
    ])

    const editor = composer('teh quick fox')

    select(editor, 1)
    fireEvent.contextMenu(editor)
    fireEvent.click(await screen.findByText('Replace teh'))

    await waitFor(() => expect(action).not.toBeNull())
    expect(menuOpenDuringSelect).toBe(false)
    expect(document.activeElement).toBe(editor)
    expect(action!.word).toBe('teh')

    expect(action!.replaceWord('the')).toBe(true)
    expect(editor.textContent).toBe('the quick fox')
    expect(action!.replaceWord('then')).toBe(false)
    expect(editor.textContent).toBe('the quick fox')
  })

  it("puts a replacement on the composer's own undo stack", async () => {
    let action: ComposerContextMenuAction | null = null

    plugin('alpha', () => [{ label: 'Fix', onSelect: received => void (action = received) }])

    const editor = composer('teh quick fox')
    const undo = mountUndo(editor)

    select(editor, 1)
    fireEvent.contextMenu(editor)
    fireEvent.click(await screen.findByText('Fix'))
    await waitFor(() => expect(action).not.toBeNull())

    expect(action!.replaceWord('the')).toBe(true)
    expect(editor.textContent).toBe('the quick fox')

    act(() => void undo.current!.undo())

    expect(editor.textContent).toBe('teh quick fox')
  })

  it('refuses a replace once the composer text under the word has changed', async () => {
    let action: ComposerContextMenuAction | null = null

    plugin('alpha', () => [{ label: 'Later', onSelect: received => void (action = received) }])

    const editor = composer('teh quick fox')

    select(editor, 1)
    fireEvent.contextMenu(editor)
    fireEvent.click(await screen.findByText('Later'))
    await waitFor(() => expect(action).not.toBeNull())

    editor.textContent = 'something else entirely'

    expect(action!.replaceWord('the')).toBe(false)
    expect(editor.textContent).toBe('something else entirely')
  })

  it("drops a plugin's items when the plugin unloads, even from an open menu", async () => {
    plugin('alpha', () => [{ label: 'Plugin row', onSelect: vi.fn() }])

    const editor = composer('teh quick fox')

    select(editor, 1)
    fireEvent.contextMenu(editor)
    expect(await screen.findByText('Plugin row')).toBeTruthy()

    act(() => disposers.splice(0).forEach(dispose => dispose()))

    await waitFor(() => expect(screen.queryByText('Plugin row')).toBeNull())
    expect(screen.getByText('Paste')).toBeTruthy()
  })
})
