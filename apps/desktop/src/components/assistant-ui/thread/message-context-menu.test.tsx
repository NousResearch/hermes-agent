import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest'

import { ComposerScopeProvider, MAIN_COMPOSER_SCOPE } from '@/app/chat/composer/scope'
import { MessageContextMenu } from '@/components/assistant-ui/thread/message-context-menu'
import { createComposerAttachmentScope } from '@/store/composer'
import { $documentSelection } from '@/store/selection'

// Radix ContextMenu uses PointerEvent; jsdom doesn't fire it by default.
// fireEvent.contextMenu triggers the right-click that opens the menu.
// The menu content renders in a portal, so we query by role.

vi.mock('@/app/chat/composer/focus', () => ({
  requestComposerFocus: vi.fn(),
  requestComposerInsert: vi.fn(),
}))

vi.mock('@/components/ui/copy-button', () => ({
  writeClipboardText: vi.fn().mockResolvedValue(undefined),
}))

vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      common: {
        addAsContext: 'Add as context',
        copy: 'Copy',
        pasteAsText: 'Paste as text',
        selectAll: 'Select All',
      },
    },
  }),
}))

beforeAll(() => {
  Element.prototype.hasPointerCapture ??= () => false
  Element.prototype.setPointerCapture ??= () => undefined
  Element.prototype.releasePointerCapture ??= () => undefined
})

/** Select real DOM text with the browser's own Range machinery, so the
 *  menu's intersects-host check exercises the actual selection API. */
function selectText(host: HTMLElement): void {
  const range = document.createRange()
  range.selectNodeContents(host)
  const selection = window.getSelection()
  selection?.removeAllRanges()
  selection?.addRange(range)
}

/** A selection in a DIFFERENT part of the document than the menu's host. */
function selectForeignText(): HTMLElement {
  const foreign = document.createElement('div')
  foreign.textContent = 'text somewhere else entirely'
  document.body.appendChild(foreign)
  selectText(foreign)

  return foreign
}

function clearSelection(): void {
  window.getSelection()?.removeAllRanges()
}

/** Radix opens a ContextMenu on contextmenu after a pointerdown positions it. */
function openContextMenu(target: HTMLElement) {
  fireEvent.pointerDown(target, { button: 2, pointerType: 'mouse' })
  fireEvent.contextMenu(target, { button: 2 })
}

describe('MessageContextMenu', () => {
  afterEach(() => {
    cleanup()
    clearSelection()
    $documentSelection.set(null)
    document.querySelectorAll('body > div:not([id])').forEach(node => {
      if (node.textContent === 'text somewhere else entirely') {
        node.remove()
      }
    })
    vi.restoreAllMocks()
  })

  it('renders children and keeps the menu closed when no text is selected', async () => {
    render(
      <MessageContextMenu messageId="msg-1">
        <div data-testid="child">Hello</div>
      </MessageContextMenu>
    )

    // Children always render — the wrapper stays mounted so the DOM tree
    // is stable across selection changes (no mid-drag remounts).
    expect(screen.getByTestId('child')).toBeTruthy()

    // No selection → trigger is disabled → right-click shows nothing from us.
    const child = screen.getByTestId('child')
    openContextMenu(child)

    await act(async () => {
      await Promise.resolve()
    })
    expect(screen.queryByText('Add as context')).toBeNull()
  })

  it('keeps Copy and Select All alongside the context items when text is selected', async () => {
    render(
      <MessageContextMenu messageId="msg-1">
        <div data-testid="child">Selectable text here</div>
      </MessageContextMenu>
    )

    selectText(screen.getByTestId('child'))
    await act(async () => {
      document.dispatchEvent(new Event('selectionchange'))
    })

    const child = screen.getByTestId('child')
    openContextMenu(child)

    // Menu items appear in a portal — the standard actions stay, our two
    // additions follow below the separator.
    expect(await screen.findByText('Copy')).toBeTruthy()
    expect(screen.getByText('Select All')).toBeTruthy()
    expect(screen.getByText('Add as context')).toBeTruthy()
    expect(screen.getByText('Paste as text')).toBeTruthy()
  })

  it('does not arm when the selection lives in another part of the document', async () => {
    render(
      <MessageContextMenu messageId="msg-1">
        <div data-testid="child">This message's own text</div>
      </MessageContextMenu>
    )

    // Text selected OUTSIDE the message (e.g. in the composer) must not
    // arm THIS message's menu — a right-click here would otherwise stage
    // foreign text.
    selectForeignText()
    await act(async () => {
      document.dispatchEvent(new Event('selectionchange'))
    })

    const child = screen.getByTestId('child')
    openContextMenu(child)

    await act(async () => {
      await Promise.resolve()
    })
    expect(screen.queryByText('Add as context')).toBeNull()
  })

  it('stages the chip into the ambient composer scope, not always the main one', async () => {
    const tileAttachments = createComposerAttachmentScope()
    const tileScope = { ...MAIN_COMPOSER_SCOPE, attachments: tileAttachments, target: 'tile:s-1' as const }

    render(
      <ComposerScopeProvider value={tileScope}>
        <MessageContextMenu messageId="msg-1">
          <div data-testid="child">Tile message text</div>
        </MessageContextMenu>
      </ComposerScopeProvider>
    )

    selectText(screen.getByTestId('child'))
    await act(async () => {
      document.dispatchEvent(new Event('selectionchange'))
    })

    openContextMenu(screen.getByTestId('child'))
    fireEvent.click(await screen.findByText('Add as context'))

    // The chip lands in the TILE's attachment scope — the same routing
    // sibling gestures (drag-drop, shift+click) use — not the main atom.
    const staged = tileAttachments.$attachments.get()
    expect(staged).toHaveLength(1)
    expect(staged[0]!.kind).toBe('text')
    expect(staged[0]!.textContent).toBe('Tile message text')
    expect(staged[0]!.sourceMessageId).toBe('msg-1')
  })

  it('routes Paste as text to the ambient composer target', async () => {
    const { requestComposerInsert } = await import('@/app/chat/composer/focus')
    const tileScope = { ...MAIN_COMPOSER_SCOPE, target: 'tile:s-2' as const }

    render(
      <ComposerScopeProvider value={tileScope}>
        <MessageContextMenu messageId="msg-1">
          <div data-testid="child">Tile message text</div>
        </MessageContextMenu>
      </ComposerScopeProvider>
    )

    selectText(screen.getByTestId('child'))
    await act(async () => {
      document.dispatchEvent(new Event('selectionchange'))
    })

    openContextMenu(screen.getByTestId('child'))
    fireEvent.click(await screen.findByText('Paste as text'))

    expect(requestComposerInsert).toHaveBeenCalledWith('> Tile message text\n\n', {
      mode: 'block',
      target: 'tile:s-2'
    })
  })
})