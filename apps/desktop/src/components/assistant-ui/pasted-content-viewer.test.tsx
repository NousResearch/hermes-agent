// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { DirectiveContent } from './directive-text'

// The viewer inserts through the composer bus; stubbing it keeps the assertion
// on the text that leaves the dialog, not on a mounted composer.
const requestComposerInsert = vi.fn()

vi.mock('@/app/chat/composer/focus', () => ({
  requestComposerInsert: (...args: unknown[]) => requestComposerInsert(...args)
}))

// `electron/composer-paste.ts` names every saved paste this way; the trailing
// `-N` covers a staged copy of the same paste.
const PASTE_PATH = '/tmp/hermes/pasted_content_20261104_091500_ab12cd.txt'
const PASTE_TEXT = `first line\nsecond line\n${'pasted body '.repeat(300)}`

function installBridge(readFileText: (path: string) => Promise<unknown>) {
  Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: { readFileText } })
}

function chipIn(container: HTMLElement): HTMLElement {
  const element = container.querySelector<HTMLElement>('[data-slot="aui_directive-chip"]')

  if (!element) {
    throw new Error('no directive chip rendered')
  }

  return element
}

beforeEach(() => {
  requestComposerInsert.mockReset()
})

afterEach(() => {
  cleanup()
  Reflect.deleteProperty(window, 'hermesDesktop')
})

describe('pasted content chip in a sent message', () => {
  // Regression: the chip used to be an inert span. The click bubbled to the
  // bubble's edit button, which reopened the message with nothing but that same
  // `@file:` ref — an empty editor next to "Pasted content".
  it('is an activatable chip that keeps its path in the tooltip', () => {
    installBridge(async () => ({ path: PASTE_PATH, text: PASTE_TEXT }))
    const { container } = render(<DirectiveContent text={`@file:${PASTE_PATH}`} />)
    const chip = chipIn(container)

    expect(chip.tagName).toBe('BUTTON')
    expect(chip.textContent).toContain('Pasted content')
    expect(chip.getAttribute('title')).toBe(PASTE_PATH)
    expect(chip.getAttribute('aria-haspopup')).toBe('dialog')
  })

  it('does not let the press reach the bubble that opens the edit composer', () => {
    installBridge(async () => ({ path: PASTE_PATH, text: PASTE_TEXT }))
    const bubbleClick = vi.fn()
    const bubblePointerDown = vi.fn()

    const { container } = render(
      <div onClick={bubbleClick} onPointerDown={bubblePointerDown}>
        <DirectiveContent text={`@file:${PASTE_PATH}`} />
      </div>
    )

    const chip = chipIn(container)

    fireEvent.pointerDown(chip)
    fireEvent.click(chip)

    expect(bubblePointerDown).not.toHaveBeenCalled()
    expect(bubbleClick).not.toHaveBeenCalled()
  })

  it('opens a read-only view of the text read from disk', async () => {
    const readFileText = vi.fn(async () => ({ path: PASTE_PATH, text: PASTE_TEXT }))
    installBridge(readFileText)
    const { container } = render(<DirectiveContent text={`@file:${PASTE_PATH}`} />)

    fireEvent.click(chipIn(container))

    const dialog = await screen.findByRole('dialog')
    await waitFor(() => expect(readFileText).toHaveBeenCalledWith(PASTE_PATH))

    expect(dialog.querySelector('[data-slot="pasted-content-text"]')?.textContent).toBe(PASTE_TEXT)
    // Read-only: nothing in the dialog can take the paste back into an editor.
    expect(dialog.querySelector('textarea, [contenteditable="true"]')).toBeNull()
  })

  it('offers copy and insert-into-composer actions for the full text', async () => {
    installBridge(async () => ({ path: PASTE_PATH, text: PASTE_TEXT }))
    const { container } = render(<DirectiveContent text={`@file:${PASTE_PATH}`} />)

    fireEvent.click(chipIn(container))

    expect(await screen.findByRole('button', { name: 'Copy full text' })).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: 'Insert into composer' }))

    await waitFor(() => expect(requestComposerInsert).toHaveBeenCalledWith(PASTE_TEXT, { mode: 'block' }))
  })

  // A cleaned-up temp file must read as gone, never as an empty paste.
  it('says the file is gone instead of showing an empty box', async () => {
    installBridge(async () => ({ error: 'ENOENT', message: 'no such file', ok: false }))
    const { container } = render(<DirectiveContent text={`@file:${PASTE_PATH}`} />)

    fireEvent.click(chipIn(container))

    const dialog = await screen.findByRole('dialog')

    expect(await screen.findByText(/no longer on disk/)).toBeTruthy()
    expect(dialog.querySelector('[data-slot="pasted-content-text"]')).toBeNull()
    expect((screen.getByRole('button', { name: 'Insert into composer' }) as HTMLButtonElement).disabled).toBe(true)
  })

  it('surfaces an unreadable paste with the underlying error', async () => {
    installBridge(async () => {
      throw new Error('permission denied')
    })
    const { container } = render(<DirectiveContent text={`@file:${PASTE_PATH}`} />)

    fireEvent.click(chipIn(container))

    expect(await screen.findByText(/permission denied/)).toBeTruthy()
  })

  it('leaves an ordinary file reference inert', () => {
    installBridge(async () => ({ path: '/home/q/notes.md', text: '# notes' }))
    const { container } = render(<DirectiveContent text="@file:/home/q/notes.md" />)
    const chip = chipIn(container)

    expect(chip.tagName).toBe('SPAN')
    expect(chip.getAttribute('aria-haspopup')).toBeNull()
  })
})
