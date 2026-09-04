import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'
import { $previewTabs, closeRightRail } from '@/store/preview'

import { ComposerDirectiveActions } from './directive-actions'
import { refChipElement } from './rich-editor'

const desktopWindow = window as unknown as { hermesDesktop?: Window['hermesDesktop'] }

const openSession = vi.fn()

vi.mock('@/app/open-session', () => ({ openSession: (...args: unknown[]) => openSession(...args) }))

/** A live contenteditable holding real chips, with the watcher bound to it —
 *  the same pair both composers mount. */
function mountEditor(chips: { kind: string; value: string }[]) {
  const editor = document.createElement('div')

  editor.contentEditable = 'true'
  editor.append(...chips.map(chip => refChipElement(chip.kind, `\`${chip.value}\``)))
  document.body.append(editor)

  render(
    <I18nProvider configClient={null} initialLocale="en">
      <ComposerDirectiveActions editorRef={{ current: editor }} />
    </I18nProvider>
  )

  return editor
}

function chips(editor: HTMLElement, kind: string) {
  return Array.from(editor.querySelectorAll(`[data-ref-kind="${kind}"]`))
}

function hover(node: Element) {
  fireEvent.pointerOver(node, { bubbles: true })
}

/** The reference the visible action pill points at, or null when there is none. */
function pillValue() {
  return document.querySelector('[data-slot="composer-directive-action"]')?.getAttribute('data-value') ?? null
}

afterEach(() => {
  cleanup()
  document.body.replaceChildren()
  closeRightRail()
  delete desktopWindow.hermesDesktop
  openSession.mockReset()
  vi.useRealTimers()
})

describe('ComposerDirectiveActions', () => {
  it('offers an action for a hovered actionable chip', () => {
    const editor = mountEditor([{ kind: 'url', value: 'https://example.com/docs' }])

    expect(pillValue()).toBeNull()

    hover(chips(editor, 'url')[0]!)

    expect(pillValue()).toBe('https://example.com/docs')
  })

  it('opens a url in the in-app browser rather than navigating the app', async () => {
    const openExternal = vi.fn().mockResolvedValue(undefined)

    desktopWindow.hermesDesktop = { openExternal } as unknown as Window['hermesDesktop']

    const editor = mountEditor([{ kind: 'url', value: 'https://example.com/docs' }])

    hover(chips(editor, 'url')[0]!)
    fireEvent.click(screen.getByRole('button'))

    expect(openExternal).not.toHaveBeenCalled()
    await vi.waitFor(() => expect($previewTabs.get().at(-1)?.target.url).toBe('https://example.com/docs'))
    expect(pillValue()).toBeNull()
  })

  it('runs the kind-specific action — a session chip opens the session', async () => {
    const editor = mountEditor([{ kind: 'session', value: 'default/20260722_204335_d62c16' }])

    hover(chips(editor, 'session')[0]!)
    fireEvent.click(screen.getByRole('button'))
    // openSessionRef lazy-imports the navigator, so the call lands a tick later.
    await vi.waitFor(() =>
      expect(openSession).toHaveBeenCalledWith('20260722_204335_d62c16', expect.any(Function), 'tab')
    )
  })

  it('leaves kinds with no action alone', () => {
    const editor = mountEditor([{ kind: 'file', value: 'src/main.tsx' }])

    hover(chips(editor, 'file')[0]!)

    expect(pillValue()).toBeNull()
  })

  it('follows the pointer from one chip to the next', () => {
    const editor = mountEditor([
      { kind: 'url', value: 'https://one.example' },
      { kind: 'url', value: 'https://two.example' }
    ])

    const [first, second] = chips(editor, 'url')

    hover(first!)

    expect(pillValue()).toBe('https://one.example')

    hover(second!)

    expect(pillValue()).toBe('https://two.example')
  })

  it('keeps the pill up while the pointer crosses onto it', () => {
    vi.useFakeTimers()

    const editor = mountEditor([{ kind: 'url', value: 'https://example.com' }])
    const chip = chips(editor, 'url')[0]!

    hover(chip)
    fireEvent.pointerOut(chip, { relatedTarget: document.body })
    fireEvent.mouseEnter(screen.getByRole('button').parentElement!)
    vi.advanceTimersByTime(500)

    expect(pillValue()).toBe('https://example.com')
  })

  it('binds to the document so a late-attached editor still gets the affordance', () => {
    // The edit composer's editor isn't reliably in the DOM when the effect
    // first runs; a document listener that reads the editor lazily works
    // regardless — this is the whole reason it binds to document, not editor.
    const editor = document.createElement('div')

    editor.contentEditable = 'true'
    editor.append(refChipElement('url', '`https://late.example`'))

    render(
      <I18nProvider configClient={null} initialLocale="en">
        <ComposerDirectiveActions editorRef={{ current: editor }} />
      </I18nProvider>
    )

    // Editor attached AFTER mount.
    document.body.append(editor)
    hover(editor.querySelector('[data-ref-kind="url"]')!)

    expect(pillValue()).toBe('https://late.example')
  })
})
