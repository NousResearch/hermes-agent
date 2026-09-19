import { EditorView } from '@codemirror/view'
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'
import { composerFocusBlockedBySurface } from '@/lib/keybinds/composer-focus-keys'
import { $activeGatewayProfile } from '@/store/profile'

import { BackworkspacePage } from './back-page'
import { $backworkspaceOpen, toggleBackworkspace } from './store'

const request = vi.fn<(...args: unknown[]) => Promise<unknown>>()
const askAgent = vi.fn<(...args: unknown[]) => Promise<string>>()

vi.mock('@/store/gateway', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  requestGatewayForAgent: (...args: unknown[]) => request(...args)
}))
vi.mock('./ask', () => ({ askBackworkspace: (...args: unknown[]) => askAgent(...args) }))
vi.mock('./store', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  toggleBackworkspace: vi.fn()
}))

function renderPage() {
  $backworkspaceOpen.set(true)

  return render(
    <I18nProvider configClient={null} initialLocale="en">
      <BackworkspacePage />
    </I18nProvider>
  )
}

afterEach(() => {
  cleanup()
  $backworkspaceOpen.set(false)
  request.mockReset()
  askAgent.mockReset()
  vi.mocked(toggleBackworkspace).mockClear()
})

describe('BackworkspacePage', () => {
  it('owns the keyboard while turned over and shows the shell again on any unmount', () => {
    // Never answers: the page stays loading, which is when focus used to fall
    // through to <body> and the hidden composer.
    request.mockImplementation(() => new Promise(() => {}))
    $activeGatewayProfile.set('still-loading')

    const { unmount } = renderPage()
    const sheet = screen.getByRole('region', { name: 'Back workspace' })
    const root = sheet.ownerDocument.documentElement

    expect(root.hasAttribute('data-backworkspace')).toBe(true)
    expect(sheet).toBe(sheet.ownerDocument.activeElement)
    expect(composerFocusBlockedBySurface()).toBe(true)

    fireEvent.keyDown(sheet, { key: 'Escape' })
    expect(toggleBackworkspace).toHaveBeenCalledTimes(1)

    // An error boundary replacing the tree unmounts the page the same way.
    unmount()
    expect(root.hasAttribute('data-backworkspace')).toBe(false)
  })

  it('offers the window profile as @hermes when a mention is typed, and Enter writes it', async () => {
    request.mockResolvedValue({ page: null })
    $activeGatewayProfile.set('default')

    renderPage()

    const host = await screen.findByLabelText('Back workspace', { selector: '.cm-content' })
    const view = EditorView.findFromDOM(host as HTMLElement)!

    act(() => {
      view.dispatch({ changes: { from: 0, insert: '@he' }, selection: { anchor: 3 } })
    })

    // The profile's own agent is offered as @hermes; the composer's bot sources
    // add the rest when Bot Mode is registered.
    const list = await screen.findByRole('listbox')

    expect(list.textContent).toContain('@hermes')

    // Escape belongs to the open list, not to the window behind it.
    fireEvent.keyDown(view.contentDOM, { key: 'Escape' })
    expect(screen.queryByRole('listbox')).toBeNull()
    expect(toggleBackworkspace).not.toHaveBeenCalled()

    act(() => {
      view.dispatch({ changes: { from: 0, insert: '@he', to: view.state.doc.length }, selection: { anchor: 3 } })
    })
    await screen.findByRole('listbox')

    fireEvent.keyDown(view.contentDOM, { key: 'Enter' })
    expect(view.state.doc.toString()).toBe('@hermes ')
    expect(screen.queryByRole('listbox')).toBeNull()
  })

  it('sends the paragraph with the mod-enter chord and writes the reply under it', async () => {
    request.mockResolvedValue({
      page: { content: '@asking what is this?', id: '20260920_101010_abcdef', path: '/p.md' }
    })
    askAgent.mockResolvedValue('a short answer')
    $activeGatewayProfile.set('asking')

    renderPage()

    const host = await screen.findByLabelText('Back workspace', { selector: '.cm-content' })
    const view = EditorView.findFromDOM(host as HTMLElement)!
    // CodeMirror resolves `Mod` per platform, exactly as it will at runtime.
    const mod = /Mac/i.test(navigator.platform) ? { metaKey: true } : { ctrlKey: true }

    fireEvent.keyDown(view.contentDOM, { key: 'Enter', ...mod })

    await vi.waitFor(() => expect(askAgent).toHaveBeenCalledTimes(1))
    expect(askAgent.mock.calls[0][1]).toBe('@asking what is this?')
    await vi.waitFor(() => expect(view.state.doc.toString()).toContain('> asking · '))
    // The chord must not also leave CodeMirror's own blank line behind.
    expect(view.state.doc.toString().startsWith('@asking what is this?\n\n>')).toBe(true)
  })

  it('opens the stored page in the editor, ready to type', async () => {
    request.mockResolvedValue({ page: { content: 'text from the file', id: '20260920_101010_abcdef' } })
    $activeGatewayProfile.set('with-a-page')

    renderPage()

    const line = await screen.findByText('text from the file')
    const editor = line.closest('.cm-content')

    expect(editor).not.toBeNull()
    expect(editor).toBe(line.ownerDocument.activeElement)
  })
})
