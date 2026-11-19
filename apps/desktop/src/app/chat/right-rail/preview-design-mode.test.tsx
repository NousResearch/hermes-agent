import { PREVIEW_DESIGN_EVENT_CHANNEL, type PreviewDesignCommand } from '@hermes/shared/preview-design'
import { act, cleanup, fireEvent, render, waitFor, within } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { markActiveComposer, onComposerInsertRequest } from '@/app/chat/composer/focus'
import { group } from '@/components/pane-shell/tree/model'
import { $layoutTree, noteActiveTreeGroup } from '@/components/pane-shell/tree/store'
import { $previewDesignRequest } from '@/store/preview-design'
import { $selectedStoredSessionId } from '@/store/session'
import { $focusedStoredSessionId } from '@/store/session-focus'

import { installGuestDesignBridge } from '../../../../electron/preview-guest-preload'

import { PreviewPane, safeDesignPageUrl } from './preview-pane'

vi.mock('./real-profile-consent-dialog', () => ({ RealProfileConsentDialog: () => null }))

afterEach(() => {
  cleanup()
  $previewDesignRequest.set(null)
  markActiveComposer('main')
  $selectedStoredSessionId.set(null)
})

function wireDesignBridge(webview: Element, doc: Document) {
  const listeners = new Map<string, (_event: unknown, payload: unknown) => void>()

  installGuestDesignBridge(doc, {
    on: (channel, listener) => listeners.set(channel, listener),
    sendToHost: (channel, payload) => {
      expect(channel).toBe(PREVIEW_DESIGN_EVENT_CHANNEL)
      webview.dispatchEvent(Object.assign(new Event('ipc-message'), { args: [payload], channel }))
    }
  })

  return (channel: string, payload: PreviewDesignCommand) => listeners.get(channel)?.({}, payload)
}

function fixture() {
  const rendered = render(
    <PreviewPane
      sessionId="session-a"
      tabId="browser-a"
      target={{ kind: 'url', label: 'Page', source: 'https://example.com', url: 'https://example.com' }}
    />
  )

  const frame = window.document.createElement('iframe')
  rendered.container.appendChild(frame)
  const doc = frame.contentDocument!
  const win = frame.contentWindow!
  const webview = rendered.container.querySelector('webview')!
  const scripts = vi.fn(async (code: string) => new Function('window', 'document', 'return (' + code + ')')(win, doc))
  const send = wireDesignBridge(webview, doc)
  Object.assign(webview, { executeJavaScript: scripts, getURL: () => 'https://example.com', send })
  const button = doc.createElement('button')
  button.textContent = 'Save'
  doc.body.appendChild(button)
  doc.elementFromPoint = () => button
  button.getBoundingClientRect = () => ({
    x: 10,
    y: 20,
    left: 10,
    top: 20,
    width: 90,
    height: 30,
    right: 100,
    bottom: 50,
    toJSON: () => ({})
  })

  return { button, doc, win, webview, scripts, ...rendered }
}

describe('Browser Design Mode', () => {
  it('removes credentials, query parameters, and fragments from prompt page context', () => {
    expect(safeDesignPageUrl('https://user:secret@example.com/account?token=private#billing')).toBe(
      'https://example.com/account'
    )
  })

  it('accepts a command from the focused conversation tile rather than the hidden primary', async () => {
    const originalTree = $layoutTree.get()
    $selectedStoredSessionId.set('session-a')
    $layoutTree.set(group(['workspace', 'session-tile:session-b'], { active: 'session-tile:session-b', id: 'design' }))
    noteActiveTreeGroup('design')
    expect($focusedStoredSessionId.get()).toBe('session-b')

    try {
      const page = render(
        <PreviewPane
          sessionId="session-b"
          tabId="browser-b"
          target={{ kind: 'url', label: 'Tile', source: 'https://example.com', url: 'https://example.com' }}
        />
      )

      const frame = window.document.createElement('iframe')
      page.container.appendChild(frame)
      const webview = page.container.querySelector('webview')!
      const send = wireDesignBridge(webview, frame.contentDocument!)
      Object.assign(webview, {
        executeJavaScript: async (code: string) =>
          new Function('window', 'document', 'return (' + code + ')')(frame.contentWindow!, frame.contentDocument!),
        getURL: () => 'https://example.com',
        send
      })

      await act(async () => $previewDesignRequest.set({ sessionId: 'session-b', tabId: 'browser-b' }))
      await waitFor(() => expect(within(page.container).getByRole('region', { name: 'DOM Layers' })).toBeTruthy())
      expect(frame.contentDocument!.querySelector('hermes-design')).toBeTruthy()
    } finally {
      await act(async () => {
        noteActiveTreeGroup(null)
        $layoutTree.set(originalTree)
      })
    }
  })

  it('starts only in the addressed Browser and stops on a conversation switch', async () => {
    const originalTree = $layoutTree.get()
    $layoutTree.set(group(['session-tile:session-a', 'session-tile:session-b'], { active: 'session-tile:session-a', id: 'switch' }))
    noteActiveTreeGroup('switch')
    const page = fixture()

    const other = render(
      <PreviewPane
        sessionId="session-b"
        tabId="browser-b"
        target={{ kind: 'url', label: 'Other', source: 'https://other.test', url: 'https://other.test' }}
      />
    )

    await act(async () => $previewDesignRequest.set({ sessionId: 'session-a', tabId: 'browser-a' }))
    await waitFor(() => expect(page.doc.querySelector('hermes-design')).toBeTruthy())
    expect(within(other.container).queryByRole('region', { name: 'DOM Layers' })).toBeNull()
    expect(page.getByRole('region', { name: 'DOM Layers' })).toBeTruthy()
    expect($previewDesignRequest.get()).toBeNull()

    try {
      await act(async () =>
        $layoutTree.set(group(['session-tile:session-a', 'session-tile:session-b'], { active: 'session-tile:session-b', id: 'switch' }))
      )
      await waitFor(() => expect(page.doc.querySelector('hermes-design')).toBeNull())
      expect(page.queryByRole('region', { name: 'DOM Layers' })).toBeNull()
    } finally {
      await act(async () => {
        noteActiveTreeGroup(null)
        $layoutTree.set(originalTree)
      })
    }
  })

  it('shows a Layers panel only in Design Mode and leaves it empty until selection', async () => {
    const page = fixture()
    expect(page.queryByRole('region', { name: 'DOM Layers' })).toBeNull()
    fireEvent.click(page.getByRole('button', { name: 'Design Mode' }))
    const layers = await page.findByRole('region', { name: 'DOM Layers' })
    expect(within(layers).queryByRole('button', { name: /button/i })).toBeNull()
    await act(async () => page.win.dispatchEvent(new MouseEvent('click', { clientX: 12, clientY: 22, bubbles: true })))
    expect(within(layers).getByRole('button', { name: 'body' })).toBeTruthy()
    expect(within(layers).getByRole('button', { name: 'button' })).toBeTruthy()
    fireEvent.click(page.getByRole('button', { name: 'Stop Design Mode' }))
    await waitFor(() => expect(page.queryByRole('region', { name: 'DOM Layers' })).toBeNull())
  })

  it('keeps the page unmarked until the user selects an element', async () => {
    const page = fixture()
    fireEvent.click(page.getByRole('button', { name: 'Design Mode' }))
    await waitFor(() => expect(page.doc.querySelector('hermes-design')).toBeTruthy())
    const shadow = page.doc.querySelector('hermes-design')!.shadowRoot!
    const selection = shadow.querySelector<HTMLElement>('[data-design-selected]')!
    page.win.dispatchEvent(new MouseEvent('mousemove', { clientX: 12, clientY: 22, bubbles: true }))
    expect(shadow.querySelector('[data-design-hover]')).toBeNull()
    expect(selection.style.display).toBe('none')
    await act(async () => page.win.dispatchEvent(new MouseEvent('click', { clientX: 12, clientY: 22, bubbles: true })))
    expect(selection.style.display).toBe('block')
  })

  it('shows DOM Layers in this Browser and selects an ancestor for the prompt', async () => {
    const page = fixture()
    const section = page.doc.createElement('section')
    page.doc.body.appendChild(section)
    section.appendChild(page.button)
    const inserts: string[] = []
    const unsubscribe = onComposerInsertRequest(detail => inserts.push(detail.text))

    try {
      fireEvent.click(page.getByRole('button', { name: 'Design Mode' }))
      await waitFor(() => expect(page.doc.querySelector('hermes-design')).toBeTruthy())
      await act(async () => page.win.dispatchEvent(new MouseEvent('click', { clientX: 12, clientY: 22, bubbles: true })))
      const layers = await page.findByRole('region', { name: 'DOM Layers' })
      fireEvent.click(within(layers).getByRole('button', { name: 'section' }))
      await waitFor(() => expect(page.container.querySelector('webview')).toBeTruthy())
      fireEvent.click(page.getByRole('button', { name: 'Add selection to prompt' }))
      await waitFor(() => expect(inserts).toHaveLength(1))
      expect(inserts[0]).toContain('Element: <section>')
      expect(page.container.querySelectorAll('webview')).toHaveLength(1)
    } finally {
      unsubscribe()
    }
  })

  it('selects in the existing guest and inserts context only on an explicit prompt action', async () => {
    const page = fixture()
    const inserts: string[] = []
    const unsubscribe = onComposerInsertRequest(detail => inserts.push(detail.text))

    try {
      fireEvent.click(page.getByRole('button', { name: 'Design Mode' }))
      await waitFor(() => expect(page.doc.querySelector('hermes-design')).toBeTruthy())
      expect(page.container.querySelectorAll('webview')).toHaveLength(1)
      page.win.dispatchEvent(new MouseEvent('mousemove', { clientX: 12, clientY: 22, bubbles: true }))
      expect(page.doc.querySelector('hermes-design')!.shadowRoot!.querySelector('[data-design-hover]')).toBeNull()
      expect(inserts).toEqual([])
      await act(async () =>
        page.win.dispatchEvent(new MouseEvent('click', { clientX: 12, clientY: 22, bubbles: true, cancelable: true }))
      )
      expect(inserts).toEqual([])
      await waitFor(() => expect(page.getByRole('button', { name: 'Add selection to prompt' })).toBeTruthy())
      fireEvent.click(page.getByRole('button', { name: 'Add selection to prompt' }))
      await waitFor(() => expect(inserts).toHaveLength(1))
      expect(inserts[0]).toContain('button')
      expect(inserts[0]).toContain('https://example.com')
      expect(inserts[0]).not.toMatch(/source file|src\/|\.tsx/)
    } finally {
      unsubscribe()
    }
  })

  it('removes its guest listeners and overlay when switched off', async () => {
    const page = fixture()
    const nativeClick = vi.fn()
    page.button.addEventListener('click', nativeClick)
    fireEvent.click(page.getByRole('button', { name: 'Design Mode' }))
    await waitFor(() => expect(page.doc.querySelector('hermes-design')).toBeTruthy())
    fireEvent.click(page.getByRole('button', { name: 'Stop Design Mode' }))
    await waitFor(() => expect(page.doc.querySelector('hermes-design')).toBeNull())
    const click = new MouseEvent('click', { bubbles: true, cancelable: true, clientX: 12, clientY: 22 })
    page.button.dispatchEvent(click)
    expect(click.defaultPrevented).toBe(false)
    expect(nativeClick).toHaveBeenCalledOnce()
    const escape = new KeyboardEvent('keydown', { cancelable: true, key: 'Escape' })
    page.win.dispatchEvent(escape)
    expect(escape.defaultPrevented).toBe(false)
  })

  it('blocks page mousedown actions while selecting', async () => {
    const page = fixture()
    const nativeDown = vi.fn()
    page.button.addEventListener('mousedown', nativeDown)
    fireEvent.click(page.getByRole('button', { name: 'Design Mode' }))
    await waitFor(() => expect(page.doc.querySelector('hermes-design')).toBeTruthy())
    const down = new MouseEvent('mousedown', { bubbles: true, cancelable: true, button: 0, clientX: 12, clientY: 22 })
    page.button.dispatchEvent(down)
    expect(down.defaultPrevented).toBe(true)
    expect(nativeDown).not.toHaveBeenCalled()
  })

  it.each(['did-navigate', 'did-navigate-in-page'])('drops selection on %s', async eventName => {
    const page = fixture()
    fireEvent.click(page.getByRole('button', { name: 'Design Mode' }))
    await waitFor(() => expect(page.doc.querySelector('hermes-design')).toBeTruthy())
    await act(async () => page.win.dispatchEvent(new MouseEvent('click', { clientX: 12, clientY: 22, bubbles: true })))
    await waitFor(() => expect(page.getByRole('button', { name: 'Add selection to prompt' })).toBeTruthy())
    await act(async () =>
      page.webview.dispatchEvent(Object.assign(new Event(eventName), { url: 'https://example.com/next' }))
    )
    await waitFor(() => expect(page.doc.querySelector('hermes-design')).toBeNull())
    expect(page.queryByRole('button', { name: 'Add selection to prompt' })).toBeNull()
    expect(page.getByRole('button', { name: 'Design Mode' })).toBeTruthy()
  })

  it('switches between Design Mode and Annotate without stacking overlays', async () => {
    const page = fixture()
    fireEvent.click(page.getByRole('button', { name: 'Design Mode' }))
    await waitFor(() => expect(page.doc.querySelector('hermes-design')).toBeTruthy())
    fireEvent.click(page.getByRole('button', { name: 'Annotate' }))
    await waitFor(() => expect(page.doc.querySelector('hermes-annotate')).toBeTruthy())
    expect(page.doc.querySelector('hermes-design')).toBeNull()
    fireEvent.click(page.getByRole('button', { name: 'Design Mode' }))
    await waitFor(() => expect(page.doc.querySelector('hermes-design')).toBeTruthy())
    expect(page.doc.querySelector('hermes-annotate')).toBeNull()
  })

  it('does not insert into another selected conversation', async () => {
    const page = fixture()
    fireEvent.click(page.getByRole('button', { name: 'Design Mode' }))
    await waitFor(() => expect(page.doc.querySelector('hermes-design')).toBeTruthy())
    await act(async () => page.win.dispatchEvent(new MouseEvent('click', { clientX: 12, clientY: 22, bubbles: true })))
    await waitFor(() => expect(page.getByRole('button', { name: 'Add selection to prompt' })).toBeTruthy())
    const inserted = vi.fn()
    const unsubscribe = onComposerInsertRequest(inserted)

    try {
      await act(async () => $selectedStoredSessionId.set('session-b'))
      expect(page.queryByRole('button', { name: 'Add selection to prompt' })).toBeNull()
      expect(inserted).not.toHaveBeenCalled()
    } finally {
      unsubscribe()
    }
  })

  it('addresses the owning composer rather than a different active tile', async () => {
    const page = fixture()
    fireEvent.click(page.getByRole('button', { name: 'Design Mode' }))
    await waitFor(() => expect(page.doc.querySelector('hermes-design')).toBeTruthy())
    await act(async () => page.win.dispatchEvent(new MouseEvent('click', { clientX: 12, clientY: 22, bubbles: true })))
    await waitFor(() => expect(page.getByRole('button', { name: 'Add selection to prompt' })).toBeTruthy())
    const otherComposer = window.document.createElement('div')
    otherComposer.dataset.composerTarget = 'tile:other-session'
    page.container.appendChild(otherComposer)
    markActiveComposer('tile:other-session')
    const targets: string[] = []
    const unsubscribe = onComposerInsertRequest(detail => targets.push(detail.target))

    try {
      fireEvent.click(page.getByRole('button', { name: 'Add selection to prompt' }))
      await waitFor(() => expect(targets).toEqual(['main']))
    } finally {
      unsubscribe()
    }
  })
})
