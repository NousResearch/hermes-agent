// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'
import type * as PreviewStoreModule from '@/store/preview'

const h = vi.hoisted(() => ({
  closeTab: vi.fn(),
  insert: vi.fn(),
  insertRefs: vi.fn(),
  newTab: vi.fn(),
  openBrowser: vi.fn()
}))

vi.mock('@/app/chat/right-rail/preview', () => ({
  PreviewTilePane: ({ tabId }: { tabId: string }) => (
    <div data-testid="preview-pane">
      {tabId}
      <webview />
    </div>
  )
}))

vi.mock('@/app/chat/composer/focus', () => ({
  requestComposerInsert: (...args: unknown[]) => h.insert(...args),
  requestComposerInsertRefs: (...args: unknown[]) => h.insertRefs(...args)
}))

vi.mock('@/store/preview', async importOriginal => {
  const actual = await importOriginal<typeof PreviewStoreModule>()

  return {
    ...actual,
    closeRightRailTab: (...args: unknown[]) => h.closeTab(...args),
    newBrowserTab: () => h.newTab(),
    openBrowserTab: () => h.openBrowser()
  }
})

import { $rightRailActiveTabId } from '@/store/layout'
import { $previewTabs } from '@/store/preview'

import { $ideActiveChat } from '../chat/store'

import { BrowserRegion } from './index'

const exampleTab = {
  id: 'url:1',
  target: { kind: 'url', label: 'Example', source: 'https://example.com', url: 'https://example.com' }
}

function renderRegion() {
  return render(
    <I18nProvider configClient={null} initialLocale="en">
      <BrowserRegion />
    </I18nProvider>
  )
}

beforeEach(() => {
  window.localStorage.clear()
  h.closeTab.mockReset()
  h.insert.mockReset()
  h.insertRefs.mockReset()
  h.newTab.mockReset()
  h.openBrowser.mockReset()
  $previewTabs.set([])
  $rightRailActiveTabId.set(null)
  $ideActiveChat.set(null)
})

afterEach(() => {
  cleanup()
})

describe('BrowserRegion', () => {
  it('shows the empty state and opens a browser tab from it', () => {
    renderRegion()

    expect(screen.getByText('No page open')).toBeTruthy()
    fireEvent.click(screen.getByText('Open browser'))

    expect(h.openBrowser).toHaveBeenCalledTimes(1)
  })

  it('renders the active tab and sends its URL to the active IDE composer', () => {
    $previewTabs.set([exampleTab as never])
    $rightRailActiveTabId.set('url:1' as never)
    $ideActiveChat.set('s1')

    renderRegion()

    expect(screen.getByTestId('preview-pane').textContent).toBe('url:1')

    fireEvent.click(screen.getByLabelText('Add page to chat'))

    expect(h.insert).toHaveBeenCalledWith('Page: https://example.com', { mode: 'block', target: 'tile:s1' })
  })

  it('picks an element through the webview and hands the chat a collapsed chip', async () => {
    $previewTabs.set([exampleTab as never])
    $rightRailActiveTabId.set('url:1' as never)
    $ideActiveChat.set('s2')

    renderRegion()

    const webview = screen.getByTestId('preview-pane').querySelector('webview') as HTMLElement & {
      executeJavaScript?: unknown
    }

    const execute = vi.fn().mockResolvedValue({ html: '<a href="/x">X</a>', selector: 'a:nth-of-type(1)' })
    webview.executeJavaScript = execute

    fireEvent.click(screen.getByLabelText('Inspect element'))

    await waitFor(() => expect(execute).toHaveBeenCalledTimes(1))
    await waitFor(() => expect(h.insertRefs).toHaveBeenCalledTimes(1))

    const [refs, options] = h.insertRefs.mock.calls[0] as [
      Array<{ kind: string; label: string; value: string }>,
      { target: string }
    ]

    expect(refs).toHaveLength(1)
    expect(refs[0]?.kind).toBe('element')
    expect(refs[0]?.label).toBe('a:nth-of-type(1)')
    expect(refs[0]?.value).toContain('<a href="/x">X</a>')
    expect(h.insert).not.toHaveBeenCalled()
    expect(options).toEqual({ target: 'tile:s2' })
  })

  it('closes a tab through the preview store', () => {
    $previewTabs.set([exampleTab as never])
    $rightRailActiveTabId.set('url:1' as never)

    renderRegion()
    fireEvent.click(screen.getByLabelText('Close Example'))

    expect(h.closeTab).toHaveBeenCalledWith('url:1')
  })

  it('keeps the strip controls outside the scrolling tab region', () => {
    $previewTabs.set([exampleTab as never])
    $rightRailActiveTabId.set('url:1' as never)

    renderRegion()

    const tablist = screen.getByRole('tablist', { name: 'Browser tabs' })

    // The + / inspect / add-page controls must not live inside the region that
    // scrolls with the tabs, or resizing the pane pushes them out of reach.
    expect(tablist.contains(screen.getByLabelText('New browser tab'))).toBe(false)
    expect(tablist.contains(screen.getByLabelText('Inspect element'))).toBe(false)
    expect(tablist.contains(screen.getByLabelText('Add page to chat'))).toBe(false)

    // And the preview pane reports its own context-menu affordance separately.
    expect(tablist.querySelectorAll('[role="tab"]').length).toBe(1)
  })
})
