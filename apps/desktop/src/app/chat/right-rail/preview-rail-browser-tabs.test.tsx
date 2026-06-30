import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { I18nProvider } from '@/i18n/context'
import {
  $browserTabs,
  clearBrowserTabs,
  createBrowserTab,
  setBrowserEnabled
} from '@/store/browser'
import { $rightRailActiveTabId } from '@/store/layout'
import { clearSessionPreviewRegistry, type PreviewTarget, setCurrentSessionPreviewTarget } from '@/store/preview'
import { setActiveSessionId } from '@/store/session'

import { ChatPreviewRail } from './preview'

function renderWithI18n(ui: React.ReactNode) {
  return render(
    <I18nProvider configClient={{ getConfig: async () => ({}), saveConfig: async () => ({ ok: true }) }}>
      {ui}
    </I18nProvider>
  )
}

function previewTarget(source: string): PreviewTarget {
  return {
    kind: 'file',
    label: source,
    path: source,
    previewKind: 'html',
    source,
    url: `file://${source}`
  }
}

describe('ChatPreviewRail browser tabs', () => {
  beforeEach(() => {
    setActiveSessionId(null)
    window.localStorage.clear()
    clearSessionPreviewRegistry()
    clearBrowserTabs()
    setBrowserEnabled(false)
  })

  afterEach(() => {
    cleanup()
    clearBrowserTabs()
    clearSessionPreviewRegistry()
    setBrowserEnabled(false)
    window.localStorage.clear()
  })

  it('renders an active browser tab as an additive rail surface once Browser is enabled', () => {
    setBrowserEnabled(true)
    const tab = createBrowserTab({ sessionId: 'session-1', url: 'https://example.com/app' })

    renderWithI18n(<ChatPreviewRail />)

    expect($rightRailActiveTabId.get()).toBe(tab.id)
    expect(screen.getByRole('tab', { name: 'example.com' })).toBeDefined()
    expect((screen.getByRole('textbox', { name: 'Browser URL' }) as HTMLInputElement).value).toBe(
      'https://example.com/app'
    )
  })

  it('opens a new Browser tab from the rail chrome without disturbing Preview', () => {
    setBrowserEnabled(true)
    setActiveSessionId('session-rail')
    setCurrentSessionPreviewTarget(previewTarget('/work/live.html'), 'tool-result')

    renderWithI18n(<ChatPreviewRail />)

    fireEvent.click(screen.getByRole('button', { name: 'New Browser tab' }))

    expect($browserTabs.get()).toHaveLength(1)
    expect($browserTabs.get()[0]).toMatchObject({ sessionId: 'session-rail', title: 'Browser', url: 'about:blank' })
    expect($rightRailActiveTabId.get()).toBe($browserTabs.get()[0]?.id)
    expect(screen.getByRole('tab', { name: 'Browser' })).toBeDefined()
  })

  it('requires an explicit rail switch before opening Browser tabs', () => {
    setCurrentSessionPreviewTarget(previewTarget('/work/live.html'), 'tool-result')

    renderWithI18n(<ChatPreviewRail />)

    expect(screen.queryByRole('button', { name: 'New Browser tab' })).toBeNull()
    fireEvent.click(screen.getByRole('button', { name: 'Enable Browser' }))

    expect($browserTabs.get()).toHaveLength(1)
    expect($browserTabs.get()[0]).toMatchObject({ title: 'Browser', url: 'about:blank' })
    expect(screen.getByRole('button', { name: 'New Browser tab' })).toBeDefined()
  })
})
