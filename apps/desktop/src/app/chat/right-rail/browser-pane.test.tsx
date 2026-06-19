import { act, cleanup, fireEvent, render } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'
import { $browserNavRequest, $browserState, openBrowser } from '@/store/browser'

import { BrowserPane } from './browser-pane'

// jsdom doesn't implement <webview>; it renders as an unknown element, which is
// fine — we drive nav state through the store and assert the chrome reacts.
function resetStores() {
  $browserState.set({
    canGoBack: false,
    canGoForward: false,
    loading: false,
    loadError: null,
    open: true,
    title: '',
    url: 'https://example.com'
  })
  $browserNavRequest.set({ nonce: 0, url: '' })
}

function renderPane() {
  return render(
    <I18nProvider configClient={null}>
      <BrowserPane />
    </I18nProvider>
  )
}

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

describe('BrowserPane chrome', () => {
  it('shows the current URL in the address bar', () => {
    resetStores()
    const { getByLabelText } = renderPane()

    expect((getByLabelText('Address bar') as HTMLInputElement).value).toBe('https://example.com')
  })

  it('disables back/forward when history is empty', () => {
    resetStores()
    const { getByLabelText } = renderPane()

    expect((getByLabelText('Back') as HTMLButtonElement).disabled).toBe(true)
    expect((getByLabelText('Forward') as HTMLButtonElement).disabled).toBe(true)
  })

  it('enables back when the webview reports history', () => {
    resetStores()
    const { getByLabelText } = renderPane()

    act(() => $browserState.set({ ...$browserState.get(), canGoBack: true }))

    expect((getByLabelText('Back') as HTMLButtonElement).disabled).toBe(false)
  })

  it('submitting the address bar navigates (emits a nav request)', () => {
    resetStores()
    const { getByLabelText } = renderPane()
    const input = getByLabelText('Address bar') as HTMLInputElement

    fireEvent.focus(input)
    fireEvent.change(input, { target: { value: 'example.org/docs' } })
    const before = $browserNavRequest.get().nonce
    fireEvent.submit(input.closest('form')!)

    expect($browserNavRequest.get().nonce).toBeGreaterThan(before)
    expect($browserNavRequest.get().url).toBe('https://example.org/docs')
  })

  it('shows a Stop control while loading and Reload otherwise', () => {
    resetStores()
    const { getByLabelText, queryByLabelText } = renderPane()

    expect(queryByLabelText('Reload')).not.toBeNull()
    expect(queryByLabelText('Stop')).toBeNull()

    act(() => $browserState.set({ ...$browserState.get(), loading: true }))

    expect(queryByLabelText('Stop')).not.toBeNull()
    expect(getByLabelText('Stop')).toBeTruthy()
  })

  it('renders a load-error overlay with a retry affordance', () => {
    resetStores()
    const { getByText } = renderPane()

    act(() => $browserState.set({ ...$browserState.get(), loadError: 'ERR_CONNECTION_REFUSED' }))

    expect(getByText('This page failed to load')).toBeTruthy()
    expect(getByText('ERR_CONNECTION_REFUSED')).toBeTruthy()
    expect(getByText('Try again')).toBeTruthy()
  })

  it('mirrors store URL changes into the address bar when not editing', () => {
    resetStores()
    const { getByLabelText } = renderPane()

    act(() => openBrowser('https://changed.example'))

    expect((getByLabelText('Address bar') as HTMLInputElement).value).toBe('https://changed.example')
  })

  it('opens the current page in the OS browser via the toolbar button (passkey escape hatch)', () => {
    resetStores()
    const openExternal = vi.fn()
    ;(window as unknown as { hermesDesktop: { openExternal: typeof openExternal } }).hermesDesktop = {
      openExternal
    }

    const { getByLabelText } = renderPane()
    act(() => $browserState.set({ ...$browserState.get(), url: 'https://accounts.google.com/signin' }))

    fireEvent.click(getByLabelText('Open in default browser'))

    expect(openExternal).toHaveBeenCalledWith('https://accounts.google.com/signin')
  })

  it('disables the open-in-browser button when there is no http(s) page', () => {
    resetStores()
    const { getByLabelText } = renderPane()

    act(() => $browserState.set({ ...$browserState.get(), url: 'about:blank' }))

    expect((getByLabelText('Open in default browser') as HTMLButtonElement).disabled).toBe(true)
  })
})
