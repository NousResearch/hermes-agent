import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, beforeEach, describe, expect, it, type MockInstance, vi } from 'vitest'

import { PRIMARY_SESSION_VIEW, type SessionView, SessionViewProvider } from '@/app/chat/session-view'
import { accentsFor, NOUS_ACCENT } from '@/components/onboarding-chat/options'
import type { HermesConnection } from '@/global'
import { I18nProvider } from '@/i18n'
import {
  answerSetupCard,
  type ClarifyRequest,
  clearClarifyRequest,
  setClarifyRequest,
  type SetupChooseSpec
} from '@/store/clarify'
import { $activeGatewayProfile } from '@/store/profile'
import { rememberServerRequest, resetServerRequestsForTests } from '@/store/server-requests'
import { $connection } from '@/store/session'
import { $accentOverride } from '@/themes/accent-override'
import { modePref, skinPref, ThemeProvider, useTheme } from '@/themes/context'
import { DEFAULT_SKIN_NAME } from '@/themes/presets'
import { $userThemes } from '@/themes/user-themes'

import { SetupChoosePending } from './setup-pending'

const SESSION = 'setup-session'

const CONNECTION: HermesConnection = {
  baseUrl: 'http://localhost:1',
  isFullscreen: false,
  nativeOverlayWidth: 0,
  windowButtonPosition: null,
  token: '',
  wsUrl: 'ws://localhost:1',
  logs: []
}

const view: SessionView = { ...PRIMARY_SESSION_VIEW, $runtimeId: atom(SESSION), $storedId: atom('stored-setup') }

let theme: ReturnType<typeof useTheme>

function ThemeProbe() {
  theme = useTheme()

  return null
}

const primary = () => window.document.documentElement.style.getPropertyValue('--theme-primary')

function renderCard(requestId: string, question: string, kind: SetupChooseSpec['kind'], preselected: string[] = []) {
  const request: ClarifyRequest = {
    questions: [{ choices: null, multiSelect: false, qid: 'q0', question }],
    requestId,
    sessionId: SESSION,
    setup: { kind, multiSelect: false, options: null, preselected }
  }

  rememberServerRequest({ fail: vi.fn(), id: requestId, method: 'setup_choose', params: {}, respond: vi.fn() })
  setClarifyRequest(request)

  return render(
    <QueryClientProvider client={new QueryClient()}>
      <I18nProvider configClient={null} initialLocale="en">
        <ThemeProvider>
          <ThemeProbe />
          <SessionViewProvider value={view}>
            <SetupChoosePending fromArgs={null} onAnswered={vi.fn()} request={request} undelivered={false} />
          </SessionViewProvider>
        </ThemeProvider>
      </I18nProvider>
    </QueryClientProvider>
  )
}

let consoleError: MockInstance<typeof console.error>

beforeEach(() => {
  window.localStorage.clear()
  $userThemes.set({})
  $activeGatewayProfile.set('default')
  $connection.set(CONNECTION)
  resetServerRequestsForTests()
  consoleError = vi.spyOn(console, 'error')
})

afterEach(() => {
  cleanup()
  clearClarifyRequest()
  resetServerRequestsForTests()
  $accentOverride.set(null)
  consoleError.mockRestore()
})

const loopErrors = () => consoleError.mock.calls.filter(([first]) => /Maximum update depth/.test(String(first)))

describe('setup cards render without redrawing themselves', () => {
  it('shows the name card and waits for an answer', async () => {
    renderCard('name-1', 'What should I call you?', 'question')

    expect(await screen.findByText('What should I call you?')).toBeTruthy()
    expect(loopErrors()).toEqual([])
  })

  it('keeps a confirmed accent through cold startup, profile switches, and later theme changes', async () => {
    skinPref.assign('other', 'ember')
    skinPref.assign('default', DEFAULT_SKIN_NAME)
    $activeGatewayProfile.set('setup')
    modePref.assign('setup', 'light')
    renderCard('accent-1', 'Pick an accent', 'accent')
    await screen.findByText('Pick an accent')

    const swatch = accentsFor(false).find(({ name }) => name === 'GitHub green')

    expect(swatch).toBeTruthy()
    act(() => expect(answerSetupCard(SESSION, swatch!.name)).toBe(true))
    const selected = theme.themeName
    const light = primary()

    expect(selected).not.toBe(DEFAULT_SKIN_NAME)
    expect(skinPref.resolve('setup')).toBe(selected)
    expect($accentOverride.get()).toBeNull()
    act(() => theme.setMode('dark'))
    const dark = primary()

    // Drop every module-level atom, preserving only localStorage as a new renderer does.
    cleanup()
    vi.resetModules()
    const fresh = await import('@/themes/context')
    expect(primary()).toBe(dark)
    const { $activeGatewayProfile: profile } = await import('@/store/profile')
    const { $connection: connection } = await import('@/store/session')
    let restored: ReturnType<typeof useTheme>

    function RestoredProbe() {
      restored = fresh.useTheme()

      return null
    }

    profile.set('setup')
    connection.set(CONNECTION)
    render(
      <fresh.ThemeProvider>
        <RestoredProbe />
      </fresh.ThemeProvider>
    )
    expect(restored!.themeName).toBe(selected)
    expect(primary()).toBe(dark)
    act(() => restored!.setMode('light'))
    expect(primary()).toBe(light)
    act(() => profile.set('other'))
    expect(restored!.themeName).toBe('ember')
    act(() => profile.set('setup'))
    expect(restored!.themeName).toBe(selected)
    expect(primary()).toBe(light)
    act(() => restored!.setTheme('midnight'))
    const replacement = primary()
    cleanup()
    render(
      <fresh.ThemeProvider>
        <RestoredProbe />
      </fresh.ThemeProvider>
    )
    expect(restored!.themeName).toBe('midnight')
    expect(primary()).toBe(replacement)
    expect(loopErrors()).toEqual([])
  })

  it('only saves the clicked accent when confirmed and restores skipped previews', async () => {
    modePref.assign('default', 'light')
    renderCard('accent-click', 'Pick an accent', 'accent')
    const original = primary()
    const swatch = accentsFor(false).find(({ hex }) => hex !== NOUS_ACCENT)!

    fireEvent.click(await screen.findByRole('button', { name: swatch.name }))
    expect(primary()).not.toBe(original)
    expect(skinPref.resolve('default')).toBe(DEFAULT_SKIN_NAME)
    expect(Object.keys($userThemes.get())).toHaveLength(0)
    act(() => clearClarifyRequest())
    expect(primary()).toBe(original)
    cleanup()

    renderCard('accent-confirm', 'Pick an accent', 'accent')
    fireEvent.click(await screen.findByRole('button', { name: swatch.name }))
    fireEvent.click(screen.getByRole('button', { name: /confirm/i }))
    expect(skinPref.resolve('default')).toBe(theme.themeName)
    expect(theme.themeName).not.toBe(DEFAULT_SKIN_NAME)
    expect(Object.keys($userThemes.get())).toHaveLength(1)
    expect($accentOverride.get()).toBeNull()
    cleanup()

    renderCard('accent-blue', 'Pick an accent', 'accent', [NOUS_ACCENT])
    fireEvent.click(await screen.findByRole('button', { name: /confirm/i }))
    expect(theme.themeName).toBe(DEFAULT_SKIN_NAME)
    expect(primary()).toBe(original)
    expect(Object.keys($userThemes.get())).toHaveLength(1)
  })
})
