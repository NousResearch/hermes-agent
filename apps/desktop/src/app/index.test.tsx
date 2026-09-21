import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { getGlobalModelOptions, getHermesConfigRecord, saveHermesConfigRecord, setModelAssignment } from '@/hermes'
import { I18nProvider } from '@/i18n'
import { requestGatewayForAgent } from '@/store/gateway'
import { startManualOnboarding } from '@/store/onboarding'
import { $activeGatewayProfile } from '@/store/profile'
import { setConnection } from '@/store/session'

import { JARVIS_ONBOARDING_VERSION, jarvisOnboardingStorageKey } from './jarvis/onboarding-state'

import AppRoot, { appCompositionMode, jarvisViewForLocation } from './index'

const windowMode = vi.hoisted(() => ({
  auxiliary: false
}))

vi.mock('@/store/windows', async importActual => ({
  ...(await importActual<Record<string, unknown>>()),
  isAuxiliaryWindow: () => windowMode.auxiliary
}))

vi.mock('@/hermes', async importActual => ({
  ...(await importActual<Record<string, unknown>>()),
  getApiRequestConnection: () => null,
  getApiRequestProfile: () => null,
  getGlobalModelOptions: vi.fn(async () => ({
    model: 'llama-3',
    provider: 'fireworks',
    providers: [{ authenticated: true, models: ['llama-3'], name: 'Fireworks AI', slug: 'fireworks' }]
  })),
  getHermesConfig: vi.fn(async () => ({
    approvals: { mode: 'manual' },
    voice: { auto_tts: false }
  })),
  getHermesConfigRecord: vi.fn(async () => ({
    approvals: { mode: 'manual' },
    voice: { auto_tts: false }
  })),
  saveHermesConfig: vi.fn(async () => ({ ok: true })),
  saveHermesConfigRecord: vi.fn(async () => ({ ok: true })),
  setModelAssignment: vi.fn(async () => ({ model: 'llama-3', ok: true, provider: 'fireworks' }))
}))

vi.mock('@/store/gateway', async importActual => ({
  ...(await importActual<Record<string, unknown>>()),
  requestGatewayForAgent: vi.fn(async () => ({}))
}))

vi.mock('@/store/onboarding', () => ({
  startManualOnboarding: vi.fn()
}))

vi.mock('./contrib', async () => {
  const router = await vi.importActual('react-router')
  const useLocation = (router as { useLocation: () => { pathname: string; search: string } }).useLocation

  return {
    ContribController: ({ layoutMode }: { layoutMode?: string }) => {
      const location = useLocation()

      return (
        <main
          data-layout-mode={layoutMode ?? 'viewport'}
          data-path={`${location.pathname}${location.search}`}
          data-testid="contrib-runtime"
        />
      )
    }
  }
})

function renderRoot(initialEntry = '/') {
  window.localStorage.setItem(
    jarvisOnboardingStorageKey(),
    JSON.stringify({
      version: JARVIS_ONBOARDING_VERSION,
      currentStep: 'approvals',
      completedSteps: ['profile', 'engine', 'model', 'voice', 'access', 'approvals']
    })
  )

  return render(
    <MemoryRouter initialEntries={[initialEntry]}>
      <I18nProvider configClient={null} initialLocale="pl">
        <AppRoot />
      </I18nProvider>
    </MemoryRouter>
  )
}

function deferred<T>() {
  let resolve!: (value: T) => void
  let reject!: (reason?: unknown) => void

  const promise = new Promise<T>((res, rej) => {
    resolve = res
    reject = rej
  })

  return { promise, reject, resolve }
}

function persistReadyApprovalsState(scope?: { connectionId: string; profile: string }) {
  window.localStorage.setItem(
    jarvisOnboardingStorageKey(scope),
    JSON.stringify({
      version: JARVIS_ONBOARDING_VERSION,
      currentStep: 'approvals',
      completedSteps: ['profile', 'engine', 'model', 'voice', 'access'],
      selections: {
        approvalsMode: 'balanced',
        engine: 'fireworks',
        model: 'llama-3',
        validatedAccessModel: 'llama-3',
        validatedAccessProvider: 'fireworks',
        validatedModel: 'llama-3',
        validatedProvider: 'fireworks'
      }
    })
  )
}

afterEach(() => {
  cleanup()
  window.localStorage.clear()
  windowMode.auxiliary = false
  setConnection(null)
  $activeGatewayProfile.set('default')
  vi.mocked(getGlobalModelOptions).mockReset()
  vi.mocked(getGlobalModelOptions).mockResolvedValue({
    model: 'llama-3',
    provider: 'fireworks',
    providers: [{ authenticated: true, models: ['llama-3'], name: 'Fireworks AI', slug: 'fireworks' }]
  })
  vi.mocked(getHermesConfigRecord).mockReset()
  vi.mocked(getHermesConfigRecord).mockResolvedValue({
    approvals: { mode: 'manual' },
    voice: { auto_tts: false }
  })
  vi.mocked(saveHermesConfigRecord).mockReset()
  vi.mocked(saveHermesConfigRecord).mockResolvedValue({ ok: true })
  vi.mocked(setModelAssignment).mockReset()
  vi.mocked(setModelAssignment).mockResolvedValue({ model: 'llama-3', ok: true, provider: 'fireworks' })
  vi.mocked(requestGatewayForAgent).mockReset()
  vi.mocked(requestGatewayForAgent).mockResolvedValue({})
  vi.mocked(startManualOnboarding).mockReset()
})

describe('desktop app root Jarvis integration', () => {
  it('reaches the Jarvis shell through the default app root while keeping the contrib runtime mounted', () => {
    renderRoot('/')

    expect(screen.getByRole('navigation', { name: 'Główna nawigacja' })).toBeTruthy()
    expect(screen.getAllByRole('navigation', { name: 'Główna nawigacja' })).toHaveLength(1)
    expect(screen.getAllByRole('main')).toHaveLength(1)
    expect(screen.getByTestId('contrib-runtime').getAttribute('data-path')).toBe('/')
    expect(screen.getByTestId('contrib-runtime').getAttribute('data-layout-mode')).toBe('embedded')
  })

  it.each([
    ['Zadania', '/cron'],
    ['Pamięć', '/settings?tab=config:memory'],
    ['Narzędzia', '/skills?tab=toolsets'],
    ['Ustawienia', '/settings'],
    ['Profil', '/profiles'],
    ['Jarvis', '/']
  ])('delegates %s to an existing production route', (label, route) => {
    renderRoot('/settings')

    fireEvent.click(screen.getByRole('button', { name: label }))

    expect(screen.getByTestId('contrib-runtime').getAttribute('data-path')).toBe(route)
  })

  it('derives Jarvis navigation state from existing runtime routes', () => {
    expect(jarvisViewForLocation('/cron', '')).toBe('tasks')
    expect(jarvisViewForLocation('/settings', '?tab=config:memory')).toBe('memory')
    expect(jarvisViewForLocation('/skills', '?tab=toolsets')).toBe('tools')
    expect(jarvisViewForLocation('/settings', '')).toBe('settings')
    expect(jarvisViewForLocation('/profiles', '')).toBe('profile')
    expect(jarvisViewForLocation('/some-session', '')).toBe('jarvis')
  })

  it('keeps HUD windows on the existing chrome-free contrib root', () => {
    windowMode.auxiliary = true

    renderRoot('/')

    expect(screen.queryByRole('navigation', { name: 'Główna nawigacja' })).toBeNull()
    expect(screen.queryByTestId('jarvis-dashboard')).toBeNull()
    expect(screen.getByTestId('contrib-runtime').getAttribute('data-layout-mode')).toBe('viewport')
  })

  it('keeps Browser Popout windows on the existing popout root', () => {
    windowMode.auxiliary = true

    renderRoot('/')

    expect(screen.queryByRole('navigation', { name: 'Główna nawigacja' })).toBeNull()
    expect(screen.getByTestId('contrib-runtime').getAttribute('data-layout-mode')).toBe('viewport')
  })

  it('keeps secondary single-chat windows on the existing chrome-free controller path', () => {
    windowMode.auxiliary = true
    window.localStorage.clear()

    render(
      <MemoryRouter initialEntries={['/session-123']}>
        <I18nProvider configClient={null} initialLocale="pl">
          <AppRoot />
        </I18nProvider>
      </MemoryRouter>
    )

    expect(screen.queryByRole('navigation', { name: 'Główna nawigacja' })).toBeNull()
    expect(screen.queryByTestId('jarvis-onboarding')).toBeNull()
    expect(screen.getByTestId('contrib-runtime').getAttribute('data-path')).toBe('/session-123')
    expect(screen.getByTestId('contrib-runtime').getAttribute('data-layout-mode')).toBe('viewport')
  })

  it('keeps watch session windows on the same secondary-window controller path', () => {
    windowMode.auxiliary = true

    renderRoot('/session-123')

    expect(screen.queryByRole('navigation', { name: 'Główna nawigacja' })).toBeNull()
    expect(screen.getByTestId('contrib-runtime').getAttribute('data-path')).toBe('/session-123')
    expect(screen.getByTestId('contrib-runtime').getAttribute('data-layout-mode')).toBe('viewport')
  })

  it('classifies product shell composition without coupling tests to the host window', () => {
    expect(appCompositionMode({ auxiliary: false })).toBe('product-shell')
    expect(appCompositionMode({ auxiliary: true })).toBe('special-window')
  })

  it('mounts the Jarvis onboarding gate for incomplete primary windows', async () => {
    window.localStorage.clear()

    render(
      <MemoryRouter initialEntries={['/']}>
        <I18nProvider configClient={null} initialLocale="pl">
          <AppRoot />
        </I18nProvider>
      </MemoryRouter>
    )

    expect(await screen.findByTestId('jarvis-onboarding')).toBeTruthy()
  })

  it('remounts onboarding by active connection and profile and ignores stale pending loads', async () => {
    window.localStorage.clear()
    let resolveDefaultConfig!: (value: { approvals: { mode: string }; voice: { auto_tts: boolean } }) => void

    vi.mocked(getHermesConfigRecord).mockImplementation(scope => {
      if (scope && typeof scope === 'object' && scope.profile === 'research') {
        return Promise.resolve({ approvals: { mode: 'manual' }, voice: { auto_tts: false } })
      }

      return new Promise(resolve => {
        resolveDefaultConfig = resolve
      })
    })
    vi.mocked(getGlobalModelOptions).mockImplementation(async (_opts, scope) =>
      scope && typeof scope === 'object' && scope.profile === 'research'
        ? {
            model: 'research-model',
            provider: 'research-provider',
            providers: [
              { authenticated: true, models: ['research-model'], name: 'Research Provider', slug: 'research-provider' }
            ]
          }
        : {
            model: 'default-model',
            provider: 'default-provider',
            providers: [{ authenticated: true, models: ['default-model'], name: 'Default Provider', slug: 'default-provider' }]
          }
    )

    render(
      <MemoryRouter initialEntries={['/']}>
        <I18nProvider configClient={null} initialLocale="pl">
          <AppRoot />
        </I18nProvider>
      </MemoryRouter>
    )

    expect(await screen.findByTestId('jarvis-onboarding')).toBeTruthy()

    act(() => {
      setConnection({ connectionId: 'remote-a', mode: 'remote', profile: 'research', registryScoped: true } as never)
      $activeGatewayProfile.set('research')
    })

    await waitFor(() =>
      expect(
        JSON.stringify(
          window.localStorage.getItem(jarvisOnboardingStorageKey({ connectionId: 'remote-a', profile: 'research' }))
        )
      ).toContain('research-provider')
    )

    resolveDefaultConfig({ approvals: { mode: 'manual' }, voice: { auto_tts: false } })

    await new Promise(resolve => setTimeout(resolve, 0))
    expect(
      window.localStorage.getItem(jarvisOnboardingStorageKey({ connectionId: 'remote-a', profile: 'research' }))
    ).toContain('research-provider')
    expect(
      window.localStorage.getItem(jarvisOnboardingStorageKey({ connectionId: 'remote-a', profile: 'research' }))
    ).not.toContain('default-provider')
  })

  it('unveils the working shell after completing the real onboarding component without reload', async () => {
    window.localStorage.setItem(
      jarvisOnboardingStorageKey(),
      JSON.stringify({
        version: JARVIS_ONBOARDING_VERSION,
        currentStep: 'approvals',
        completedSteps: ['profile', 'engine', 'model', 'voice', 'access'],
        selections: {
          approvalsMode: 'balanced',
          engine: 'fireworks',
          model: 'llama-3',
          validatedAccessModel: 'llama-3',
          validatedAccessProvider: 'fireworks',
          validatedModel: 'llama-3',
          validatedProvider: 'fireworks'
        }
      })
    )

    render(
      <MemoryRouter initialEntries={['/']}>
        <I18nProvider configClient={null} initialLocale="pl">
          <AppRoot />
        </I18nProvider>
      </MemoryRouter>
    )

    expect(await screen.findByTestId('jarvis-onboarding')).toBeTruthy()
    expect(screen.getByTestId('contrib-runtime').getAttribute('data-layout-mode')).toBe('embedded')

    fireEvent.click(screen.getByRole('button', { name: 'Zakończ' }))

    await waitFor(() => expect(screen.queryByTestId('jarvis-onboarding')).toBeNull())
    expect(screen.getByRole('navigation', { name: 'Główna nawigacja' })).toBeTruthy()
    expect(screen.getByTestId('contrib-runtime').getAttribute('data-layout-mode')).toBe('embedded')
  })

  it('rolls back only the old scope when onboarding is remounted after model assignment', async () => {
    window.localStorage.clear()
    persistReadyApprovalsState({ connectionId: 'local', profile: 'default' })
    const modelWrite = deferred<{ model: string; ok: boolean; provider: string }>()

    vi.mocked(setModelAssignment)
      .mockReturnValueOnce(modelWrite.promise)
      .mockResolvedValueOnce({ model: 'llama-3', ok: true, provider: 'fireworks' })

    render(
      <MemoryRouter initialEntries={['/']}>
        <I18nProvider configClient={null} initialLocale="pl">
          <AppRoot />
        </I18nProvider>
      </MemoryRouter>
    )

    fireEvent.click(await screen.findByRole('button', { name: 'Zakończ' }))
    await waitFor(() => expect(setModelAssignment).toHaveBeenCalledTimes(1))

    act(() => {
      setConnection({ connectionId: 'remote-b', mode: 'remote', profile: 'research', registryScoped: true } as never)
      $activeGatewayProfile.set('research')
    })

    await act(async () => {
      modelWrite.resolve({ model: 'llama-3', ok: true, provider: 'fireworks' })
      await modelWrite.promise
    })

    await waitFor(() => expect(setModelAssignment).toHaveBeenCalledTimes(2))
    expect(setModelAssignment).toHaveBeenNthCalledWith(
      1,
      { model: 'llama-3', provider: 'fireworks', scope: 'main' },
      { connectionId: 'local', profile: 'default' }
    )
    expect(setModelAssignment).toHaveBeenNthCalledWith(
      2,
      { model: 'llama-3', provider: 'fireworks', scope: 'main' },
      { connectionId: 'local', profile: 'default' }
    )
    expect(saveHermesConfigRecord).not.toHaveBeenCalled()
    expect(requestGatewayForAgent).not.toHaveBeenCalled()
    expect(
      JSON.parse(window.localStorage.getItem(jarvisOnboardingStorageKey({ connectionId: 'local', profile: 'default' })) ?? '{}')
        .completedSteps
    ).not.toContain('approvals')
  })

  it('rolls back config and model on the old scope when onboarding is remounted after config save', async () => {
    window.localStorage.clear()
    persistReadyApprovalsState({ connectionId: 'local', profile: 'default' })
    const configWrite = deferred<{ ok: boolean }>()

    vi.mocked(saveHermesConfigRecord)
      .mockReturnValueOnce(configWrite.promise)
      .mockResolvedValueOnce({ ok: true })
    vi.mocked(setModelAssignment).mockResolvedValue({ model: 'llama-3', ok: true, provider: 'fireworks' })

    render(
      <MemoryRouter initialEntries={['/']}>
        <I18nProvider configClient={null} initialLocale="pl">
          <AppRoot />
        </I18nProvider>
      </MemoryRouter>
    )

    fireEvent.click(await screen.findByRole('button', { name: 'Zakończ' }))
    await waitFor(() => expect(saveHermesConfigRecord).toHaveBeenCalledTimes(1))

    act(() => {
      setConnection({ connectionId: 'remote-b', mode: 'remote', profile: 'research', registryScoped: true } as never)
      $activeGatewayProfile.set('research')
    })

    await act(async () => {
      configWrite.resolve({ ok: true })
      await configWrite.promise
    })

    await waitFor(() => expect(saveHermesConfigRecord).toHaveBeenCalledTimes(2))
    await waitFor(() => expect(setModelAssignment).toHaveBeenCalledTimes(2))
    expect(saveHermesConfigRecord).toHaveBeenNthCalledWith(
      1,
      { approvals: { mode: 'smart' }, voice: { auto_tts: false } },
      { connectionId: 'local', profile: 'default' }
    )
    expect(saveHermesConfigRecord).toHaveBeenNthCalledWith(
      2,
      { approvals: { mode: 'manual' }, voice: { auto_tts: false } },
      { connectionId: 'local', profile: 'default' }
    )
    expect(setModelAssignment).toHaveBeenNthCalledWith(
      2,
      { model: 'llama-3', provider: 'fireworks', scope: 'main' },
      { connectionId: 'local', profile: 'default' }
    )
    expect(requestGatewayForAgent).not.toHaveBeenCalled()
    expect(
      JSON.parse(window.localStorage.getItem(jarvisOnboardingStorageKey({ connectionId: 'local', profile: 'default' })) ?? '{}')
        .completedSteps
    ).not.toContain('approvals')
  })

  it('passes the initiating remote scope when Jarvis opens secure provider setup', async () => {
    window.localStorage.clear()
    setConnection({ connectionId: 'remote-a', mode: 'remote', profile: 'research', registryScoped: true } as never)
    $activeGatewayProfile.set('research')
    window.localStorage.setItem(
      jarvisOnboardingStorageKey({ connectionId: 'remote-a', profile: 'research' }),
      JSON.stringify({
        version: JARVIS_ONBOARDING_VERSION,
        currentStep: 'access',
        completedSteps: ['profile', 'engine', 'model', 'voice'],
        selections: { engine: 'fireworks', model: 'llama-3' }
      })
    )

    render(
      <MemoryRouter initialEntries={['/']}>
        <I18nProvider configClient={null} initialLocale="pl">
          <AppRoot />
        </I18nProvider>
      </MemoryRouter>
    )

    fireEvent.click(await screen.findByRole('button', { name: 'Otwórz bezpieczną konfigurację dostawcy' }))

    expect(startManualOnboarding).toHaveBeenCalledWith(null, { connectionId: 'remote-a', profile: 'research' })
  })
})
