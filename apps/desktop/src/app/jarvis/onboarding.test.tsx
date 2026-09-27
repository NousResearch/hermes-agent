import { cleanup, configure, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { StrictMode } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type * as HermesApi from '@/hermes'
import { I18nProvider } from '@/i18n'
import { pl } from '@/i18n/pl'
import { startManualOnboarding } from '@/store/onboarding'

import { JarvisOnboarding } from './onboarding'
import {
  dismissJarvisOnboarding,
  JARVIS_ONBOARDING_STATE_KEY,
  JARVIS_ONBOARDING_STEPS,
  JARVIS_ONBOARDING_VERSION,
  jarvisOnboardingComplete,
  jarvisOnboardingStorageKey,
  readJarvisOnboardingState,
  sanitizeJarvisOnboardingState,
  serializeJarvisOnboardingState,
  shouldShowJarvisOnboarding,
  writeJarvisOnboardingState
} from './onboarding-state'

// The wizard resolves providers over the gateway, so a busy CI runner needs more
// than the library defaults (1s async / 5s per test) — otherwise these tests flake
// on timing rather than on behaviour.
vi.setConfig({ testTimeout: 20_000, hookTimeout: 20_000 })
configure({ asyncUtilTimeout: 5_000 })

vi.mock('@/store/onboarding', () => ({
  startManualOnboarding: vi.fn()
}))

const savedEnv = vi.hoisted(() => new Map<string, string>())

vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<typeof HermesApi>()),
  setEnvVar: vi.fn(async (key: string, value: string) => {
    savedEnv.set(key, value)

    return { ok: true }
  }),
  validateProviderCredential: vi.fn(async () => ({ message: '', ok: true, reachable: true }))
}))

const TEST_SCOPE = { connectionId: 'local', profile: 'default' }

/** The computer step probes a real backend; tests that are not about the probe
 *  keep it quiet with a platform that has no desktop driver. */
const OFFLINE_COMPUTER_STATUS = {
  loadStatus: async () => ({
    accessibility: null,
    can_grant: false,
    checks: [],
    error: null,
    installed: false,
    platform: 'linux',
    platform_supported: false,
    ready: null,
    screen_recording: null,
    screen_recording_capturable: null,
    source: null,
    version: null
  })
}

function onboardingStorageKey() {
  return jarvisOnboardingStorageKey(TEST_SCOPE)
}

function readStoredOnboardingState() {
  return readJarvisOnboardingState(window.localStorage, TEST_SCOPE)
}

function renderOnboarding(props: Partial<React.ComponentProps<typeof JarvisOnboarding>> = {}) {
  return render(
    <I18nProvider configClient={null} initialLocale="pl">
      <JarvisOnboarding
        initialStep="welcome"
        loadConfig={async () => ({
          approvals: { mode: 'manual' },
          stt: { enabled: false },
          voice: { auto_tts: false }
        })}
        loadModelOptions={async () => ({
          model: 'llama-3',
          provider: 'fireworks',
          providers: [
            {
              authenticated: true,
              models: ['llama-3', 'qwen-3'],
              name: 'Fireworks AI',
              slug: 'fireworks'
            }
          ]
        })}
        onComplete={vi.fn()}
        requestGateway={async () => undefined}
        saveConfig={async () => ({ ok: true })}
        saveModel={async () => ({ model: 'llama-3', ok: true, provider: 'fireworks' })}
        scope={TEST_SCOPE}
        {...props}
      />
    </I18nProvider>
  )
}

function persistReadyApprovalsState() {
  window.localStorage.setItem(
    onboardingStorageKey(),
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

it('keeps setup usable after StrictMode remount and loads providers for the required step', async () => {
  render(
    <StrictMode>
      <I18nProvider configClient={null} initialLocale="pl">
        <JarvisOnboarding
          initialStep="welcome"
          loadConfig={async () => ({})}
          loadModelOptions={async () => ({
            provider: 'openrouter',
            model: 'demo',
            providers: [{ slug: 'openrouter', name: 'OpenRouter', authenticated: true, models: ['demo'] }]
          })}
          onComplete={vi.fn()}
          scope={TEST_SCOPE}
        />
      </I18nProvider>
    </StrictMode>
  )
  await waitFor(() =>
    expect((screen.getByRole('button', { name: pl.jarvisOnboarding.actions.next }) as HTMLButtonElement).disabled).toBe(
      false
    )
  )
  fireEvent.click(screen.getByRole('button', { name: /Silnik|Własne API/ }))
  await waitFor(() => expect(screen.getByRole('button', { name: /OpenRouter.*1/ })).toBeTruthy())
})

it('opens fresh onboarding after the user activates the logo', async () => {
  renderOnboarding({ initialStep: undefined })
  expect(screen.getByTestId('czesiek-onboarding-splash')).toBeTruthy()
  expect(screen.queryByTestId('jarvis-onboarding')).toBeNull()
  fireEvent.click(screen.getByRole('button', { name: 'Kliknij logo, aby rozpocząć' }))
  await waitFor(() => expect(screen.getByTestId('jarvis-onboarding')).toBeTruthy())
})

function persistReadyComputerState() {
  window.localStorage.setItem(
    onboardingStorageKey(),
    JSON.stringify({
      version: JARVIS_ONBOARDING_VERSION,
      currentStep: 'computer',
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

function deferred<T>() {
  let resolve!: (value: T) => void
  let reject!: (reason?: unknown) => void

  const promise = new Promise<T>((res, rej) => {
    resolve = res
    reject = rej
  })

  return { promise, reject, resolve }
}

beforeEach(() => {
  window.localStorage.clear()
  vi.mocked(startManualOnboarding).mockReset()
})

afterEach(() => {
  cleanup()
  window.localStorage.clear()
})

describe('Agent CzesiekOnboarding state', () => {
  it('resumes versioned progress from scoped localStorage without serializing unknown fields', () => {
    const serialized = serializeJarvisOnboardingState({
      version: JARVIS_ONBOARDING_VERSION,
      currentStep: 'voice',
      completedSteps: ['profile', 'engine', 'model'],
      selections: {
        approvalsMode: 'balanced',
        engine: 'fireworks',
        model: 'llama-3',
        profile: 'default',
        voiceMode: 'spoken',
        apiKey: 'sk-should-not-persist'
      } as never
    })

    expect(serialized).not.toContain('sk-should-not-persist')
    expect(serialized).toContain('llama-3')

    window.localStorage.setItem(onboardingStorageKey(), serialized)

    expect(readStoredOnboardingState()?.currentStep).toBe('voice')
  })

  it('rejects mismatched or corrupt persisted state', () => {
    window.localStorage.setItem(
      onboardingStorageKey(),
      JSON.stringify({ version: JARVIS_ONBOARDING_VERSION + 1, currentStep: 'approvals', completedSteps: [] })
    )

    expect(readStoredOnboardingState()).toBeNull()

    window.localStorage.setItem(onboardingStorageKey(), '{')

    expect(readStoredOnboardingState()).toBeNull()
  })

  it('keeps serialized state to progress keys and allow-listed selections', () => {
    const clean = sanitizeJarvisOnboardingState({
      version: JARVIS_ONBOARDING_VERSION,
      currentStep: 'model',
      completedSteps: ['profile'],
      selections: { credential: 'drop', engine: 'fireworks', password: 'drop' } as never
    })

    expect(Object.keys(clean).sort()).toEqual(['completedSteps', 'currentStep', 'selections', 'version'])
    expect(clean.selections).toEqual({ engine: 'fireworks' })
  })

  it('scopes completion by connection and profile', () => {
    const serialized = serializeJarvisOnboardingState({
      version: JARVIS_ONBOARDING_VERSION,
      currentStep: 'approvals',
      completedSteps: ['profile', 'engine', 'model', 'voice', 'access', 'approvals'],
      selections: { engine: 'fireworks', model: 'llama-3' }
    })

    window.localStorage.setItem(jarvisOnboardingStorageKey({ connectionId: 'local', profile: 'default' }), serialized)

    expect(readJarvisOnboardingState(window.localStorage, { connectionId: 'local', profile: 'default' })).not.toBeNull()
    expect(readJarvisOnboardingState(window.localStorage, { connectionId: 'local', profile: 'research' })).toBeNull()
  })

  it('closes the wizard without claiming completion and stays closed on reload', () => {
    expect(dismissJarvisOnboarding(window.localStorage, TEST_SCOPE)).toBe(true)

    const stored = readStoredOnboardingState()

    expect(stored?.skipped).toBe(true)
    expect(stored?.completedSteps).toEqual([])
    // A skip is not a completion: nothing was chosen, so nothing is claimed.
    expect(jarvisOnboardingComplete(stored)).toBe(false)
    expect(shouldShowJarvisOnboarding(stored)).toBe(false)
  })

  it('scopes a closed wizard to its connection and profile', () => {
    dismissJarvisOnboarding(window.localStorage, TEST_SCOPE)

    expect(shouldShowJarvisOnboarding(readJarvisOnboardingState(window.localStorage, TEST_SCOPE))).toBe(false)
    expect(
      shouldShowJarvisOnboarding(
        readJarvisOnboardingState(window.localStorage, { connectionId: 'local', profile: 'somebody-else' })
      )
    ).toBe(true)
  })

  it('reports a refused dismiss instead of pretending the wizard is gone', () => {
    const refusing = {
      getItem: () => null,
      setItem: () => {
        throw new Error('storage refused the write')
      }
    } as unknown as Storage

    expect(dismissJarvisOnboarding(refusing, TEST_SCOPE)).toBe(false)
  })

  it('reports write failure when storage is unavailable', () => {
    const state = {
      version: JARVIS_ONBOARDING_VERSION,
      currentStep: 'profile' as const,
      completedSteps: [],
      selections: { engine: 'fireworks', rogue: 'drop' } as never
    }

    expect(
      readJarvisOnboardingState(undefined, {
        connectionId: 'local',
        profile: 'default'
      })
    ).toBeNull()
    expect(writeJarvisOnboardingState(state, undefined)).toBe(false)
    expect(serializeJarvisOnboardingState(state)).not.toContain('rogue')
  })
})

describe('Agent CzesiekOnboarding', () => {
  it('blocks Next with the real provider configuration message, then unlocks after retry success', async () => {
    const providerConfigurationCheck = vi
      .fn()
      .mockResolvedValueOnce({ ok: false, message: 'Nieprawidłowy klucz' })
      .mockResolvedValueOnce({ ok: true })

    renderOnboarding({ initialStep: 'model', providerConfigurationCheck })

    await screen.findByRole('heading', { name: 'Model' })
    await screen.findByDisplayValue('llama-3')
    fireEvent.click(screen.getByRole('button', { name: 'Sprawdź konfigurację' }))

    expect(await screen.findByText('Nieprawidłowy klucz')).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Dalej' })).toHaveProperty('disabled', true)

    fireEvent.click(screen.getByRole('button', { name: 'Sprawdź konfigurację' }))

    await waitFor(() => expect(screen.getByRole('button', { name: 'Dalej' })).toHaveProperty('disabled', false))
  })

  it('resumes on the persisted step after reload', async () => {
    window.localStorage.setItem(
      onboardingStorageKey(),
      JSON.stringify({
        version: JARVIS_ONBOARDING_VERSION,
        currentStep: 'access',
        completedSteps: ['profile', 'engine', 'model', 'voice'],
        selections: { engine: 'fireworks', model: 'llama-3' }
      })
    )

    renderOnboarding()

    expect(await screen.findByRole('heading', { name: 'Dostępy' })).toBeTruthy()
  })

  it('completes only after config save succeeds and keeps retry available on save failure', async () => {
    const onComplete = vi.fn()
    const saveConfig = vi.fn().mockRejectedValueOnce(new Error('Disk full')).mockResolvedValueOnce({ ok: true })
    persistReadyApprovalsState()

    renderOnboarding({
      initialStep: 'approvals',
      onComplete,
      saveConfig,
      providerConfigurationCheck: async () => ({ ok: true })
    })

    fireEvent.click(await screen.findByRole('radio', { name: 'Zrównoważony' }))
    fireEvent.click(screen.getByRole('button', { name: 'Zakończ' }))

    await waitFor(() => expect(saveConfig).toHaveBeenCalledTimes(1))
    expect(await screen.findByText('Disk full')).toBeTruthy()
    expect(onComplete).not.toHaveBeenCalled()
    expect(readStoredOnboardingState()?.completedSteps).not.toContain('approvals')

    fireEvent.click(screen.getByRole('button', { name: 'Zakończ' }))

    await waitFor(() => expect(saveConfig).toHaveBeenCalledTimes(2))
    await waitFor(() => expect(readStoredOnboardingState()?.completedSteps).toContain('approvals'))
  })

  it('does not complete when the final state cannot be serialized and retry succeeds', async () => {
    const onComplete = vi.fn()
    const originalSetItem = window.localStorage.setItem.bind(window.localStorage)
    const setItem = vi.spyOn(window.localStorage, 'setItem')
    let failFinalWrite = true

    setItem.mockImplementation(function setItemWithFinalWriteFailure(this: Storage, key: string, value: string) {
      if (
        failFinalWrite &&
        key.startsWith(`${JARVIS_ONBOARDING_STATE_KEY}:`) &&
        value.includes(`"completedSteps":${JSON.stringify(JARVIS_ONBOARDING_STEPS)}`)
      ) {
        failFinalWrite = false
        throw new Error('Storage unavailable')
      }

      return Reflect.apply(originalSetItem, this, [key, value])
    })

    try {
      persistReadyApprovalsState()
      renderOnboarding({
        initialStep: 'approvals',
        onComplete,
        providerConfigurationCheck: async () => ({ ok: true })
      })

      fireEvent.click(await screen.findByRole('button', { name: 'Zakończ' }))

      expect(await screen.findByText('Nie udało się zapisać onboardingu.')).toBeTruthy()
      expect(onComplete).not.toHaveBeenCalled()

      fireEvent.click(screen.getByRole('button', { name: 'Zakończ' }))

      await waitFor(() => expect(onComplete).toHaveBeenCalledOnce())
      await waitFor(() => expect(readStoredOnboardingState()?.completedSteps).toContain('approvals'))
    } finally {
      setItem.mockRestore()
    }
  })

  it('does not assign the main model during the default configuration check', async () => {
    const saveModel = vi.fn().mockResolvedValue({ ok: false })

    renderOnboarding({
      initialStep: 'model',
      saveModel
    })

    await screen.findByRole('heading', { name: 'Model' })
    await screen.findByDisplayValue('llama-3')
    fireEvent.click(screen.getByRole('button', { name: 'Sprawdź konfigurację' }))

    await screen.findByText('Konfiguracja sprawdzona')
    expect(saveModel).not.toHaveBeenCalled()
    expect(screen.getByRole('button', { name: 'Dalej' })).toHaveProperty('disabled', false)
  })

  it('blocks configuration check when the refreshed provider is unauthenticated', async () => {
    renderOnboarding({
      initialStep: 'model',
      loadModelOptions: async () => ({
        model: 'llama-3',
        provider: 'fireworks',
        providers: [
          {
            authenticated: false,
            models: ['llama-3'],
            name: 'Fireworks AI',
            slug: 'fireworks'
          }
        ]
      })
    })

    await screen.findByRole('heading', { name: 'Model' })
    await screen.findByDisplayValue('llama-3')
    fireEvent.click(screen.getByRole('button', { name: 'Sprawdź konfigurację' }))

    expect(await screen.findByText(/Ten dostawca nie jest jeszcze gotowy/)).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Dalej' })).toHaveProperty('disabled', true)
  })

  it('allows skipping access while secure setup and refresh still validate credentials', async () => {
    const loadModelOptions = vi
      .fn()
      .mockResolvedValueOnce({
        model: 'llama-3',
        provider: 'fireworks',
        providers: [{ authenticated: false, models: ['llama-3'], name: 'Fireworks AI', slug: 'fireworks' }]
      })
      .mockResolvedValueOnce({
        model: 'llama-3',
        provider: 'fireworks',
        providers: [{ authenticated: true, models: ['llama-3'], name: 'Fireworks AI', slug: 'fireworks' }]
      })

    renderOnboarding({ initialStep: 'access', loadModelOptions })

    await screen.findByRole('heading', { name: 'Dostępy' })
    await waitFor(() => expect(readStoredOnboardingState()?.selections?.engine).toBe('fireworks'))
    expect(screen.getByRole('button', { name: 'Dalej' })).toHaveProperty('disabled', false)

    fireEvent.click(screen.getByRole('button', { name: 'Otwórz bezpieczną konfigurację dostawcy' }))
    expect(startManualOnboarding).toHaveBeenCalledWith(null, TEST_SCOPE)
    expect(await screen.findByText(/Bezpieczna konfiguracja została otwarta/)).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Dalej' })).toHaveProperty('disabled', false)

    fireEvent.click(screen.getByRole('button', { name: 'Odśwież i sprawdź dostęp' }))

    expect(await screen.findByText('Dostęp dostawcy jest skonfigurowany')).toBeTruthy()
    await waitFor(() => expect(screen.getByRole('button', { name: 'Dalej' })).toHaveProperty('disabled', false))
  })

  it('invalidates persisted model, access, and final completion when provider changes', async () => {
    window.localStorage.setItem(
      onboardingStorageKey(),
      JSON.stringify({
        version: JARVIS_ONBOARDING_VERSION,
        currentStep: 'engine',
        completedSteps: ['profile', 'engine', 'model', 'access', 'approvals'],
        selections: {
          engine: 'fireworks',
          model: 'llama-3',
          validatedAccessModel: 'llama-3',
          validatedAccessProvider: 'fireworks',
          validatedModel: 'llama-3',
          validatedProvider: 'fireworks'
        }
      })
    )

    renderOnboarding({
      loadModelOptions: async () => ({
        model: 'llama-3',
        provider: 'fireworks',
        providers: [
          { authenticated: true, models: ['llama-3'], name: 'Fireworks AI', slug: 'fireworks' },
          { authenticated: true, models: ['qwen-3'], name: 'OpenRouter', slug: 'openrouter' }
        ]
      })
    })

    fireEvent.click(await screen.findByRole('button', { name: /OpenRouter/ }))

    await waitFor(() => expect(readStoredOnboardingState()?.selections?.engine).toBe('openrouter'))
    const stored = readStoredOnboardingState()

    expect(stored?.completedSteps).not.toContain('model')
    expect(stored?.completedSteps).not.toContain('access')
    expect(stored?.completedSteps).not.toContain('approvals')
    expect(stored?.selections?.validatedProvider).toBeUndefined()
    expect(stored?.selections?.validatedAccessProvider).toBeUndefined()
  })

  it('ignores a slow validation success after the selected model changes', async () => {
    const pending = deferred<{ ok: true } | { message: string; ok: false }>()

    const providerConfigurationCheck = vi.fn(({ model }) =>
      model === 'llama-3' ? pending.promise : Promise.resolve({ ok: true as const })
    )

    renderOnboarding({
      initialStep: 'model',
      providerConfigurationCheck,
      loadModelOptions: async () => ({
        model: 'llama-3',
        provider: 'fireworks',
        providers: [{ authenticated: true, models: ['llama-3', 'qwen-3'], name: 'Fireworks AI', slug: 'fireworks' }]
      })
    })

    await screen.findByRole('heading', { name: 'Model' })
    await screen.findByDisplayValue('llama-3')
    fireEvent.click(screen.getByRole('button', { name: 'Sprawdź konfigurację' }))
    fireEvent.change(screen.getByRole('combobox'), { target: { value: 'qwen-3' } })
    fireEvent.click(screen.getByRole('button', { name: 'Sprawdź konfigurację' }))

    await waitFor(() => expect(readStoredOnboardingState()?.selections?.validatedModel).toBe('qwen-3'))
    await waitFor(() => expect(screen.getByRole('button', { name: 'Dalej' })).toHaveProperty('disabled', false))

    pending.resolve({ ok: true })

    await waitFor(() => expect(readStoredOnboardingState()?.selections?.model).toBe('qwen-3'))
    expect(readStoredOnboardingState()?.selections?.validatedModel).toBe('qwen-3')
    expect(readStoredOnboardingState()?.completedSteps).toContain('model')
    expect(providerConfigurationCheck).toHaveBeenCalledTimes(2)
  })

  it('rolls back model assignment when final config save fails', async () => {
    persistReadyApprovalsState()
    const saveConfig = vi.fn().mockResolvedValueOnce({ ok: false })
    const saveModel = vi.fn().mockResolvedValue({ ok: true })

    renderOnboarding({ initialStep: 'approvals', saveConfig, saveModel })

    fireEvent.click(await screen.findByRole('button', { name: 'Zakończ' }))

    await waitFor(() => expect(saveModel).toHaveBeenCalledTimes(2))
    expect(saveModel).toHaveBeenNthCalledWith(1, { model: 'llama-3', provider: 'fireworks' }, expect.any(Object))
    expect(saveModel).toHaveBeenNthCalledWith(2, { model: 'llama-3', provider: 'fireworks' }, expect.any(Object))
    expect(readStoredOnboardingState()?.completedSteps).not.toContain('approvals')
  })

  it('resyncs runtime after rollback when the first reload applies then rejects', async () => {
    persistReadyApprovalsState()

    const requestGateway = vi
      .fn()
      .mockRejectedValueOnce(new Error('lost reload response'))
      .mockResolvedValueOnce({ ok: true })

    const saveConfig = vi.fn().mockResolvedValue({ ok: true })
    const saveModel = vi.fn().mockResolvedValue({ ok: true })
    const onComplete = vi.fn()

    renderOnboarding({ initialStep: 'approvals', onComplete, requestGateway, saveConfig, saveModel })

    fireEvent.click(await screen.findByRole('button', { name: 'Zakończ' }))

    await waitFor(() => expect(requestGateway).toHaveBeenCalledTimes(2))
    expect(requestGateway).toHaveBeenNthCalledWith(1, 'reload.env')
    expect(requestGateway).toHaveBeenNthCalledWith(2, 'reload.env')
    expect(saveConfig).toHaveBeenNthCalledWith(
      1,
      {
        approvals: { mode: 'smart' },
        custom_prompt: expect.stringContaining('Cześkiem'),
        stt: { enabled: false },
        voice: { auto_tts: false, engine: 'classic' }
      },
      TEST_SCOPE
    )
    expect(saveConfig).toHaveBeenNthCalledWith(
      2,
      { approvals: { mode: 'manual' }, stt: { enabled: false }, voice: { auto_tts: false } },
      TEST_SCOPE
    )
    expect(saveModel).toHaveBeenCalledTimes(2)
    expect(onComplete).not.toHaveBeenCalled()
    expect(readStoredOnboardingState()?.completedSteps).not.toContain('approvals')
  })

  it('resyncs runtime after rollback when completion gate write fails after reload', async () => {
    persistReadyApprovalsState()
    const requestGateway = vi.fn().mockResolvedValue({ ok: true })
    const onComplete = vi.fn()
    const originalSetItem = window.localStorage.setItem.bind(window.localStorage)
    const setItem = vi.spyOn(window.localStorage, 'setItem')

    setItem.mockImplementation(function failFinalGateWrite(this: Storage, key: string, value: string) {
      if (
        key.startsWith(`${JARVIS_ONBOARDING_STATE_KEY}:`) &&
        value.includes(`"completedSteps":${JSON.stringify(JARVIS_ONBOARDING_STEPS)}`)
      ) {
        throw new Error('gate unavailable')
      }

      return Reflect.apply(originalSetItem, this, [key, value])
    })

    try {
      renderOnboarding({ initialStep: 'approvals', onComplete, requestGateway })

      fireEvent.click(await screen.findByRole('button', { name: 'Zakończ' }))

      await waitFor(() => expect(requestGateway).toHaveBeenCalledTimes(2))
      expect(requestGateway).toHaveBeenNthCalledWith(1, 'reload.env')
      expect(requestGateway).toHaveBeenNthCalledWith(2, 'reload.env')
      expect(await screen.findByText('Nie udało się zapisać onboardingu.')).toBeTruthy()
      expect(onComplete).not.toHaveBeenCalled()
      expect(readStoredOnboardingState()?.completedSteps).not.toContain('approvals')
    } finally {
      setItem.mockRestore()
    }
  })

  it('surfaces recovery error and does not complete when rollback resync fails', async () => {
    persistReadyApprovalsState()

    const requestGateway = vi
      .fn()
      .mockRejectedValueOnce(new Error('lost reload response'))
      .mockRejectedValueOnce(new Error('resync failed'))

    const onComplete = vi.fn()

    renderOnboarding({ initialStep: 'approvals', onComplete, requestGateway })

    fireEvent.click(await screen.findByRole('button', { name: 'Zakończ' }))

    expect(await screen.findByText(/resync failed/)).toBeTruthy()
    expect(await screen.findByText(/Automatyczny rollback się nie udał/)).toBeTruthy()
    expect(requestGateway).toHaveBeenCalledTimes(2)
    expect(onComplete).not.toHaveBeenCalled()
    expect(readStoredOnboardingState()?.completedSteps).not.toContain('approvals')
  })

  it('surfaces recovery failure instead of completing when rollback fails', async () => {
    persistReadyApprovalsState()
    const saveConfig = vi.fn().mockResolvedValueOnce({ ok: false })
    const saveModel = vi.fn().mockResolvedValueOnce({ ok: true }).mockRejectedValueOnce(new Error('rollback blocked'))
    const onComplete = vi.fn()

    renderOnboarding({ initialStep: 'approvals', onComplete, saveConfig, saveModel })

    fireEvent.click(await screen.findByRole('button', { name: 'Zakończ' }))

    expect(await screen.findByText(/rollback blocked/)).toBeTruthy()
    expect(await screen.findByText(/Automatyczny rollback się nie udał/)).toBeTruthy()
    expect(onComplete).not.toHaveBeenCalled()
    expect(readStoredOnboardingState()?.completedSteps).not.toContain('approvals')
  })

  it('blocks final completion when the final assignment requires confirmation', async () => {
    persistReadyApprovalsState()

    const saveModel = vi
      .fn()
      .mockResolvedValue({ confirm_message: 'Confirm paid model', confirm_required: true, ok: false })

    const onComplete = vi.fn()

    renderOnboarding({ initialStep: 'approvals', onComplete, saveModel })

    fireEvent.click(await screen.findByRole('button', { name: 'Zakończ' }))

    expect(await screen.findByText('Confirm paid model')).toBeTruthy()
    expect(onComplete).not.toHaveBeenCalled()
    expect(readStoredOnboardingState()?.completedSteps).not.toContain('approvals')
  })

  it('blocks final completion before assignment when the previous model snapshot is empty', async () => {
    persistReadyApprovalsState()
    const saveConfig = vi.fn().mockResolvedValue({ ok: true })
    const saveModel = vi.fn().mockResolvedValue({ ok: true })

    renderOnboarding({
      initialStep: 'approvals',
      loadModelOptions: async () => ({
        model: '',
        provider: '',
        providers: [{ authenticated: true, models: ['llama-3'], name: 'Fireworks AI', slug: 'fireworks' }]
      }),
      saveConfig,
      saveModel
    })

    fireEvent.click(await screen.findByRole('button', { name: 'Zakończ' }))

    expect(await screen.findByText(/Nie można bezpiecznie zapisać onboardingu/)).toBeTruthy()
    expect(saveModel).not.toHaveBeenCalled()
    expect(saveConfig).not.toHaveBeenCalled()
    expect(readStoredOnboardingState()?.completedSteps).not.toContain('approvals')
  })

  it('makes Gemini Live the conversation layer: its key goes to GEMINI_API_KEY and setup writes the provider', async () => {
    renderOnboarding({ initialStep: 'voice' })

    const openaiKeyPanel = document.querySelector('[data-live-key="live"]') as HTMLElement
    const geminiKeyPanel = document.querySelector('[data-live-key="gemini"]') as HTMLElement
    expect(openaiKeyPanel).toBeTruthy()
    expect(geminiKeyPanel).toBeTruthy()

    fireEvent.change(within(openaiKeyPanel).getByLabelText(pl.jarvisOnboarding.voice.liveKeyLabel), {
      target: { value: 'sk-test-openai' }
    })
    fireEvent.click(within(openaiKeyPanel).getByRole('button', { name: pl.jarvisOnboarding.voice.liveKeySave }))
    await waitFor(() => expect(savedEnv.get('OPENAI_API_KEY')).toBe('sk-test-openai'))

    fireEvent.click(await screen.findByRole('button', { name: pl.jarvisOnboarding.voice.gemini }))
    fireEvent.change(within(geminiKeyPanel).getByLabelText(pl.jarvisOnboarding.voice.geminiKeyLabel), {
      target: { value: 'AIza-test' }
    })
    fireEvent.click(within(geminiKeyPanel).getByRole('button', { name: pl.jarvisOnboarding.voice.liveKeySave }))

    await waitFor(() => expect(savedEnv.get('GEMINI_API_KEY')).toBe('AIza-test'))
    expect(readStoredOnboardingState()?.selections?.voiceMode).toBe('gemini')

    cleanup()
    persistReadyApprovalsState()
    const stored = readStoredOnboardingState()!

    window.localStorage.setItem(
      onboardingStorageKey(),
      JSON.stringify({ ...stored, selections: { ...stored.selections, voiceMode: 'gemini' } })
    )

    const saveConfig = vi.fn(async () => ({ ok: true }))
    const onComplete = vi.fn()

    renderOnboarding({ initialStep: 'approvals', onComplete, saveConfig })
    fireEvent.click(await screen.findByRole('button', { name: 'Zakończ' }))

    await waitFor(() => expect(onComplete).toHaveBeenCalledOnce())
    expect(saveConfig).toHaveBeenCalledWith(
      expect.objectContaining({
        voice: expect.objectContaining({
          engine: 'realtime',
          realtime: expect.objectContaining({ provider: 'gemini' })
        })
      }),
      TEST_SCOPE
    )
  })

  it('opens on how Agent Czesiek works, and remembers what the person wants connected', async () => {
    renderOnboarding({ initialStep: 'welcome' })

    const welcome = await screen.findByTestId('jarvis-onboarding-welcome')

    expect(within(welcome).getByText(pl.jarvisOnboarding.welcome.pillars.approvals.title)).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: new RegExp(pl.jarvisOnboarding.steps.connections) }))

    const connections = await screen.findByTestId('jarvis-onboarding-connections')

    const google = within(connections).getByRole('checkbox', {
      name: new RegExp(pl.jarvisConnections.entries.google.name)
    })

    fireEvent.click(google)
    await waitFor(() => expect(readStoredOnboardingState()?.selections?.connections).toEqual(['google']))
    expect(google.getAttribute('aria-checked')).toBe('true')

    fireEvent.click(google)
    await waitFor(() => expect(readStoredOnboardingState()?.selections?.connections).toEqual([]))
  })

  it('applies the chosen computer capabilities only after setup commits', async () => {
    persistReadyComputerState()
    const setToolsetEnabled = vi.fn(async () => ({ ok: true }))
    const onComplete = vi.fn()

    renderOnboarding({
      computerStatus: OFFLINE_COMPUTER_STATUS,
      initialStep: 'computer',
      onComplete,
      setToolsetEnabled
    })

    fireEvent.click(await screen.findByRole('radio', { name: 'Operator' }))
    expect(setToolsetEnabled).not.toHaveBeenCalled()

    fireEvent.click(screen.getByRole('button', { name: 'Dalej' }))
    // Connections come between the computer and approvals; choosing none is fine.
    await screen.findByTestId('jarvis-onboarding-connections')
    fireEvent.click(screen.getByRole('button', { name: 'Dalej' }))
    fireEvent.click(await screen.findByRole('button', { name: 'Zakończ' }))

    await waitFor(() => expect(onComplete).toHaveBeenCalledOnce())
    expect(setToolsetEnabled).toHaveBeenCalledWith('computer_use', true, TEST_SCOPE)
    expect(setToolsetEnabled).toHaveBeenCalledWith('terminal', true, TEST_SCOPE)
    expect(readStoredOnboardingState()?.selections?.computerMode).toBe('operator')
  })

  it('revokes desktop control when the user steps back down to conversation only', async () => {
    persistReadyComputerState()
    const setToolsetEnabled = vi.fn(async () => ({ ok: true }))

    renderOnboarding({ computerStatus: OFFLINE_COMPUTER_STATUS, initialStep: 'computer', setToolsetEnabled })

    fireEvent.click(await screen.findByRole('radio', { name: 'Rozmowa' }))
    fireEvent.click(screen.getByRole('button', { name: 'Dalej' }))
    // Connections come between the computer and approvals; choosing none is fine.
    await screen.findByTestId('jarvis-onboarding-connections')
    fireEvent.click(screen.getByRole('button', { name: 'Dalej' }))
    fireEvent.click(await screen.findByRole('button', { name: 'Zakończ' }))

    await waitFor(() => expect(setToolsetEnabled).toHaveBeenCalledWith('computer_use', false, TEST_SCOPE))
    expect(setToolsetEnabled).toHaveBeenCalledWith('terminal', false, TEST_SCOPE)
    expect(setToolsetEnabled).toHaveBeenCalledWith('web', true, TEST_SCOPE)
  })

  it('completes setup even when the backend refuses a toolset', async () => {
    persistReadyApprovalsState()
    const onComplete = vi.fn()

    renderOnboarding({
      initialStep: 'approvals',
      onComplete,
      setToolsetEnabled: vi.fn(async () => {
        throw new Error('toolset unavailable')
      })
    })

    fireEvent.click(await screen.findByRole('button', { name: 'Zakończ' }))

    await waitFor(() => expect(onComplete).toHaveBeenCalledOnce())
    expect(readStoredOnboardingState()?.completedSteps).toContain('approvals')
  })

  it('shows real desktop-driver readiness only for the mode that needs it', async () => {
    const loadStatus = vi.fn(async () => ({
      accessibility: false,
      can_grant: true,
      checks: [],
      error: null,
      installed: true,
      platform: 'darwin',
      platform_supported: true,
      ready: false,
      screen_recording: false,
      screen_recording_capturable: null,
      source: null,
      version: 'cua-driver 0.5.1'
    }))

    renderOnboarding({ computerStatus: { loadStatus }, initialStep: 'computer' })

    fireEvent.click(await screen.findByRole('radio', { name: 'Pracuje za Ciebie' }))
    expect(screen.queryByTestId('jarvis-computer-status')).toBeNull()
    expect(loadStatus).toHaveBeenCalled()

    fireEvent.click(screen.getByRole('radio', { name: 'Operator' }))

    expect(await screen.findByText('Sterowanie pulpitem wymaga uprawnień')).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Nadaj uprawnienia' })).toBeTruthy()
  })

  it('supports radiogroup arrow navigation for approvals', async () => {
    renderOnboarding({
      initialStep: 'approvals',
      loadConfig: async () => ({
        approvals: { mode: 'auto' },
        stt: { enabled: false },
        voice: { auto_tts: false }
      })
    })

    const balanced = await screen.findByRole('radio', { name: 'Zrównoważony' })
    await waitFor(() => expect(balanced.getAttribute('aria-checked')).toBe('true'))
    balanced.focus()
    fireEvent.keyDown(screen.getByRole('radiogroup', { name: 'Zgody' }), { key: 'ArrowRight' })

    await waitFor(() => expect(screen.getByRole('radio', { name: 'Ścisły' }).getAttribute('aria-checked')).toBe('true'))
  })
})

describe('Agent CzesiekOnboarding OpenRouter quick start', () => {
  beforeEach(() => {
    savedEnv.clear()
    window.localStorage.clear()
  })

  afterEach(cleanup)

  it('turns a pasted OpenRouter key into the DeepSeek work model and moves on', async () => {
    renderOnboarding({
      computerStatus: OFFLINE_COMPUTER_STATUS,
      initialStep: 'engine',
      loadModelOptions: async () => ({
        model: '',
        provider: '',
        providers: savedEnv.has('OPENROUTER_API_KEY')
          ? [
              {
                authenticated: true,
                models: ['openai/gpt-5.5', 'deepseek/deepseek-v4.1-flash'],
                name: 'OpenRouter',
                slug: 'openrouter'
              }
            ]
          : []
      })
    })

    const key = await screen.findByLabelText('Klucz API OpenRouter')
    fireEvent.change(key, { target: { value: 'sk-or-v1-test' } })
    fireEvent.click(screen.getByRole('button', { name: 'Połącz' }))

    await waitFor(() => expect(readStoredOnboardingState()?.currentStep).toBe('model'))
    expect(savedEnv.get('OPENROUTER_API_KEY')).toBe('sk-or-v1-test')
    expect(readStoredOnboardingState()?.selections).toMatchObject({
      engine: 'openrouter',
      model: 'deepseek/deepseek-v4.1-flash'
    })
  })

  it('closes setup from the header and remembers the skip without completing it', async () => {
    const onDismiss = vi.fn()

    renderOnboarding({ onDismiss })

    fireEvent.click(screen.getByRole('button', { name: 'Zamknij konfigurację' }))

    expect(onDismiss).toHaveBeenCalledTimes(1)
    await waitFor(() => expect(readStoredOnboardingState()?.skipped).toBe(true))
    expect(jarvisOnboardingComplete(readStoredOnboardingState())).toBe(false)
  })

  it('offers finishing setup later from the footer', async () => {
    const onDismiss = vi.fn()

    renderOnboarding({ onDismiss })

    fireEvent.click(screen.getByRole('button', { name: 'Dokończę później' }))

    expect(onDismiss).toHaveBeenCalledTimes(1)
    await waitFor(() => expect(readStoredOnboardingState()?.skipped).toBe(true))
  })
})
