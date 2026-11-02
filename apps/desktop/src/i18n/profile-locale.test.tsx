import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { setApiRequestConnection } from '@/api/client'
import type { HermesApiRequest } from '@/global'
import type { HermesConfigRecord } from '@/hermes'
import { $gateway } from '@/store/gateway'
import { $activeGatewayProfile } from '@/store/profile'
import { $connection, $gatewayState } from '@/store/session'

import { useI18n } from './context'
import { ProfileI18nProvider as I18nProvider } from './profile-provider'
import { resetAppLocaleRegistry } from './registry'
import { $requestedLocale } from './runtime'

function Probe() {
  const { locale, setLocale, saveError, isSavingLocale } = useI18n()

  return (
    <>
      <span data-testid="locale">{locale}</span>
      <span data-testid="error">{saveError?.message}</span>
      <span data-testid="saving">{String(isSavingLocale)}</span>
      <button onClick={() => void setLocale('ja').catch(() => undefined)}>Japanese</button>
    </>
  )
}

function route(profile: string, connectionId: string | null = null) {
  setApiRequestConnection(connectionId)
  $connection.set(connectionId ? ({ connectionId } as never) : null)
  $activeGatewayProfile.set(profile)
}

afterEach(() => {
  cleanup()
  $gateway.set(null)
  $gatewayState.set('idle')
  resetAppLocaleRegistry()
  $requestedLocale.set(null)
  route('default')
  Reflect.deleteProperty(window, 'hermesDesktop')
  vi.restoreAllMocks()
})

// Regression for #113980: exercise the real provider, API origin binding and
// profile routing, including an explicit pick before leaving and a fresh mount.
it('reads and persists the owning profile through A → B → A and restart', async () => {
  const configs: Record<string, HermesConfigRecord> = {
    default: { display: { language: 'en' } },
    coder: { display: { language: 'zh', skin: 'mono' } }
  }

  const api = vi.fn(async (request: HermesApiRequest) => {
    const key = request.profile || 'default'

    if (request.method === 'PUT') {
      configs[key] = structuredClone((request.body as { config: HermesConfigRecord }).config)

      return { ok: true }
    }

    return structuredClone(configs[key])
  })

  window.hermesDesktop = { api } as never

  const view = render(
    <I18nProvider>
      <Probe />
    </I18nProvider>
  )

  await waitFor(() => expect(api).toHaveBeenCalled())
  act(() => route('coder'))
  await waitFor(() => expect(screen.getByTestId('locale').textContent).toBe('zh'))
  fireEvent.click(screen.getByText('Japanese'))
  await waitFor(() => expect(configs.coder.display).toEqual({ language: 'ja', skin: 'mono' }))
  act(() => route('default'))
  await waitFor(() => expect(screen.getByTestId('locale').textContent).toBe('en'))
  act(() => route('coder'))
  await waitFor(() => expect(screen.getByTestId('locale').textContent).toBe('ja'))
  expect(configs.default.display).toEqual({ language: 'en' })
  view.unmount()
  render(
    <I18nProvider>
      <Probe />
    </I18nProvider>
  )
  await waitFor(() => expect(screen.getByTestId('locale').textContent).toBe('ja'))
})

it('isolates stale reads and failed saves when the same profile moves between connections', async () => {
  let finishBoot!: (config: HermesConfigRecord) => void
  let finishSaveRead!: (config: HermesConfigRecord) => void
  let rejectSave!: (reason: Error) => void
  let reads = 0

  const api = vi.fn((request: HermesApiRequest) => {
    if (request.method === 'PUT') {
      return new Promise((_, reject) => {
        rejectSave = reject
      })
    }

    reads += 1

    if (reads === 1) {
      return new Promise(resolve => {
        finishBoot = resolve
      })
    }

    if (reads === 3) {
      return new Promise(resolve => {
        finishSaveRead = resolve
      })
    }

    return Promise.resolve({ display: { language: request.connectionId === 'remote' ? 'zh' : 'en' } })
  })

  window.hermesDesktop = { api } as never
  render(
    <I18nProvider>
      <Probe />
    </I18nProvider>
  )
  act(() => route('default', 'remote'))
  await waitFor(() => expect(screen.getByTestId('locale').textContent).toBe('zh'))
  await act(async () => finishBoot({ display: { language: 'ar' } }))
  expect(screen.getByTestId('locale').textContent).toBe('zh')
  fireEvent.click(screen.getByText('Japanese'))
  await waitFor(() => expect(reads).toBe(3))
  act(() => route('default', 'local'))
  await waitFor(() => expect(screen.getByTestId('locale').textContent).toBe('en'))
  await act(async () => finishSaveRead({ display: { language: 'zh' } }))
  expect(api).toHaveBeenLastCalledWith(expect.objectContaining({ connectionId: 'remote', method: 'PUT' }))
  await act(async () => rejectSave(new Error('remote save refused')))
  expect(screen.getByTestId('locale').textContent).toBe('en')
  expect(screen.getByTestId('error').textContent).toBe('')
  expect(screen.getByTestId('saving').textContent).toBe('false')

  // Legacy remotes may not have registry IDs. A new backend URL is still a
  // new config owner even when the visible profile retains its name.
  act(() => {
    setApiRequestConnection(null)
    $connection.set({ baseUrl: 'https://first.example' } as never)
  })
  await waitFor(() => expect(reads).toBe(5))
  api.mockImplementation(() => Promise.resolve({ display: { language: 'ar' } }))
  act(() => $connection.set({ baseUrl: 'https://second.example' } as never))
  await waitFor(() => expect(screen.getByTestId('locale').textContent).toBe('ar'))
})

// A backend whose hermes-lang-ko pack carries Desktop strings.
function koPackRequest() {
  return vi.fn(async (method: string) =>
    method === 'i18n.languages'
      ? {
          languages: [
            {
              id: 'ko',
              endonym: '한국어',
              rtl: false,
              source: 'plugin:hermes-lang-ko',
              surfaces: ['core', 'tui', 'desktop']
            }
          ]
        }
      : { lang: 'ko', surface: 'desktop', messages: { 'common.save': '저장' } }
  )
}

// A saved `display.language: ko` with an OS-language probe the test resolves.
function savedKoWithDeferredProbe() {
  let resolveProfile: (profile: { locale: string }) => void = () => undefined

  const machineProfile = new Promise<{ locale: string }>(resolve => {
    resolveProfile = resolve
  })

  window.hermesDesktop = {
    api: vi.fn(async () => ({ display: { language: 'ko' } })),
    getMachineProfile: vi.fn(() => machineProfile)
  } as never

  return async (locale: string) => {
    await act(async () => {
      resolveProfile({ locale })
      await machineProfile
    })
    await act(async () => undefined)
  }
}

function mountWithOpenSocket() {
  act(() => $gatewayState.set('open'))
  render(
    <I18nProvider>
      <Probe />
    </I18nProvider>
  )
}

// Boot reports the socket `open` before the client is published in `$gateway`.
// The backend-pack sync must still run once the client arrives, or a language
// only a pack renders (ko via hermes-lang-ko) never leaves English.
it('pulls the backend language pack once the gateway client is published after the socket opened', async () => {
  window.hermesDesktop = { api: vi.fn(async () => ({ display: { language: 'ko' } })) } as never
  const request = koPackRequest()

  mountWithOpenSocket()
  await waitFor(() => expect(window.hermesDesktop.api).toHaveBeenCalled())
  expect(request).not.toHaveBeenCalled()

  act(() => $gateway.set({ request } as never))
  await waitFor(() => expect(screen.getByTestId('locale').textContent).toBe('ko'))
  expect(request).toHaveBeenCalledWith('i18n.catalog', { lang: 'ko', surface: 'desktop' })
})

// The OS-language probe can finish after the pack sync promoted the saved
// language; the probe's answer must not replace it.
it('keeps a pack-rendered saved language when the OS-language probe resolves after the pack', async () => {
  const resolveProbe = savedKoWithDeferredProbe()

  mountWithOpenSocket()
  await waitFor(() => expect(window.hermesDesktop.getMachineProfile).toHaveBeenCalled())

  act(() => $gateway.set({ request: koPackRequest() } as never))
  await waitFor(() => expect(screen.getByTestId('locale').textContent).toBe('ko'))

  await resolveProbe('en-US')
  expect(screen.getByTestId('locale').textContent).toBe('ko')
})

// The other order: the OS guess applies first, then the pack promotes the
// saved language.
it('promotes a saved pack language that arrives after the OS-language probe', async () => {
  const resolveProbe = savedKoWithDeferredProbe()

  mountWithOpenSocket()
  await waitFor(() => expect(window.hermesDesktop.getMachineProfile).toHaveBeenCalled())

  await resolveProbe('en-US')
  expect(screen.getByTestId('locale').textContent).toBe('en')

  act(() => $gateway.set({ request: koPackRequest() } as never))
  await waitFor(() => expect(screen.getByTestId('locale').textContent).toBe('ko'))
})
