import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { setApiRequestConnection } from '@/api/client'
import type { HermesApiRequest } from '@/global'
import type { HermesConfigRecord } from '@/hermes'
import { $activeGatewayProfile } from '@/store/profile'
import { $connection } from '@/store/session'

import { useI18n } from './context'
import { ProfileI18nProvider as I18nProvider } from './profile-provider'

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
