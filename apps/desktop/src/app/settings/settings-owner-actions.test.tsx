// @vitest-environment jsdom
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type * as HermesModule from '@/hermes'

const { confirm, getHermesConfigDefaults, getHermesConfigRecord, saveHermesConfig } = vi.hoisted(() => ({
  confirm: vi.fn(),
  getHermesConfigDefaults: vi.fn(),
  getHermesConfigRecord: vi.fn(),
  saveHermesConfig: vi.fn()
}))

vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<typeof HermesModule>()),
  getHermesConfigDefaults,
  getHermesConfigRecord,
  saveHermesConfig
}))
vi.mock('@/store/confirm', () => ({ confirm }))
vi.mock('./about-settings', () => ({ AboutSettings: () => <div>About</div> }))

import { $notifications } from '@/store/notifications'
import { $connection } from '@/store/session'
import { $settingsScopeOverride } from '@/store/settings-scope'

import { SettingsView } from './index'

function deferred<T>() {
  let reject!: (reason: Error) => void
  let resolve!: (value: T) => void

  const promise = new Promise<T>((yes, no) => {
    resolve = yes
    reject = no
  })

  return { promise, reject, resolve }
}

function setOwner(connectionId: string) {
  $connection.set({ connectionId, mode: 'remote', baseUrl: `https://${connectionId}.invalid` } as never)
}

function renderSettings() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })

  return render(
    <MemoryRouter initialEntries={['/settings?tab=about']}>
      <QueryClientProvider client={client}>
        <SettingsView onClose={vi.fn()} />
      </QueryClientProvider>
    </MemoryRouter>
  )
}

beforeEach(() => {
  $notifications.set([])
  $settingsScopeOverride.set(null)
  setOwner('owner-a')
  confirm.mockResolvedValue(true)
  saveHermesConfig.mockResolvedValue({ ok: true })
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

describe('Settings owner action errors', () => {
  it.each([
    ['Export config', getHermesConfigRecord],
    ['Reset to defaults', getHermesConfigDefaults]
  ])('hides stale-owner %s rejection while preserving the captured request owner', async (label, request) => {
    const pending = deferred<Record<string, unknown>>()
    request.mockReturnValueOnce(pending.promise)
    renderSettings()

    fireEvent.click((await screen.findAllByRole('button', { name: label }))[0])
    await waitFor(() => expect(request).toHaveBeenCalledWith(expect.objectContaining({ connectionId: 'owner-a' })))
    await act(async () => setOwner('owner-b'))
    await act(async () => pending.reject(new Error('owner A failed')))

    expect($notifications.get()).toEqual([])
  })

  it.each([
    ['Export config', getHermesConfigRecord],
    ['Reset to defaults', getHermesConfigDefaults]
  ])('surfaces current-owner %s rejection', async (label, request) => {
    request.mockRejectedValueOnce(new Error('current owner failed'))
    renderSettings()

    fireEvent.click((await screen.findAllByRole('button', { name: label }))[0])
    await waitFor(() => expect($notifications.get().length).toBe(1))
    expect($notifications.get()[0]?.kind).toBe('error')
  })
})
