import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const mocks = vi.hoisted(() => ({
  getWebhooks: vi.fn(),
  enableWebhooks: vi.fn(),
  createWebhook: vi.fn(),
  deleteWebhook: vi.fn(),
  setWebhookEnabled: vi.fn(),
  runGatewayRestart: vi.fn(),
  notify: vi.fn(),
  notifyError: vi.fn()
}))

vi.mock('@/hermes', () => ({
  getWebhooks: (profile?: string) => mocks.getWebhooks(profile),
  enableWebhooks: (profile?: string) => mocks.enableWebhooks(profile),
  createWebhook: (body: unknown, profile?: string) => mocks.createWebhook(body, profile),
  deleteWebhook: (name: string, profile?: string) => mocks.deleteWebhook(name, profile),
  setWebhookEnabled: (name: string, enabled: boolean, profile?: string) => mocks.setWebhookEnabled(name, enabled, profile)
}))

vi.mock('@/store/profile', async () => {
  const { atom } = await import('nanostores')

  return {
    $activeGatewayProfile: atom('alpha'),
    $profileScope: atom('__all__'),
    normalizeProfileKey: (value: string | null | undefined) => String(value ?? '').trim() || 'default'
  }
})

vi.mock('@/store/system-actions', () => ({
  runGatewayRestart: (profile?: string) => mocks.runGatewayRestart(profile)
}))

vi.mock('@/store/notifications', () => ({ notify: mocks.notify, notifyError: mocks.notifyError }))

import { $activeGatewayProfile } from '@/store/profile'

import { WebhooksView } from './index'

function deferred<T>() {
  let resolve!: (value: T) => void
  let reject!: (reason?: unknown) => void

  const promise = new Promise<T>((res, rej) => {
    resolve = res
    reject = rej
  })

  return { promise, reject, resolve }
}

function renderView() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })

  return render(
    <QueryClientProvider client={client}>
      <WebhooksView onClose={() => undefined} />
    </QueryClientProvider>
  )
}

beforeEach(() => {
  vi.stubGlobal(
    'ResizeObserver',
    class ResizeObserver {
      observe() {}
      unobserve() {}
      disconnect() {}
    }
  )
  Element.prototype.scrollIntoView = vi.fn()
  Element.prototype.hasPointerCapture = vi.fn(() => false)
  Element.prototype.releasePointerCapture = vi.fn()

  $activeGatewayProfile.set('alpha')
  mocks.getWebhooks.mockResolvedValue({ enabled: true, subscriptions: [] })
  mocks.enableWebhooks.mockResolvedValue({ restart_started: true })
  mocks.deleteWebhook.mockResolvedValue({ ok: true })
  mocks.setWebhookEnabled.mockResolvedValue({ enabled: true, name: 'hook', ok: true })
  mocks.runGatewayRestart.mockResolvedValue(undefined)
})

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
  vi.clearAllMocks()
})

describe('WebhooksView profile ownership', () => {
  it('drops the create draft when the backend profile changes under All profiles', async () => {
    renderView()

    fireEvent.click(await screen.findByRole('button', { name: /new subscription/i }))
    fireEvent.change(screen.getByLabelText(/^name$/i), { target: { value: 'alpha-hook' } })
    expect(screen.getByDisplayValue('alpha-hook')).toBeTruthy()

    act(() => $activeGatewayProfile.set('beta'))

    await waitFor(() => expect(mocks.getWebhooks).toHaveBeenCalledWith('beta'))
    expect(screen.queryByDisplayValue('alpha-hook')).toBeNull()
  })

  it('ignores a create completion owned by the previous profile', async () => {
    const create = deferred<{ secret: string; url: string }>()
    mocks.createWebhook.mockReturnValue(create.promise)
    renderView()

    fireEvent.click(await screen.findByRole('button', { name: /new subscription/i }))
    fireEvent.change(screen.getByLabelText(/^name$/i), { target: { value: 'alpha-hook' } })
    fireEvent.click(screen.getByRole('button', { name: /^create$/i }))

    await waitFor(() =>
      expect(mocks.createWebhook).toHaveBeenCalledWith(expect.objectContaining({ name: 'alpha-hook' }), 'alpha')
    )

    act(() => $activeGatewayProfile.set('beta'))
    await act(async () => create.resolve({ secret: 'alpha-secret', url: 'https://alpha.example/hook' }))

    await waitFor(() => expect(mocks.getWebhooks).toHaveBeenCalledWith('beta'))
    expect(screen.queryByText('alpha-secret')).toBeNull()
    expect(mocks.notify).not.toHaveBeenCalled()
  })
})
