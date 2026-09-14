import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import { type ProfileScope, profileScopeKey, setApiRequestConnection, setApiRequestProfile } from '@/api/client'
import { $activeGatewayProfile } from '@/store/profile'

import { DelegationModelSettings } from './delegation-model-settings'
import { delegationModelsCopy } from './delegation-models-copy'

const server = vi.hoisted(() => ({
  get: vi.fn(),
  options: vi.fn(),
  records: {} as Record<string, Record<string, unknown>>,
  save: vi.fn(),
  schema: vi.fn()
}))

vi.mock('@/hermes', async () => ({
  ...(await import('@/api/client')),
  getGlobalModelOptions: (...args: unknown[]) => server.options(...args),
  getHermesConfigSchema: (...args: unknown[]) => server.schema(...args),
  getHermesConfigRecord: (...args: unknown[]) => server.get(...args),
  saveHermesConfigRecord: (...args: unknown[]) => server.save(...args)
}))
vi.mock('@/lib/query-client', () => ({
  queryClient: { invalidateQueries: vi.fn() },
  writeCache: () => () => {}
}))
vi.mock('@/store/gateway', async () => ({ $gateway: (await import('nanostores')).atom(null) }))
vi.mock('@/store/profile', async () => ({ $activeGatewayProfile: (await import('nanostores')).atom<string | null>('alpha') }))

const copy = delegationModelsCopy('en')
const main = { default: 'head-model', provider: 'head' }
const click = (element: Element) => fireEvent.click(element)

beforeAll(() => {
  Element.prototype.scrollIntoView = vi.fn()
  Element.prototype.hasPointerCapture = vi.fn(() => false)
  Element.prototype.releasePointerCapture = vi.fn()
})
beforeEach(() => {
  vi.clearAllMocks()
  setApiRequestConnection('gateway-a')
  setApiRequestProfile('alpha')
  $activeGatewayProfile.set('alpha')
  server.records = {
    'gateway-a::alpha': {
      delegation: { max_iterations: 17 },
      fallback_providers: [{ model: 'head-backup', provider: 'head' }],
      model: main
    },
    'gateway-a::beta': { delegation: { model: 'beta-worker', provider: '' }, model: main }
  }
  server.get.mockImplementation(async (scope: ProfileScope) => structuredClone(server.records[profileScopeKey(scope)]))
  server.schema.mockResolvedValue({ capabilities: { delegation_fallbacks: true }, fields: {} })
  server.options.mockResolvedValue({
    providers: [{ authenticated: true, models: ['worker-one', 'worker-two'], name: 'Worker provider', slug: 'worker' }]
  })
  server.save.mockImplementation(async (patch: { delegation: Record<string, unknown> }, scope: ProfileScope) => {
    const key = profileScopeKey(scope)
    const previous = server.records[key]

    server.records[key] = { ...previous, delegation: { ...(previous.delegation as object), ...patch.delegation } }

    return { ok: true }
  })
})
afterEach(() => {
  cleanup()
  setApiRequestConnection(null)
  setApiRequestProfile(null)
  $activeGatewayProfile.set('default')
})

async function open(profile: string | null = 'alpha') {
  const client = new QueryClient({ defaultOptions: { queries: { gcTime: 0, retry: false } } })

  const ui = (next: string | null) => (
    <QueryClientProvider client={client}>
      <DelegationModelSettings scopeProfile={next ?? undefined} />
    </QueryClientProvider>
  )

  const view = render(ui(profile))

  await screen.findByRole('combobox', { name: copy.provider })

  return { client, ui, user: { click }, view }
}

async function chooseWorker(user: { click: typeof click }) {
  await user.click(screen.getByRole('combobox', { name: copy.provider }))
  await user.click(screen.getByRole('option', { name: 'Worker provider' }))
  expect(server.save).not.toHaveBeenCalled()
  expect((screen.getByRole('button', { name: copy.apply }) as HTMLButtonElement).disabled).toBe(true)
  await user.click(screen.getByRole('combobox', { name: copy.model }))
  await user.click(screen.getByRole('option', { name: 'worker-two' }))
}

describe('Models screen delegation persistence', () => {
  it('does not expose unused fallback controls on an older backend, even with saved keys', async () => {
    server.schema.mockResolvedValue({ fields: {} })
    server.records['gateway-a::alpha'].delegation = {
      fallback_providers: [{ model: 'worker-one', provider: 'worker' }]
    }
    const client = new QueryClient({ defaultOptions: { queries: { gcTime: 0, retry: false } } })

    render(
      <QueryClientProvider client={client}>
        <DelegationModelSettings scopeProfile="alpha" />
      </QueryClientProvider>
    )
    await screen.findByText(copy.unsupported)
    expect(screen.queryByRole('combobox')).toBeNull()
    expect(server.save).not.toHaveBeenCalled()
    expect(server.options).not.toHaveBeenCalled()
    expect(server.schema).toHaveBeenCalledWith({ connectionId: 'gateway-a', profile: 'alpha' })
  })

  it('saves one complete pair to the catalog owner and confirms it without touching the main model', async () => {
    const { user } = await open()

    await chooseWorker(user)
    expect(server.save).not.toHaveBeenCalled()
    await user.click(screen.getByRole('button', { name: copy.apply }))
    await waitFor(() => expect(server.save).toHaveBeenCalledTimes(1))
    expect(server.save.mock.calls[0]).toEqual([
      {
        delegation: {
          api_key: '',
          api_mode: '',
          base_url: '',
          model: 'worker-two',
          provider: 'worker',
          request_overrides: null
        }
      },
      { connectionId: 'gateway-a', profile: 'alpha' }
    ])
    expect(server.options).toHaveBeenCalledWith(undefined, { connectionId: 'gateway-a', profile: 'alpha' })
    await waitFor(() => expect((screen.getByRole('button', { name: copy.apply }) as HTMLButtonElement).disabled).toBe(true))
    expect(server.records['gateway-a::alpha'].model).toEqual(main)
    expect(server.records['gateway-a::alpha'].delegation).toMatchObject({ max_iterations: 17 })
    expect(server.records['gateway-a::alpha'].fallback_providers).toEqual([{ model: 'head-backup', provider: 'head' }])
  })

  it.each(['gateway-a', 'local'])('saves an implicit default profile on %s', async connectionId => {
    setApiRequestConnection(connectionId === 'local' ? null : connectionId)
    setApiRequestProfile(null)
    $activeGatewayProfile.set('default')
    const key = `${connectionId}::default`

    server.records[key] = { delegation: { max_iterations: 17 }, model: main }
    const { user } = await open(null)

    await chooseWorker(user)
    await user.click(screen.getByRole('button', { name: copy.apply }))
    await waitFor(() => expect(server.save).toHaveBeenCalledTimes(1))
    expect(server.save.mock.calls[0][1]).toEqual({ connectionId, profile: 'default' })
    expect(server.options).toHaveBeenCalledWith(undefined, { connectionId, profile: 'default' })
    await waitFor(() => expect((screen.getByRole('button', { name: copy.apply }) as HTMLButtonElement).disabled).toBe(true))
    expect(server.records[key].delegation).toMatchObject({
      max_iterations: 17,
      model: 'worker-two',
      provider: 'worker'
    })
    expect(server.records[key].model).toEqual(main)
  })

  it('retains edits and allows retry after a failed write', async () => {
    const { user } = await open()

    await chooseWorker(user)
    server.save.mockRejectedValueOnce(new Error('offline'))
    await user.click(screen.getByRole('button', { name: copy.apply }))
    await screen.findByText(copy.failed)
    expect(screen.getByRole('combobox', { name: copy.model }).textContent).toContain('worker-two')
    expect((screen.getByRole('button', { name: copy.apply }) as HTMLButtonElement).disabled).toBe(false)
    await user.click(screen.getByRole('button', { name: copy.apply }))
    await waitFor(() => expect(server.save).toHaveBeenCalledTimes(2))
  })

  it('resets drafts on a profile switch, even when no prior write occurred', async () => {
    const { ui, user, view } = await open()

    await chooseWorker(user)
    view.rerender(ui('beta'))
    await waitFor(() =>
      expect((screen.getByRole('textbox', { name: copy.model }) as HTMLInputElement).value).toBe('beta-worker')
    )
    expect(server.save).not.toHaveBeenCalled()
    expect(server.options).toHaveBeenCalledWith(undefined, { connectionId: 'gateway-a', profile: 'beta' })
  })

  it('does not apply an old save completion to the newly selected profile', async () => {
    const { ui, user, view } = await open()

    await chooseWorker(user)
    let finish!: (value: { ok: boolean }) => void

    server.save.mockImplementationOnce(() => new Promise(resolve => { finish = resolve }))
    await user.click(screen.getByRole('button', { name: copy.apply }))
    await waitFor(() => expect(server.save).toHaveBeenCalledTimes(1))
    const alphaReads = server.get.mock.calls.filter(([scope]) => scope.profile === 'alpha').length

    view.rerender(ui('beta'))
    await waitFor(() =>
      expect((screen.getByRole('textbox', { name: copy.model }) as HTMLInputElement).value).toBe('beta-worker')
    )
    finish({ ok: true })
    await waitFor(() => expect(screen.getByRole('textbox', { name: copy.model })).toBeTruthy())
    expect(server.get.mock.calls.filter(([scope]) => scope.profile === 'alpha')).toHaveLength(alphaReads)
    expect(server.records['gateway-a::beta'].delegation).toEqual({ model: 'beta-worker', provider: '' })
  })

  it('preserves endpoint and credential-reference metadata through a reorder', async () => {
    const first = { base_url: 'https://one.invalid/v1', key_env: 'ONE_KEY', model: 'worker-one', provider: 'worker' }
    const second = { base_url: 'https://two.invalid/v1', key_env: 'TWO_KEY', model: 'worker-two', provider: 'worker' }

    server.records['gateway-a::alpha'].delegation = { fallback_providers: [first, second] }
    const { user } = await open()

    await user.click(screen.getByRole('button', { name: `${copy.up} 2` }))
    expect(server.save).not.toHaveBeenCalled()
    await user.click(screen.getByRole('button', { name: copy.apply }))
    await waitFor(() => expect(server.save).toHaveBeenCalledTimes(1))
    expect(server.save.mock.calls[0][0]).toEqual({
      delegation: {
        fallback_chain: null,
        fallback_model: null,
        fallback_providers: [second, first]
      }
    })
  })

  it('detects a competing config update rather than silently overwriting it', async () => {
    const { client, user } = await open()

    await chooseWorker(user)
    server.records['gateway-a::alpha'].delegation = { model: 'newer-external-choice', provider: '' }
    await client.invalidateQueries()
    await screen.findByText(copy.conflict)
    expect((screen.getByRole('button', { name: copy.apply }) as HTMLButtonElement).disabled).toBe(true)
    expect(server.save).not.toHaveBeenCalled()
  })
})
