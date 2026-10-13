import { QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { PRIMARY_SESSION_VIEW, SessionViewProvider } from '@/app/chat/session-view'
import { hermesConfigKey } from '@/app/hooks/use-config-record'
import type { HermesApiRequest } from '@/global'
import { setApiRequestConnection, setApiRequestProfile } from '@/hermes'
import { I18nProvider } from '@/i18n'
import { queryClient } from '@/lib/query-client'
import { $activeGatewayProfile } from '@/store/profile'
import { $connection } from '@/store/session'
import type { HermesConfigRecord } from '@/types/hermes'

import { ComposerScopeProvider, MAIN_COMPOSER_SCOPE } from './scope'
import { WorkhorseModelPill } from './workhorse-model-pill'

const mocks = vi.hoisted(() => ({ request: vi.fn(), notifyError: vi.fn() }))
vi.mock('@/store/gateway', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  requestGatewayForAgent: mocks.request
}))
vi.mock('@/store/notifications', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  notifyError: mocks.notifyError
}))

const owner = { connectionId: 'connection-a', profile: 'worker-a' }

const tileView = {
  ...PRIMARY_SESSION_VIEW,
  kind: 'tile' as const,
  $runtimeId: atom<string | null>('tile-a'),
  $model: atom('parent-model'),
  $provider: atom('p'),
  $reasoningEffort: atom('medium')
}

const other = { connectionId: 'connection-b', profile: 'worker-b' }
const directRoute = { base_url: 'https://old.invalid/v1', api_key: 'old-test-key', api_mode: 'responses' }
const clearedRoute = { base_url: '', api_key: '', api_mode: '' }

const catalog = {
  provider: 'p',
  model: 'parent-model',
  providers: [
    {
      slug: 'p',
      name: 'Provider',
      models: ['old-model', 'new-model', 'last-model'],
      capabilities: {
        'old-model': { reasoning: true, can_disable_reasoning: true },
        'new-model': { reasoning: true, can_disable_reasoning: true }
      }
    }
  ]
}

let records: Map<string, HermesConfigRecord>
let api: ReturnType<typeof vi.fn>
let save: ReturnType<typeof vi.fn<(request: HermesApiRequest) => Promise<{ ok: boolean }>>>

const slot = (route: { connectionId?: string | null; profile?: string | null }) =>
  `${route.connectionId}::${route.profile}`

const cached = (route = owner) =>
  queryClient.getQueryData<HermesConfigRecord>(hermesConfigKey(route, route.connectionId))

const puts = () => api.mock.calls.map(([r]) => r as HermesApiRequest).filter(r => r.method === 'PUT')

function activate(route: typeof owner) {
  setApiRequestConnection(route.connectionId)
  setApiRequestProfile(route.profile)
  $connection.set({ connectionId: route.connectionId, mode: 'remote' } as never)
  $activeGatewayProfile.set(route.profile)
}

function deferred() {
  let resolve!: (value: { ok: boolean }) => void
  let reject!: (reason: Error) => void

  const promise = new Promise<{ ok: boolean }>((yes, no) => {
    resolve = yes
    reject = no
  })

  return { promise, resolve, reject }
}

function ui(route = owner, locale: 'en' | 'es' = 'en') {
  return (
    <QueryClientProvider client={queryClient}>
      <I18nProvider configClient={null} initialLocale={locale}>
        <ComposerScopeProvider value={{ ...MAIN_COMPOSER_SCOPE, ...route }}>
          <SessionViewProvider value={tileView}>
            <WorkhorseModelPill disabled={false} />
          </SessionViewProvider>
        </ComposerScopeProvider>
      </I18nProvider>
    </QueryClientProvider>
  )
}

async function mount(route = owner) {
  const result = render(ui(route))
  await waitFor(() => expect(screen.getByTestId('workhorse-model-pill').hasAttribute('disabled')).toBe(false))

  return result
}

async function pick(model: string) {
  fireEvent.keyDown(screen.getByTestId('workhorse-model-pill'), { key: 'Enter' })
  const search = await screen.findByRole('textbox', { name: 'Search models' })
  fireEvent.change(search, { target: { value: model } })
  await screen.findByText(
    model
      .split('-')
      .map(word => word[0].toUpperCase() + word.slice(1))
      .join(' ')
  )
  fireEvent.keyDown(search, { key: 'Enter' })
  await waitFor(() => expect(screen.queryByRole('menu')).toBeNull())
}

async function openEffort() {
  fireEvent.keyDown(screen.getByTestId('workhorse-model-pill'), { key: 'Enter' })
  const trigger = await screen.findByTestId('workhorse-reasoning-menu')
  fireEvent.keyDown(trigger, { key: 'ArrowRight' })

  return screen.findByRole('switch')
}

beforeEach(() => {
  queryClient.clear()
  queryClient.setDefaultOptions({ queries: { retry: false } })
  records = new Map(
    [owner, other].map(route => [
      slot(route),
      {
        model: { default: 'parent-model', provider: 'parent-provider' },
        delegation: {
          model: 'old-model',
          provider: 'p',
          reasoning_effort: 'none',
          max_iterations: 42,
          max_concurrent_children: 2,
          ...directRoute
        }
      }
    ])
  )
  save = vi.fn().mockResolvedValue({ ok: true })
  api = vi.fn(async (request: HermesApiRequest) => {
    if (request.method === 'PUT') {
      const result = await save(request)

      if (result.ok) {
        const previous = records.get(slot(request))!
        const patch = (request.body as { config: HermesConfigRecord }).config
        records.set(slot(request), {
          ...previous,
          delegation: { ...(previous.delegation as object), ...(patch.delegation as object) }
        })
      }

      return result
    }

    if (request.path.startsWith('/api/config')) {
      return structuredClone(records.get(slot(request)))
    }

    if (request.path.startsWith('/api/model/')) {
      return catalog
    }

    return {}
  })
  Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: { api } })
  Element.prototype.scrollIntoView ??= () => undefined
  mocks.request.mockReset().mockResolvedValue(catalog)
  mocks.notifyError.mockReset()
  activate(owner)
})
afterEach(() => {
  cleanup()
  queryClient.clear()
  setApiRequestConnection(null)
  setApiRequestProfile(null)
  $connection.set(null)
  $activeGatewayProfile.set('default')
  Reflect.deleteProperty(window, 'hermesDesktop')
})

describe('workhorse owner and mutation regressions (real picker, Thinking, config hook)', () => {
  it('reads and writes the composer owner while chrome is on B; clears direct route only', async () => {
    activate(other)
    await mount()
    await pick('new-model')
    await waitFor(() => expect(puts()).toHaveLength(1))
    expect(puts()[0]).toMatchObject({
      ...owner,
      body: { config: { delegation: { model: 'new-model', provider: 'p', ...clearedRoute } } }
    })
    expect(mocks.request).toHaveBeenCalledWith(
      owner.connectionId,
      owner.profile,
      'model.options',
      expect.objectContaining({ profile: owner.profile, session_id: 'tile-a' })
    )
    await waitFor(() =>
      expect(cached()?.delegation).toMatchObject({ model: 'new-model', max_iterations: 42, max_concurrent_children: 2 })
    )
    expect(records.get(slot(owner))?.model).toEqual({ default: 'parent-model', provider: 'parent-provider' })
    expect(tileView.$model.get()).toBe('parent-model')
    expect(tileView.$reasoningEffort.get()).toBe('medium')
    expect(mocks.request.mock.calls.every(([, , method]) => method === 'model.options')).toBe(true)
    expect(records.get(slot(other))?.delegation).toMatchObject({ model: 'old-model', ...directRoute })
  })

  it('pins the cache and PUT to A when connection and composer change during save', async () => {
    const pending = deferred()
    save.mockReturnValueOnce(pending.promise)
    const view = await mount()
    await pick('new-model')
    await waitFor(() => expect(save).toHaveBeenCalledTimes(1))
    act(() => activate(other))
    view.rerender(ui(other))
    await waitFor(() => expect(cached(other)?.delegation).toMatchObject({ model: 'old-model' }))
    await act(async () => pending.resolve({ ok: true }))
    await waitFor(() => expect(cached()?.delegation).toMatchObject({ model: 'new-model' }))
    expect(cached(other)?.delegation).toMatchObject({ model: 'old-model', ...directRoute })
    fireEvent.keyDown(screen.getByTestId('workhorse-model-pill'), { key: 'Enter' })
    expect(await screen.findByText('Subagents · Old Model')).toBeTruthy()
  })

  it('reactivates Thinking from none through the real shared switch', async () => {
    await mount()
    fireEvent.click(await openEffort())
    await waitFor(() => expect(puts()).toHaveLength(1))
    expect(puts()[0]).toMatchObject({ body: { config: { delegation: { reasoning_effort: 'medium' } } } })
    await waitFor(() => expect(screen.getByRole('switch').getAttribute('aria-checked')).toBe('true'))
  })

  it('explicitly restores model/provider/route inheritance without touching effort or limits', async () => {
    await mount()
    fireEvent.keyDown(screen.getByTestId('workhorse-model-pill'), { key: 'Enter' })
    fireEvent.click(await screen.findByRole('menuitem', { name: 'Inherit parent model and route' }))
    await waitFor(() =>
      expect(cached()?.delegation).toMatchObject({
        model: '',
        provider: '',
        ...clearedRoute,
        reasoning_effort: 'none',
        max_iterations: 42,
        max_concurrent_children: 2
      })
    )
    expect(screen.getByTestId('workhorse-model-pill')).toBeTruthy()
  })

  it('restores effort inheritance independently, including when model inherits', async () => {
    records.set(slot(owner), { delegation: { reasoning_effort: 'high' } })
    render(ui())
    await waitFor(() => expect(screen.getByTestId('workhorse-model-pill').hasAttribute('disabled')).toBe(false))
    await openEffort()
    fireEvent.click(screen.getByRole('menuitem', { name: 'Inherit parent effort' }))
    await waitFor(() => expect(cached()?.delegation).toEqual({ reasoning_effort: '' }))
  })

  it('serializes repeated model choices and never paints an obsolete successful response', async () => {
    const first = deferred(),
      last = deferred()

    save.mockReturnValueOnce(first.promise).mockReturnValueOnce(last.promise)
    await mount()
    await pick('new-model')
    await waitFor(() => expect(save).toHaveBeenCalledTimes(1))
    await pick('last-model')
    expect(save).toHaveBeenCalledTimes(1)
    await act(async () => first.resolve({ ok: true }))
    await waitFor(() => expect(save).toHaveBeenCalledTimes(2))
    expect(cached()?.delegation).toMatchObject({ model: 'old-model' })
    await act(async () => last.resolve({ ok: true }))
    await waitFor(() => expect(cached()?.delegation).toMatchObject({ model: 'last-model' }))
    expect(records.get(slot(owner))?.delegation).toMatchObject({ model: 'last-model' })
  })

  it('reports a latest failed save and reconciles the last confirmed selection', async () => {
    const first = deferred(),
      last = deferred()

    save.mockReturnValueOnce(first.promise).mockReturnValueOnce(last.promise)
    await mount()
    await pick('new-model')
    await waitFor(() => expect(save).toHaveBeenCalledTimes(1))
    await pick('last-model')
    await act(async () => first.resolve({ ok: true }))
    await waitFor(() => expect(save).toHaveBeenCalledTimes(2))
    await act(async () => last.reject(new Error('disk full')))
    await waitFor(() => expect(mocks.notifyError).toHaveBeenCalledTimes(1))
    expect(cached()?.delegation).toMatchObject({ model: 'new-model' })
    expect(records.get(slot(owner))?.delegation).toMatchObject({ model: 'new-model' })
    fireEvent.keyDown(screen.getByTestId('workhorse-model-pill'), { key: 'Enter' })
    expect(await screen.findByText('Subagents · New Model')).toBeTruthy()
  })

  it('does not treat ok:false as a successful save', async () => {
    save.mockResolvedValueOnce({ ok: false })
    await mount()
    await pick('new-model')
    await waitFor(() => expect(mocks.notifyError).toHaveBeenCalledTimes(1))
    expect(cached()?.delegation).toMatchObject({ model: 'old-model' })
    await pick('new-model')
    await waitFor(() => expect(cached()?.delegation).toMatchObject({ model: 'new-model' }))
  })

  it('serializes model and effort independently without losing either field', async () => {
    const pending = deferred()
    save.mockReturnValueOnce(pending.promise)
    await mount()
    await pick('new-model')
    await waitFor(() => expect(save).toHaveBeenCalledTimes(1))
    await openEffort()
    fireEvent.click(screen.getByRole('menuitemradio', { name: 'High' }))
    expect(save).toHaveBeenCalledTimes(1)
    await act(async () => pending.resolve({ ok: true }))
    await waitFor(() => expect(cached()?.delegation).toMatchObject({ model: 'new-model', reasoning_effort: 'high' }))
    expect(records.get(slot(owner))?.delegation).toMatchObject({
      model: 'new-model',
      reasoning_effort: 'high',
      max_iterations: 42
    })
  })

  it('serializes rapid effort choices and suppresses superseded failures', async () => {
    const pending = deferred()
    save.mockReturnValueOnce(pending.promise)
    await mount()
    await openEffort()
    fireEvent.click(screen.getByRole('menuitemradio', { name: 'High' }))
    await waitFor(() => expect(save).toHaveBeenCalledTimes(1))
    fireEvent.click(screen.getByRole('menuitemradio', { name: 'Low' }))
    expect(save).toHaveBeenCalledTimes(1)
    await act(async () => pending.reject(new Error('obsolete failure')))
    await waitFor(() => expect(cached()?.delegation).toMatchObject({ reasoning_effort: 'low' }))
    expect(mocks.notifyError).not.toHaveBeenCalled()
    expect(records.get(slot(owner))?.delegation).toMatchObject({ reasoning_effort: 'low' })
  })

  it('allows another profile on the same connection to save while A is pending', async () => {
    const sibling = { ...owner, profile: 'worker-b' }
    records.set(slot(sibling), structuredClone(records.get(slot(other))!))
    const pending = deferred()
    save.mockReturnValueOnce(pending.promise)
    const view = await mount()
    await pick('new-model')
    await waitFor(() => expect(save).toHaveBeenCalledTimes(1))
    view.rerender(ui(sibling))
    await waitFor(() => expect(cached(sibling)?.delegation).toMatchObject({ model: 'old-model' }))
    await pick('last-model')
    await waitFor(() => expect(cached(sibling)?.delegation).toMatchObject({ model: 'last-model' }))
    await act(async () => pending.resolve({ ok: true }))
    await waitFor(() => expect(cached()?.delegation).toMatchObject({ model: 'new-model' }))
    expect(cached(sibling)?.delegation).toMatchObject({ model: 'last-model' })
  })

  it('keeps a failed owner catalog visible instead of falling back to another connection', async () => {
    mocks.request.mockRejectedValue(new Error('owner catalog unavailable'))
    activate(other)
    await mount()
    fireEvent.keyDown(screen.getByTestId('workhorse-model-pill'), { key: 'Enter' })
    expect(await screen.findByText('owner catalog unavailable')).toBeTruthy()
    expect(api.mock.calls.some(([r]) => r.path.startsWith('/api/model/'))).toBe(false)
    expect(puts()).toHaveLength(0)
  })

  it('uses the active owner for the main composer without an explicit scope', async () => {
    render(
      <QueryClientProvider client={queryClient}>
        <I18nProvider configClient={null} initialLocale="en">
          <WorkhorseModelPill disabled={false} />
        </I18nProvider>
      </QueryClientProvider>
    )
    await waitFor(() => expect(screen.getByTestId('workhorse-model-pill').hasAttribute('disabled')).toBe(false))
    await pick('new-model')
    await waitFor(() => expect(cached()?.delegation).toMatchObject({ model: 'new-model' }))
    expect(puts()[0]).toMatchObject(owner)
  })

  it('ignores a late config read started while the write was pending', async () => {
    const pending = deferred()
    save.mockReturnValueOnce(pending.promise)
    await mount()
    const staleRecord = structuredClone(records.get(slot(owner)))
    let releaseRead!: (record: HermesConfigRecord | undefined) => void

    const delayedRead = new Promise<HermesConfigRecord | undefined>(resolve => {
      releaseRead = resolve
    })

    await pick('new-model')
    await waitFor(() => expect(save).toHaveBeenCalledTimes(1))
    api.mockImplementationOnce(() => delayedRead)
    let refetch!: Promise<void>
    act(() => {
      refetch = queryClient.invalidateQueries({ queryKey: hermesConfigKey(owner), exact: true })
    })
    await act(async () => pending.resolve({ ok: true }))
    await waitFor(() => expect(cached()?.delegation).toMatchObject({ model: 'new-model' }))
    await act(async () => {
      releaseRead(staleRecord)
      await refetch
    })
    expect(cached()?.delegation).toMatchObject({ model: 'new-model' })
  })

  it('labels the icon accessibly in Spanish', async () => {
    render(ui(owner, 'es'))
    await waitFor(() =>
      expect(screen.getByTestId('workhorse-model-pill').getAttribute('aria-label')).toBe('Subagentes')
    )
  })
})
