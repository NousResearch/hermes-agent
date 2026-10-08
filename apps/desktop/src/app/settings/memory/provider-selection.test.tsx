import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { $apiRequestScope, type ResolvedOwner } from '@/api/client'
import { CATALOG_ORIGIN } from '@/lib/plugin-catalog'
import { $alwaysExternalLinks } from '@/store/external-links'
import { $pluginInstallRequest, closePluginInstallRequest } from '@/store/plugin-install-request'

import { hermesConfigKey } from '../../hooks/use-config-record'

import { MemoryProviderSelection } from './provider-selection'
import { field } from './test-fixtures'

let client: QueryClient

beforeEach(() => {
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  Object.defineProperty(Element.prototype, 'scrollIntoView', { configurable: true, value: vi.fn() })
})

function renderSelection(
  desktop: { api: unknown; openExternal?: unknown },
  owner: ResolvedOwner = { connectionId: null, profile: null }
) {
  vi.stubGlobal('hermesDesktop', desktop)
  $apiRequestScope.set(owner)

  return render(
    <QueryClientProvider client={client}>
      <MemoryProviderSelection />
    </QueryClientProvider>
  )
}

async function inspectProvider(name: string) {
  fireEvent.click(await screen.findByRole('combobox', { name: 'Memory Provider' }))
  fireEvent.click(await screen.findByRole('option', { name }))
}

afterEach(() => {
  cleanup()
  client.clear()
  vi.unstubAllGlobals()
  closePluginInstallRequest()
  $apiRequestScope.set({ connectionId: null, profile: null })
  $alwaysExternalLinks.set(false)
})

it('configures and connects an inactive provider before explicit Use, with owner-pinned readiness and readback', async () => {
  let active = 'builtin'
  let configured = false
  let connected = false

  const api = vi.fn(async (request: { path: string; method?: string; body?: unknown }) => {
    if (request.path.endsWith('/oauth/start')) {
      connected = true

      return { state: 'pending', connected: false }
    }

    if (request.path.endsWith('/oauth/status')) {
      return { state: connected ? 'complete' : 'idle', connected, auth: connected ? 'oauth' : null }
    }

    if (request.path.includes('/config')) {
      if (request.method === 'PUT') {
        configured = true

        return { ok: true }
      }

      return {
        name: 'fixture',
        label: 'Fixture',
        docs_url: '',
        supports_save_only: true,
        fields: [field({ key: 'workspace', label: 'Workspace', kind: 'text', group: '', inline: false })]
      }
    }

    if (request.path === '/api/memory/provider') {
      active = 'fixture'

      return { ok: true, active }
    }

    return {
      active,
      builtin_files: { memory: 0, user: 0 },
      providers: [{ name: 'fixture', configured, status: configured && connected ? 'ready' : 'needs_config' }]
    }
  })

  renderSelection({ api }, { connectionId: 'a', profile: 'alpha' })
  await inspectProvider('Fixture')
  // Not ready: the row offers a re-check, never a Use that the backend would refuse.
  expect(await screen.findByRole('button', { name: 'Retry' })).toBeTruthy()
  expect(screen.queryByRole('button', { name: 'Use provider' })).toBeNull()
  fireEvent.click(await screen.findByRole('button', { name: 'Full config…' }))
  fireEvent.change(await screen.findByRole('textbox'), { target: { value: 'alpha-space' } })
  fireEvent.click(screen.getByRole('button', { name: 'Save changes' }))
  await waitFor(() => expect(configured).toBe(true))
  expect(active).toBe('builtin')
  expect(api.mock.calls.find(([r]) => r.method === 'PUT')?.[0]).toMatchObject({
    connectionId: 'a',
    profile: 'alpha',
    body: { values: { workspace: 'alpha-space' }, activate: false }
  })
  expect(screen.queryByRole('button', { name: 'Use provider' })).toBeNull()
  fireEvent.click(await screen.findByRole('button', { name: 'Connect' }))
  expect(await screen.findByRole('button', { name: 'Use provider' })).toBeTruthy()
  expect(screen.getByText('Active: Built-in')).toBeTruthy()
  expect(api.mock.calls.some(([r]) => r.path === '/api/memory/provider')).toBe(false)
  fireEvent.click(screen.getByRole('button', { name: 'Use provider' }))
  // Once active, the row reads as settled: no status line, no Use.
  await waitFor(() => expect(screen.queryByText('Active: Built-in')).toBeNull())
  expect(screen.queryByRole('button', { name: 'Use provider' })).toBeNull()
  expect(api.mock.calls.filter(([r]) => r.path === '/api/memory/provider')).toHaveLength(1)
  expect(api.mock.calls.every(([r]) => (r as any).connectionId === 'a' && (r as any).profile === 'alpha')).toBe(true)
})

it('pins an in-flight Use and authoritative readback to A while B becomes foreground', async () => {
  let finish!: (value: unknown) => void

  const pending = new Promise(resolve => {
    finish = resolve
  })

  let selectedA = false

  const api = vi.fn((request: any) => {
    if (request.method === 'PUT') {
      return pending
    }

    if (request.path.includes('/config')) {
      return Promise.resolve({ fields: [], supports_save_only: true })
    }

    return Promise.resolve({
      active: request.profile === 'a' && selectedA ? 'ready-a' : '',
      builtin_files: { memory: 0, user: 0 },
      providers: [{ name: `ready-${request.profile}`, featured: true, configured: true, status: 'ready' }]
    })
  })

  renderSelection({ api }, { connectionId: 'a-host', profile: 'a' })
  await inspectProvider('Ready A')
  fireEvent.click(await screen.findByRole('button', { name: 'Use provider' }))
  act(() => $apiRequestScope.set({ connectionId: 'b-host', profile: 'b' }))
  await screen.findByRole('combobox', { name: 'Memory Provider' })
  await act(async () => {
    selectedA = true
    finish({ ok: true })
    await pending
  })
  await waitFor(() =>
    expect(client.getQueryData(['memory-discovery', 'a-host', 'a'])).toMatchObject({ active: 'ready-a' })
  )
  expect(screen.getByRole('combobox', { name: 'Memory Provider' }).textContent).toBe('Built-in')
  expect(api.mock.calls.filter(([request]) => request.method === 'PUT').map(([request]) => request)).toEqual([
    {
      connectionId: 'a-host',
      profile: 'a',
      priority: 'foreground',
      path: '/api/memory/provider',
      method: 'PUT',
      body: { provider: 'ready-a' }
    }
  ])
})

it('requires both mutation acceptance and the captured owner readback before reporting activation', async () => {
  const owner = { connectionId: 'a-host', profile: 'a' }
  const provider = 'ready-memory'

  for (const accepted of [true, false]) {
    const activeByProfile = { a: 'builtin', b: 'builtin' }

    const api = vi.fn(async (request: { path: string; method?: string; profile?: string | null }) => {
      if (request.method === 'PUT') {
        // Accepted-but-misrouted writes must not count as success in A. A
        // rejected response must not count even if a later read would match.
        activeByProfile[accepted ? 'b' : 'a'] = provider

        return { ok: accepted }
      }

      if (request.path.includes('/config')) {
        return { fields: [], supports_save_only: true }
      }

      return {
        active: activeByProfile[request.profile as 'a' | 'b'],
        builtin_files: { memory: 0, user: 0 },
        providers: [{ name: provider, featured: true, configured: true, status: 'ready' }]
      }
    })

    const configKey = hermesConfigKey(owner.profile, owner.connectionId)
    client.setQueryData(configKey, { memory: { provider: '' } })

    const view = renderSelection({ api }, owner)

    await inspectProvider('Ready Memory')
    fireEvent.click(await screen.findByRole('button', { name: 'Use provider' }))
    expect(await screen.findByRole('alert')).toBeTruthy()
    expect(screen.getByText('Active: Built-in')).toBeTruthy()
    expect(client.getQueryState(configKey)?.isInvalidated).toBe(false)
    expect((screen.getByRole('button', { name: 'Use provider' }) as HTMLButtonElement).disabled).toBe(false)
    expect(api.mock.calls.every(([request]) => request.profile === owner.profile)).toBe(true)

    if (accepted) {
      expect(api.mock.calls.at(-1)?.[0]).toMatchObject({ path: '/api/memory', ...owner })
      expect(activeByProfile.b).toBe(provider)
    }

    view.unmount()
    client.clear()
  }
})

it('preserves a missing active identity without loading its settings and keeps install separate from builtin recovery', async () => {
  const owner = { connectionId: 'a-host', profile: 'a' }
  const provider = 'missing-memory'
  let active = provider

  const api = vi.fn(async (request: { path: string; method?: string; body?: { provider: string } }) => {
    if (request.method === 'PUT') {
      active = request.body!.provider

      return { ok: true }
    }

    if (request.path.includes('/config')) {
      return { fields: [], supports_save_only: true }
    }

    return {
      active,
      builtin_files: { memory: 0, user: 0 },
      providers: [{ name: provider, featured: true, configured: false, status: 'missing' }],
      catalog_providers: [
        { name: provider, title: 'Recover memory', featured: true, repo: 'https://github.com/example/memory' }
      ]
    }
  })

  renderSelection({ api }, owner)
  const selector = await screen.findByRole('combobox', { name: 'Memory Provider' })
  await waitFor(() => expect(selector.textContent).toContain('Recover memory (Missing)'))
  expect(api.mock.calls.some(([request]) => request.path.includes('/providers/'))).toBe(false)
  expect(screen.queryByRole('button', { name: 'Configure' })).toBeNull()
  fireEvent.click(screen.getByRole('button', { name: 'Review & install' }))
  expect($pluginInstallRequest.get()).toMatchObject({ catalogName: provider, memory: { name: provider, owner } })
  expect(active).toBe(provider)
  expect(api.mock.calls.some(([request]) => request.method === 'PUT')).toBe(false)
  fireEvent.click(selector)
  expect(await screen.findByRole('option', { name: 'Recover memory (Missing)' })).toBeTruthy()
  fireEvent.click(screen.getByRole('option', { name: 'Built-in' }))
  fireEvent.click(screen.getByRole('button', { name: 'Use provider' }))
  await waitFor(() => expect(screen.queryByRole('button', { name: 'Use provider' })).toBeNull())
  expect(screen.queryByText(/^Active:/)).toBeNull()
  expect(api.mock.calls.filter(([request]) => request.method === 'PUT').map(([request]) => request)).toEqual([
    { ...owner, priority: 'foreground', path: '/api/memory/provider', method: 'PUT', body: { provider: '' } }
  ])
  await inspectProvider('Recover memory (Missing)')
  expect(screen.getByRole('button', { name: 'Review & install' })).toBeTruthy()
})

it.each([undefined, true])(
  'saves active provider configuration using advertised capability %s',
  async supports_save_only => {
    const provider = 'legacy-memory'

    const api = vi.fn(async (request: { path: string; method?: string }) => {
      if (request.path.includes('/config')) {
        if (request.method === 'PUT') {
          return { ok: true }
        }

        return {
          name: provider,
          label: 'Legacy memory',
          supports_save_only,
          fields: [
            field({ key: 'workspace', label: 'Workspace', kind: 'text', value: 'old-space', group: '', inline: true })
          ]
        }
      }

      return {
        active: provider,
        builtin_files: { memory: 0, user: 0 },
        providers: [{ name: provider, status: 'ready', configured: true }]
      }
    })

    renderSelection({ api }, { connectionId: 'legacy-host', profile: 'legacy' })
    const workspace = await screen.findByRole('textbox')
    fireEvent.change(workspace, { target: { value: 'new-space' } })
    fireEvent.blur(workspace)
    await waitFor(() => expect(api.mock.calls.some(([request]) => request.method === 'PUT')).toBe(true))
    expect(api.mock.calls.find(([request]) => request.method === 'PUT')?.[0]).toMatchObject({
      connectionId: 'legacy-host',
      profile: 'legacy',
      body: { values: { workspace: 'new-space' }, activate: supports_save_only !== true }
    })
    expect(screen.getByRole('combobox', { name: 'Memory Provider' }).textContent).toBe('Legacy Memory')
  }
)

it.each([undefined, false])(
  'retires stale configuration and refuses inactive save capability %s',
  async supports_save_only => {
    let release!: (value: unknown) => void

    const pending = new Promise(resolve => {
      release = resolve
    })

    const api = vi.fn((r: any) =>
      r.path.includes('/config')
        ? pending
        : Promise.resolve({
            active: 'builtin',
            builtin_files: { memory: 0, user: 0 },
            providers: [{ name: r.profile + '-memory', featured: true, configured: false, status: 'unavailable' }]
          })
    )

    renderSelection({ api }, { connectionId: null, profile: 'a' })
    await inspectProvider('A Memory')
    act(() => $apiRequestScope.set({ connectionId: 'b', profile: 'b' }))
    await act(async () => {
      release({ name: 'a-memory', label: 'A only', fields: [], supports_save_only })
      await pending
    })
    expect(screen.queryByText('A only')).toBeNull()
    await inspectProvider('B Memory')
    expect(await screen.findByText(/Configure this provider with/)).toBeTruthy()
    expect(screen.queryByRole('button', { name: 'Full config…' })).toBeNull()
    expect(api.mock.calls.every(([r]) => !r.method)).toBe(true)
  }
)

it('offers catalog installs inside the provider dropdown without changing selection, with a separated Memory marketplace action', async () => {
  const owner = { connectionId: 'remote-a', profile: 'a' }

  const entry = {
    name: 'catalog-only',
    title: 'Catalog memory',
    featured: true,
    repo: 'https://github.com/example/memory',
    subdir: 'plugin',
    sha: 'a'.repeat(40)
  }

  const api = vi.fn(async (_request: { path: string }) => ({
    active: 'builtin',
    providers: [{ name: 'installed', status: 'ready' }],
    catalog_providers: [entry, { ...entry, name: 'installed' }, { ...entry, name: 'not-eligible', featured: false }],
    builtin_files: { memory: 0, user: 0 }
  }))

  const openExternal = vi.fn()
  $alwaysExternalLinks.set(true)
  renderSelection({ api, openExternal }, owner)
  const selector = await screen.findByRole('combobox', { name: 'Memory Provider' })
  expect(screen.queryByText('Featured memory')).toBeNull()
  expect($pluginInstallRequest.get()).toBeNull()
  fireEvent.click(selector)
  // The installed id renders once, in the Installed group, even though the catalog also lists it.
  expect(screen.getAllByRole('option', { name: 'Installed' })).toHaveLength(1)
  expect(screen.getByRole('group', { name: 'Installed' })).toBeTruthy()
  expect(screen.queryByText('not-eligible')).toBeNull()
  expect(screen.getByRole('group', { name: 'Available to install' })).toBeTruthy()
  const explore = screen.getByRole('option', { name: 'Explore all…' })
  expect(explore.previousElementSibling?.getAttribute('data-slot')).toBe('select-separator')
  fireEvent.click(screen.getByRole('option', { name: 'Catalog memory' }))
  expect($pluginInstallRequest.get()).toBeNull()
  expect(screen.getByText('Installation required')).toBeTruthy()
  expect(api.mock.calls.every(([request]) => request.path === '/api/memory')).toBe(true)
  expect(selector.textContent).toBe('Catalog memory')
  expect(screen.getAllByText('Catalog memory')).toHaveLength(1)
  fireEvent.click(screen.getByRole('button', { name: 'Review & install' }))
  expect($pluginInstallRequest.get()).toMatchObject({
    catalogName: entry.name,
    repo: entry.repo + '#plugin',
    sha: entry.sha,
    enable: true,
    profile: 'a',
    legacyHint: 'agent',
    memory: { name: entry.name, owner }
  })
  expect(selector.textContent).toBe('Catalog memory')
  expect(api.mock.calls.every(([r]) => (r as any).path === '/api/memory')).toBe(true)
  fireEvent.click(selector)
  fireEvent.click(screen.getByRole('option', { name: 'Explore all…' }))
  expect(openExternal).toHaveBeenCalledWith(`${CATALOG_ORIGIN}/docs/plugins?kind=memory`)
  expect(selector.textContent).toBe('Catalog memory')
})

it('keeps the provider dropdown and marketplace available without optional catalog metadata', async () => {
  renderSelection({
    api: vi.fn(async () => ({ active: '', providers: [], builtin_files: { memory: 0, user: 0 } }))
  })
  fireEvent.click(await screen.findByRole('combobox', { name: 'Memory Provider' }))
  expect(screen.getByRole('option', { name: 'Built-in' })).toBeTruthy()
  expect(screen.getByRole('option', { name: 'Explore all…' })).toBeTruthy()
  expect($pluginInstallRequest.get()).toBeNull()
})
