import http from 'node:http'
import type { AddressInfo } from 'node:net'

import { QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { atom } from 'nanostores'
import { createRef } from 'react'
import { MemoryRouter } from 'react-router'
import { afterEach, expect, it, vi } from 'vitest'

// Keep config hooks, settings scope, switch hook and API routing real; stub
// only the transport and unrelated shell surfaces. No live backend is used.
vi.mock('@/store/profile', () => ({
  $activeGatewayProfile: atom('default'),
  normalizeProfileKey: (value?: string | null) => value?.trim() || 'default'
}))
vi.mock('@/store/projects', () => ({
  repoDiscoveryPolicyFromConfig: () => ({}),
  repoDiscoveryPolicySignature: () => '',
  scanAndRecordRepos: vi.fn()
}))
vi.mock('./profile-scope', () => ({ SettingsProfileScope: () => null }))

import { profileScopeKey, setApiRequestProfile, setApiRequestConnection as setRequestConnection } from '@/api/client'
import { saveHermesConfig } from '@/api/config'
import type { HermesApiRequest } from '@/global'
import { invalidateProfileScopedQueries, queryClient } from '@/lib/query-client'
import { $notifications } from '@/store/notifications'
import { $connection } from '@/store/session'
import { $settingsOwner, $settingsScopeOverride } from '@/store/settings-scope'

import { hermesConfigKey } from '../hooks/use-config-record'

import { ConfigSettings } from './config-settings'

function modelResponse(path: string) {
  if (path === '/api/model/info') {
    return { provider: 'fixture', model: 'fixture-model' }
  }

  if (path.startsWith('/api/model/options')) {
    return { providers: [{ slug: 'fixture', name: 'Fixture', models: ['fixture-model'] }] }
  }

  if (path === '/api/model/auxiliary') {
    return { tasks: [] }
  }

  if (path === '/api/model/moa') {
    return null
  }

  if (path === '/api/model/set') {
    return { ok: true, provider: 'fixture', model: 'fixture-model' }
  }

  return { available: false }
}

function deferred<T>() {
  let resolve!: (value: T) => void
  let reject!: (reason: Error) => void

  const promise = new Promise<T>((yes, no) => {
    resolve = yes
    reject = no
  })

  return { promise, resolve, reject }
}

async function flush() {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(25)
  })
}

function page(section = 'model') {
  return (
    <MemoryRouter>
      <QueryClientProvider client={queryClient}>
        <ConfigSettings activeSectionId={section} importInputRef={createRef<HTMLInputElement>()} />
      </QueryClientProvider>
    </MemoryRouter>
  )
}

function setApiRequestConnection(connectionId: string) {
  setRequestConnection(connectionId)
  $connection.set({ ...$connection.get(), connectionId } as NonNullable<ReturnType<typeof $connection.get>>)
}

function setup() {
  vi.useFakeTimers()
  queryClient.clear()
  queryClient.setDefaultOptions({ queries: { retry: false, refetchOnWindowFocus: false } })
  setApiRequestProfile('default')
  $settingsScopeOverride.set(null)
}

afterEach(() => {
  cleanup()
  queryClient.clear()
  vi.useRealTimers()
  vi.unstubAllGlobals()
})

it('unknown and missing descriptors never send Settings requests to local', async () => {
  setup()
  const api = vi.fn(async (_request: HermesApiRequest) => ({}))
  vi.stubGlobal('hermesDesktop', { api })

  for (const connection of [
    null,
    { source: 'env', baseUrl: 'https://legacy.invalid' },
    { mode: 'remote' },
    { mode: 'remote', connectionId: null, baseUrl: 'https://legacy.invalid' },
    { mode: 'local', connectionId: ' ' }
  ]) {
    $connection.set(connection as ReturnType<typeof $connection.get>)
    const view = render(page())
    await flush()
    expect.soft(api).not.toHaveBeenCalled()
    view.unmount()
    api.mockClear()
  }

  // Once Electron publishes an authoritative owner, Settings recovers.
  $connection.set({ mode: 'remote', connectionId: 'registered-env' } as ReturnType<typeof $connection.get>)
  render(page())
  await flush()
  expect(api.mock.calls.length).toBeGreaterThan(0)
  expect(api.mock.calls.every(([request]) => request.connectionId === 'registered-env')).toBe(true)
})

it('keeps a pending autosave mounted across an equivalent legacy reconnect', async () => {
  setup()
  setRequestConnection('')
  const writes: HermesApiRequest[] = []

  const descriptor = {
    mode: 'remote',
    source: 'settings',
    baseUrl: 'https://legacy.invalid',
    token: 'fixture',
    authMode: 'token',
    headers: { 'Cf-Access-Client-Id': 'fixture-client' },
    isFullscreen: false,
    logs: [],
    nativeOverlayWidth: 0,
    windowButtonPosition: null,
    wsUrl: 'wss://legacy.invalid/api/ws'
  } satisfies NonNullable<ReturnType<typeof $connection.get>>

  const api = vi.fn(async (request: HermesApiRequest) => {
    if (request.method === 'PUT') {
      writes.push(request)

      return { ok: true }
    }

    if (request.path === '/api/config') {
      return { checkpoints: { enabled: false } }
    }

    if (request.path === '/api/config/schema') {
      return { fields: {} }
    }

    return { available: false }
  })

  vi.stubGlobal('hermesDesktop', { api })
  $connection.set(descriptor)
  render(page('safety'))
  await flush()
  await flush()
  fireEvent.click(screen.getByRole('switch'))
  await act(async () => {
    $connection.set({ ...descriptor, headers: { ...descriptor.headers } })
    await vi.advanceTimersByTimeAsync(600)
  })

  expect(writes).toHaveLength(1)
  expect(writes[0].legacyConnection?.headers).toEqual(descriptor.headers)
})

it('the production pinned model child can restart its exact owner after code skew', async () => {
  setup()
  const owner = { connectionId: 'local', profile: 'default' }
  const recycleBackend = vi.fn(async () => ({ ok: true }))

  const api = vi.fn(async (request: HermesApiRequest) => {
    if (request.path === '/api/config') {
      return {}
    }

    if (request.path === '/api/config/schema') {
      return { fields: {} }
    }

    if (request.path === '/api/model/info' && !recycleBackend.mock.calls.length) {
      throw new Error('503: Restart required: stale module')
    }

    return modelResponse(request.path)
  })

  vi.stubGlobal('hermesDesktop', { api, recycleBackend })
  setApiRequestConnection('local')
  render(page())
  await flush()
  await flush()
  const restart = screen.getByRole('button', { name: /Restart backend/i })
  expect.soft((restart as HTMLButtonElement).disabled).toBe(false)
  fireEvent.click(restart)
  await flush()
  expect(recycleBackend).toHaveBeenCalledWith(owner)
})

it('same-name source switches isolate an unavailable owner’s late config error in both directions', async () => {
  for (const [unavailable, healthy] of [
    ['cloud-test', 'local'],
    ['local', 'cloud-test']
  ]) {
    for (const blockedPath of ['/api/config', '/api/config/schema', '/api/model/info']) {
      for (const override of [null, 'other']) {
        setup()
        $settingsScopeOverride.set(override)
        const old = deferred<unknown>()

        const api = vi.fn((request: { path: string; connectionId?: string }) => {
          if (request.path === blockedPath && request.connectionId === unavailable) {
            return old.promise
          }

          if (request.path === '/api/config') {
            return Promise.resolve({})
          }

          if (request.path === '/api/config/schema') {
            return Promise.resolve({ fields: {} })
          }

          return Promise.resolve(modelResponse(request.path))
        })

        vi.stubGlobal('hermesDesktop', { api })
        setApiRequestConnection(unavailable)
        const view = render(page())
        await flush()
        await act(async () => {
          setApiRequestConnection(healthy)
          invalidateProfileScopedQueries()
          view.rerender(page())
        })
        await flush()
        await act(async () => {
          old.reject(new Error('owner unavailable'))
        })
        await flush()
        expect
          .soft(
            api.mock.calls.some(([r]) => r.path === blockedPath && r.connectionId === healthy),
            `${unavailable} -> ${healthy}, scope ${override}: fetch new owner`
          )
          .toBe(true)
        expect
          .soft(
            screen.queryByRole('button', { name: 'Apply' }),
            `${unavailable} -> ${healthy}: old error must not block model page`
          )
          .not.toBeNull()
        fireEvent.click(screen.getByRole('button', { name: 'Apply' }))
        await flush()

        for (const path of [
          '/api/model/info',
          '/api/model/options?explicit_only=1',
          '/api/model/auxiliary',
          '/api/model/moa',
          '/api/model/set',
          '/api/audio/elevenlabs/voices'
        ]) {
          expect(api.mock.calls.some(([r]) => r.path === path && r.connectionId === healthy)).toBe(true)
        }

        cleanup()
        queryClient.clear()
        // Control: the same healthy transport mounts normally without the old flight.
        render(page())
        await flush()
        await flush()
        expect(screen.queryByRole('button', { name: 'Apply' })).not.toBeNull()
        cleanup()
        queryClient.clear()
      }
    }
  }
})

it.each([
  ['local', 'cloud-test'],
  ['legacy', 'cloud-test'],
  ['legacy', 'local'],
  ['legacy', 'legacy-replaced']
])('queued autosaves never move %s drafts to %s, even after remount', async (origin, next) => {
  setup()
  const firstSave = deferred<{ ok: boolean }>()
  const secondSave = deferred<{ ok: boolean }>()
  const writes: HermesApiRequest[] = []

  const api = vi.fn((request: HermesApiRequest) => {
    if (request.method === 'PUT') {
      writes.push(request)

      return writes.length === 1 ? firstSave.promise : secondSave.promise
    }

    if (request.path === '/api/config') {
      return Promise.resolve({
        owner: request.legacyConnection?.baseUrl ?? request.connectionId,
        checkpoints: { enabled: false }
      })
    }

    if (request.path === '/api/config/schema') {
      return Promise.resolve({ fields: {} })
    }

    return Promise.resolve({ available: false })
  })

  vi.stubGlobal('hermesDesktop', { api })
  setApiRequestConnection(origin)

  if (origin === 'legacy') {
    $connection.set({
      mode: 'remote',
      source: 'env',
      baseUrl: 'https://legacy.invalid',
      token: 'fixture'
    } as ReturnType<typeof $connection.get>)
  }

  const owner = $settingsOwner.get()!
  const ownerKey = profileScopeKey(owner)
  const view = render(page('safety'))
  await flush()
  await flush()
  fireEvent.click(screen.getByRole('switch'))
  await act(async () => {
    await vi.advanceTimersByTimeAsync(600)
  })
  expect(writes).toHaveLength(1)
  fireEvent.click(screen.getByRole('switch'))
  await act(async () => {
    await vi.advanceTimersByTimeAsync(600)
  })
  // Force remount too: cache keys + remount alone cannot revoke a queued save.
  view.unmount()
  setApiRequestConnection(next)

  if (next === 'legacy-replaced') {
    $connection.set({
      mode: 'remote',
      source: 'settings',
      baseUrl: 'https://replacement.invalid',
      token: 'replacement'
    } as ReturnType<typeof $connection.get>)
  }

  render(page('safety'))
  await flush()
  await act(async () => {
    firstSave.resolve({ ok: true })
  })
  await flush()
  expect.soft(writes).toHaveLength(1)
  expect.soft(writes.filter(write => write.connectionId !== (origin === 'legacy' ? undefined : origin))).toEqual([])
  expect
    .soft(queryClient.getQueryData(hermesConfigKey($settingsOwner.get()!)))
    .toMatchObject({ owner: next === 'legacy-replaced' ? 'https://replacement.invalid' : next })
  secondSave.resolve({ ok: true })
  await flush()
  // The real API helper also pins a callback invoked after unmount/foreground switch.
  await saveHermesConfig({ checkpoints: { enabled: true } }, owner)
  expect(writes.at(-1)?.connectionId).toBe(origin === 'legacy' ? undefined : origin)
  expect(profileScopeKey(owner)).toBe(ownerKey)

  if (owner.legacyConnection) {
    expect(profileScopeKey({ ...owner, legacyConnection: { ...owner.legacyConnection } })).not.toBe(ownerKey)
    expect(writes.at(-1)).toMatchObject({ legacyConnection: owner.legacyConnection })
  }
})

it('model confirmation keeps its originating HTTP owner after switching hosts', async () => {
  setup()
  vi.useRealTimers()
  $notifications.set([])
  const received: Array<{ owner: string; profile: string | null; body: Record<string, unknown> }> = []
  const servers = new Map<string, http.Server>()
  const urls = new Map<string, string>()

  try {
    for (const owner of ['local', 'cloud-test']) {
      const server = http.createServer(async (request, response) => {
        const url = new URL(request.url!, 'http://fixture')
        let body = ''

        for await (const chunk of request) {
          body += chunk
        }

        let result: unknown = modelResponse(url.pathname)

        if (url.pathname === '/api/config') {
          result = {}
        }

        if (url.pathname === '/api/config/schema') {
          result = { fields: {} }
        }

        if (url.pathname === '/api/model/set') {
          const assignment = JSON.parse(body)
          received.push({ owner, profile: url.searchParams.get('profile'), body: assignment })
          result = assignment.confirm_expensive_model
            ? modelResponse(url.pathname)
            : { confirm_required: true, confirm_message: 'Fixture model needs approval' }
        }

        response.setHeader('Content-Type', 'application/json')
        response.end(JSON.stringify(result))
      })

      servers.set(owner, server)
      await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve))
      urls.set(owner, `http://127.0.0.1:${(server.address() as AddressInfo).port}`)
    }

    // Substitute only IPC: real renderer API helpers feed two real HTTP
    // origins. Assertions below are server-side receipts, not mock calls.
    vi.stubGlobal('hermesDesktop', {
      api: (request: HermesApiRequest) =>
        new Promise((resolve, reject) => {
          const owner = request.connectionId

          if (!owner || !urls.has(owner)) {
            reject(new Error('Missing owner pin'))

            return
          }

          const url = new URL(request.path, urls.get(owner))

          if (request.profile) {
            url.searchParams.set('profile', request.profile)
          }

          const call = http.request(url, { method: request.method ?? 'GET' }, response => {
            let body = ''
            response.on('data', chunk => {
              body += chunk
            })
            response.on('end', () => resolve(JSON.parse(body)))
          })

          call.on('error', reject)
          call.end(request.body ? JSON.stringify(request.body) : undefined)
        })
    })
    setApiRequestConnection('local')
    $settingsScopeOverride.set('other')
    render(page())
    fireEvent.click(await screen.findByRole('button', { name: 'Apply' }))
    await waitFor(() => expect($notifications.get().some(n => n.message === 'Fixture model needs approval')).toBe(true))
    const approve = $notifications.get().find(n => n.message === 'Fixture model needs approval')!.action!
    await act(async () => {
      setApiRequestConnection('cloud-test')
    })
    await screen.findByRole('button', { name: 'Apply' })
    await act(async () => {
      approve.onClick()
    })
    await waitFor(() => expect(received).toHaveLength(2))
    expect(received.map(({ owner, profile }) => ({ owner, profile }))).toEqual([
      { owner: 'local', profile: 'other' },
      { owner: 'local', profile: 'other' }
    ])
    expect(received[1].body).toMatchObject({ scope: 'main', confirm_expensive_model: true })
    expect(queryClient.getQueryData(hermesConfigKey({ connectionId: 'cloud-test', profile: 'other' }))).toEqual({})
  } finally {
    cleanup()

    for (const server of servers.values()) {
      server.closeAllConnections()
      await new Promise<void>(resolve => server.close(() => resolve()))
    }
  }
})
