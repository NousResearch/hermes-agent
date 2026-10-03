import { backendScopeKey } from '@hermes/shared'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import type { HermesApiRequest } from '@/global'

import type { BrowserProfileRetirement, BrowserWorkspace } from '../../../electron/browser-workspace-types'
import { BrowserWorkspaces } from '../../../electron/browser-workspaces'

const storageKey = 'hermes.desktop.previewTabs.v2'

const tab = (id: string) => ({
  id: `url:${id}` as const,
  pinned: false,
  target: { kind: 'url' as const, label: id, source: `https://example.test/${id}`, url: `https://example.test/${id}` }
})

function deferred<T>() {
  let resolve!: (value: T) => void
  const promise = new Promise<T>(done => { resolve = done })

  return { promise, resolve }
}

beforeEach(() => {
  vi.resetModules()
  window.localStorage.clear()
  window.history.replaceState({}, '', '/')
})

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
  delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop
  window.localStorage.clear()
})

// No rename/migration/store mocks: the dialog calls the real API helper and
// real renderer retirement, whose preload boundary delegates to main's owner.
// Only the backend, preload transport and active socket/teardown are fixtures.
it.each([
  { connectionId: 'remote', ambient: 'remote', nextAmbient: null, from: 'conn:remote::alpha', to: 'conn:remote::renamed', controlScope: 'alpha' },
  { connectionId: 'local', ambient: null, nextAmbient: 'remote', from: 'alpha', to: 'renamed', controlScope: 'conn:remote::alpha' }
])('renames ambient $connectionId ownership without retiring the same-named other source', async scenario => {
  const sessions = await import('@/store/session')
  const states = await import('@/store/session-states')
  const preview = await import('@/store/preview')
  const workspaces = await import('@/store/browser-workspaces')
  const gateway = await import('@/store/gateway')
  const client = await import('@/api/client')
  const { getProfiles } = await import('@/api/profiles')
  let activeConnectionId: null | string = scenario.ambient
  const activeConnection = vi.spyOn(gateway, 'activeGatewayConnectionId').mockImplementation(() => activeConnectionId)
  const order: string[] = []
  const retireLocal = vi.spyOn(gateway, 'retireLocalProfileGateways').mockImplementation(() => { order.push('local-gateways') })
  const { RenameProfileDialog } = await import('./rename-profile-dialog')
  const queued: BrowserWorkspace[] = []
  const runtime = new BrowserWorkspaces((_recipients, state) => queued.push(state))

  function open(connectionId: string, renderer: number) {
    const conversation = { kind: 'session' as const, id: `${connectionId}-chat`, connectionId, profile: 'alpha' }

    const state = runtime.open(1, {
      tab: tab(connectionId),
      scope: backendScopeKey(connectionId, 'alpha'),
      destination: { kind: 'composer', surfaceId: conversation.id, target: conversation.id, windowId: 'chat', conversation }
    }, conversation)

    runtime.attach(state.id, renderer)

    return runtime.snapshots(renderer)[0]!
  }

  const local = open('local', 2)
  const remote = open('remote', 3)
  const isLocal = scenario.connectionId === 'local'
  const target = isLocal ? local : remote
  const control = isLocal ? remote : local
  const renderer = isLocal ? 2 : 3
  const controlRenderer = isLocal ? 3 : 2
  expect(target.owner.scope).toBe(scenario.from)
  expect(control.owner.scope).toBe(scenario.controlScope)
  sessions.setSessionOwnerHint('local-chat', { connectionId: 'local', profile: 'alpha' })
  sessions.setSessionOwnerHint('remote-chat', { connectionId: 'remote', profile: 'alpha' })
  const localHint = sessions.getSessionOwnerHint('local-chat')
  const remoteHint = sessions.getSessionOwnerHint('remote-chat')
  // The local control also proves the existing local tile migration still runs.
  states.$sessionTiles.set([
    { storedSessionId: 'local-chat', ownerRoute: { connectionId: 'local', profile: 'alpha' } },
    { storedSessionId: 'remote-chat', ownerRoute: { connectionId: 'remote', profile: 'alpha' } }
  ])
  preview.setPreviewScope(scenario.from)
  workspaces.receiveBrowserWorkspace(local)
  workspaces.receiveBrowserWorkspace(remote)
  expect(JSON.parse(window.localStorage.getItem(storageKey)!)).toEqual({
    alpha: [tab('local')], 'conn:remote::alpha': [tab('remote')]
  })

  queued.length = 0
  const oldTarget = { windowId: target.id, tabId: target.activeTabId!, owner: target.owner, selectionVersion: target.selectionVersion }

  // A real page publication queued before the rename must not revive alpha.
  const latePage = runtime.command(renderer, target.id, {
    kind: 'page', tabId: oldTarget.tabId, url: 'https://example.test/latest', title: 'latest'
  })!

  const latestTab = { ...tab(scenario.connectionId), target: { ...tab(scenario.connectionId).target, url: 'https://example.test/latest', label: 'latest' } }
  const patch = deferred<{ name: string; ok: boolean; path: string }>()
  const retirement = deferred<void>()
  const retirementStarted = deferred<void>()
  const refreshCompleted = deferred<void>()

  const api = vi.fn((request: HermesApiRequest) => {
    if (request.method === 'PATCH' && request.path === '/api/profiles/alpha') {
      order.push('patch')

      return patch.promise
    }

    if (request.path === '/api/profiles' && !request.method) {
      order.push('refresh')

      return Promise.resolve({ profiles: [] })
    }

    throw new Error(`Unexpected fixture API request: ${request.method} ${request.path}`)
  })

  const retireProfile = vi.fn(async (change: BrowserProfileRetirement) => {
    order.push('browser-retirement')
    const snapshots = runtime.retireProfile(change)
    retirementStarted.resolve(undefined)
    await retirement.promise

    return snapshots
  })

  const acknowledge = vi.fn((id: string, revision: number) => runtime.acknowledge(1, id, revision))
  Object.assign(window, { hermesDesktop: { api, browserWorkspace: { retireProfile, acknowledge } } })
  client.setApiRequestConnection(activeConnectionId)
  client.setApiRequestProfile('alpha')
  const onClose = vi.fn()

  const onRenamed = vi.fn(async (_name: string) => {
    await getProfiles()
    refreshCompleted.resolve(undefined)
  })

  const view = render(<RenameProfileDialog currentName="alpha" onClose={onClose} onRenamed={onRenamed} open />)
  activeConnection.mockClear()
  fireEvent.change(screen.getByLabelText(/new name/i), { target: { value: 'renamed' } })
  fireEvent.click(screen.getByRole('button', { name: /^rename$/i }))

  try {
    await waitFor(() => expect(api).toHaveBeenCalledOnce())
    // Local stays ambient/untagged: pinning it to 'local' would bypass legacy
    // per-profile remote overrides. Remote must carry its registry source.
    expect(api.mock.calls[0]![0]).toEqual({
      ...(isLocal ? {} : { connectionId: 'remote' }),
      profile: 'alpha', path: '/api/profiles/alpha', method: 'PATCH', body: { new_name: 'renamed' }
    })
    expect(retireProfile).not.toHaveBeenCalled()
    expect(onRenamed).not.toHaveBeenCalled()

    if (isLocal) {
      expect(retireLocal).toHaveBeenCalledExactlyOnceWith('alpha')
      expect(activeConnection.mock.invocationCallOrder[0]).toBeLessThan(retireLocal.mock.invocationCallOrder[0]!)
      expect(order).toEqual(['local-gateways', 'patch'])
    }

    // Foreground moves while PATCH is pending. Only the originally captured
    // owner may retire; re-reading the active source after the await is wrong.
    activeConnectionId = scenario.nextAmbient
    client.setApiRequestConnection(activeConnectionId)
    await act(async () => {
      patch.resolve({ name: 'renamed', ok: true, path: '/fixture/renamed' })
      // Arrival is the barrier; a wrong owner is a settled result, not a retry.
      await retirementStarted.promise
    })
    expect(retireProfile).toHaveBeenCalledExactlyOnceWith({
      connectionId: scenario.connectionId, profile: 'alpha', replacementProfile: 'renamed'
    })
    expect(activeConnection).toHaveBeenCalledOnce()

    if (!isLocal) { expect(retireLocal).not.toHaveBeenCalled() }
    expect(runtime.ownerForRenderer(renderer)).toBeNull()
    expect(runtime.snapshots(controlRenderer)).toEqual([control])
    expect(onRenamed).not.toHaveBeenCalled()
    expect(onClose).not.toHaveBeenCalled()
    expect(api).toHaveBeenCalledOnce()
    expect(screen.queryByText('Profile renamed')).toBeNull()
    expect(JSON.parse(window.localStorage.getItem(storageKey)!)[scenario.to]).toBeUndefined()
  } finally {
    // Drain both fixture gates even when an ownership/order assertion fails.
    // The real migration reconciles main's snapshots and the real API refresh
    // finishes before cleanup can reset modules or remove the preload bridge.
    await act(async () => {
      patch.resolve({ name: 'renamed', ok: true, path: '/fixture/renamed' })
      retirement.resolve(undefined)
      await refreshCompleted.promise
    })
  }

  await waitFor(() => expect(screen.getByText('Profile renamed')).toBeTruthy())
  expect(onRenamed).toHaveBeenCalledExactlyOnceWith('renamed')
  expect(order).toEqual([...(isLocal ? ['local-gateways'] : []), 'patch', 'browser-retirement', 'refresh'])
  expect(api).toHaveBeenCalledTimes(2)
  expect(acknowledge).toHaveBeenCalledWith(target.id, expect.any(Number))
  expect(states.$sessionTiles.get().map(tile => tile.ownerRoute)).toEqual([
    { connectionId: 'local', profile: isLocal ? 'renamed' : 'alpha' },
    { connectionId: 'remote', profile: 'alpha' }
  ])
  expect(sessions.getSessionOwnerHint('local-chat')).toEqual(isLocal ? { ...localHint, profile: 'renamed' } : localHint)
  expect(sessions.getSessionOwnerHint('remote-chat')).toEqual(remoteHint)

  for (const state of queued) { workspaces.receiveBrowserWorkspace(state) }
  workspaces.receiveBrowserWorkspace(latePage)
  runtime.close(target.id)
  expect(runtime.command(renderer, target.id, { kind: 'page', tabId: oldTarget.tabId, url: 'https://example.test/late', title: 'late' })).toBeNull()
  expect(runtime.relay(1, { id: 'late', kind: 'act', target: oldTarget, requester: target.owner.conversation, payload: { kind: 'click' } })).toBeNull()
  expect(preview.$poppedBrowserTabIds.get().has(oldTarget.tabId)).toBe(false)
  expect(runtime.ownerForRenderer(controlRenderer)).toEqual(control.owner)
  const expectedStored = { [scenario.to]: [latestTab], [scenario.controlScope]: [tab(isLocal ? 'remote' : 'local')] }
  expect(JSON.parse(window.localStorage.getItem(storageKey)!)).toEqual(expectedStored)

  view.unmount()
  vi.resetModules()
  const reloaded = await import('@/store/preview')
  const reloadedWorkspaces = await import('@/store/browser-workspaces')

  for (const state of runtime.snapshots(1)) { reloadedWorkspaces.receiveBrowserWorkspace(state) }
  reloaded.setPreviewScope(scenario.to)
  expect(reloaded.$previewTabs.get()).toEqual([latestTab])
  reloaded.setPreviewScope(scenario.from)
  expect(reloaded.$previewTabs.get()).toEqual([])
  expect(JSON.parse(window.localStorage.getItem(storageKey)!)).toEqual(expectedStored)
  expect(runtime.snapshots(controlRenderer)).toEqual([control])
})
