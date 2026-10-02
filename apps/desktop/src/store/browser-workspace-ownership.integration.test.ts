import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import type { BrowserProfileRetirement, BrowserRequestTarget, BrowserWorkspace, BrowserWorkspaceOpen } from '../../electron/browser-workspace-types'
import { browserWorkspaceRoute, BrowserWorkspaces } from '../../electron/browser-workspaces'


const key = 'hermes.desktop.previewTabs.v2'

const tab = (id: string, url = `https://example.test/${id}`) => ({
  id: `url:${id}` as const, pinned: false, target: { kind: 'url' as const, label: id, source: url, url }
})

const route = { connectionId: 'route-B', profile: 'default' }

const registry = {
  connections: [{ id: 'route-A', kind: 'ssh' }, { id: 'route-B', kind: 'ssh' }], primary: 'route-A'
}

beforeEach(() => {
  vi.resetModules()
  window.localStorage.clear()
  window.history.replaceState({}, '', '/')
  document.body.replaceChildren()
})
afterEach(() => {
  document.body.replaceChildren()
  delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop
})

async function fixture() {
  const sessions = await import('./session')
  const states = await import('./session-states')
  const preview = await import('./preview')
  const workspaces = await import('./browser-workspaces')
  const handoff = await import('@/lib/preview-annotate/handoff')
  sessions.setSessionOwnerHint('stored-A', route)
  sessions.setSessionOwnerHint('stored-B', route)
  const composer = document.createElement('div')
  composer.dataset.composerTarget = 'main'
  composer.dataset.composerSurfaceId = 'same-mounted-primary'
  composer.dataset.browserSessionId = 'stored-A'
  document.body.append(composer)
  sessions.$activeSessionId.set('runtime-A')
  sessions.$selectedStoredSessionId.set('stored-A')
  preview.setPreviewScope('conn:route-B::default')
  const runtime = new BrowserWorkspaces(() => {})
  const request = { tab: { ...tab('seed'), sessionId: 'stored-A' }, scope: 'conn:route-B::default', destination: handoff.capturePreviewAnnotateDestination() }
  const state = runtime.open(1, request, browserWorkspaceRoute(request, registry)!)
  runtime.attach(state.id, 2)
  runtime.command(2, state.id, { kind: 'new' })
  workspaces.receiveBrowserWorkspace(runtime.snapshots(1)[0]!)
  const listeners = new Set<(packet: unknown) => void>()
  const dispatched: BrowserRequestTarget[] = []

  const send = vi.fn((value: unknown) => {
    const packet = value as { id: string; kind: string; target: BrowserRequestTarget }

    if (runtime.relay(1, packet) === 2) {
      dispatched.push(packet.target)
      const reply = { id: packet.id, kind: packet.kind, target: packet.target, result: { success: true, text: 'fixture page' } }
      expect(runtime.relay(2, reply)).toBe(1)

      for (const listener of listeners) {listener(reply)}
    }
  })

  Object.assign(window, { hermesDesktop: {
    browserWorkspace: { retireSession: (id: string) => runtime.retireSession(1, id) },
    windowRelay: { send, onMessage: (listener: (packet: unknown) => void) => {
      listeners.add(listener)

      return () => listeners.delete(listener)
    } }
  } })
  const { handleServerRequest } = await import('@/app/session/hooks/use-message-stream/gateway-event/server-requests')

  async function dispatch(session: string, method = 'preview.act', connectionId = route.connectionId) {
    const respond = vi.fn()
    const request = { id: crypto.randomUUID(), method, connectionId, profile: 'default', params: { session_id: session, action: 'click' }, respond, decline: vi.fn(), fail: vi.fn() }
    const deps = { activeSessionIdRef: { current: sessions.$activeSessionId.get() }, sessionStateByRuntimeIdRef: { current: new Map([['runtime-A', { storedSessionId: 'stored-A' }], ['runtime-B', { storedSessionId: 'stored-B' }]]) }, sessionInterrupted: () => false, updateSessionState: vi.fn(), upsertToolCall: vi.fn() }
    handleServerRequest(request as never, deps as never, sessions.$activeSessionId.get())
    await vi.waitFor(() => expect(respond).toHaveBeenCalledOnce())

    return respond
  }

  return { sessions, states, preview, workspaces, handoff, composer, runtime, state, dispatch, dispatched, listeners }
}

it('binds dispatch to durable session and source: A → B → A, hosted background read, retirement', async () => {
  const f = await fixture()
  await f.dispatch('runtime-A')
  expect(f.dispatched).toHaveLength(1)
  expect(f.dispatched[0]!.tabId).not.toBe('url:seed')
  f.composer.dataset.browserSessionId = 'stored-B'
  f.sessions.$selectedStoredSessionId.set('stored-B')
  f.sessions.$activeSessionId.set('runtime-B')
  expect(f.workspaces.selectedPopoutTarget()).toBeNull()
  await f.dispatch('runtime-B')
  await f.dispatch('runtime-B', 'preview.read')
  expect(f.dispatched).toHaveLength(1)
  f.states.$sessionTiles.set([{ storedSessionId: 'stored-A', runtimeId: 'runtime-A', ownerRoute: route }] as never)
  // A hosted background tile may read ITS detached page, never B's foreground.
  await f.dispatch('runtime-A', 'preview.read')
  expect(f.dispatched).toHaveLength(2)
  await f.dispatch('runtime-A')
  expect(f.dispatched).toHaveLength(2)
  f.composer.dataset.browserSessionId = 'stored-A'
  f.sessions.$selectedStoredSessionId.set('stored-A')
  f.sessions.$activeSessionId.set('runtime-A')
  await f.dispatch('runtime-A', 'preview.act', 'route-A')
  expect(f.dispatched).toHaveLength(2)
  await f.dispatch('runtime-A')
  expect(f.dispatched).toHaveLength(3)
  const { retireBrowserSession } = await import('./browser-conversation')
  retireBrowserSession('stored-A')
  await f.dispatch('runtime-A')
  await f.dispatch('runtime-A', 'preview.read')
  expect(f.dispatched).toHaveLength(3)
  expect(f.runtime.snapshots(1)[0]!.closed).toBe(true)
})

it('sends the actual in-place chat route through open and main authority, not primary A', async () => {
  const f = await fixture()

  const open = vi.fn((_id: string, request: BrowserWorkspaceOpen) => {
    const actualRoute = browserWorkspaceRoute(request, registry)!
    const state = f.runtime.open(1, request, actualRoute)
    f.runtime.attach(state.id, 3)


    return Promise.resolve({ ok: true })
  })

  Object.assign(window.hermesDesktop!, { openBrowserWindow: open })
  f.preview.newBrowserTab()
  const id = f.preview.$previewTabs.get().at(-1)!.id
  f.preview.popOutBrowserTab(id)
  await vi.waitFor(() => expect(open).toHaveBeenCalledOnce())
  expect(open.mock.calls[0]![1].destination?.conversation).toEqual({ kind: 'session', id: 'stored-A', ...route })
  expect(f.runtime.ownerForRenderer(3)).toMatchObject({ ...route, registryScoped: true })
  expect(browserWorkspaceRoute(open.mock.calls[0]![1], { ...registry, connections: [] })).toBeNull()
})

it('delivers comments only to the exact durable recipient despite a reused primary surface', async () => {
  const f = await fixture()
  const insert = vi.fn()
  window.addEventListener('hermes:composer-insert', insert)

  try {
    // focus.ts is loaded by production imports and subscribes to the real handoff.
    await import('@/app/chat/composer/focus')
    const current = f.runtime.snapshots(1)[0]!
    const request = { type: 'preview-annotate-handoff', requestId: 'comment-1', tabId: current.activeTabId, destination: current.owner.destination, images: [], prompt: 'fixture comment', count: 1 }
    expect(f.runtime.relayComment(2, request)).toBe(1)
    f.composer.dataset.browserSessionId = 'stored-B'

    for (const listener of f.listeners) {listener(request)}
    await Promise.resolve()
    expect(insert).not.toHaveBeenCalled()
    f.composer.dataset.browserSessionId = 'stored-A'

    for (const listener of f.listeners) {listener({ ...request, requestId: 'comment-2' })}
    // Delivery is synchronous after validation: no timer may insert into B.
    expect(insert).toHaveBeenCalledOnce()
    f.composer.dataset.browserSessionId = 'stored-B'
    await Promise.resolve()
    expect(insert).toHaveBeenCalledOnce()
  } finally {window.removeEventListener('hermes:composer-insert', insert)}
})

it('reconciles another window’s final-tab/profile deletion before page and selection snapshots and reload', async () => {
  window.localStorage.setItem(key, JSON.stringify({ alpha: [tab('a')], beta: [tab('b')] }))
  const { setPreviewScope, $previewTabs } = await import('./preview')
  const { receiveBrowserWorkspace } = await import('./browser-workspaces')
  setPreviewScope('alpha')
  $previewTabs.set([...$previewTabs.get(), { id: 'artifact:transient', target: { kind: 'artifact', label: 'runtime', source: '', url: 'transient' } }])
  window.localStorage.setItem(key, JSON.stringify({ alpha: [tab('a')] }))
  const state = { id: 'alpha-window', revision: 1, selectionVersion: 0, selectionIntentVersion: 0, tabs: [tab('a')], activeTabId: 'url:a', removed: [], docked: [], closed: false, owner: { connectionId: 'alpha', profile: 'alpha', scope: 'alpha', destination: null } }
  receiveBrowserWorkspace(state)
  receiveBrowserWorkspace({ ...state, revision: 2, selectionIntentVersion: 1 })
  expect(JSON.parse(window.localStorage.getItem(key)!).beta).toBeUndefined()
  expect($previewTabs.get().some(item => item.id === 'artifact:transient')).toBe(true)
  vi.resetModules()
  const reloaded = await import('./preview')
  reloaded.setPreviewScope('beta')
  expect(reloaded.$previewTabs.get()).toEqual([])
})

it('takes latest main page on cold reload, then preserves live guests across metadata and selection updates', async () => {
  const oldTab = tab('seed', 'https://example.test/old')
  oldTab.target.label = 'Stale title'
  window.localStorage.setItem(key, JSON.stringify({ default: [oldTab] }))
  window.history.replaceState({}, '', '/?win=browser&browserWindow=workspace')
  const { receiveBrowserWorkspace } = await import('./browser-workspaces')
  const { $previewTabs } = await import('./preview')
  const state = { id: 'workspace', revision: 9, selectionVersion: 0, selectionIntentVersion: 0, tabs: [tab('seed', 'https://example.test/new'), tab('second')], activeTabId: 'url:seed', removed: [], docked: [], closed: false, owner: { connectionId: 'connection', profile: 'default', scope: 'default', destination: null } }
  receiveBrowserWorkspace(state)
  const mounted = $previewTabs.get()[0]
  expect(mounted?.target.url).toBe('https://example.test/new')
  expect(mounted?.target.label).toBe('seed')
  receiveBrowserWorkspace({ ...state, revision: 10, selectionVersion: 1, selectionIntentVersion: 1, activeTabId: 'url:second', tabs: [tab('seed', 'https://example.test/newer'), tab('second')] })
  expect($previewTabs.get()[0]).toBe(mounted)
  expect(JSON.parse(window.localStorage.getItem(key)!).default[0].target.url).toBe('https://example.test/old')
})

it.each(['delete', 'rename'] as const)('applies the real profile %s lifecycle before stale page/close/snapshot and reload', async kind => {
  const states = await import('./session-states')
  const preview = await import('./preview')
  const workspaces = await import('./browser-workspaces')
  const queued: BrowserWorkspace[] = []
  const runtime = new BrowserWorkspaces((_recipients, state) => queued.push(state))

  function open(connectionId: string, id: string, scope: string, renderer: number) {
    const conversation = { kind: 'session' as const, id, connectionId, profile: 'alpha' }
    const state = runtime.open(1, { tab: tab(id), scope, destination: { kind: 'composer', surfaceId: id, target: id, windowId: 'chat', conversation } }, conversation)
    runtime.attach(state.id, renderer)

    return runtime.snapshots(renderer)[0]!
  }

  const local = open('local', 'local', 'alpha', 2)
  const remote = open('remote', 'remote', 'conn:remote::alpha', 3)
  preview.setPreviewScope('alpha')
  workspaces.receiveBrowserWorkspace(local)
  workspaces.receiveBrowserWorkspace(remote)
  const extra = runtime.command(2, local.id, { kind: 'new' })!
  workspaces.receiveBrowserWorkspace(extra)
  // The close publication is still queued when profile retirement starts.
  runtime.command(2, local.id, { kind: 'close', tabId: extra.activeTabId! })
  queued.length = 0
  const oldTarget = { windowId: local.id, tabId: local.activeTabId!, owner: local.owner, selectionVersion: local.selectionVersion }
  const before = runtime.command(2, local.id, { kind: 'page', tabId: oldTarget.tabId, url: 'https://example.test/latest', title: 'latest' })!
  Object.assign(window, { hermesDesktop: { browserWorkspace: {
    retireProfile: async (change: BrowserProfileRetirement) => runtime.retireProfile(change),
    acknowledge: (id: string, revision: number) => runtime.acknowledge(1, id, revision)
  } } })

  if (kind === 'delete') {await states.dropTilesForProfile('alpha', { connectionId: 'local', profile: 'alpha' })}
  else {await states.migrateTilesForProfile('alpha', 'renamed')}

  for (const state of queued) {workspaces.receiveBrowserWorkspace(state)}
  workspaces.receiveBrowserWorkspace(before)
  runtime.close(local.id)
  expect(runtime.command(2, local.id, { kind: 'page', tabId: oldTarget.tabId, url: 'https://example.test/late', title: 'late' })).toBeNull()
  expect(runtime.relay(1, { id: 'late', kind: 'act', target: oldTarget, requester: local.owner.conversation, payload: { kind: 'click' } })).toBeNull()
  expect(preview.$poppedBrowserTabIds.get().has(oldTarget.tabId)).toBe(false)
  let stored = JSON.parse(window.localStorage.getItem(key)!)
  expect(stored.alpha).toBeUndefined()
  expect(stored['conn:remote::alpha']).toEqual([tab('remote')])

  if (kind === 'rename') {expect(stored.renamed).toEqual([{ ...tab('local'), target: { ...tab('local').target, url: 'https://example.test/latest', label: 'latest' } }])}
  vi.resetModules()
  const reloaded = await import('./preview')
  const reloadedWorkspaces = await import('./browser-workspaces')

  for (const state of runtime.snapshots(1)) {reloadedWorkspaces.receiveBrowserWorkspace(state)}
  reloaded.setPreviewScope(kind === 'rename' ? 'renamed' : 'alpha')
  expect(reloaded.$previewTabs.get().map(item => item.id)).toEqual(kind === 'rename' ? ['url:local'] : [])
  stored = JSON.parse(window.localStorage.getItem(key)!)
  expect(stored.alpha).toBeUndefined()
  expect(runtime.ownerForRenderer(3)).toEqual(remote.owner)
})

it.each(['Stop', 'request.cancel'] as const)('forwards originating %s to the exact detached pending action on a stable tab', async reason => {
  const f = await fixture()
  vi.useFakeTimers()
  const scheduleDeadline = vi.spyOn(window, 'setTimeout')
  const clearDeadline = vi.spyOn(window, 'clearTimeout')
  const scheduleWatch = vi.spyOn(globalThis, 'setInterval')
  const clearWatch = vi.spyOn(globalThis, 'clearInterval')

  try {
    const packets: Array<{ id: string; kind: string; target: BrowserRequestTarget }> = []

    window.hermesDesktop!.windowRelay!.send = value => {
      const packet = value as typeof packets[number]
      expect(f.runtime.relay(1, packet)).toBe(2)
      packets.push(packet)
    }

    const { handleServerRequest } = await import('@/app/session/hooks/use-message-stream/gateway-event/server-requests')
    const { handleInputRequestEvent } = await import('@/app/session/hooks/use-message-stream/gateway-event/input-requests')
    let interrupted = false
    const respond = vi.fn()
    const deps = { activeSessionIdRef: { current: 'runtime-A' }, sessionStateByRuntimeIdRef: { current: new Map([['runtime-A', { storedSessionId: 'stored-A' }]]) }, sessionInterrupted: () => interrupted, updateSessionState: vi.fn(), upsertToolCall: vi.fn() }
    const request = { id: 'originating', method: 'preview.act', connectionId: route.connectionId, profile: 'default', params: { session_id: 'runtime-A', action: 'type', text: 'long pending typing' }, respond, decline: vi.fn(), fail: vi.fn() }
    handleServerRequest(request as never, deps as never, 'runtime-A')
    expect(packets.map(packet => packet.kind)).toEqual(['act'])
    const deadlineIndex = scheduleDeadline.mock.calls.findIndex(([, delay]) => delay === 20_000)
    const watchIndex = scheduleWatch.mock.calls.findIndex(([, delay]) => delay === 50)
    expect(deadlineIndex).toBeGreaterThanOrEqual(0)
    expect(watchIndex).toBeGreaterThanOrEqual(0)
    const deadline = scheduleDeadline.mock.results[deadlineIndex]!.value
    const watch = scheduleWatch.mock.results[watchIndex]!.value

    if (reason === 'Stop') {interrupted = true; await vi.advanceTimersByTimeAsync(50)}
    else {handleInputRequestEvent({ deps, event: { type: 'request.cancel' }, payload: { id: request.id, reason: 'timeout' }, sessionId: 'runtime-A' } as never)}

    await vi.advanceTimersByTimeAsync(0)
    expect(packets.map(packet => packet.kind)).toEqual(['act', 'cancel'])
    expect(packets[1]!.target).toEqual(packets[0]!.target)
    expect(packets[1]!.id).toBe(packets[0]!.id)
    expect(f.runtime.snapshots(1)[0]!.selectionVersion).toBe(packets[0]!.target.selectionVersion)
    expect(respond).toHaveBeenCalledOnce()
    expect(JSON.parse(respond.mock.calls[0]![0].value)).toMatchObject({ success: false })
    // Assert cleanup of this request's timers, not unrelated lazy-store timers.
    expect(clearDeadline).toHaveBeenCalledWith(deadline)
    expect(clearWatch).toHaveBeenCalledWith(watch)
  } finally {vi.restoreAllMocks(); vi.useRealTimers()}
})

it('does not revive a deleted visible bucket on a later local edit after a different scope snapshot', async () => {
  window.localStorage.setItem(key, JSON.stringify({ alpha: [tab('a')], beta: [tab('deleted')] }))
  const preview = await import('./preview')
  const { receiveBrowserWorkspace } = await import('./browser-workspaces')
  preview.setPreviewScope('beta')
  window.localStorage.setItem(key, JSON.stringify({ alpha: [tab('a')] }))
  receiveBrowserWorkspace({ id: 'alpha-window', revision: 1, selectionVersion: 0, selectionIntentVersion: 0, tabs: [tab('a')], activeTabId: 'url:a', removed: [], docked: [], closed: false, owner: { connectionId: 'alpha', profile: 'alpha', scope: 'alpha', destination: null } })
  expect(preview.$previewTabs.get()).toEqual([])
  preview.newBrowserTab()
  const stored = JSON.parse(window.localStorage.getItem(key)!)
  expect(stored.beta).toHaveLength(1)
  expect(stored.beta[0].id).not.toBe('url:deleted')
})

it('round-trips session metadata through main, persistence, rotation and deletion without stale restoration', async () => {
  const preview = await import('./preview')
  const workspaces = await import('./browser-workspaces')
  const { retireBrowserSession } = await import('./browser-conversation')
  const queued: BrowserWorkspace[] = []
  const runtime = new BrowserWorkspaces((_recipients, state) => queued.push(state))
  const conversation = { kind: 'session' as const, id: 'owned-a', connectionId: 'local', profile: 'alpha' }
  preview.setPreviewScope('alpha')
  preview.openPreview(tab('owned').target, 'owned-a')
  const seed = preview.$previewTabs.get()[0]!

  const state = runtime.open(1, {
    tab: { ...seed, id: seed.id as `url:${string}`, target: tab('owned').target }, scope: 'alpha',
    destination: { kind: 'composer', surfaceId: 'original', target: 'original', windowId: 'chat', conversation }
  }, conversation)

  runtime.attach(state.id, 2)
  Object.assign(window, { hermesDesktop: { browserWorkspace: {
    updateOwnership: (update: Parameters<BrowserWorkspaces['updateOwnership']>[1]) => runtime.updateOwnership(1, update),
    retireSession: (id: string) => runtime.retireSession(1, id),
    acknowledge: (id: string, revision: number) => runtime.acknowledge(1, id, revision)
  } } })

  const drain = () => {for (const snapshot of queued.splice(0)) {workspaces.receiveBrowserWorkspace(snapshot)}}
  drain()
  const second = runtime.command(2, state.id, { kind: 'new' })!.activeTabId!
  drain()
  preview.setPreviewTabPinned(seed.id, true)
  drain()
  preview.rekeyPreviewTabsSession('owned-a', 'owned-next')
  drain()
  const live = runtime.snapshots(1)[0]!
  expect(live.tabs.map(row => [row.id, row.sessionId, row.pinned])).toEqual([
    [seed.id, 'owned-next', true], [second, 'owned-next', false]
  ])
  const stale = runtime.command(2, state.id, { kind: 'page', tabId: second, url: 'https://example.test/latest', title: 'Latest' })!
  retireBrowserSession('owned-a')
  preview.prunePreviewTabsForSession('owned-a')
  drain()
  workspaces.receiveBrowserWorkspace(stale)
  const restored = preview.decodePreviewTabs(JSON.stringify(JSON.parse(window.localStorage.getItem(key)!).alpha))
  expect(restored.map(row => row.id)).toEqual([seed.id])
  expect(restored[0]).toMatchObject({ pinned: true })
  expect(restored[0]!.sessionId).toBeUndefined()
  const { readPreviewAnnotateDestination } = await import('@/lib/preview-annotate/handoff')
  expect(readPreviewAnnotateDestination(seed.id)).toBeNull()
  expect(runtime.snapshots(1)).toEqual([])
})

it.each([false, true])('adopts detached-created pending tabs through the opener (close seed before binding: %s)', async closeSeed => {
  const sessions = await import('./session')
  const states = await import('./session-states')
  const preview = await import('./preview')
  const workspaces = await import('./browser-workspaces')
  const { $pendingRuntimeByTab } = await import('./preview-ownership')
  const { createClientSessionState } = await import('@/lib/chat-runtime')
  const { capturePreviewAnnotateDestination } = await import('@/lib/preview-annotate/handoff')
  const runtimeId = 'pending-runtime'
  const storedId = 'pending-stored'
  const scope = 'conn:route-B::default'
  sessions.setSessionOwnerHint(runtimeId, route)
  sessions.setSessionOwnerHint(storedId, route)
  sessions.$selectedStoredSessionId.set(null)
  sessions.$activeSessionId.set(runtimeId)
  states.publishSessionState(runtimeId, createClientSessionState(null))
  const composer = document.createElement('div')
  composer.dataset.composerTarget = 'main'
  composer.dataset.composerSurfaceId = 'pending-primary'
  composer.dataset.browserSessionId = runtimeId
  document.body.append(composer)
  preview.setPreviewScope(scope)
  preview.openPreview(tab('pending').target, null, runtimeId)
  const seed = preview.$previewTabs.get()[0]!
  const queued: BrowserWorkspace[] = []
  const runtime = new BrowserWorkspaces((_recipients, snapshot) => queued.push(snapshot))

  const request = {
    tab: { ...seed, id: seed.id as `url:${string}`, target: tab('pending').target, pendingRuntimeId: $pendingRuntimeByTab.get().get(seed.id) },
    scope, destination: capturePreviewAnnotateDestination()
  }

  const state = runtime.open(1, request, browserWorkspaceRoute(request, registry)!)
  runtime.attach(state.id, 2)
  Object.assign(window, { hermesDesktop: { browserWorkspace: {
    updateOwnership: (update: Parameters<BrowserWorkspaces['updateOwnership']>[1]) => runtime.updateOwnership(1, update),
    acknowledge: (id: string, revision: number) => runtime.acknowledge(1, id, revision)
  } } })

  // Deliver publications outside main's callback: ownership may publish again.
  const drain = () => {
    for (let batch = 0; queued.length && batch < 20; batch++) {
      for (const snapshot of queued.splice(0)) {workspaces.receiveBrowserWorkspace(snapshot)}
    }

    expect(queued).toHaveLength(0)
  }

  drain()
  const created = runtime.command(2, state.id, { kind: 'new' })!
  const second = created.activeTabId!
  expect(second).not.toBe(seed.id)
  expect(created.tabs.find(row => row.id === second)).toMatchObject({ pendingRuntimeId: runtimeId, pinned: false })
  drain()

  if (closeSeed) {
    runtime.command(2, state.id, { kind: 'close', tabId: seed.id })
    drain()
    expect($pendingRuntimeByTab.get().has(seed.id)).toBe(false)
  }

  // Hold a real pre-binding page publication until the local binding is newer.
  runtime.command(2, state.id, { kind: 'page', tabId: second, url: 'https://example.test/pending-page', title: 'Pending page' })
  expect(queued).toHaveLength(1)
  const delayed = queued.shift()!
  const oldTarget = { windowId: state.id, tabId: second, owner: delayed.owner, selectionVersion: delayed.selectionVersion }
  states.publishSessionState(runtimeId, createClientSessionState(storedId))
  // Direct behavioral RED: unchanged source leaves this non-seed row unbound.
  expect(runtime.snapshots(1)[0]!.tabs.find(row => row.id === second)?.sessionId).toBe(storedId)
  expect(preview.$previewTabs.get().find(row => row.id === second)?.sessionId).toBe(storedId)
  expect($pendingRuntimeByTab.get().has(second)).toBe(false)
  expect(delayed.tabs.find(row => row.id === second)?.pendingRuntimeId).toBe(runtimeId)
  preview.setPreviewTabPinned(second, true)
  workspaces.receiveBrowserWorkspace(delayed)
  expect(preview.$previewTabs.get().find(row => row.id === second)).toMatchObject({
    sessionId: storedId, pinned: true, target: { url: 'https://example.test/pending-page', label: 'Pending page' }
  })
  expect($pendingRuntimeByTab.get().has(second)).toBe(false)
  drain()
  preview.setPreviewTabPinned(second, false)
  drain()
  composer.dataset.browserSessionId = storedId
  sessions.$selectedStoredSessionId.set(storedId)
  drain()
  const live = runtime.snapshots(1)[0]!
  expect(live.tabs.map(row => [row.id, row.sessionId, row.pendingRuntimeId, row.pinned])).toEqual(
    (closeSeed ? [second] : [seed.id, second]).map(id => [id, storedId, undefined, false])
  )
  const conversation = { kind: 'session' as const, id: storedId, ...route }
  expect(live.owner.conversation).toEqual(conversation)
  expect(live.owner.destination?.conversation).toEqual(conversation)
  const owner = { profile: scope, runtimeId, sessionId: storedId }
  const allowed = preview.previewTabIdsVisibleTo(owner)
  expect(allowed).toContain(second)
  expect(preview.previewTabIdsVisibleTo({ ...owner, runtimeId: 'other-runtime', sessionId: 'other-stored' })).not.toContain(second)
  const target = workspaces.selectedPopoutTarget(conversation, allowed)
  expect(target).toEqual({ windowId: state.id, tabId: second, owner: live.owner, selectionVersion: live.selectionVersion })

  for (const kind of ['read', 'act'] as const) {
    const packet = { id: `pending-${kind}`, kind, target, requester: conversation, payload: kind === 'read' ? {} : { kind: 'click', ref: '@e1' } }
    expect(runtime.relay(1, { ...packet, target: oldTarget })).toBeNull()
    expect(runtime.relay(1, { ...packet, target: { ...target, tabId: seed.id } })).toBeNull()
    expect(runtime.relay(1, { ...packet, requester: undefined })).toBeNull()
    expect(runtime.relay(1, { ...packet, requester: { ...conversation, id: 'other-stored' } })).toBeNull()
    expect(runtime.relay(1, { ...packet, requester: { ...conversation, connectionId: 'route-A' } })).toBeNull()
    expect(runtime.relay(1, packet)).toBe(2)
    expect(runtime.relay(2, { id: packet.id, kind, target, result: { success: true } })).toBe(1)
  }
})

it('authorizes owner-visible tabs before exact read and drive resolution, including background pins', async () => {
  const preview = await import('./preview')
  const { readActivePreview, registerPreviewPageReader } = await import('@/app/chat/right-rail/preview-reader')
  const { registerPreviewScriptRunner, activePreviewScriptRunner } = await import('@/app/chat/right-rail/preview-script-runner')
  const { actOnActivePreview } = await import('@/app/chat/right-rail/preview-act')
  const { $selectedStoredSessionId } = await import('./session')
  preview.setPreviewScope('alpha')
  preview.$previewTabs.set([
    { ...tab('a'), sessionId: 'session-a' },
    { ...tab('b'), sessionId: 'session-b' },
    { ...tab('pin'), sessionId: 'session-b', pinned: true }
  ])
  $selectedStoredSessionId.set('session-b')
  const owner = { profile: 'alpha', runtimeId: 'runtime-a', sessionId: 'session-a' }
  const run = vi.fn(async () => JSON.stringify({ success: true }))
  const stopRun = registerPreviewScriptRunner('url:b', run)
  const stopRead = registerPreviewPageReader('url:a', async () => ({ text: 'A only', title: 'Live A', url: 'https://example.test/live-a' }))

  try {
    expect(await readActivePreview({}, owner, 'url:b')).toBeNull()
    expect(activePreviewScriptRunner(owner, 'url:b')).toBeNull()
    expect(await actOnActivePreview({ kind: 'click', ref: '@e1' }, undefined, owner, { tabId: 'url:b', valid: () => true })).toMatchObject({ success: false })
    expect(run).not.toHaveBeenCalled()
    const read = await readActivePreview({}, owner, 'url:a')
    expect(read).toMatchObject({ text: 'A only', title: 'Live A', active_tab_id: 'url:a' })
    expect(read!.tabs!.map(row => row.id)).toEqual(['url:a', 'url:pin'])
    expect(await readActivePreview({}, owner, 'url:pin')).not.toBeNull()
    expect(await readActivePreview({}, { ...owner, profile: 'beta' }, 'url:pin')).toBeNull()
    expect(await readActivePreview({}, owner, 'url:missing')).toBeNull()
  } finally {stopRun(); stopRead()}
})
