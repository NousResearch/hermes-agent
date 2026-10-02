import { describe, expect, it, vi } from 'vitest'

import type { BrowserRequestTarget, BrowserWorkspaceOpen } from './browser-workspace-types'
import { browserWorkspaceRoute, BrowserWorkspaces } from './browser-workspaces'
import { resolveDesktopConnectionRequest } from './desktop-profile'
import { registrySshScopeForWindowRoute, WindowConnectionRouteRegistry } from './window-connection-route'

const request = (id = 'seed', scope = 'alpha'): BrowserWorkspaceOpen => ({
  tab: {
    id: `url:${id}`,
    sessionId: 'stored-a',
    pinned: false,
    target: { kind: 'url', label: id, source: 'https://example.test/', url: 'https://example.test/' }
  },
  scope,
  destination: { kind: 'composer', surfaceId: 'composer-a', target: 'session-a', windowId: 'chat-a',
    conversation: { kind: 'session', id: 'stored-a', connectionId: scope === 'beta' ? 'connection-b' : 'connection-a', profile: scope } }
})

const route = { connectionId: 'connection-a', profile: 'alpha' }

function fixture() {
  const changed = vi.fn()
  const runtime = new BrowserWorkspaces(changed)
  const state = runtime.open(1, request(), route)
  runtime.attach(state.id, 2)
  const current = () => runtime.snapshots(2)[0]!

  const target = (): BrowserRequestTarget => ({
    windowId: state.id,
    tabId: current().activeTabId!,
    selectionVersion: current().selectionVersion,
    owner: current().owner
  })

  return { runtime, changed, state, current, target }
}

describe('detached browser runtime authority', () => {
  it('shares a same-profile pin without transferring annotation authority or native-input epochs', () => {
    const { runtime, state, current, target } = fixture()
    runtime.updateOwnership(99, { tabs: [{ id: 'url:seed', sessionId: 'stored-a', pinned: true }] })
    expect(current().tabs[0]!.pinned).toBe(false)
    runtime.updateOwnership(1, { tabs: [{ id: 'url:seed', sessionId: 'stored-a', pinned: true }] })
    const captured = target()
    const requester = { ...captured.owner.conversation!, id: 'stored-b' }
    const packet = { id: 'shared-pin', kind: 'act', target: captured, requester, payload: { kind: 'click' } }
    expect(runtime.relay(99, packet)).toBeNull()
    expect(runtime.relay(1, { ...packet, requester: { ...requester, profile: 'beta' } })).toBeNull()
    expect(runtime.relay(1, packet)).toBe(2)
    runtime.command(2, state.id, { kind: 'select', tabId: captured.tabId })
    expect(current().selectionVersion).toBe(captured.selectionVersion)
    expect(runtime.relay(2, { id: packet.id, kind: 'act', target: captured, result: { success: true } })).toBe(1)
    expect(runtime.relay(2, { id: packet.id, kind: 'act', target: captured, result: {} })).toBeNull()
    expect(current().owner.destination).toEqual(state.owner.destination)
    const comment = { type: 'preview-annotate-handoff', requestId: 'original-comment', tabId: captured.tabId, destination: state.owner.destination }
    expect(runtime.relayComment(2, { ...comment, destination: { ...state.owner.destination, conversation: requester } })).toBeNull()
    expect(runtime.relayComment(2, comment)).toBe(1)
    runtime.updateOwnership(1, { tabs: [{ id: 'url:seed', sessionId: 'stored-a', pinned: false }] })
    expect(runtime.relay(1, { ...packet, id: 'after-unpin', target: target() })).toBeNull()
    expect(runtime.command(2, state.id, { kind: 'new' })!.tabs.at(-1)).toMatchObject({ sessionId: 'stored-a', pinned: false })
  })

  it('binds and rotates detached ownership once, then deletes unpinned tabs without reviving pin annotations', () => {
    const { runtime, state, current, target } = fixture()
    const metadataFree = request('metadata-free')
    delete metadataFree.tab.pinned
    delete metadataFree.tab.sessionId
    expect(runtime.open(3, metadataFree, route).tabs[0]!.pinned).toBe(false)
    const second = runtime.command(2, state.id, { kind: 'new' })!.activeTabId!
    runtime.updateOwnership(1, {
      tabs: [{ id: 'url:seed', sessionId: 'stored-next', pinned: true }, { id: second, sessionId: 'stored-next', pinned: false }],
      rotation: { previousId: 'stored-a', nextId: 'stored-next' }
    })
    const rotated = current()
    expect(rotated.owner.conversation?.id).toBe('stored-next')
    expect(rotated.owner.destination?.conversation?.id).toBe('stored-next')
    runtime.updateOwnership(1, { tabs: [], rotation: { previousId: 'stored-next', nextId: 'stored-a' } })
    expect(current()).toEqual(rotated)
    const pending = { id: 'before-delete', kind: 'act', target: target(), requester: target().owner.conversation, payload: { kind: 'type' } }
    expect(runtime.relay(1, pending)).toBe(2)
    expect(runtime.retireSession(1, 'stored-a')).toEqual([state.id])
    expect(current()).toMatchObject({ closed: true, tabs: [], removed: [second], owner: { destination: null } })
    expect(current().docked).toHaveLength(1)
    expect(current().docked[0]).toMatchObject({ id: 'url:seed', pinned: true })
    expect(current().docked[0]!.sessionId).toBeUndefined()
    expect(runtime.relay(1, { id: pending.id, kind: 'cancel', target: pending.target })).toBe(2)
    runtime.close(state.id)
    expect(current().docked).toHaveLength(1)
    runtime.acknowledge(1, state.id, current().revision)
    expect(runtime.snapshots(1)).toEqual([])
  })

  it('cancels only the original pending request, including after its target retires, without replay revival', () => {
    const { runtime, state, target } = fixture()
    const packet = { id: 'cancel-me', kind: 'act', target: target(), requester: target().owner.conversation, payload: { kind: 'type' } }
    expect(runtime.relay(1, packet)).toBe(2)
    const cancel = { id: packet.id, kind: 'cancel', target: packet.target }
    expect(runtime.relay(99, cancel)).toBeNull()
    expect(runtime.relay(2, cancel)).toBeNull()
    expect(runtime.relay(1, { ...cancel, id: 'unknown' })).toBeNull()
    expect(runtime.relay(1, { ...cancel, target: { ...target(), tabId: 'url:wrong' } })).toBeNull()
    expect(runtime.relay(1, { ...cancel, target: { ...target(), owner: { ...target().owner, scope: 'wrong' } } })).toBeNull()
    runtime.command(2, state.id, { kind: 'new' })
    expect(runtime.relay(1, { ...cancel, target: target() })).toBeNull()
    expect(runtime.relay(1, cancel)).toBe(2)
    expect(runtime.relay(1, cancel)).toBeNull()
    expect(runtime.relay(2, { id: packet.id, kind: 'act', target: packet.target, result: { success: true } })).toBeNull()
    expect(runtime.relay(1, { ...packet, target: target() })).toBeNull()
    const survivor = { ...packet, id: 'survivor', target: target() }
    expect(runtime.relay(1, survivor)).toBe(2)
    runtime.close(state.id)
    runtime.acknowledge(1, state.id, runtime.snapshots(1)[0]!.revision)
    expect(runtime.relay(1, { id: survivor.id, kind: 'cancel', target: survivor.target })).toBe(2)
  })

  it.each(['delete', 'rename'] as const)('retires exact connection/profile authority on %s across openers, never via normal close', kind => {
    const { runtime, state, target } = fixture()
    const peer = runtime.open(3, request('peer'), route)
    runtime.attach(peer.id, 4)
    const remoteRequest = request('remote', 'conn:remote::alpha')
    remoteRequest.destination!.conversation = { kind: 'session', id: 'remote', connectionId: 'remote', profile: 'alpha' }
    const remote = runtime.open(5, remoteRequest, { connectionId: 'remote', profile: 'alpha' })
    runtime.attach(remote.id, 6)
    const packet = { id: 'profile-act', kind: 'act', target: target(), requester: target().owner.conversation, payload: { kind: 'type' } }
    expect(runtime.relay(1, packet)).toBe(2)
    const retired = runtime.retireProfile({ ...route, ...(kind === 'rename' ? { replacementProfile: 'renamed' } : {}) })
    expect(retired.map(item => item.id)).toEqual([state.id, peer.id])

    for (const item of retired) {
      expect(item).toMatchObject({ closed: true, tabs: [], activeTabId: null })
      expect(item.docked.length).toBe(kind === 'rename' ? 1 : 0)
    }

    expect(runtime.command(2, state.id, { kind: 'page', tabId: 'url:seed', url: 'https://example.test/late', title: 'late' })).toBeNull()
    runtime.close(state.id)
    expect(runtime.relay(1, { ...packet, id: 'late' })).toBeNull()
    expect(runtime.relay(1, { id: packet.id, kind: 'cancel', target: packet.target })).toBe(2)
    expect(runtime.ownerForRenderer(2)).toBeNull()
    expect(runtime.ownerForRenderer(6)).toEqual(remote.owner)
    expect(runtime.command(6, remote.id, { kind: 'new' })?.tabs).toHaveLength(2)
  })

  it('pins the conversation route through the native connection and SSH reach resolvers across foreground switches', () => {
    const runtime = new BrowserWorkspaces(() => {})

    const registry = { version: 2 as const, launchMode: 'primary' as const, lastUsed: 'connection-a', primary: 'connection-a', connections: [
      { id: 'connection-a', kind: 'ssh' as const, label: 'A' },
      { id: 'connection-b', kind: 'ssh' as const, label: 'B' }
    ] }

    const open = request('remote', 'beta')
    open.destination!.conversation!.profile = 'default'
    const routes = new WindowConnectionRouteRegistry()
    routes.set(1, { connectionId: 'connection-a', profile: 'default', registryScoped: true })
    const state = runtime.open(1, open, browserWorkspaceRoute(open, registry)!)
    runtime.attach(state.id, 2)
    routes.set(2, runtime.ownerForRenderer(2))
    routes.set(1, { connectionId: 'connection-b', profile: 'default', registryScoped: true })
    routes.set(1, { connectionId: 'connection-a', profile: 'default', registryScoped: true })
    expect(resolveDesktopConnectionRequest(undefined, routes.get(2), 'default')).toEqual({ connectionId: 'connection-b', profile: 'default' })
    expect(registrySshScopeForWindowRoute(routes.get(2), registry)).toContain('connection-b')
    expect(runtime.snapshots(1)[0]!.closed).toBe(false)
    expect(browserWorkspaceRoute(open, { connections: [] })).toBeNull()
    expect(() => runtime.open(1, open, { connectionId: 'connection-a', profile: 'default' })).toThrow('route')
  })

  it('rejects a copied target with a different requester and retires exact sessions and comments', () => {
    const { runtime, state, target } = fixture()
    const packet = { id: 'foreign', kind: 'read', target: target(), payload: {}, requester: { ...target().owner.conversation!, id: 'stored-b' } }
    expect(runtime.relay(1, packet)).toBeNull()
    expect(runtime.relay(1, { ...packet, requester: { ...packet.requester, id: 'stored-a', connectionId: 'connection-b' } })).toBeNull()
    expect(runtime.relay(1, { ...packet, requester: target().owner.conversation })).toBe(2)
    const comment = { type: 'preview-annotate-handoff', requestId: 'comments', tabId: 'url:seed', destination: state.owner.destination }
    expect(runtime.relayComment(2, { ...comment, destination: { ...comment.destination, conversation: packet.requester } })).toBeNull()
    expect(runtime.relayComment(2, comment)).toBe(1)
    expect(runtime.relayComment(99, { type: 'preview-annotate-handoff-ack', requestId: 'comments' })).toBeNull()
    expect(runtime.retireSession(99, 'stored-a')).toEqual([])
    expect(runtime.retireSession(1, 'stored-b')).toEqual([])
    expect(runtime.retireSession(1, 'stored-a')).toEqual([state.id])
    expect(runtime.relay(1, { ...packet, requester: target().owner.conversation })).toBeNull()
    expect(runtime.relayComment(2, { ...comment, requestId: 'late' })).toBeNull()
    runtime.acknowledge(1, state.id, runtime.snapshots(1)[0]!.revision)
    expect(runtime.relayComment(2, { ...comment, requestId: 'after-receipt' })).toBeNull()
  })

  it('keeps ordered second/third tabs and stable window identity after closing the seed', () => {
    const { runtime, state, current } = fixture()
    const second = runtime.command(2, state.id, { kind: 'new' })!.activeTabId!
    const third = runtime.command(2, state.id, { kind: 'new' })!.activeTabId!
    expect(new Set([state.id, 'url:seed', second, third]).size).toBe(4)
    expect(current().tabs.map(tab => tab.id)).toEqual(['url:seed', second, third])
    runtime.command(2, state.id, { kind: 'close', tabId: 'url:seed' })
    expect(current()).toMatchObject({ id: state.id, activeTabId: third, closed: false })
    runtime.command(2, state.id, { kind: 'close', tabId: third })
    expect(current().activeTabId).toBe(second)
    runtime.command(2, state.id, { kind: 'new' })
    expect(current().tabs).toHaveLength(2)
  })

  it('docks one tab, keeps its final page, then native-close docks survivors exactly once', () => {
    const { runtime, state, current, changed } = fixture()
    const second = runtime.command(2, state.id, { kind: 'new' })!.activeTabId!
    runtime.command(2, state.id, { kind: 'page', tabId: second, url: 'https://example.test/second', title: 'Second' })
    runtime.command(2, state.id, { kind: 'dock', tabId: second })
    expect(current().tabs.map(tab => tab.id)).toEqual(['url:seed'])
    expect(current().docked[0]?.target).toMatchObject({ url: 'https://example.test/second', label: 'Second' })
    runtime.close(state.id)
    const calls = changed.mock.calls.length
    runtime.close(state.id)
    expect(changed).toHaveBeenCalledTimes(calls)
    expect(current().docked.map(tab => tab.id)).toEqual([second, 'url:seed'])
    expect(current()).toMatchObject({ tabs: [], activeTabId: null, closed: true })
  })

  it('final-tab close deletes rather than resurrecting and failed-open rollback returns the seed', () => {
    const { runtime, state, current } = fixture()
    runtime.command(2, state.id, { kind: 'close', tabId: 'url:seed' })
    runtime.close(state.id)
    expect(current()).toMatchObject({ tabs: [], removed: ['url:seed'], docked: [], closed: true })
    const failed = runtime.open(1, request('failed'), route)
    runtime.close(failed.id)
    expect(
      runtime
        .snapshots(1)
        .find(item => item.id === failed.id)
        ?.docked.map(tab => tab.id)
    ).toEqual(['url:failed'])
  })

  it('restricts snapshots and writes to exact registered hosts, not guests/other windows', () => {
    const { runtime, state } = fixture()
    expect(runtime.snapshots(99)).toEqual([])
    expect(runtime.command(1, state.id, { kind: 'new' })).toBeNull()
    expect(runtime.command(99, state.id, { kind: 'new' })).toBeNull()
    expect(runtime.command(2, 'missing', { kind: 'new' })).toBeNull()
    expect(() => runtime.open(99, request(), route)).toThrow('another window')
    expect(runtime.open(1, request(), route).id).toBe(state.id)
    expect(() => runtime.attach(state.id, 99)).toThrow()
    expect(
      runtime.command(2, state.id, { kind: 'page', tabId: 'url:seed', url: 'javascript:alert(1)', title: '' })
    ).toBeNull()
  })

  it('copies snapshots and owner input instead of sharing mutable renderer references', () => {
    const { runtime, state } = fixture()
    state.tabs.length = 0
    const snapshot = runtime.snapshots(1)[0]!
    snapshot.owner.profile = 'wrong'
    snapshot.tabs[0]!.target.url = 'https://wrong.test'
    expect(runtime.snapshots(2)[0]!.owner.profile).toBe('alpha')
    expect(runtime.snapshots(2)[0]!.tabs[0]!.target.url).toBe('https://example.test/')
  })

  it('routes a request and reply only between exact opener and workspace, including a non-seed tab', () => {
    const { runtime, state, target } = fixture()
    const other = runtime.open(3, request('other', 'beta'), { connectionId: 'connection-b', profile: 'beta' })
    runtime.attach(other.id, 4)
    runtime.command(2, state.id, { kind: 'new' })
    const packet = { id: 'act-1', kind: 'act', target: target(), requester: target().owner.conversation, payload: { kind: 'click' } }
    expect(packet.target.tabId).not.toBe('url:seed')
    expect(runtime.relay(3, packet)).toBeNull()
    expect(runtime.relay(4, packet)).toBeNull()
    expect(runtime.relay(1, packet)).toBe(2)
    expect(runtime.relay(1, packet)).toBeNull()
    const reply = { id: packet.id, kind: 'act', target: packet.target, result: { success: true } }
    expect(runtime.relay(4, reply)).toBeNull()
    expect(runtime.relay(2, reply)).toBe(1)
    expect(runtime.relay(2, reply)).toBeNull()
    expect(runtime.relay(1, { id: packet.id, kind: 'cancel', target: packet.target })).toBeNull()
  })

  it.each(['click', 'type'])('keeps a %s request effect-bound through same-active selection renewal', kind => {
    const { runtime, state, current, target } = fixture()
    const captured = target()
    const packet = { id: `renew-${kind}`, kind: 'act', target: captured, requester: captured.owner.conversation, payload: { kind } }
    expect(runtime.relay(1, packet)).toBe(2)
    const before = current()
    runtime.command(2, state.id, { kind: 'select', tabId: captured.tabId })
    runtime.command(2, state.id, { kind: 'select', tabId: captured.tabId })
    const reply = { id: packet.id, kind: 'act', target: captured, result: { success: true, acted: kind } }
    expect(runtime.relay(99, reply)).toBeNull()
    expect(runtime.relay(2, reply)).toBe(1)
    expect(runtime.relay(2, reply)).toBeNull()
    expect(current().selectionIntentVersion).toBe(before.selectionIntentVersion + 2)
    expect(current().selectionVersion).toBe(captured.selectionVersion)
  })

  it.each(['switch-away-and-back', 'closed', 'moved', 'window-closed', 'wrong-owner'] as const)(
    'rejects an in-flight action after %s, even with a refreshed reply target', transition => {
      const { runtime, state, target } = fixture()
      const captured = target()
      const packet = { id: 'retired-act', kind: 'act', target: captured, requester: captured.owner.conversation, payload: { kind: 'type' } }
      expect(runtime.relay(1, packet)).toBe(2)
      let replyTarget = captured

      if (transition === 'switch-away-and-back') {
        runtime.command(2, state.id, { kind: 'new' })
        runtime.command(2, state.id, { kind: 'select', tabId: captured.tabId })
      } else if (transition === 'window-closed') {
        runtime.close(state.id)
      } else if (transition === 'wrong-owner') {
        replyTarget = { ...captured, owner: { ...captured.owner, scope: 'foreign' } }
      } else {
        runtime.command(2, state.id, { kind: transition === 'closed' ? 'close' : 'dock', tabId: captured.tabId })
      }

      const reply = { id: packet.id, kind: 'act', target: replyTarget, result: { success: true } }
      expect(runtime.relay(2, reply)).toBeNull()
      expect(runtime.relay(1, { ...packet, id: 'late', target: replyTarget })).toBeNull()

      if (transition === 'switch-away-and-back') {
        // A valid current epoch cannot be substituted onto an old request ID.
        expect(runtime.relay(2, { ...reply, target: target() })).toBeNull()
        expect(runtime.relay(1, { ...packet, id: 'fresh', target: target() })).toBe(2)
      }
    }
  )

  it('rejects missing/malformed/wrong-owner/stale targets and late replies after switch-away-and-back', () => {
    const { runtime, state, target } = fixture()
    const packet = { id: 'read-1', kind: 'read', target: target(), requester: target().owner.conversation, payload: {} }

    for (const bad of [
      null,
      {},
      { ...packet, target: null },
      { ...packet, target: { ...packet.target, owner: { ...route, scope: 'wrong', destination: null } } }
    ]) {
      expect(runtime.relay(1, bad)).toBeNull()
    }

    expect(runtime.relay(1, packet)).toBe(2)
    runtime.command(2, state.id, { kind: 'new' })
    runtime.command(2, state.id, { kind: 'select', tabId: 'url:seed' })
    expect(runtime.relay(2, { id: packet.id, kind: 'read', target: packet.target, result: {} })).toBeNull()
    expect(runtime.relay(1, { ...packet, id: 'stale' })).toBeNull()
    const fresh = { ...packet, id: 'fresh', target: target() }
    expect(runtime.relay(1, fresh)).toBe(2)
    runtime.command(2, state.id, { kind: 'close', tabId: 'url:seed' })
    expect(runtime.relay(2, { id: fresh.id, kind: 'read', target: fresh.target, result: {} })).toBeNull()
  })

  it('replays unacknowledged transfers, then retires receipts so reload cannot resurrect a docked tab', () => {
    const { runtime, state, current } = fixture()
    const second = runtime.command(2, state.id, { kind: 'new' })!.activeTabId!
    runtime.command(2, state.id, { kind: 'dock', tabId: second })
    const revision = current().revision
    expect(current().docked.map(tab => tab.id)).toEqual([second])
    runtime.acknowledge(99, state.id, revision)
    runtime.acknowledge(1, state.id, revision - 1)
    expect(current().docked).toHaveLength(1)
    runtime.acknowledge(1, state.id, revision)
    expect(current().docked).toEqual([])
    expect(current().tabs.map(tab => tab.id)).toEqual(['url:seed'])
    runtime.close(state.id)
    expect(runtime.snapshots(1)[0]!.docked).toHaveLength(1)
    runtime.acknowledge(1, state.id, current().revision)
    expect(runtime.snapshots(1)).toEqual([])
    expect(runtime.snapshots(2)).toEqual([])
  })

  it('re-homes only the opener being retired and leaves other owners alive', () => {
    const { runtime, state } = fixture()
    const other = runtime.open(3, request('other', 'beta'), { connectionId: 'connection-b', profile: 'beta' })
    runtime.attach(other.id, 4)
    expect(runtime.closeOwned(1)).toEqual([state.id])
    expect(runtime.isRenderer(2)).toBe(false)
    expect(runtime.isRenderer(4)).toBe(true)
    expect(runtime.closeOwned(1)).toEqual([])
  })
})
