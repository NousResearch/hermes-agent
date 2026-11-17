import { beforeEach, expect, it, vi } from 'vitest'

beforeEach(() => {
  window.localStorage.clear()
  vi.resetModules()
})

it('restores the hidden choice without sharing it with another session or owner', async () => {
  const store = await import('./composer-status-drawer')
  const scope = { connectionId: 'local', profile: 'default', targetProfile: 'default', sessionId: 'chat-a' }
  const key = store.statusDrawerKey(scope)
  const otherSession = store.statusDrawerKey({ ...scope, sessionId: 'chat-b' })
  const otherProfile = store.statusDrawerKey({ ...scope, profile: 'coder', targetProfile: 'coder' })
  const otherConnection = store.statusDrawerKey({ ...scope, connectionId: 'remote' })

  store.setStatusDrawerCollapsed(key, true)
  store.setStatusDrawerCollapsed(otherSession, true)
  store.setStatusDrawerCollapsed(otherSession, false)
  vi.resetModules()

  const restored = await import('./composer-status-drawer')
  expect(restored.$collapsedStatusDrawers.get()).toEqual([key])
  expect(restored.$collapsedStatusDrawers.get()).not.toContain(otherProfile)
  expect(restored.$collapsedStatusDrawers.get()).not.toContain(otherConnection)
})

it('follows local profile renames and scopes profile removal to the exact owner', async () => {
  const store = await import('./composer-status-drawer')
  const scope = { connectionId: 'local', profile: 'coder', targetProfile: 'coder', sessionId: 'chat' }
  const local = store.statusDrawerKey(scope)
  const remote = store.statusDrawerKey({ ...scope, connectionId: 'remote' })
  const renamed = store.statusDrawerKey({ ...scope, profile: 'work', targetProfile: 'work' })

  store.setStatusDrawerCollapsed(local, true)
  store.setStatusDrawerCollapsed(remote, true)
  store.migrateStatusDrawersForProfile('coder', 'work')
  expect(store.$collapsedStatusDrawers.get()).toEqual([renamed, remote])
  store.dropStatusDrawersForProfile('work')
  expect(store.$collapsedStatusDrawers.get()).toEqual([remote])
  store.dropStatusDrawersForProfile('coder', { connectionId: 'remote', profile: 'coder', targetProfile: 'coder' })
  expect(store.$collapsedStatusDrawers.get()).toEqual([])
})
