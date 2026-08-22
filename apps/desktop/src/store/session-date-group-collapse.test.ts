import { beforeEach, describe, expect, it, vi } from 'vitest'

import type { HermesConnection } from '@/global'

const connection = (overrides: Partial<HermesConnection> = {}): HermesConnection =>
  ({
    baseUrl: 'http://127.0.0.1:8642',
    mode: 'remote',
    profile: 'default',
    ...overrides
  }) as HermesConnection

describe('session date-group collapse persistence', () => {
  beforeEach(() => {
    localStorage.clear()
    vi.resetModules()
  })

  it('restores collapsed technical keys after a store remount', async () => {
    const first = await import('./session-date-group-collapse')
    const scope = first.sessionDateGroupScope(connection(), 'default')

    first.setSessionDateGroupCollapsed(scope, 'week:2026-07-20', true)
    expect(first.getCollapsedSessionDateGroups(scope)).toEqual(new Set(['week:2026-07-20']))

    vi.resetModules()
    const remounted = await import('./session-date-group-collapse')

    expect(remounted.getCollapsedSessionDateGroups(scope)).toEqual(new Set(['week:2026-07-20']))
  })

  it('isolates state between profiles on the same backend', async () => {
    const store = await import('./session-date-group-collapse')
    const defaultScope = store.sessionDateGroupScope(connection(), 'default')
    const workScope = store.sessionDateGroupScope(connection(), 'work')

    store.setSessionDateGroupCollapsed(defaultScope, 'today', true)

    expect(store.getCollapsedSessionDateGroups(defaultScope)).toEqual(new Set(['today']))
    expect(store.getCollapsedSessionDateGroups(workScope)).toEqual(new Set())
  })

  it('isolates state between backends for the same profile', async () => {
    const store = await import('./session-date-group-collapse')
    const localScope = store.sessionDateGroupScope(connection({ baseUrl: 'http://127.0.0.1:8642' }), 'default')
    const remoteScope = store.sessionDateGroupScope(connection({ baseUrl: 'https://gateway.example.test/' }), 'default')

    store.collapseAllSessionDateGroups(localScope, ['today', 'yesterday'])

    expect(store.getCollapsedSessionDateGroups(localScope)).toEqual(new Set(['today', 'yesterday']))
    expect(store.getCollapsedSessionDateGroups(remoteScope)).toEqual(new Set())
  })

  it('expands only currently known groups and retains unrelated persisted groups', async () => {
    const store = await import('./session-date-group-collapse')
    const scope = store.sessionDateGroupScope(connection(), 'default')

    store.collapseAllSessionDateGroups(scope, ['today', 'yesterday', 'month:2025-12'])
    store.expandAllSessionDateGroups(scope, ['today', 'yesterday'])

    expect(store.getCollapsedSessionDateGroups(scope)).toEqual(new Set(['month:2025-12']))
  })

  it('normalizes equivalent backend URLs and falls back to the connection profile', async () => {
    const store = await import('./session-date-group-collapse')

    expect(store.sessionDateGroupScope(connection({ baseUrl: 'https://gateway.example.test/' }), null)).toBe(
      store.sessionDateGroupScope(connection({ baseUrl: 'https://gateway.example.test' }), 'default')
    )
  })

  it('keeps a distinct scope for the real All profiles view', async () => {
    const store = await import('./session-date-group-collapse')

    const effectiveProfile = store.resolveSessionDateGroupProfile(connection(), '__all__', {
      allProfilesKey: '__all__',
      showAllProfiles: true
    })

    expect(effectiveProfile).toBe('__all__')
    expect(store.sessionDateGroupScope(connection(), effectiveProfile)).not.toBe(
      store.sessionDateGroupScope(connection(), 'default')
    )
  })

  it('keeps a concrete selected profile', async () => {
    const store = await import('./session-date-group-collapse')

    expect(
      store.resolveSessionDateGroupProfile(connection({ profile: 'work' }), 'work', {
        allProfilesKey: '__all__',
        showAllProfiles: false
      })
    ).toBe('work')
  })

  it('falls back to the active connection profile for a residual All profiles value', async () => {
    const store = await import('./session-date-group-collapse')

    expect(
      store.resolveSessionDateGroupProfile(connection({ profile: 'solo' }), '__all__', {
        allProfilesKey: '__all__',
        showAllProfiles: false
      })
    ).toBe('solo')
  })

  it('preserves backend and effective-profile isolation after scope resolution', async () => {
    const store = await import('./session-date-group-collapse')
    const options = { allProfilesKey: '__all__', showAllProfiles: false }
    const local = connection({ baseUrl: 'http://127.0.0.1:8642', profile: 'default' })
    const remote = connection({ baseUrl: 'https://gateway.example.test', profile: 'default' })

    const localDefault = store.sessionDateGroupScope(
      local,
      store.resolveSessionDateGroupProfile(local, 'default', options)
    )

    const localWork = store.sessionDateGroupScope(local, store.resolveSessionDateGroupProfile(local, 'work', options))

    const remoteDefault = store.sessionDateGroupScope(
      remote,
      store.resolveSessionDateGroupProfile(remote, 'default', options)
    )

    expect(new Set([localDefault, localWork, remoteDefault])).toHaveLength(3)
  })
})
