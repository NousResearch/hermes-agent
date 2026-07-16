import { cleanup, renderHook, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { useProfileScopeSessionRefresh } from './use-profile-scope-session-refresh'

const ALL_PROFILES_SCOPE = '__all__'

describe('useProfileScopeSessionRefresh', () => {
  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
  })

  it('does not refresh on the initial render', () => {
    const refreshSessions = vi.fn(async () => undefined)

    renderHook(() =>
      useProfileScopeSessionRefresh({ gatewayState: 'open', profileScope: 'default', refreshSessions })
    )

    expect(refreshSessions).not.toHaveBeenCalled()
  })

  it('refreshes when switching between concrete profiles', async () => {
    const refreshSessions = vi.fn(async () => undefined)

    const { rerender } = renderHook(
      ({ profileScope }) =>
        useProfileScopeSessionRefresh({ gatewayState: 'open', profileScope, refreshSessions }),
      { initialProps: { profileScope: 'default' } }
    )

    rerender({ profileScope: 'work' })

    await waitFor(() => expect(refreshSessions).toHaveBeenCalledTimes(1))
  })

  it('refreshes when switching from a concrete profile to All Profiles', async () => {
    const refreshSessions = vi.fn(async () => undefined)

    const { rerender } = renderHook(
      ({ profileScope }) =>
        useProfileScopeSessionRefresh({ gatewayState: 'open', profileScope, refreshSessions }),
      { initialProps: { profileScope: 'default' } }
    )

    rerender({ profileScope: ALL_PROFILES_SCOPE })

    await waitFor(() => expect(refreshSessions).toHaveBeenCalledTimes(1))
  })

  it('refreshes when switching from All Profiles to the current profile', async () => {
    const refreshSessions = vi.fn(async () => undefined)

    const { rerender } = renderHook(
      ({ profileScope }) =>
        useProfileScopeSessionRefresh({ gatewayState: 'open', profileScope, refreshSessions }),
      { initialProps: { profileScope: ALL_PROFILES_SCOPE } }
    )

    rerender({ profileScope: 'work' })

    await waitFor(() => expect(refreshSessions).toHaveBeenCalledTimes(1))
  })

  it('does not refresh a scope change while the gateway is closed', () => {
    const refreshSessions = vi.fn(async () => undefined)

    const { rerender } = renderHook(
      ({ profileScope }) =>
        useProfileScopeSessionRefresh({ gatewayState: 'closed', profileScope, refreshSessions }),
      { initialProps: { profileScope: 'default' } }
    )

    rerender({ profileScope: ALL_PROFILES_SCOPE })

    expect(refreshSessions).not.toHaveBeenCalled()
  })
})
