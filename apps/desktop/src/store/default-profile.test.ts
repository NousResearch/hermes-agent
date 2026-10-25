import { act, cleanup, renderHook } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { useDefaultProfilePreference } from '@/app/gateway/hooks/use-default-profile-preference'
import type { DesktopProfileRoute } from '@/global'
import { deferred } from '@/test/deferred'

import * as preference from './default-profile'

// The native bridge is the persistence boundary; the renderer owns only a mirror.
afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
})

describe('default profile preference', () => {
  it('subscribes before loading, mirrors cross-window changes, and tears down', async () => {
    const oldRoute = { connectionId: null, profile: 'personal' }
    const nextRoute = { connectionId: 'lab', profile: 'research' }
    const read = deferred<typeof oldRoute>()
    let listener!: (route: DesktopProfileRoute | null) => void
    const off = vi.fn()

    const onDefaultChanged = vi.fn(callback => {
      listener = callback

      return off
    })

    const getDefault = vi.fn(() => {
      expect(onDefaultChanged).toHaveBeenCalledTimes(1)

      return read.promise
    })

    vi.stubGlobal('window', { hermesDesktop: { profile: { getDefault, onDefaultChanged } } })
    const { unmount } = renderHook(useDefaultProfilePreference)

    await act(async () => {
      listener(nextRoute)
      read.resolve(oldRoute)
      await read.promise
    })
    expect(preference.$defaultProfileRoute.get()).toEqual(nextRoute)
    unmount()
    expect(off).toHaveBeenCalledTimes(1)
  })

  it('publishes only successful native writes and rejects stale refreshes', async () => {
    const first = { connectionId: null, profile: 'personal' }
    const second = { connectionId: 'lab', profile: 'research' }
    const read = deferred<typeof first>()
    const getDefault = vi.fn(() => read.promise)
    const setDefault = vi.fn(async () => second)
    vi.stubGlobal('window', { hermesDesktop: { profile: { getDefault, setDefault } } })

    const refreshing = preference.refreshDefaultProfile()
    await preference.setDefaultProfile(second)
    read.resolve(first)
    await refreshing
    expect(preference.$defaultProfileRoute.get()).toEqual(second)

    setDefault.mockRejectedValueOnce(new Error('read-only preferences'))
    await expect(preference.setDefaultProfile(first)).rejects.toThrow('read-only preferences')
    expect(preference.$defaultProfileRoute.get()).toEqual(second)
    expect(setDefault).toHaveBeenLastCalledWith(first)
  })
})
