import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { CustomEndpoint, CustomEndpointsResponse } from '@/types/hermes'

const activeProfile = atom('default')
const activateCustomEndpoint = vi.fn()
const deleteCustomEndpoint = vi.fn()
const getCustomEndpoints = vi.fn()
const saveCustomEndpoint = vi.fn()
const validateCustomEndpoint = vi.fn()
const notify = vi.fn()
const notifyError = vi.fn()
const triggerHaptic = vi.fn()

vi.mock('@/hermes', () => ({
  activateCustomEndpoint: (id: string) => activateCustomEndpoint(id),
  deleteCustomEndpoint: (id: string) => deleteCustomEndpoint(id),
  getCustomEndpoints: () => getCustomEndpoints(),
  saveCustomEndpoint: (body: unknown) => saveCustomEndpoint(body),
  validateCustomEndpoint: (body: unknown) => validateCustomEndpoint(body)
}))

vi.mock('@/store/profile', () => ({
  $activeGatewayProfile: activeProfile
}))

vi.mock('@/store/notifications', () => ({
  notify: (message: unknown) => notify(message),
  notifyError: (error: unknown, message: string) => notifyError(error, message)
}))

vi.mock('@/lib/haptics', () => ({
  triggerHaptic: (kind: string) => triggerHaptic(kind)
}))

function endpoint(id: string, name: string): CustomEndpoint {
  return {
    base_url: `https://${id}.invalid/v1`,
    discover_models: true,
    has_api_key: false,
    id,
    is_current: true,
    model: `${id}-model`,
    models: [`${id}-model`],
    name,
    source: 'providers'
  }
}

function response(item: CustomEndpoint): CustomEndpointsResponse {
  return {
    current: {
      base_url: item.base_url,
      model: item.model,
      provider: item.id
    },
    endpoints: [item]
  }
}

function deferred<T>() {
  let resolve!: (value: T) => void

  const promise = new Promise<T>(done => {
    resolve = done
  })

  return { promise, resolve }
}

beforeEach(() => {
  activeProfile.set('default')
  vi.clearAllMocks()
})

afterEach(() => {
  cleanup()
})

describe('CustomEndpointsSettings profile isolation', () => {
  it('ignores a profile A load that resolves after profile B', async () => {
    const profileA = response(endpoint('profile-a', 'Profile A endpoint'))
    const profileB = response(endpoint('profile-b', 'Profile B endpoint'))
    const profileALoad = deferred<CustomEndpointsResponse>()
    const profileBLoad = deferred<CustomEndpointsResponse>()

    getCustomEndpoints.mockReturnValueOnce(profileALoad.promise).mockReturnValueOnce(profileBLoad.promise)

    const { CustomEndpointsSettings } = await import('./custom-endpoints-settings')
    render(<CustomEndpointsSettings />)

    await waitFor(() => expect(getCustomEndpoints).toHaveBeenCalledTimes(1))

    await act(async () => {
      activeProfile.set('worker')
    })

    await waitFor(() => expect(getCustomEndpoints).toHaveBeenCalledTimes(2))

    await act(async () => {
      profileBLoad.resolve(profileB)
      await profileBLoad.promise
    })

    expect(await screen.findByText('Profile B endpoint')).toBeTruthy()

    await act(async () => {
      profileALoad.resolve(profileA)
      await profileALoad.promise
    })

    expect(screen.getByText('Profile B endpoint')).toBeTruthy()
    expect(screen.queryByText('Profile A endpoint')).toBeNull()
  })

  it('drops profile A state and ignores its pending save after switching to profile B', async () => {
    const profileA = response(endpoint('profile-a', 'Profile A endpoint'))
    const profileB = response(endpoint('profile-b', 'Profile B endpoint'))
    const profileBLoad = deferred<CustomEndpointsResponse>()
    const profileASave = deferred<CustomEndpointsResponse>()
    const onConfigSaved = vi.fn()
    const onMainModelChanged = vi.fn()

    getCustomEndpoints.mockResolvedValueOnce(profileA).mockReturnValueOnce(profileBLoad.promise)
    saveCustomEndpoint.mockReturnValueOnce(profileASave.promise)

    const { CustomEndpointsSettings } = await import('./custom-endpoints-settings')
    render(<CustomEndpointsSettings onConfigSaved={onConfigSaved} onMainModelChanged={onMainModelChanged} />)

    expect(await screen.findByText('Profile A endpoint')).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: 'Save' }))
    await waitFor(() => expect(saveCustomEndpoint).toHaveBeenCalledTimes(1))

    await act(async () => {
      activeProfile.set('worker')
      profileASave.resolve({ ...profileA, id: 'profile-a', ok: true })
      await profileASave.promise
    })

    await waitFor(() => expect(getCustomEndpoints).toHaveBeenCalledTimes(2))
    expect(screen.queryByText('Profile A endpoint')).toBeNull()
    expect(screen.queryByRole('button', { name: 'Save' })).toBeNull()
    expect(onConfigSaved).not.toHaveBeenCalled()
    expect(onMainModelChanged).not.toHaveBeenCalled()
    expect(triggerHaptic).not.toHaveBeenCalled()

    await act(async () => {
      profileBLoad.resolve(profileB)
      await profileBLoad.promise
    })

    expect(await screen.findByText('Profile B endpoint')).toBeTruthy()

    expect(screen.getByText('Profile B endpoint')).toBeTruthy()
    expect(screen.queryByText('Profile A endpoint')).toBeNull()
    expect(saveCustomEndpoint).toHaveBeenCalledTimes(1)
    expect(onConfigSaved).not.toHaveBeenCalled()
    expect(onMainModelChanged).not.toHaveBeenCalled()
    expect(triggerHaptic).not.toHaveBeenCalled()
  })
})
