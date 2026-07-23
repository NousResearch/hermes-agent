import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import { DropdownMenu, DropdownMenuContent } from '@/components/ui/dropdown-menu'
import { $activeGatewayProfile } from '@/store/profile'
import { $activeSessionId, $currentModel, $currentProvider } from '@/store/session'
import type { MoaConfigResponse, MoaPresetConfig, ModelOptionsResponse } from '@/types/hermes'

import { ModelMenuCloseContext, ModelMenuPanel } from './model-menu-panel'

const hermesMocks = vi.hoisted(() => ({
  getGlobalModelOptions: vi.fn(),
  getMoaModels: vi.fn(),
  getProfiles: vi.fn(),
  setApiRequestProfile: vi.fn()
}))

vi.mock('@/hermes', () => ({
  STARTUP_REQUEST_TIMEOUT_MS: 60_000,
  getGlobalModelOptions: (...args: unknown[]) => hermesMocks.getGlobalModelOptions(...args),
  getMoaModels: () => hermesMocks.getMoaModels(),
  getProfiles: () => hermesMocks.getProfiles(),
  setApiRequestProfile: (profile: string | null) => hermesMocks.setApiRequestProfile(profile)
}))

// Radix calls these on open; jsdom does not implement them.
beforeAll(() => {
  Element.prototype.scrollIntoView = vi.fn()
  Element.prototype.hasPointerCapture = vi.fn(() => false)
  Element.prototype.releasePointerCapture = vi.fn()
})

const baseSlot = { model: 'hermes-4', provider: 'nous' }

const presetConfig = (enabled: boolean): MoaPresetConfig => ({
  aggregator: baseSlot,
  aggregator_temperature: null,
  enabled,
  fanout: 'user_turn',
  max_tokens: 4096,
  reference_max_tokens: 600,
  reference_models: [baseSlot],
  reference_temperature: null
})

function moaConfig(presets: Record<string, MoaPresetConfig>): MoaConfigResponse {
  const names = Object.keys(presets)

  return {
    active_preset: '',
    aggregator: baseSlot,
    aggregator_temperature: null,
    default_preset: names[0] ?? '',
    enabled: true,
    max_tokens: 4096,
    presets,
    reference_models: [baseSlot],
    reference_temperature: null
  }
}

function configFromEntries(entries: ReadonlyArray<readonly [string, boolean]>): MoaConfigResponse {
  return moaConfig(Object.fromEntries(entries.map(([name, enabled]) => [name, presetConfig(enabled)])))
}

function modelOptions(models: string[]): ModelOptionsResponse {
  return {
    providers: [{ models, name: 'Mixture of Agents', slug: 'moa' }]
  }
}

function deferred<T>() {
  let resolve!: (value: T) => void
  let reject!: (reason?: unknown) => void

  const promise = new Promise<T>((res, rej) => {
    resolve = res
    reject = rej
  })

  return { promise, reject, resolve }
}

const createCloseMenu = () => vi.fn<() => void>()

const createSelectModel = () =>
  vi.fn<(selection: { model: string; provider: string; sessionId?: null | string }) => Promise<boolean> | void>()

function renderPanel({
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } }),
  closeMenu = createCloseMenu(),
  onSelectModel = createSelectModel()
}: {
  client?: QueryClient
  closeMenu?: ReturnType<typeof createCloseMenu>
  onSelectModel?: ReturnType<typeof createSelectModel>
} = {}) {
  const view = render(
    <QueryClientProvider client={client}>
      <ModelMenuCloseContext.Provider value={() => closeMenu()}>
        <DropdownMenu open>
          <DropdownMenuContent>
            <ModelMenuPanel
              onManageMoaPresets={vi.fn()}
              onSelectModel={selection => onSelectModel(selection)}
              requestGateway={vi.fn()}
            />
          </DropdownMenuContent>
        </DropdownMenu>
      </ModelMenuCloseContext.Provider>
    </QueryClientProvider>
  )

  return { client, closeMenu, onSelectModel, view }
}

beforeEach(() => {
  hermesMocks.getGlobalModelOptions.mockReset()
  hermesMocks.getMoaModels.mockReset()
  hermesMocks.getProfiles.mockReset()
  hermesMocks.setApiRequestProfile.mockClear()
  hermesMocks.getGlobalModelOptions.mockResolvedValue(modelOptions(['enabled']))
  hermesMocks.getMoaModels.mockResolvedValue(configFromEntries([['enabled', true]]))
  hermesMocks.getProfiles.mockResolvedValue({ profiles: [] })
  $activeGatewayProfile.set('default')
  $activeSessionId.set('runtime-1')
  $currentModel.set('')
  $currentProvider.set('')
})

afterEach(() => {
  cleanup()
  $activeGatewayProfile.set('default')
  vi.clearAllMocks()
})

describe('ModelMenuPanel profile-scoped MoA preset guards', () => {
  it('renders only enabled own config entries from the virtual provider catalog', async () => {
    hermesMocks.getGlobalModelOptions.mockResolvedValue(modelOptions(['enabled', 'disabled', 'catalog-only']))
    hermesMocks.getMoaModels.mockResolvedValue(
      configFromEntries([
        ['enabled', true],
        ['disabled', false]
      ])
    )

    renderPanel()

    expect(await screen.findByText('MoA: enabled')).toBeTruthy()
    expect(screen.queryByText('MoA: disabled')).toBeNull()
    expect(screen.queryByText('MoA: catalog-only')).toBeNull()
    expect(screen.getByText('Manage MoA presets…')).toBeTruthy()
  })

  it('hides preset rows when top-level MoA is disabled', async () => {
    const disabled = configFromEntries([['enabled', true]])
    disabled.enabled = false
    hermesMocks.getMoaModels.mockResolvedValue(disabled)

    renderPanel()

    await waitFor(() => expect(hermesMocks.getMoaModels).toHaveBeenCalledOnce())
    expect(screen.queryByText('MoA: enabled')).toBeNull()
    expect(screen.getByText('Manage MoA presets…')).toBeTruthy()
  })

  it('revalidates an enabled row and fails closed when it was freshly disabled', async () => {
    const enabled = configFromEntries([['volatile', true]])
    const disabled = configFromEntries([['volatile', false]])
    hermesMocks.getGlobalModelOptions.mockResolvedValue(modelOptions(['volatile']))
    hermesMocks.getMoaModels.mockResolvedValueOnce(enabled).mockResolvedValueOnce(disabled)
    const { client, closeMenu, onSelectModel } = renderPanel()

    fireEvent.click(await screen.findByText('MoA: volatile'))

    await waitFor(() => expect(hermesMocks.getMoaModels).toHaveBeenCalledTimes(2))
    await waitFor(() => expect(screen.queryByText('MoA: volatile')).toBeNull())
    expect(client.getQueryData(['moa-menu-config', 'default'])).toEqual(disabled)
    expect(onSelectModel).not.toHaveBeenCalled()
    expect(closeMenu).not.toHaveBeenCalled()
  })

  it('removes a row when fresh activation authority disables top-level MoA', async () => {
    const enabled = configFromEntries([['volatile', true]])
    const disabled = configFromEntries([['volatile', true]])
    disabled.enabled = false
    hermesMocks.getGlobalModelOptions.mockResolvedValue(modelOptions(['volatile']))
    hermesMocks.getMoaModels.mockResolvedValueOnce(enabled).mockResolvedValueOnce(disabled)
    const { closeMenu, onSelectModel } = renderPanel()

    fireEvent.click(await screen.findByText('MoA: volatile'))

    await waitFor(() => expect(hermesMocks.getMoaModels).toHaveBeenCalledTimes(2))
    await waitFor(() => expect(screen.queryByText('MoA: volatile')).toBeNull())
    expect(onSelectModel).not.toHaveBeenCalled()
    expect(closeMenu).not.toHaveBeenCalled()
  })

  it('fails closed when the initial profile-scoped config GET is unavailable', async () => {
    hermesMocks.getGlobalModelOptions.mockResolvedValue(modelOptions(['unverified']))
    hermesMocks.getMoaModels.mockRejectedValue(new Error('moa unavailable'))
    const { closeMenu, onSelectModel } = renderPanel()

    await waitFor(() => expect(hermesMocks.getMoaModels).toHaveBeenCalledOnce())
    expect(screen.queryByText('MoA: unverified')).toBeNull()
    expect(screen.getByText('Manage MoA presets…')).toBeTruthy()
    expect(onSelectModel).not.toHaveBeenCalled()
    expect(closeMenu).not.toHaveBeenCalled()
  })

  it('fails closed and removes the row when activation revalidation rejects', async () => {
    hermesMocks.getGlobalModelOptions.mockResolvedValue(modelOptions(['fragile']))
    hermesMocks.getMoaModels
      .mockResolvedValueOnce(configFromEntries([['fragile', true]]))
      .mockRejectedValueOnce(new Error('profile GET failed'))
    const { closeMenu, onSelectModel } = renderPanel()

    fireEvent.click(await screen.findByText('MoA: fragile'))

    await waitFor(() => expect(hermesMocks.getMoaModels).toHaveBeenCalledTimes(2))
    await waitFor(() => expect(screen.queryByText('MoA: fragile')).toBeNull())
    expect(onSelectModel).not.toHaveBeenCalled()
    expect(closeMenu).not.toHaveBeenCalled()
  })

  it('lets the latest same-profile activation win when the older validation resolves last', async () => {
    const initial = configFromEntries([
      ['preset-a', true],
      ['preset-b', true]
    ])

    const freshPresetA = deferred<MoaConfigResponse>()
    const freshPresetB = deferred<MoaConfigResponse>()

    const presetAResponse = configFromEntries([
      ['preset-a', true],
      ['preset-b', false]
    ])

    const presetBResponse = configFromEntries([
      ['preset-a', false],
      ['preset-b', true]
    ])

    hermesMocks.getGlobalModelOptions.mockResolvedValue(modelOptions(['preset-a', 'preset-b']))
    hermesMocks.getMoaModels
      .mockResolvedValueOnce(initial)
      .mockReturnValueOnce(freshPresetA.promise)
      .mockReturnValueOnce(freshPresetB.promise)
    const { client, closeMenu, onSelectModel } = renderPanel()

    fireEvent.click(await screen.findByText('MoA: preset-a'))
    await waitFor(() => expect(hermesMocks.getMoaModels).toHaveBeenCalledTimes(2))
    fireEvent.click(screen.getByText('MoA: preset-b'))
    await waitFor(() => expect(hermesMocks.getMoaModels).toHaveBeenCalledTimes(3))

    await act(async () => {
      freshPresetB.resolve(presetBResponse)
      await freshPresetB.promise
    })
    await waitFor(() =>
      expect(onSelectModel).toHaveBeenCalledWith({ model: 'preset-b', provider: 'moa', sessionId: 'runtime-1' })
    )

    await act(async () => {
      freshPresetA.resolve(presetAResponse)
      await freshPresetA.promise
    })

    expect(onSelectModel.mock.calls.map(([selection]) => selection)).toEqual([
      { model: 'preset-b', provider: 'moa', sessionId: 'runtime-1' }
    ])
    expect(client.getQueryData(['moa-menu-config', 'default'])).toEqual(presetBResponse)
    expect(closeMenu).toHaveBeenCalledOnce()
  })

  it('keeps an older activation invalid after the newer same-profile validation fails', async () => {
    const initial = configFromEntries([
      ['preset-a', true],
      ['preset-b', true]
    ])

    const freshPresetA = deferred<MoaConfigResponse>()
    const freshPresetB = deferred<MoaConfigResponse>()
    hermesMocks.getGlobalModelOptions.mockResolvedValue(modelOptions(['preset-a', 'preset-b']))
    hermesMocks.getMoaModels
      .mockResolvedValueOnce(initial)
      .mockReturnValueOnce(freshPresetA.promise)
      .mockReturnValueOnce(freshPresetB.promise)
    const { client, closeMenu, onSelectModel } = renderPanel()

    fireEvent.click(await screen.findByText('MoA: preset-a'))
    await waitFor(() => expect(hermesMocks.getMoaModels).toHaveBeenCalledTimes(2))
    fireEvent.click(screen.getByText('MoA: preset-b'))
    await waitFor(() => expect(hermesMocks.getMoaModels).toHaveBeenCalledTimes(3))

    await act(async () => {
      freshPresetB.reject(new Error('newer validation failed'))
      await expect(freshPresetB.promise).rejects.toThrow('newer validation failed')
    })
    await waitFor(() => expect(client.getQueryData(['moa-menu-config', 'default'])).toBeNull())

    await act(async () => {
      freshPresetA.resolve(initial)
      await freshPresetA.promise
    })

    expect(onSelectModel).not.toHaveBeenCalled()
    expect(client.getQueryData(['moa-menu-config', 'default'])).toBeNull()
    expect(closeMenu).not.toHaveBeenCalled()
  })

  it('does not persist or close after a pending activation outlives the menu lifecycle', async () => {
    const config = configFromEntries([['guarded', true]])
    const fresh = deferred<MoaConfigResponse>()
    hermesMocks.getGlobalModelOptions.mockResolvedValue(modelOptions(['guarded']))
    hermesMocks.getMoaModels.mockResolvedValueOnce(config).mockReturnValueOnce(fresh.promise)
    const { closeMenu, onSelectModel, view } = renderPanel()

    fireEvent.click(await screen.findByText('MoA: guarded'))
    await waitFor(() => expect(hermesMocks.getMoaModels).toHaveBeenCalledTimes(2))
    view.unmount()

    await act(async () => {
      fresh.resolve(config)
      await fresh.promise
    })

    expect(onSelectModel).not.toHaveBeenCalled()
    expect(closeMenu).not.toHaveBeenCalled()
  })

  it('invalidates a pending activation when the menu session source is replaced', async () => {
    const config = configFromEntries([['guarded', true]])
    const fresh = deferred<MoaConfigResponse>()
    hermesMocks.getGlobalModelOptions.mockResolvedValue(modelOptions(['guarded']))
    hermesMocks.getMoaModels.mockResolvedValueOnce(config).mockReturnValueOnce(fresh.promise)
    const { closeMenu, onSelectModel } = renderPanel()

    fireEvent.click(await screen.findByText('MoA: guarded'))
    await waitFor(() => expect(hermesMocks.getMoaModels).toHaveBeenCalledTimes(2))

    act(() => {
      $activeSessionId.set('runtime-2')
    })

    await act(async () => {
      fresh.resolve(config)
      await fresh.promise
    })

    expect(onSelectModel).not.toHaveBeenCalled()
    expect(closeMenu).not.toHaveBeenCalled()
  })

  it('drops profile A state immediately and never paints a stale A response into profile B', async () => {
    const profileAResponse = deferred<MoaConfigResponse>()
    const profileBResponse = deferred<MoaConfigResponse>()
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    hermesMocks.getGlobalModelOptions.mockResolvedValue(modelOptions(['profile-a', 'profile-b']))
    hermesMocks.getMoaModels.mockReturnValueOnce(profileAResponse.promise).mockReturnValueOnce(profileBResponse.promise)
    $activeGatewayProfile.set('profile-a')
    client.setQueryData(['moa-menu-config', 'profile-a'], configFromEntries([['profile-a', true]]))
    renderPanel({ client })

    expect(await screen.findByText('MoA: profile-a')).toBeTruthy()
    await waitFor(() => expect(hermesMocks.getMoaModels).toHaveBeenCalledOnce())

    act(() => {
      $activeGatewayProfile.set('profile-b')
    })

    expect(screen.queryByText('MoA: profile-a')).toBeNull()

    await act(async () => {
      profileBResponse.resolve(configFromEntries([['profile-b', true]]))
      await profileBResponse.promise
    })

    expect(await screen.findByText('MoA: profile-b')).toBeTruthy()

    await act(async () => {
      profileAResponse.resolve(configFromEntries([['profile-a', true]]))
      await profileAResponse.promise
    })

    expect(screen.queryByText('MoA: profile-a')).toBeNull()
    expect(screen.getByText('MoA: profile-b')).toBeTruthy()
  })

  it('does not select profile A after activation revalidation outlives a switch to profile B', async () => {
    const freshProfileA = deferred<MoaConfigResponse>()
    const profileA = configFromEntries([['guarded', true]])
    hermesMocks.getGlobalModelOptions.mockResolvedValue(modelOptions(['guarded']))
    hermesMocks.getMoaModels
      .mockResolvedValueOnce(profileA)
      .mockReturnValueOnce(freshProfileA.promise)
      .mockResolvedValueOnce(configFromEntries([['guarded', false]]))
    $activeGatewayProfile.set('profile-a')
    const { closeMenu, onSelectModel } = renderPanel()

    fireEvent.click(await screen.findByText('MoA: guarded'))
    await waitFor(() => expect(hermesMocks.getMoaModels).toHaveBeenCalledTimes(2))

    act(() => {
      $activeGatewayProfile.set('profile-b')
    })

    await waitFor(() => expect(hermesMocks.getMoaModels).toHaveBeenCalledTimes(3))
    await act(async () => {
      freshProfileA.resolve(profileA)
      await freshProfileA.promise
    })

    expect(onSelectModel).not.toHaveBeenCalled()
    expect(closeMenu).not.toHaveBeenCalled()
    expect(screen.queryByText('MoA: guarded')).toBeNull()
  })

  it('does not close the new profile menu when a persistent selection outlives the profile switch', async () => {
    const selection = deferred<boolean>()
    const config = configFromEntries([['guarded', true]])
    hermesMocks.getGlobalModelOptions.mockResolvedValue(modelOptions(['guarded']))
    hermesMocks.getMoaModels.mockResolvedValue(config)
    $activeGatewayProfile.set('profile-a')
    const closeMenu = createCloseMenu()
    const onSelectModel = vi.fn(() => selection.promise)

    renderPanel({ closeMenu, onSelectModel })

    fireEvent.click(await screen.findByText('MoA: guarded'))
    await waitFor(() =>
      expect(onSelectModel).toHaveBeenCalledWith({ model: 'guarded', provider: 'moa', sessionId: 'runtime-1' })
    )

    act(() => {
      $activeGatewayProfile.set('profile-b')
    })

    await act(async () => {
      selection.resolve(true)
      await selection.promise
    })

    expect(closeMenu).not.toHaveBeenCalled()
  })

  it('renders and activates own reserved names while rejecting an inherited catalog pointer', async () => {
    const presets = Object.create({ inherited: presetConfig(true) }) as Record<string, MoaPresetConfig>

    for (const name of ['__proto__', 'constructor', 'toString']) {
      Object.defineProperty(presets, name, {
        configurable: true,
        enumerable: true,
        value: presetConfig(true),
        writable: true
      })
    }

    const config = moaConfig(presets)
    hermesMocks.getGlobalModelOptions.mockResolvedValue(
      modelOptions(['__proto__', 'constructor', 'toString', 'inherited'])
    )
    hermesMocks.getMoaModels.mockResolvedValue(config)
    const { closeMenu, onSelectModel } = renderPanel()

    for (const name of ['__proto__', 'constructor', 'toString']) {
      fireEvent.click(await screen.findByText(`MoA: ${name}`))
      await waitFor(() =>
        expect(onSelectModel).toHaveBeenCalledWith({ model: name, provider: 'moa', sessionId: 'runtime-1' })
      )
    }

    expect(screen.queryByText('MoA: inherited')).toBeNull()
    expect(onSelectModel.mock.calls.map(([selection]) => selection)).toEqual([
      { model: '__proto__', provider: 'moa', sessionId: 'runtime-1' },
      { model: 'constructor', provider: 'moa', sessionId: 'runtime-1' },
      { model: 'toString', provider: 'moa', sessionId: 'runtime-1' }
    ])
    expect(closeMenu).toHaveBeenCalledTimes(3)
  })

  it('refreshes MoA config independently so its GET failure cannot fail provider refresh', async () => {
    const initialOptions = modelOptions(['enabled'])
    const refreshedOptions = modelOptions(['enabled', 'new-catalog-entry'])
    hermesMocks.getGlobalModelOptions.mockResolvedValueOnce(initialOptions).mockResolvedValueOnce(refreshedOptions)
    hermesMocks.getMoaModels
      .mockResolvedValueOnce(configFromEntries([['enabled', true]]))
      .mockRejectedValueOnce(new Error('moa refresh failed'))
    const { client } = renderPanel()

    expect(await screen.findByText('MoA: enabled')).toBeTruthy()
    fireEvent.click(screen.getByText('Refresh Models'))

    await waitFor(() =>
      expect(client.getQueryData(['model-options', 'default', 'runtime-1'])).toEqual(refreshedOptions)
    )
    await waitFor(() => expect(hermesMocks.getMoaModels).toHaveBeenCalledTimes(2))
    await waitFor(() => expect(screen.queryByText('MoA: enabled')).toBeNull())
  })
})
