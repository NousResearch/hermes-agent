import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import { $activeGatewayProfile } from '@/store/profile'
import type { MoaConfigResponse, ModelOptionProvider } from '@/types/hermes'

import { MoaPresetStudio } from './moa-preset-studio'

const saveMoaModels = vi.fn()
const getMoaModels = vi.fn()

beforeAll(() => {
  Element.prototype.scrollIntoView = vi.fn()
  Element.prototype.hasPointerCapture = vi.fn(() => false)
  Element.prototype.releasePointerCapture = vi.fn()
})

vi.mock('@/hermes', () => ({
  createMoaModelsSaveRequest: (body: unknown) => () => saveMoaModels(body),
  getMoaModels: () => getMoaModels(),
  saveMoaModels: (body: unknown) => saveMoaModels(body),
  setApiRequestProfile: vi.fn()
}))

const providers: ModelOptionProvider[] = [
  { authenticated: true, models: ['hermes-4'], name: 'Nous', slug: 'nous' },
  {
    authenticated: true,
    models: ['anthropic/claude-opus-4.8'],
    name: 'OpenRouter',
    slug: 'openrouter'
  },
  { authenticated: true, models: ['default'], name: 'Mixture of Agents', slug: 'moa' }
]

const configFixture = (): MoaConfigResponse => ({
  default_preset: 'default',
  active_preset: '',
  presets: {
    default: {
      reference_models: [{ continuity_id: 'advisor', provider: 'nous', model: 'hermes-4' }],
      aggregator: { provider: 'openrouter', model: 'anthropic/claude-opus-4.8' },
      reference_temperature: null,
      aggregator_temperature: null,
      max_tokens: 4096,
      reference_max_tokens: 600,
      fanout: 'user_turn',
      enabled: true
    }
  },
  reference_models: [{ continuity_id: 'advisor', provider: 'nous', model: 'hermes-4' }],
  aggregator: { provider: 'openrouter', model: 'anthropic/claude-opus-4.8' },
  reference_temperature: 0,
  aggregator_temperature: 0,
  max_tokens: 4096,
  enabled: true
})

function renderStudio(
  config = configFixture(),
  onUseMoaPreset?: (name: string) => boolean | Promise<boolean> | Promise<void> | void
) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })

  const view = render(
    <QueryClientProvider client={client}>
      <MoaPresetStudio config={config} onUseMoaPreset={onUseMoaPreset} providers={providers} />
    </QueryClientProvider>
  )

  return Object.assign(view, { client })
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

beforeEach(() => {
  getMoaModels.mockReset()
  saveMoaModels.mockReset()
  getMoaModels.mockResolvedValue(configFixture())
  $activeGatewayProfile.set('default')
})

afterEach(() => {
  cleanup()
  $activeGatewayProfile.set('default')
  vi.useRealTimers()
})

describe('MoaPresetStudio persistence feedback', () => {
  it('opens a blank unsaved preset and saves only after its required slots are complete', async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true })
    saveMoaModels.mockImplementation((body: MoaConfigResponse) => Promise.resolve(body))
    renderStudio()

    expect(screen.getByRole('button', { name: 'Add preset' })).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Duplicate preset' })).toBeTruthy()

    fireEvent.change(screen.getByRole('textbox', { name: 'New preset name' }), {
      target: { value: 'fresh preset' }
    })
    fireEvent.click(screen.getByRole('button', { name: 'Add preset' }))

    expect(saveMoaModels).not.toHaveBeenCalled()
    expect(screen.getByRole('combobox', { name: 'Preset' }).textContent).toContain('fresh preset')
    expect(screen.getByRole('combobox', { name: 'Reference 1 provider' }).textContent).not.toContain('Nous')
    expect(screen.getByRole('combobox', { name: 'Aggregator provider' }).textContent).not.toContain('OpenRouter')

    fireEvent.click(screen.getByRole('combobox', { name: 'Reference 1 provider' }))
    fireEvent.click(await screen.findByRole('option', { name: 'Nous' }))
    fireEvent.click(screen.getByRole('combobox', { name: 'Reference 1 model' }))
    fireEvent.click(await screen.findByRole('option', { name: 'hermes-4' }))
    fireEvent.click(screen.getByRole('combobox', { name: 'Aggregator provider' }))
    fireEvent.click(await screen.findByRole('option', { name: 'OpenRouter' }))
    fireEvent.click(screen.getByRole('combobox', { name: 'Aggregator model' }))
    fireEvent.click(await screen.findByRole('option', { name: 'anthropic/claude-opus-4.8' }))

    await act(async () => {
      await vi.advanceTimersByTimeAsync(700)
    })

    expect(saveMoaModels).not.toHaveBeenCalled()
    fireEvent.click(screen.getByRole('button', { name: 'Save changes' }))
    expect(saveMoaModels).toHaveBeenCalledOnce()
    const sent = saveMoaModels.mock.calls[0][0] as MoaConfigResponse
    expect(sent.presets['fresh preset']).toMatchObject({
      aggregator: { model: 'anthropic/claude-opus-4.8', provider: 'openrouter' },
      enabled: true,
      reference_models: [{ model: 'hermes-4', provider: 'nous' }]
    })
    expect(sent.presets['fresh preset'].reference_models[0].continuity_id).toBeUndefined()
  })

  it('discards an unfinished blank preset on unmount without writing it', () => {
    const view = renderStudio()

    fireEvent.change(screen.getByRole('textbox', { name: 'New preset name' }), {
      target: { value: 'unfinished' }
    })
    fireEvent.click(screen.getByRole('button', { name: 'Add preset' }))
    view.unmount()

    expect(saveMoaModels).not.toHaveBeenCalled()
  })

  it('invalidates model options and mounted-menu MoA config after an accepted save', async () => {
    saveMoaModels.mockImplementation((body: MoaConfigResponse) => Promise.resolve(body))
    const { client } = renderStudio()
    const invalidate = vi.spyOn(client, 'invalidateQueries')

    fireEvent.change(screen.getByRole('textbox', { name: 'New preset name' }), { target: { value: 'copy' } })
    fireEvent.click(screen.getByRole('button', { name: 'Duplicate preset' }))

    expect(saveMoaModels).not.toHaveBeenCalled()
    fireEvent.click(screen.getByRole('button', { name: 'Save changes' }))
    await waitFor(() => expect(saveMoaModels).toHaveBeenCalledOnce())
    await waitFor(() => expect(invalidate).toHaveBeenCalledWith({ queryKey: ['model-options'] }))
    expect(invalidate).toHaveBeenCalledWith({ queryKey: ['moa-menu-config'] })
  })

  it('retains a failed manual save as an editable draft and exposes retry status', async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true })
    saveMoaModels.mockRejectedValueOnce(new Error('disk is read-only'))
    const onUseMoaPreset = vi.fn()
    renderStudio(configFixture(), onUseMoaPreset)

    const enabled = screen.getByRole('switch', { name: 'Preset enabled' })
    expect(enabled.getAttribute('aria-checked')).toBe('true')

    fireEvent.click(enabled)
    expect(enabled.getAttribute('aria-checked')).toBe('false')

    expect(saveMoaModels).not.toHaveBeenCalled()
    fireEvent.click(screen.getByRole('button', { name: 'Save changes' }))

    await waitFor(() => expect(screen.getByRole('alert').textContent).toContain('disk is read-only'))
    expect(screen.getByText('Save failed — changes remain unsaved.')).toBeTruthy()
    expect(screen.getByRole('switch', { name: 'Preset enabled' }).getAttribute('aria-checked')).toBe('false')
    expect((screen.getByRole('button', { name: 'Save changes' }) as HTMLButtonElement).disabled).toBe(false)

    const usePreset = screen.getByRole('button', { name: 'Use in this chat' }) as HTMLButtonElement
    expect(usePreset.disabled).toBe(true)
    fireEvent.click(usePreset)
    expect(getMoaModels).not.toHaveBeenCalled()
    expect(onUseMoaPreset).not.toHaveBeenCalled()
  })

  it('ignores an older save completion after a new authoritative config arrives', async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true })
    const pending = deferred<MoaConfigResponse>()
    saveMoaModels.mockReturnValueOnce(pending.promise)
    const view = renderStudio()

    fireEvent.click(screen.getByRole('switch', { name: 'Preset enabled' }))
    fireEvent.click(screen.getByRole('button', { name: 'Save changes' }))

    expect(saveMoaModels).toHaveBeenCalledTimes(1)

    const profileB = configFixture()
    profileB.default_preset = 'profile-b'
    profileB.presets = { 'profile-b': { ...profileB.presets.default, enabled: true } }
    view.rerender(
      <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
        <MoaPresetStudio config={profileB} providers={providers} />
      </QueryClientProvider>
    )

    await act(async () => {
      pending.resolve(configFixture())
      await pending.promise
    })

    expect(screen.getByRole('combobox', { name: 'Preset' }).textContent).toContain('profile-b')
    expect(screen.getByRole('switch', { name: 'Preset enabled' }).getAttribute('aria-checked')).toBe('true')
  })

  it('blocks preset switching while a manual save is pending and permits it after acceptance', async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true })
    const pending = deferred<MoaConfigResponse>()
    const config = configFixture()
    config.presets.backup = {
      ...config.presets.default,
      reference_models: config.presets.default.reference_models.map(slot => ({ ...slot }))
    }
    saveMoaModels.mockReturnValueOnce(pending.promise)
    renderStudio(config)

    fireEvent.click(screen.getByRole('switch', { name: 'Preset enabled' }))
    fireEvent.click(screen.getByRole('button', { name: 'Save changes' }))

    const saved = saveMoaModels.mock.calls[0][0] as MoaConfigResponse
    expect(screen.getByRole('combobox', { name: 'Preset' }).matches(':disabled')).toBe(true)

    await act(async () => {
      pending.resolve(saved)
      await pending.promise
    })

    fireEvent.click(screen.getByRole('combobox', { name: 'Preset' }))
    fireEvent.click(await screen.findByRole('option', { name: 'backup' }))
    expect(screen.getByRole('combobox', { name: 'Preset' }).textContent).toContain('backup')
  })

  it('retains accumulated field and lifecycle edits when the manual save fails', async () => {
    saveMoaModels.mockRejectedValueOnce(new Error('manual save failed'))
    renderStudio()

    fireEvent.click(screen.getByRole('switch', { name: 'Preset enabled' }))
    fireEvent.change(screen.getByRole('textbox', { name: 'New preset name' }), { target: { value: 'copy' } })
    fireEvent.click(screen.getByRole('button', { name: 'Duplicate preset' }))

    expect(saveMoaModels).not.toHaveBeenCalled()
    expect(screen.getByRole('combobox', { name: 'Preset' }).textContent).toContain('copy')
    fireEvent.click(screen.getByRole('button', { name: 'Save changes' }))

    await waitFor(() => expect(screen.getByRole('alert').textContent).toContain('manual save failed'))
    expect(screen.getByRole('combobox', { name: 'Preset' }).textContent).toContain('copy')
    expect(screen.getByRole('switch', { name: 'Preset enabled' }).getAttribute('aria-checked')).toBe('false')
  })
})

describe('MoaPresetStudio manual save lifecycle', () => {
  it('clears manual-save busy state after the PUT settles', async () => {
    vi.useFakeTimers()
    const lifecycle = deferred<MoaConfigResponse>()
    saveMoaModels.mockReturnValueOnce(lifecycle.promise)
    renderStudio()

    fireEvent.change(screen.getByRole('textbox', { name: 'New preset name' }), { target: { value: 'copy' } })
    fireEvent.click(screen.getByRole('button', { name: 'Duplicate preset' }))

    const setDefault = screen.getByRole('button', { name: 'Set default' }) as HTMLButtonElement
    expect(setDefault.disabled).toBe(false)

    fireEvent.click(screen.getByRole('button', { name: 'Save changes' }))
    expect(setDefault.disabled).toBe(true)
    const lifecycleBody = saveMoaModels.mock.calls[0][0] as MoaConfigResponse

    await act(async () => {
      lifecycle.resolve(lifecycleBody)
      await lifecycle.promise
    })

    expect(setDefault.disabled).toBe(false)
  })

  it('disables mutable controls and blocks overlapping operations while Save is pending', async () => {
    vi.useFakeTimers()
    const lifecycle = deferred<MoaConfigResponse>()
    saveMoaModels.mockReturnValueOnce(lifecycle.promise)
    renderStudio()

    fireEvent.change(screen.getByRole('textbox', { name: 'New preset name' }), { target: { value: 'copy' } })
    fireEvent.click(screen.getByRole('button', { name: 'Duplicate preset' }))
    fireEvent.click(screen.getByRole('button', { name: 'Save changes' }))

    const presetSelect = screen.getByRole('combobox', { name: 'Preset' })
    const presetName = screen.getByRole('textbox', { name: 'Preset name' })
    const enabled = screen.getByRole('switch', { name: 'Preset enabled' })
    expect(presetSelect.matches(':disabled')).toBe(true)
    expect(presetName.matches(':disabled')).toBe(true)
    expect(enabled.matches(':disabled')).toBe(true)

    fireEvent.click(enabled)
    fireEvent.click(screen.getByRole('button', { name: 'Set default' }))

    expect(enabled.getAttribute('aria-checked')).toBe('true')
    expect(saveMoaModels).toHaveBeenCalledTimes(1)

    const lifecycleBody = saveMoaModels.mock.calls[0][0] as MoaConfigResponse
    await act(async () => {
      lifecycle.resolve(lifecycleBody)
      await lifecycle.promise
    })

    expect(presetSelect.matches(':disabled')).toBe(false)
    expect(presetName.matches(':disabled')).toBe(false)
    expect(enabled.matches(':disabled')).toBe(false)
  })
})

describe('MoaPresetStudio activation', () => {
  it('keeps Use disabled from a just-enabled edit through its accepted save, then revalidates before selecting', async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true })
    const pendingSave = deferred<MoaConfigResponse>()
    const config = configFixture()
    config.presets.default.enabled = false
    saveMoaModels.mockReturnValueOnce(pendingSave.promise)
    const onUseMoaPreset = vi.fn(() => true)
    renderStudio(config, onUseMoaPreset)

    fireEvent.click(screen.getByRole('switch', { name: 'Preset enabled' }))

    const usePreset = screen.getByRole('button', { name: 'Use in this chat' }) as HTMLButtonElement
    expect(usePreset.disabled).toBe(true)
    expect(screen.getByRole('status').textContent).toContain('Unsaved changes')
    fireEvent.click(usePreset)
    expect(getMoaModels).not.toHaveBeenCalled()
    expect(onUseMoaPreset).not.toHaveBeenCalled()

    expect(saveMoaModels).not.toHaveBeenCalled()
    fireEvent.click(screen.getByRole('button', { name: 'Save changes' }))
    expect(screen.getByRole('status').textContent).toContain('Saving')
    expect(saveMoaModels).toHaveBeenCalledOnce()
    expect(usePreset.disabled).toBe(true)
    const saved = saveMoaModels.mock.calls[0][0] as MoaConfigResponse
    getMoaModels.mockResolvedValue(saved)

    await act(async () => {
      pendingSave.resolve(saved)
      await pendingSave.promise
    })

    await waitFor(() => expect(usePreset.disabled).toBe(false))
    fireEvent.click(usePreset)

    await waitFor(() => expect(getMoaModels).toHaveBeenCalledOnce())
    await waitFor(() => expect(onUseMoaPreset).toHaveBeenCalledWith('default'))
  })

  it('does not activate a disabled selected preset', () => {
    const config = configFixture()
    config.presets.default.enabled = false
    const onUseMoaPreset = vi.fn()
    renderStudio(config, onUseMoaPreset)

    const usePreset = screen.getByRole('button', { name: 'Use in this chat' }) as HTMLButtonElement
    expect(usePreset.disabled).toBe(true)

    fireEvent.click(usePreset)
    expect(getMoaModels).not.toHaveBeenCalled()
    expect(onUseMoaPreset).not.toHaveBeenCalled()
  })

  it('does not activate while top-level MoA is disabled', () => {
    const config = configFixture()
    config.enabled = false
    const onUseMoaPreset = vi.fn()
    renderStudio(config, onUseMoaPreset)

    const usePreset = screen.getByRole('button', { name: 'Use in this chat' }) as HTMLButtonElement
    expect(usePreset.disabled).toBe(true)

    fireEvent.click(usePreset)
    expect(getMoaModels).not.toHaveBeenCalled()
    expect(onUseMoaPreset).not.toHaveBeenCalled()
  })

  it('fails closed on a transient fresh GET error and permits a retry', async () => {
    const config = configFixture()
    getMoaModels.mockRejectedValueOnce(new Error('temporary GET failure')).mockResolvedValueOnce(config)
    const onUseMoaPreset = vi.fn(() => true)
    renderStudio(config, onUseMoaPreset)

    const usePreset = screen.getByRole('button', { name: 'Use in this chat' }) as HTMLButtonElement
    fireEvent.click(usePreset)

    await waitFor(() => expect(getMoaModels).toHaveBeenCalledOnce())
    await waitFor(() => expect(usePreset.disabled).toBe(false))
    expect(onUseMoaPreset).not.toHaveBeenCalled()
    expect(screen.queryByRole('alert')).toBeNull()

    fireEvent.click(usePreset)

    await waitFor(() => expect(getMoaModels).toHaveBeenCalledTimes(2))
    await waitFor(() => expect(onUseMoaPreset).toHaveBeenCalledWith('default'))
  })

  it.each([
    {
      fresh: () => {
        const config = configFixture()
        config.enabled = false

        return config
      },
      label: 'top-level disabled state'
    },
    {
      fresh: () => {
        const config = configFixture()
        Reflect.deleteProperty(config, 'enabled')

        return config
      },
      label: 'missing top-level enabled state'
    },
    {
      fresh: () => {
        const config = configFixture()
        Reflect.deleteProperty(config, 'enabled')
        Object.setPrototypeOf(config, { enabled: true })

        return config
      },
      label: 'inherited top-level enabled state'
    },
    {
      fresh: () => {
        const config = configFixture()
        config.presets.default.enabled = false

        return config
      },
      label: 'disabled preset state'
    },
    {
      fresh: () => {
        const config = configFixture()
        config.presets = {}

        return config
      },
      label: 'missing preset state'
    },
    {
      fresh: () => {
        const config = configFixture()
        config.presets = Object.create({ default: config.presets.default }) as MoaConfigResponse['presets']

        return config
      },
      label: 'inherited preset state'
    }
  ])('fails closed on fresh $label', async ({ fresh }) => {
    const authoritative = fresh()
    getMoaModels.mockResolvedValue(authoritative)
    const onUseMoaPreset = vi.fn(() => true)
    const { client } = renderStudio(configFixture(), onUseMoaPreset)

    const usePreset = screen.getByRole('button', { name: 'Use in this chat' }) as HTMLButtonElement
    fireEvent.click(usePreset)

    await waitFor(() => expect(getMoaModels).toHaveBeenCalledOnce())
    await waitFor(() => expect(usePreset.disabled).toBe(false))
    expect(client.getQueryData(['moa-menu-config', 'default'])).toEqual(authoritative)
    expect(onUseMoaPreset).not.toHaveBeenCalled()
  })

  it('does not select after the active profile changes while fresh validation is pending', async () => {
    const fresh = deferred<MoaConfigResponse>()
    getMoaModels.mockReturnValueOnce(fresh.promise)
    $activeGatewayProfile.set('profile-a')
    const onUseMoaPreset = vi.fn(() => true)
    renderStudio(configFixture(), onUseMoaPreset)

    fireEvent.click(screen.getByRole('button', { name: 'Use in this chat' }))
    expect(getMoaModels).toHaveBeenCalledOnce()

    act(() => {
      $activeGatewayProfile.set('profile-b')
    })
    await act(async () => {
      fresh.resolve(configFixture())
      await fresh.promise
    })

    expect(onUseMoaPreset).not.toHaveBeenCalled()
  })

  it('does not select after the authoritative config source changes while fresh validation is pending', async () => {
    const fresh = deferred<MoaConfigResponse>()
    getMoaModels.mockReturnValueOnce(fresh.promise)
    const onUseMoaPreset = vi.fn(() => true)
    const view = renderStudio(configFixture(), onUseMoaPreset)

    fireEvent.click(screen.getByRole('button', { name: 'Use in this chat' }))
    expect(getMoaModels).toHaveBeenCalledOnce()

    const replacement = configFixture()
    replacement.max_tokens = 8192
    view.rerender(
      <QueryClientProvider client={view.client}>
        <MoaPresetStudio config={replacement} onUseMoaPreset={onUseMoaPreset} providers={providers} />
      </QueryClientProvider>
    )

    await act(async () => {
      fresh.resolve(configFixture())
      await fresh.promise
    })

    expect(onUseMoaPreset).not.toHaveBeenCalled()
  })

  it('does not select the old name after selection changes with a new config source', async () => {
    const fresh = deferred<MoaConfigResponse>()
    const config = configFixture()
    config.presets.backup = { ...config.presets.default }
    getMoaModels.mockReturnValueOnce(fresh.promise)
    const onUseMoaPreset = vi.fn(() => true)
    const view = renderStudio(config, onUseMoaPreset)

    fireEvent.click(screen.getByRole('combobox', { name: 'Preset' }))
    fireEvent.click(await screen.findByRole('option', { name: 'backup' }))
    fireEvent.click(screen.getByRole('button', { name: 'Use in this chat' }))
    expect(getMoaModels).toHaveBeenCalledOnce()

    view.rerender(
      <QueryClientProvider client={view.client}>
        <MoaPresetStudio config={configFixture()} onUseMoaPreset={onUseMoaPreset} providers={providers} />
      </QueryClientProvider>
    )

    await act(async () => {
      fresh.resolve(config)
      await fresh.promise
    })

    expect(screen.getByRole('combobox', { name: 'Preset' }).textContent).toContain('default')
    expect(onUseMoaPreset).not.toHaveBeenCalled()
  })

  it('keeps all controls disabled until the async model selection settles', async () => {
    const selection = deferred<boolean>()
    const config = configFixture()
    getMoaModels.mockResolvedValue(config)
    const onUseMoaPreset = vi.fn(() => selection.promise)
    renderStudio(config, onUseMoaPreset)

    const usePreset = screen.getByRole('button', { name: 'Use in this chat' }) as HTMLButtonElement
    const presetSelect = screen.getByRole('combobox', { name: 'Preset' })
    fireEvent.click(usePreset)

    await waitFor(() => expect(onUseMoaPreset).toHaveBeenCalledWith('default'))
    expect(usePreset.disabled).toBe(true)
    expect(presetSelect.matches(':disabled')).toBe(true)
    fireEvent.click(usePreset)
    expect(getMoaModels).toHaveBeenCalledOnce()

    await act(async () => {
      selection.resolve(true)
      await selection.promise
    })

    await waitFor(() => expect(usePreset.disabled).toBe(false))
    expect(presetSelect.matches(':disabled')).toBe(false)
  })
})

describe('MoaPresetStudio preset deletion', () => {
  it('opens confirmation, preserves the preset on cancel, and saves deletion only after Save changes', async () => {
    const config = configFixture()
    config.presets.backup = {
      ...config.presets.default,
      reference_models: config.presets.default.reference_models.map(slot => ({ ...slot }))
    }
    saveMoaModels.mockImplementation((body: MoaConfigResponse) => Promise.resolve(body))
    renderStudio(config)

    fireEvent.click(screen.getByRole('button', { name: 'Delete preset' }))

    expect(screen.getByRole('dialog')).toBeTruthy()
    expect(
      screen.getByText('Remove the “default” MoA preset from this draft? Click Save changes to apply.')
    ).toBeTruthy()
    expect(saveMoaModels).not.toHaveBeenCalled()

    fireEvent.click(screen.getByRole('button', { name: 'Cancel' }))

    await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull())
    expect(screen.getByRole('combobox', { name: 'Preset' }).textContent).toContain('default')
    expect(saveMoaModels).not.toHaveBeenCalled()

    fireEvent.click(screen.getByRole('button', { name: 'Delete preset' }))
    fireEvent.click(within(screen.getByRole('dialog')).getByRole('button', { name: 'Delete preset' }))

    await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull())
    expect(saveMoaModels).not.toHaveBeenCalled()
    expect(screen.getByRole('combobox', { name: 'Preset' }).textContent).toContain('backup')
    fireEvent.click(screen.getByRole('button', { name: 'Save changes' }))

    await waitFor(() => expect(saveMoaModels).toHaveBeenCalledTimes(1))
    const sent = saveMoaModels.mock.calls[0][0] as MoaConfigResponse
    expect(sent.presets.default).toBeUndefined()
    expect(sent.presets.backup).toBeTruthy()
    expect(sent.default_preset).toBe('backup')
    expect(screen.getByRole('combobox', { name: 'Preset' }).textContent).toContain('backup')
  })

  it('prevents deleting the final preset', () => {
    renderStudio()

    const deleteButton = screen.getByRole('button', { name: 'Delete preset' })
    expect((deleteButton as HTMLButtonElement).disabled).toBe(true)
    fireEvent.click(deleteButton)

    expect(screen.queryByRole('dialog')).toBeNull()
    expect(saveMoaModels).not.toHaveBeenCalled()
  })
})
