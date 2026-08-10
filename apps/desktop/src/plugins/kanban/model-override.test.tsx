import { DropdownMenu, DropdownMenuContent, ModelCatalogMenu, type ModelMenuController } from '@hermes/plugin-sdk'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import {
  EMPTY_OVERRIDE,
  isInherited,
  overrideCreateFields,
  overrideLabel,
  overridePatch,
  type TaskModelOverride
} from './model-override'

// Radix calls these on open; jsdom doesn't implement them.
beforeAll(() => {
  Element.prototype.scrollIntoView = vi.fn()
  Element.prototype.hasPointerCapture = vi.fn(() => false)
  Element.prototype.releasePointerCapture = vi.fn()
})

const getGlobalModelOptions = vi.fn()

vi.mock('@/hermes', () => ({
  getGlobalModelOptions: (...args: unknown[]) => getGlobalModelOptions(...args),
  setApiRequestProfile: vi.fn()
}))

beforeEach(() => {
  getGlobalModelOptions.mockResolvedValue({
    providers: [{ models: ['gemini-3.1-pro', 'gemini-2.5-flash'], name: 'Google', slug: 'google' }]
  })
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

describe('override label', () => {
  it('reads as inherited until something is pinned', () => {
    expect(isInherited(EMPTY_OVERRIDE)).toBe(true)
    expect(overrideLabel(EMPTY_OVERRIDE, 'Profile default')).toBe('Profile default')
  })

  it('shows provider: model · Effort', () => {
    expect(overrideLabel({ effort: 'high', model: 'gemini-3.1-pro', provider: 'google' }, 'x')).toBe(
      'google: gemini-3.1-pro · High'
    )
  })

  it('a depth-only pin is still an override', () => {
    const depthOnly: TaskModelOverride = { effort: 'ultra', model: '', provider: '' }

    expect(isInherited(depthOnly)).toBe(false)
    expect(overrideLabel(depthOnly, 'Profile default')).toBe('Profile default · Ultra')
  })
})

// The override drives the SHARED composer menu through a controller. This is
// the seam that keeps the two surfaces identical, so exercise the real
// component rather than a stand-in.
describe('the shared catalog menu, driven by an override controller', () => {
  function renderMenu(value: TaskModelOverride = EMPTY_OVERRIDE) {
    const onChange = vi.fn()

    const controller: ModelMenuController = {
      applyPreset: (preset, row) => onChange({ effort: preset.effort ?? '', model: row.model, provider: row.provider }),
      current: { effort: value.effort, fast: false, model: value.model, provider: value.provider },
      presetFor: () => ({}),
      select: (model, provider) => onChange({ ...value, model, provider }),
      setOptions: (patch, row) => {
        if (patch.effort !== undefined) {
          onChange({ effort: patch.effort, model: row.model, provider: row.provider })
        }
      }
    }

    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })

    render(
      <QueryClientProvider client={client}>
        <DropdownMenu open>
          <DropdownMenuContent>
            <ModelCatalogMenu controller={controller} />
          </DropdownMenuContent>
        </DropdownMenu>
      </QueryClientProvider>
    )

    return onChange
  }

  it('lists the catalog and reports a picked model', async () => {
    const onChange = renderMenu()

    // Rows render the display name; the controller receives the raw id.
    fireEvent.click(await screen.findByText(/Gemini 3\.1 Pro/i))

    const picked = onChange.mock.calls.at(-1)![0]

    expect(picked.model).toBe('gemini-3.1-pro')
    expect(picked.provider).toBe('google')
  })

  it('never renders MoA presets — a preset is not a worker model', async () => {
    getGlobalModelOptions.mockResolvedValue({
      providers: [
        { models: ['gemini-3.1-pro'], name: 'Google', slug: 'google' },
        { models: ['BeastMode'], name: 'Mixture of Agents', slug: 'moa' }
      ]
    })

    renderMenu()
    await screen.findByText(/Gemini 3\.1 Pro/i)

    expect(screen.queryByText(/BeastMode/)).toBeNull()
  })
})

// A PATCH body can't say "set to NULL" with a missing key, so clears are
// explicit flags; a create should instead OMIT what the user never touched so
// the backend's own defaults still apply.
describe('REST field mapping', () => {
  it('create omits untouched fields', () => {
    expect(overrideCreateFields(EMPTY_OVERRIDE)).toEqual({})
    expect(overrideCreateFields({ effort: 'high', model: 'm', provider: 'p' })).toEqual({
      model_override: 'm',
      provider_override: 'p',
      reasoning_effort: 'high'
    })
  })

  it('patch clears explicitly', () => {
    expect(overridePatch(EMPTY_OVERRIDE)).toEqual({ clear_model_override: true, clear_reasoning_effort: true })
  })

  it('patch can pin depth alone, leaving the model to the profile', () => {
    expect(overridePatch({ effort: 'ultra', model: '', provider: '' })).toEqual({
      clear_model_override: true,
      reasoning_effort: 'ultra'
    })
  })

  it('a provider is never sent without its model', () => {
    expect(overrideCreateFields({ effort: '', model: '', provider: 'google' })).toEqual({})
  })
})
