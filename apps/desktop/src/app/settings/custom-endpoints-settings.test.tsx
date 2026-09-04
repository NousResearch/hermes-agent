import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { CustomEndpoint, CustomEndpointUpdate } from '@/types/hermes'

import { CustomEndpointsSettings } from './custom-endpoints-settings'

// #101764: Test finds N models, Save persisted all N and flooded the picker.
// There was no way to keep only some. These tests pin the selection contract.
vi.mock('@/hermes', () => ({
  activateCustomEndpoint: vi.fn(),
  deleteCustomEndpoint: vi.fn(),
  getCustomEndpoints: vi.fn(),
  saveCustomEndpoint: vi.fn(),
  validateCustomEndpoint: vi.fn()
}))

vi.mock('@/store/confirm', () => ({ confirm: vi.fn().mockResolvedValue(true) }))
vi.mock('@/store/notifications', () => ({ notify: vi.fn(), notifyError: vi.fn() }))
vi.mock('@/lib/haptics', () => ({ triggerHaptic: vi.fn() }))

import * as hermes from '@/hermes'

const mocked = vi.mocked(hermes)

// A plan-limited endpoint: the provider advertises many, the user pays for two.
const DISCOVERED = [
  'glm-5.3-flash',
  'qwen3.8-flash',
  'expensive-opus-a',
  'expensive-opus-b',
  'expensive-opus-c'
]

const ENDPOINT: CustomEndpoint = {
  base_url: 'https://b.ai/v1',
  discover_models: true,
  has_api_key: true,
  id: 'b-ai',
  is_current: true,
  model: 'glm-5.3-flash',
  models: DISCOVERED,
  name: 'B.AI'
}

function savedPayload(): CustomEndpointUpdate {
  expect(mocked.saveCustomEndpoint).toHaveBeenCalledTimes(1)

  return mocked.saveCustomEndpoint.mock.calls[0][0]
}

beforeEach(() => {
  mocked.getCustomEndpoints.mockResolvedValue({
    current: { base_url: ENDPOINT.base_url, model: ENDPOINT.model, provider: ENDPOINT.id },
    endpoints: [ENDPOINT]
  })
  mocked.saveCustomEndpoint.mockResolvedValue({
    current: { base_url: ENDPOINT.base_url, model: ENDPOINT.model, provider: ENDPOINT.id },
    endpoints: [ENDPOINT],
    id: ENDPOINT.id,
    ok: true
  })
  mocked.validateCustomEndpoint.mockResolvedValue({
    message: '',
    models: DISCOVERED,
    ok: true,
    reachable: true
  })
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

async function renderPane() {
  const result = render(<CustomEndpointsSettings />)

  // The editor hydrates from the current endpoint; wait for the picker.
  await screen.findByText(/Models to keep/i)

  return result
}

describe('#101764 custom endpoint model selection', () => {
  it('lists every discovered model as a checkbox, all selected by default', async () => {
    await renderPane()

    expect(screen.getByText(`Models to keep (${DISCOVERED.length}/${DISCOVERED.length})`)).toBeTruthy()

    // The endpoint row also shows each model's name, so assert presence via
    // getAllByText and require every model to be rendered at least once.
    for (const model of DISCOVERED) {
      expect(screen.getAllByText(model).length).toBeGreaterThan(0)
    }
  })

  it('saves only the checked models and asserts catalogue authority', async () => {
    await renderPane()

    // Deselect the three models this plan does not cover.
    for (const model of ['expensive-opus-a', 'expensive-opus-b', 'expensive-opus-c']) {
      const row = screen.getByText(model).closest('label')

      expect(row).toBeTruthy()
      fireEvent.click(row!.querySelector('button, input')!)
    }

    expect(screen.getByText(`Models to keep (2/${DISCOVERED.length})`)).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: /^Save$/i }))

    await waitFor(() => expect(mocked.saveCustomEndpoint).toHaveBeenCalled())

    const payload = savedPayload()

    expect(payload.models).toEqual(['glm-5.3-flash', 'qwen3.8-flash'])
    // Without this flag the backend merges additively and deselection is a
    // no-op — the whole point of the issue.
    expect(payload.replace_models).toBe(true)
  })

  it('always keeps the default model even when unchecked', async () => {
    await renderPane()

    fireEvent.click(screen.getByRole('button', { name: /Select none/i }))
    expect(screen.getByText(`Models to keep (0/${DISCOVERED.length})`)).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: /^Save$/i }))
    await waitFor(() => expect(mocked.saveCustomEndpoint).toHaveBeenCalled())

    // A saved endpoint with zero models would leave the picker empty and the
    // provider unusable; the typed default is non-negotiable.
    expect(savedPayload().models).toEqual(['glm-5.3-flash'])
  })

  it('Select all re-checks the full discovered catalogue', async () => {
    await renderPane()

    fireEvent.click(screen.getByRole('button', { name: /Select none/i }))
    fireEvent.click(screen.getByRole('button', { name: /Select all/i }))

    fireEvent.click(screen.getByRole('button', { name: /^Save$/i }))
    await waitFor(() => expect(mocked.saveCustomEndpoint).toHaveBeenCalled())

    expect(savedPayload().models).toEqual(DISCOVERED)
  })

  it('re-running Test preserves the existing selection instead of re-checking everything', async () => {
    await renderPane()

    for (const model of ['expensive-opus-a', 'expensive-opus-b', 'expensive-opus-c']) {
      fireEvent.click(screen.getByText(model).closest('label')!.querySelector('button, input')!)
    }

    fireEvent.click(screen.getByRole('button', { name: /^Test$/i }))
    await waitFor(() => expect(mocked.validateCustomEndpoint).toHaveBeenCalled())

    // Still 2 of 5 — a probe must not silently undo the user's narrowing.
    await waitFor(() => expect(screen.getByText(`Models to keep (2/${DISCOVERED.length})`)).toBeTruthy())
  })

  it('drops selected models the endpoint no longer advertises', async () => {
    await renderPane()

    mocked.validateCustomEndpoint.mockResolvedValueOnce({
      message: '',
      models: ['glm-5.3-flash'],
      ok: true,
      reachable: true
    })

    fireEvent.click(screen.getByRole('button', { name: /^Test$/i }))
    await waitFor(() => expect(mocked.validateCustomEndpoint).toHaveBeenCalled())

    await waitFor(() => expect(screen.getByText('Models to keep (1/1)')).toBeTruthy())
  })

  it('Test does not claim catalogue authority', async () => {
    await renderPane()

    fireEvent.click(screen.getByRole('button', { name: /^Test$/i }))
    await waitFor(() => expect(mocked.validateCustomEndpoint).toHaveBeenCalled())

    // Probing is not saving: a validate payload must never carry the flag
    // that authorizes deletion.
    expect(mocked.validateCustomEndpoint.mock.calls[0][0].replace_models).toBeFalsy()
  })

  it('shows the stored catalogue size in the endpoint list', async () => {
    await renderPane()

    // A flooded save should be visible without opening the editor.
    expect(screen.getByText(`${DISCOVERED.length} models`)).toBeTruthy()
  })

  it('hides the picker when nothing has been discovered', async () => {
    mocked.getCustomEndpoints.mockResolvedValue({
      current: { base_url: '', model: '', provider: '' },
      endpoints: []
    })

    render(<CustomEndpointsSettings />)

    await screen.findByText(/No custom endpoints/i)
    expect(screen.queryByText(/Models to keep/i)).toBeNull()
  })
})
