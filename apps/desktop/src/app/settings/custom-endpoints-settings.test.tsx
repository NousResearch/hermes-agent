import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'
import { stubResizeObserver } from '@/test/jsdom'
import type { CustomEndpoint, CustomEndpointsResponse } from '@/types/hermes'

import { CustomEndpointsSettings } from './custom-endpoints-settings'

stubResizeObserver()

vi.mock('@/hermes', () => ({
  activateCustomEndpoint: vi.fn(),
  deleteCustomEndpoint: vi.fn(),
  getCustomEndpoints: vi.fn(),
  saveCustomEndpoint: vi.fn(),
  validateCustomEndpoint: vi.fn()
}))

import * as hermes from '@/hermes'

const mocked = vi.mocked(hermes)

const SAVED_ENDPOINT: CustomEndpoint = {
  api_key_preview: 'sk-s...kI0c',
  base_url: 'https://spark.example.com/v1',
  context_length: null,
  discover_models: true,
  has_api_key: true,
  id: 'custom',
  is_current: true,
  model: 'qwen3.8-27b',
  models: ['qwen3.8-27b'],
  name: 'Custom',
  source: 'providers'
}

function listing(endpoints: CustomEndpoint[], id?: string): CustomEndpointsResponse {
  return {
    current: { base_url: SAVED_ENDPOINT.base_url, model: SAVED_ENDPOINT.model, provider: 'custom' },
    endpoints,
    id,
    ok: true
  }
}

async function renderSettings() {
  await act(async () => {
    render(
      <MemoryRouter>
        <I18nProvider>
          <CustomEndpointsSettings />
        </I18nProvider>
      </MemoryRouter>
    )
  })
  await screen.findByText('Edit Endpoint')
}

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

describe('CustomEndpointsSettings API key', () => {
  it('tells the user a key is on file instead of showing an empty field', async () => {
    mocked.getCustomEndpoints.mockResolvedValue(listing([SAVED_ENDPOINT]))

    await renderSettings()

    expect((screen.getByPlaceholderText('Leave blank to keep the saved key (sk-s...kI0c)') as HTMLInputElement).value).toBe('')
    expect(screen.getByText('sk-s...kI0c')).toBeTruthy()
  })

  it('says when a saved endpoint has no key on file', async () => {
    mocked.getCustomEndpoints.mockResolvedValue(
      listing([{ ...SAVED_ENDPOINT, api_key_preview: null, has_api_key: false }])
    )

    await renderSettings()

    expect(screen.getByPlaceholderText('No key saved for this endpoint (optional)')).toBeTruthy()
  })

  it('surfaces an unresolved key_env even though no key is usable', async () => {
    mocked.getCustomEndpoints.mockResolvedValue(
      listing([{ ...SAVED_ENDPOINT, api_key_preview: '${HERMES_CUSTOM_CUSTOM_API_KEY} (not set)', has_api_key: false }])
    )

    await renderSettings()

    expect(screen.getByText('${HERMES_CUSTOM_CUSTOM_API_KEY} (not set)')).toBeTruthy()
    expect(screen.getByPlaceholderText('No key saved for this endpoint (optional)')).toBeTruthy()
  })

  it('tests a saved endpoint by id with a blank key so the backend uses the key on file', async () => {
    mocked.getCustomEndpoints.mockResolvedValue(listing([SAVED_ENDPOINT]))
    mocked.validateCustomEndpoint.mockResolvedValue({ message: '', models: ['qwen3.8-27b'], ok: true, reachable: true })

    await renderSettings()
    fireEvent.click(screen.getByRole('button', { name: 'Test' }))

    await waitFor(() => expect(mocked.validateCustomEndpoint).toHaveBeenCalledTimes(1))
    expect(mocked.validateCustomEndpoint).toHaveBeenCalledWith(
      expect.objectContaining({ id: 'custom', api_key: undefined, base_url: SAVED_ENDPOINT.base_url })
    )
  })

  it('keeps the field blank after Save but shows the newly saved key preview', async () => {
    const before: CustomEndpoint = { ...SAVED_ENDPOINT, api_key_preview: null, has_api_key: false }
    mocked.getCustomEndpoints.mockResolvedValue(listing([before]))
    mocked.saveCustomEndpoint.mockResolvedValue(listing([SAVED_ENDPOINT], 'custom'))

    await renderSettings()
    const field = screen.getByPlaceholderText('No key saved for this endpoint (optional)')
    fireEvent.change(field, { target: { value: 'sk-typed-key' } })
    fireEvent.click(screen.getByRole('button', { name: 'Save' }))

    await waitFor(() => expect(mocked.saveCustomEndpoint).toHaveBeenCalledTimes(1))
    expect(mocked.saveCustomEndpoint).toHaveBeenCalledWith(expect.objectContaining({ api_key: 'sk-typed-key' }))
    await screen.findByPlaceholderText('Leave blank to keep the saved key (sk-s...kI0c)')
    expect((screen.getByPlaceholderText('Leave blank to keep the saved key (sk-s...kI0c)') as HTMLInputElement).value).toBe('')
  })
})
