import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const getHindsightConfig = vi.fn()
const saveHindsightConfig = vi.fn()

vi.mock('@/hermes', () => ({
  getHindsightConfig: () => getHindsightConfig(),
  saveHindsightConfig: (body: unknown) => saveHindsightConfig(body)
}))

vi.mock('@/store/notifications', () => ({
  notify: vi.fn(),
  notifyError: vi.fn()
}))

beforeEach(() => {
  getHindsightConfig.mockResolvedValue({
    mode: 'cloud',
    api_url: 'https://api.hindsight.vectorize.io',
    bank_id: 'hermes',
    recall_budget: 'mid',
    api_key_set: false
  })
  saveHindsightConfig.mockResolvedValue({ ok: true })
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

async function renderHindsightSettings() {
  const { HindsightSettings } = await import('./hindsight-settings')

  return render(<HindsightSettings />)
}

describe('HindsightSettings', () => {
  it('loads the Hindsight config fields', async () => {
    await renderHindsightSettings()

    expect(await screen.findByDisplayValue('https://api.hindsight.vectorize.io')).toBeTruthy()
    expect(screen.getByDisplayValue('hermes')).toBeTruthy()
    expect(screen.getByText('Cloud')).toBeTruthy()
    expect(screen.getAllByText('Hindsight Cloud API (lightweight, just needs an API key)').length).toBeGreaterThan(0)
    expect(screen.getByText('mid')).toBeTruthy()
  })

  it('collapses and expands the Hindsight fields', async () => {
    await renderHindsightSettings()

    expect(await screen.findByLabelText('API URL')).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: /Hindsight settings/ }))
    expect(screen.queryByLabelText('API URL')).toBeNull()
    fireEvent.click(screen.getByRole('button', { name: /Hindsight settings/ }))
    expect(await screen.findByLabelText('API URL')).toBeTruthy()
  })

  it('saves the configured Hindsight values without requiring an API key replacement', async () => {
    getHindsightConfig.mockResolvedValue({
      mode: 'cloud',
      api_url: 'https://api.hindsight.vectorize.io',
      bank_id: 'hermes',
      recall_budget: 'mid',
      api_key_set: true
    })

    await renderHindsightSettings()

    const apiUrl = await screen.findByLabelText('API URL')
    fireEvent.change(apiUrl, { target: { value: 'http://localhost:8888' } })
    fireEvent.change(screen.getByLabelText('Bank ID'), { target: { value: 'ben-bank' } })
    fireEvent.click(screen.getByRole('button', { name: 'Save' }))

    await waitFor(() =>
      expect(saveHindsightConfig).toHaveBeenCalledWith({
        mode: 'cloud',
        api_url: 'http://localhost:8888',
        api_key: '',
        bank_id: 'ben-bank',
        recall_budget: 'mid'
      })
    )
  })
})
