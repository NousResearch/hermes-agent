import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, expect, test, vi } from 'vitest'

import { I18nProvider } from '@/i18n'

import { LiveModelPicker } from './live-model-picker'
const api = vi.hoisted(() => ({ getHermesConfigRecord: vi.fn(), saveHermesConfigRecord: vi.fn() }))
vi.mock('@/api/config', () => api)
afterEach(cleanup)
vi.stubGlobal('ResizeObserver', class { observe() {} unobserve() {} disconnect() {} })
test('changing voice preserves the worker model and other settings', async () => {
  api.getHermesConfigRecord.mockResolvedValue({ model: { default: 'worker-model' }, voice: { realtime: { gemini: { voice: 'Puck' } } } })
  api.saveHermesConfigRecord.mockResolvedValue({ ok: true })
  render(<MemoryRouter><I18nProvider configClient={null} initialLocale="pl"><LiveModelPicker connected /></I18nProvider></MemoryRouter>)
  fireEvent.click(screen.getByRole('button', { name: /Rozmowa Live/ }))
  fireEvent.click(await screen.findByRole('button', { name: 'OpenAI Realtime' }))
  await waitFor(() => expect(api.saveHermesConfigRecord).toHaveBeenCalled())
  expect(api.saveHermesConfigRecord.mock.calls[0][0]).toEqual({ model: { default: 'worker-model' }, voice: { engine: 'realtime', realtime: { provider: 'openai', model: 'gpt-realtime', gemini: { voice: 'Puck' } } } })
  expect(api.saveHermesConfigRecord.mock.calls[0][1]).toEqual(api.getHermesConfigRecord.mock.calls[0][0])
})
