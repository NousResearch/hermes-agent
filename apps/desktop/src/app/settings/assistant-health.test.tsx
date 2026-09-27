import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, test, vi } from 'vitest'

import type { RealtimeVoiceHandlers, RealtimeVoiceSession } from '@/lib/realtime-voice'

import { AssistantHealth } from './assistant-health'

const mocks = vi.hoisted(() => ({ start: vi.fn() }))
vi.mock('@/hermes', () => ({
  createRealtimeVoiceSession: vi.fn(),
  getApiRequestConnection: () => null,
  getStatus: async () => ({}),
  getEnvVars: async () => ({ GEMINI_API_KEY: { is_set: true, value: 'secret-must-not-render' } }),
  getToolsets: async () => [{ enabled: true, configured: true, label: 'Przeglądarka' }]
}))
vi.mock('@/lib/live-voice/start', () => ({ startLiveVoice: mocks.start }))
vi.mock('./recovery-settings', () => ({ RecoverySettings: () => null }))
afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

test('stored API credentials remain explicitly unverified and are never displayed', async () => {
  render(<AssistantHealth />)
  fireEvent.click(screen.getByRole('button', { name: 'Sprawdź stan' }))
  expect(await screen.findByText(/Sam zapis nie potwierdza ważności/)).toBeTruthy()
  expect(screen.getByText('Klucze API · Do sprawdzenia')).toBeTruthy()
  expect(screen.queryByText(/secret-must-not-render/)).toBeNull()
})

test('a fatal error while connecting stops a late voice session instead of leaving the mic open', async () => {
  let finish!: (session: RealtimeVoiceSession) => void
  let handlers!: RealtimeVoiceHandlers
  mocks.start.mockImplementation((value: RealtimeVoiceHandlers) => {
    handlers = value

    return new Promise<RealtimeVoiceSession>(resolve => {
      finish = resolve
    })
  })
  render(<AssistantHealth />)
  fireEvent.click(screen.getByRole('button', { name: 'Przetestuj rozmowę' }))
  await act(async () => {
    handlers.onError('401 invalid key')
  })
  expect(screen.getByText(/Wybrane API odrzuciło dostęp/)).toBeTruthy()
  const stop = vi.fn()
  await act(async () => {
    finish({ stop, setMuted: vi.fn() })
  })
  expect(stop).toHaveBeenCalledOnce()
})
