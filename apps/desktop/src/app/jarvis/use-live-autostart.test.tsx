import { cleanup, renderHook, waitFor } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, expect, test, vi } from 'vitest'
const api = vi.hoisted(() => ({ status: vi.fn(), start: vi.fn() }))
vi.mock('@/api/voice-realtime', () => ({ getRealtimeVoiceStatus: api.status }))
vi.mock('@/store/composer', () => ({ requestVoiceConversationStart: api.start }))
vi.mock('@/store/onboarding', () => ({ $desktopOnboarding: atom({ configured: true, requested: false }) }))
vi.mock('@/store/profile', () => ({ $activeGatewayProfile: atom('autostart-test') }))
vi.mock('@/store/connections', () => ({ $activeConnectionId: atom('local') }))
import { useLiveAutostart } from './use-live-autostart'
afterEach(cleanup)
test('starts once after readiness and does not reopen an ended call on navigation', async () => {
  Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: { api: vi.fn() } })
  api.status.mockResolvedValue({ available: true })
  const first = renderHook(() => useLiveAutostart(true))
  await waitFor(() => expect(api.start).toHaveBeenCalledTimes(1))
  first.unmount()
  renderHook(() => useLiveAutostart(true))
  expect(api.start).toHaveBeenCalledTimes(1)
  expect(api.status).toHaveBeenCalledTimes(1)
})
