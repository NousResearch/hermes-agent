import { beforeEach, expect, test, vi } from 'vitest'

const getTerminalBackends = vi.fn()
const selectTerminalBackend = vi.fn()

vi.mock('@/hermes', () => ({
  getTerminalBackends: (...args: unknown[]) => getTerminalBackends(...args),
  selectTerminalBackend: (...args: unknown[]) => selectTerminalBackend(...args)
}))

import { $notifications, clearNotifications } from './notifications'
import { $routeRequest } from './recovery-requests'
import { unavailableTerminalBackend, warnIfTerminalBackendUnavailable } from './terminal-backend-warning'

const row = (name: string, status: 'needs_setup' | 'ready' | 'unavailable', active = false) => ({
  name,
  label: name === 'docker' ? 'Docker' : name,
  description: '',
  active,
  status,
  detail: status === 'ready' ? '' : 'Docker daemon not reachable'
})

beforeEach(() => {
  clearNotifications()
  getTerminalBackends.mockReset()
  selectTerminalBackend.mockReset()
})

// Before this, a Docker/SSH terminal backend whose probe failed showed only a
// "Needs setup" pill inside Skills → Tools → Terminal; nothing told the user
// shell commands could not run.
test('a selected non-local backend that is not ready warns once with Use Local / Open settings', async () => {
  getTerminalBackends.mockResolvedValue({ active: 'docker', backends: [row('local', 'ready'), row('docker', 'unavailable', true)] })
  selectTerminalBackend.mockResolvedValue({ ok: true, backend: 'local' })

  expect(await warnIfTerminalBackendUnavailable()).toBe(true)

  const toast = $notifications.get()[0]
  expect(toast?.kind).toBe('warning')
  expect(toast?.title).toMatch(/Terminal commands are unavailable/)
  expect(toast?.message).toContain('Docker')
  expect(toast?.detail).toBe('Docker daemon not reachable')
  // Plain language: the toast names the tool (Docker), never "backend".
  expect(`${toast?.title} ${toast?.message} ${toast?.action?.label}`).not.toMatch(/backend/i)

  toast?.action?.onClick()
  expect($routeRequest.get()?.path).toBe('/capabilities?tab=toolsets')

  toast?.secondaryAction?.onClick()
  expect(selectTerminalBackend).toHaveBeenCalledWith('local')
})

test('local or ready backends and probe failures stay silent', async () => {
  expect(unavailableTerminalBackend([row('local', 'ready', true), row('docker', 'unavailable')])).toBeNull()
  expect(unavailableTerminalBackend([row('docker', 'ready', true)])).toBeNull()

  getTerminalBackends.mockRejectedValue(new Error('404'))
  expect(await warnIfTerminalBackendUnavailable()).toBe(false)
  expect($notifications.get()).toHaveLength(0)
})
