import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { TerminalBackendsResponse } from '@/types/hermes'

const getTerminalBackends = vi.fn()
const selectTerminalBackend = vi.fn()

vi.mock('@/hermes', () => ({
  getTerminalBackends: () => getTerminalBackends(),
  selectTerminalBackend: (backend: string) => selectTerminalBackend(backend)
}))

vi.mock('@/store/notifications', () => ({
  notify: vi.fn(),
  notifyError: vi.fn()
}))

function backends(overrides: Partial<TerminalBackendsResponse> = {}): TerminalBackendsResponse {
  return {
    active: 'local',
    backends: [
      {
        name: 'local',
        label: 'Local',
        description: 'Run commands directly on this machine. No isolation.',
        active: true,
        status: 'ready',
        detail: ''
      },
      {
        name: 'docker',
        label: 'Docker',
        description: 'Run commands in an isolated Docker container.',
        active: false,
        status: 'needs_setup',
        detail: 'Docker daemon not reachable — start Docker and retry.'
      },
      {
        name: 'ssh',
        label: 'SSH',
        description: 'Run commands on a remote host over SSH.',
        active: false,
        status: 'ready',
        detail: 'hermes@devbox'
      }
    ],
    ...overrides
  }
}

beforeEach(() => {
  getTerminalBackends.mockResolvedValue(backends())
  selectTerminalBackend.mockResolvedValue({ ok: true, backend: 'ssh' })
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

async function renderPanel(onConfiguredChange = vi.fn()) {
  const { TerminalBackendPanel } = await import('./terminal-backend-panel')
  await act(async () => {
    render(<TerminalBackendPanel onConfiguredChange={onConfiguredChange} />)
  })
}

describe('TerminalBackendPanel', () => {
  it('lists backends with status pills from the backends endpoint', async () => {
    await renderPanel()

    expect(await screen.findByText('Local')).toBeTruthy()
    expect(screen.getByText('Docker')).toBeTruthy()
    expect(screen.getByText('SSH')).toBeTruthy()
    // Ready backends show the Ready pill; needs_setup shows the warn pill.
    expect(screen.getAllByText('Ready').length).toBeGreaterThanOrEqual(2)
    expect(screen.getByText('Needs setup')).toBeTruthy()
    expect(getTerminalBackends).toHaveBeenCalled()
  }, 15_000)

  it('shows setup guidance detail for a needs_setup backend', async () => {
    await renderPanel()

    expect(await screen.findByText(/Docker daemon not reachable/)).toBeTruthy()
  })

  it('marks the active backend with an In use pill', async () => {
    await renderPanel()

    const local = await screen.findByRole('button', { name: /Local/ })
    expect(local.getAttribute('aria-pressed')).toBe('true')
    expect(screen.getByText('In use')).toBeTruthy()
  })

  it('selects a backend when clicked and reports the change', async () => {
    const onConfiguredChange = vi.fn()
    await renderPanel(onConfiguredChange)

    fireEvent.click(await screen.findByRole('button', { name: /SSH/ }))

    await waitFor(() => expect(selectTerminalBackend).toHaveBeenCalledWith('ssh'))
    await waitFor(() => expect(onConfiguredChange).toHaveBeenCalled())
    // Active highlight moves without a refetch.
    const ssh = screen.getByRole('button', { name: /SSH/ })
    expect(ssh.getAttribute('aria-pressed')).toBe('true')
  })

  it('allows selecting a needs_setup backend (guidance instead of blocking)', async () => {
    selectTerminalBackend.mockResolvedValue({ ok: true, backend: 'docker' })
    await renderPanel()

    fireEvent.click(await screen.findByRole('button', { name: /Docker/ }))

    await waitFor(() => expect(selectTerminalBackend).toHaveBeenCalledWith('docker'))
    // The guidance detail stays visible on the now-active row.
    expect(screen.getByText(/Docker daemon not reachable/)).toBeTruthy()
  })

  it('does not re-select the already active backend', async () => {
    await renderPanel()

    fireEvent.click(await screen.findByRole('button', { name: /Local/ }))

    await new Promise(resolve => setTimeout(resolve, 50))
    expect(selectTerminalBackend).not.toHaveBeenCalled()
  })
})
