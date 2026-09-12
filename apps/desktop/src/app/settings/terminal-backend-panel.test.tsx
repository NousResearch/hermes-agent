import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { StrictMode } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { TerminalBackendsResponse } from '@/types/hermes'

const getTerminalBackends = vi.fn()
const selectTerminalBackend = vi.fn()
const scopedConfigCache = vi.fn()
const hermesConfigCacheWriter = vi.fn((_profile?: unknown) => scopedConfigCache)

vi.mock('@/hermes', () => ({
  getTerminalBackends: (profile: unknown) => getTerminalBackends(profile),
  profileScopeKey: (profile?: { connectionId?: string; profile?: string }) =>
    profile ? `${profile.connectionId ?? 'local'}::${profile.profile ?? 'default'}` : 'default',
  selectTerminalBackend: (backend: string, profile: unknown) => selectTerminalBackend(backend, profile)
}))

vi.mock('@/store/notifications', () => ({
  notify: vi.fn(),
  notifyError: vi.fn()
}))

vi.mock('../hooks/use-config-record', () => ({
  hermesConfigCacheWriter: (profile: unknown) => hermesConfigCacheWriter(profile)
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

function deferred<T>() {
  let resolve!: (value: T) => void

  const promise = new Promise<T>(next => {
    resolve = next
  })

  return { promise, resolve }
}

beforeEach(() => {
  getTerminalBackends.mockResolvedValue(backends())
  selectTerminalBackend.mockResolvedValue({ ok: true, backend: 'ssh' })
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

describe('TerminalBackendPanel', () => {
  it('lists backends with status pills from the backends endpoint', async () => {
    const { TerminalBackendPanel } = await import('./terminal-backend-panel')
    render(<TerminalBackendPanel onConfiguredChange={vi.fn()} />)

    expect(await screen.findByText('Local')).toBeTruthy()
    expect(screen.getByText('Docker')).toBeTruthy()
    expect(screen.getByText('SSH')).toBeTruthy()
    // Ready backends show the Ready pill; needs_setup shows the warn pill.
    expect(screen.getAllByText('Ready').length).toBeGreaterThanOrEqual(2)
    expect(screen.getByText('Needs setup')).toBeTruthy()
    expect(getTerminalBackends).toHaveBeenCalled()
  })

  it('loads backends under React StrictMode effect replay', async () => {
    const { TerminalBackendPanel } = await import('./terminal-backend-panel')
    render(
      <StrictMode>
        <TerminalBackendPanel onConfiguredChange={vi.fn()} />
      </StrictMode>
    )

    expect(await screen.findByText('Local')).toBeTruthy()
    expect(getTerminalBackends).toHaveBeenCalledTimes(2)
  })

  it('shows setup guidance detail for a needs_setup backend', async () => {
    const { TerminalBackendPanel } = await import('./terminal-backend-panel')
    render(<TerminalBackendPanel onConfiguredChange={vi.fn()} />)

    expect(await screen.findByText(/Docker daemon not reachable/)).toBeTruthy()
  })

  it('marks the active backend with an In use pill', async () => {
    const { TerminalBackendPanel } = await import('./terminal-backend-panel')
    render(<TerminalBackendPanel onConfiguredChange={vi.fn()} />)

    const local = await screen.findByRole('button', { name: /Local/ })
    expect(local.getAttribute('aria-pressed')).toBe('true')
    expect(screen.getByText('In use')).toBeTruthy()
  })

  it('selects a backend when clicked and reports the change', async () => {
    const onConfiguredChange = vi.fn()
    const profile = { connectionId: 'remote-1', profile: 'other' }
    const { TerminalBackendPanel } = await import('./terminal-backend-panel')
    render(<TerminalBackendPanel onConfiguredChange={onConfiguredChange} profile={profile} />)

    fireEvent.click(await screen.findByRole('button', { name: /SSH/ }))

    await waitFor(() => expect(getTerminalBackends).toHaveBeenCalledWith(profile))
    await waitFor(() => expect(selectTerminalBackend).toHaveBeenCalledWith('ssh', profile))
    await waitFor(() => expect(onConfiguredChange).toHaveBeenCalled())
    expect(hermesConfigCacheWriter).toHaveBeenCalledWith(profile)
    expect(scopedConfigCache).toHaveBeenCalledTimes(1)

    const updateCache = scopedConfigCache.mock.calls[0][0] as (
      current: Record<string, unknown> | undefined
    ) => Record<string, unknown> | undefined

    expect(updateCache({ terminal: { backend: 'local' } })).toEqual({ terminal: { backend: 'ssh' } })
    // Active highlight moves without a refetch.
    const ssh = screen.getByRole('button', { name: /SSH/ })
    expect(ssh.getAttribute('aria-pressed')).toBe('true')
  })

  it('allows selecting a needs_setup backend (guidance instead of blocking)', async () => {
    selectTerminalBackend.mockResolvedValue({ ok: true, backend: 'docker' })
    const { TerminalBackendPanel } = await import('./terminal-backend-panel')
    render(<TerminalBackendPanel onConfiguredChange={vi.fn()} />)

    fireEvent.click(await screen.findByRole('button', { name: /Docker/ }))

    await waitFor(() => expect(selectTerminalBackend).toHaveBeenCalledWith('docker', undefined))
    // The guidance detail stays visible on the now-active row.
    expect(screen.getByText(/Docker daemon not reachable/)).toBeTruthy()
  })


  it('does not re-select the already active backend', async () => {
    const { TerminalBackendPanel } = await import('./terminal-backend-panel')
    render(<TerminalBackendPanel onConfiguredChange={vi.fn()} />)

    fireEvent.click(await screen.findByRole('button', { name: /Local/ }))

    await new Promise(resolve => setTimeout(resolve, 50))
    expect(selectTerminalBackend).not.toHaveBeenCalled()
  })

  it('ignores a stale backend response after the profile changes', async () => {
    const oldResponse = deferred<TerminalBackendsResponse>()
    const newResponse = backends({ active: 'ssh' })
    newResponse.backends = newResponse.backends.map(backend => ({
      ...backend,
      active: backend.name === 'ssh'
    }))
    getTerminalBackends
      .mockReturnValueOnce(oldResponse.promise)
      .mockResolvedValueOnce(newResponse)
    const { TerminalBackendPanel } = await import('./terminal-backend-panel')
    const oldProfile = { connectionId: 'remote-1', profile: 'old' }
    const newProfile = { connectionId: 'remote-1', profile: 'new' }

    const view = render(
      <TerminalBackendPanel onConfiguredChange={vi.fn()} profile={oldProfile} />
    )

    view.rerender(<TerminalBackendPanel onConfiguredChange={vi.fn()} profile={newProfile} />)
    const ssh = await screen.findByRole('button', { name: /SSH/ })
    expect(ssh.getAttribute('aria-pressed')).toBe('true')

    oldResponse.resolve(backends({ active: 'local' }))
    await new Promise(resolve => setTimeout(resolve, 0))

    expect(screen.getByRole('button', { name: /SSH/ }).getAttribute('aria-pressed')).toBe('true')
  })

  it('suppresses selection side effects after the profile changes', async () => {
    const selection = deferred<{ ok: boolean; backend: string }>()
    selectTerminalBackend.mockReturnValueOnce(selection.promise)
    const onConfiguredChange = vi.fn()
    const { TerminalBackendPanel } = await import('./terminal-backend-panel')
    const oldProfile = { connectionId: 'remote-1', profile: 'old' }
    const newProfile = { connectionId: 'remote-1', profile: 'new' }

    const view = render(
      <TerminalBackendPanel onConfiguredChange={onConfiguredChange} profile={oldProfile} />
    )

    fireEvent.click(await screen.findByRole('button', { name: /SSH/ }))

    view.rerender(
      <TerminalBackendPanel onConfiguredChange={onConfiguredChange} profile={newProfile} />
    )
    selection.resolve({ ok: true, backend: 'ssh' })
    await new Promise(resolve => setTimeout(resolve, 0))

    expect(scopedConfigCache).not.toHaveBeenCalled()
    expect(onConfiguredChange).not.toHaveBeenCalled()
  })

  it('unlocks selection for a new profile while the old selection is pending', async () => {
    const oldSelection = deferred<{ ok: boolean; backend: string }>()
    selectTerminalBackend
      .mockReturnValueOnce(oldSelection.promise)
      .mockResolvedValueOnce({ ok: true, backend: 'docker' })
    const { TerminalBackendPanel } = await import('./terminal-backend-panel')
    const oldProfile = { connectionId: 'remote-1', profile: 'old' }
    const newProfile = { connectionId: 'remote-1', profile: 'new' }
    const view = render(<TerminalBackendPanel profile={oldProfile} />)

    fireEvent.click(await screen.findByRole('button', { name: /SSH/ }))
    await waitFor(() => expect(selectTerminalBackend).toHaveBeenCalledWith('ssh', oldProfile))

    view.rerender(<TerminalBackendPanel profile={newProfile} />)
    const docker = await screen.findByRole('button', { name: /Docker/ })

    expect(docker.hasAttribute('disabled')).toBe(false)
    fireEvent.click(docker)
    await waitFor(() => expect(selectTerminalBackend).toHaveBeenCalledWith('docker', newProfile))

    oldSelection.resolve({ ok: true, backend: 'ssh' })
  })
})
