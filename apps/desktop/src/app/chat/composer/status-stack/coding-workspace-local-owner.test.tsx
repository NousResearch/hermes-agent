import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { HermesGateway } from '@/hermes'
import { createClientSessionState } from '@/lib/chat-runtime'
import { $repoStatusByCwd, $repoWorktreesByCwd } from '@/store/coding-status'
import { activeGateway, activeGatewayConnectionId, closeSecondaryGateways, configureGatewayRegistry, ensureGatewayForAgent, ensureGatewayForProfile, requestGatewayForProfile, setPrimaryGateway, setPrimaryGatewayConnectionId } from '@/store/gateway'
import { $connection, _resetSessionOwnerHintsForTests, setSessionOwnerHint, setSessions } from '@/store/session'
import { $sessionStates, knownOwnerForSession } from '@/store/session-states'
import type { CodingWorkspaceBinding } from '@/types/hermes'

import { CodingStatusRow } from './coding-row'

vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  HermesGateway: class {
    connectionState = 'closed'
    connect = vi.fn(async () => { this.connectionState = 'open' })
    close = vi.fn(() => { this.connectionState = 'closed' })
    request = vi.fn(async () => ({}))
    onEvent = vi.fn(() => () => {})
    onState = vi.fn(() => () => {})
  }
}))

const binding: CodingWorkspaceBinding = {
  requestId: 'owner', projectId: 'project', sourcePath: '/repo',
  cwd: '/repo/.worktrees/owner', repoRoot: '/repo', branch: 'task/stored'
}

async function activateLegacy(mode: 'local' | 'remote') {
  const revealPath = vi.fn(async () => true)

  const repoStatus = vi.fn(async () => ({
    added: 12, removed: 3, ahead: 0, behind: 0, untracked: 0,
    branch: 'task/live', defaultBranch: 'main', detached: false
  }))

  const getConnection = vi.fn(async (profile?: string) => ({
    connectionId: 'local', profile, mode, port: 5151, token: 'test'
  }))

  const getConnectionFor = vi.fn(async ({ connectionId, profile }) => ({
    connectionId, profile, mode: 'local', port: 5152, token: 'test'
  }))

  Object.assign(window, { hermesDesktop: { getConnection, getConnectionFor, revealPath, git: { repoStatus } } })
  configureGatewayRegistry({ onEvent: vi.fn() })
  setPrimaryGateway(new HermesGateway() as never, 'default')
  await ensureGatewayForProfile('default')
  setPrimaryGatewayConnectionId('local')
  await ensureGatewayForProfile('coder')
  // The native descriptor names local, but the registry records the actual
  // legacy null-connection socket. It must not become registry local::coder.
  expect($connection.get()).toMatchObject({ connectionId: 'local', profile: 'coder', mode })
  expect(activeGatewayConnectionId()).toBeNull()
  setSessions([{ id: 'stored-owner', profile: 'coder' }] as never)
  $sessionStates.set({ runtime: { ...createClientSessionState('stored-owner'), codingWorkspace: binding } })
  expect(knownOwnerForSession('runtime')).toBe('coder')

  return { revealPath, repoStatus, getConnection, getConnectionFor }
}

afterEach(() => {
  cleanup()
  closeSecondaryGateways()
  $connection.set(null)
  $sessionStates.set({})
  $repoStatusByCwd.set({})
  $repoWorktreesByCwd.set({})
  setSessions([])
  _resetSessionOwnerHintsForTests()
  delete (window as { hermesDesktop?: unknown }).hermesDesktop
  vi.clearAllMocks()
})

it('reveals and probes a legacy-local named profile using its actual active route, not its descriptor id', async () => {
  const { revealPath, repoStatus, getConnectionFor } = await activateLegacy('local')
  render(<CodingStatusRow sessionId="runtime" />)
  await waitFor(() => expect(repoStatus).toHaveBeenCalledWith(binding.cwd))
  const summary = await screen.findByRole('button', { name: 'repo · Worktree · task/live' })
  expect(screen.getByText('12')).toBeTruthy()
  fireEvent.pointerDown(summary, { button: 0, ctrlKey: false, pointerType: 'mouse' })
  fireEvent.click(await screen.findByRole('menuitem', { name: 'Open folder' }))
  expect(revealPath).toHaveBeenCalledWith(binding.cwd)
  expect(getConnectionFor).not.toHaveBeenCalled()

  // Same name and even the same local descriptor do NOT prove the same route.
  await act(async () => { await ensureGatewayForAgent('local', 'coder') })
  expect(activeGatewayConnectionId()).toBe('local')
  expect(screen.queryByText('12')).toBeNull()
  fireEvent.pointerDown(screen.getByRole('button', { name: 'repo · Worktree · task/stored' }), { button: 0, ctrlKey: false, pointerType: 'mouse' })
  await screen.findByRole('menuitem', { name: 'Copy path' })
  expect(screen.queryByRole('menuitem', { name: 'Open folder' })).toBeNull()
})

it('accepts a bare-profile owner when its real profile router selects the local primary socket', async () => {
  const { revealPath, repoStatus } = await activateLegacy('local')
  closeSecondaryGateways()
  const primary = new HermesGateway()
  await primary.connect({ port: 5151, token: 'test' } as never)
  setPrimaryGateway(primary, 'coder')
  await ensureGatewayForProfile('coder')
  setPrimaryGatewayConnectionId('local')
  // Unlike a same-name registry secondary, gatewayForProfile('coder') selects
  // the primary by construction even though its descriptor has a registry id.
  expect(activeGatewayConnectionId()).toBe('local')
  await requestGatewayForProfile('coder', 'session.info', { session_id: 'runtime' })
  expect(activeGateway()).toBe(primary)
  expect(primary.request).toHaveBeenCalledWith('session.info', { session_id: 'runtime' })
  render(<CodingStatusRow sessionId="runtime" />)
  await waitFor(() => expect(repoStatus).toHaveBeenCalledWith(binding.cwd))
  fireEvent.pointerDown(screen.getByRole('button', { name: 'repo · Worktree · task/live' }), { button: 0, ctrlKey: false, pointerType: 'mouse' })
  fireEvent.click(await screen.findByRole('menuitem', { name: 'Open folder' }))
  expect(revealPath).toHaveBeenCalledWith(binding.cwd)
})

it('reveals a registry tile only after its own descriptor proves local, without foreground Git', async () => {
  const { revealPath, repoStatus, getConnectionFor } = await activateLegacy('local')
  setSessionOwnerHint('stored-owner', { connectionId: 'local', profile: 'coder' })
  render(<CodingStatusRow sessionId="runtime" />)
  fireEvent.pointerDown(screen.getByRole('button', { name: 'repo · Worktree · task/stored' }), { button: 0, ctrlKey: false, pointerType: 'mouse' })
  fireEvent.click(await screen.findByRole('menuitem', { name: 'Open folder' }))
  expect(getConnectionFor).toHaveBeenCalledWith({ connectionId: 'local', profile: 'coder' })
  expect(revealPath).toHaveBeenCalledWith(binding.cwd)
  expect(activeGatewayConnectionId()).toBeNull()
  expect(repoStatus).not.toHaveBeenCalled()
})

it('ignores a late local descriptor after the surface changes to a remote owner', async () => {
  const { revealPath, getConnectionFor } = await activateLegacy('local')
  let finish!: (value: Awaited<ReturnType<typeof getConnectionFor>>) => void
  getConnectionFor.mockImplementation(() => new Promise(resolve => { finish = resolve }))
  setSessionOwnerHint('stored-owner', { connectionId: 'local', profile: 'coder' })
  const view = render(<CodingStatusRow sessionId="runtime" />)
  await waitFor(() => expect(getConnectionFor).toHaveBeenCalledOnce())
  setSessionOwnerHint('stored-remote', { connectionId: 'remote', profile: 'coder', mode: 'remote' })
  $sessionStates.set({ ...$sessionStates.get(), other: { ...createClientSessionState('stored-remote'), codingWorkspace: binding } })
  view.rerender(<CodingStatusRow sessionId="other" />)
  await act(async () => { finish({ connectionId: 'local', profile: 'coder', mode: 'local', port: 5152, token: 'test' }) })
  fireEvent.pointerDown(screen.getByRole('button', { name: 'repo · Worktree · task/stored' }), { button: 0, ctrlKey: false, pointerType: 'mouse' })
  await screen.findByRole('menuitem', { name: 'Copy path' })
  expect(screen.queryByRole('menuitem', { name: 'Open folder' })).toBeNull()
  expect(revealPath).not.toHaveBeenCalled()
})

it.each(['remote override', 'other profile', 'unverified registry owner', 'other source', 'remote owner', 'wrong descriptor profile'] as const)(
  'keeps %s paths out of native reveal and Git even with an available local bridge', async scenario => {
    const { revealPath, repoStatus, getConnectionFor } = await activateLegacy(scenario === 'remote override' ? 'remote' : 'local')

    if (scenario === 'unverified registry owner') {getConnectionFor.mockRejectedValue(new Error('Owner descriptor unavailable'))}

    if (scenario === 'other source') {getConnectionFor.mockResolvedValue({ connectionId: 'another-source', profile: 'coder', mode: 'remote', port: 5152, token: 'test' })}

    if (scenario === 'wrong descriptor profile') {getConnectionFor.mockResolvedValue({ connectionId: 'local', profile: 'other', mode: 'local', port: 5152, token: 'test' })}

    if (scenario === 'other profile') {
      setSessions([{ id: 'stored-owner', profile: 'other' }] as never)
    } else if (scenario !== 'remote override') {
      setSessionOwnerHint('stored-owner', {
        connectionId: scenario === 'other source' ? 'another-source' : 'local',
        profile: 'coder', mode: scenario === 'remote owner' ? 'remote' : 'local'
      })
    }

    render(<CodingStatusRow repoPath="/WRONG" sessionId="runtime" />)
    fireEvent.pointerDown(screen.getByRole('button', { name: 'repo · Worktree · task/stored' }), { button: 0, ctrlKey: false, pointerType: 'mouse' })
    await screen.findByRole('menuitem', { name: 'Copy path' })
    expect(screen.queryByRole('menuitem', { name: 'Open folder' })).toBeNull()
    // Drain the real coding-status debounce, not just the render microtask.
    await act(async () => { await new Promise(resolve => setTimeout(resolve, 150)) })
    expect(screen.queryByRole('menuitem', { name: 'Open folder' })).toBeNull()
    expect(repoStatus).not.toHaveBeenCalled()
    expect(revealPath).not.toHaveBeenCalled()
    expect(screen.queryByText('12')).toBeNull()
  }
)
