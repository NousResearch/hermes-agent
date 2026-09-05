import { act, cleanup, fireEvent, render, renderHook, screen } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, expect, it, vi } from 'vitest'

import { CodingStatusRow } from '@/app/chat/composer/status-stack/coding-row'
import { useSessionStateCache } from '@/app/session/hooks/use-session-state-cache'
import { requestGatewayForProfile } from '@/store/gateway'
import { $activeSessionId, $connection, $currentModel, $messages, _resetSessionOwnerHintsForTests, setSessions } from '@/store/session'
import { $sessionStates, sessionTileDelegate } from '@/store/session-states'
import type { CodingWorkspaceBinding } from '@/types/hermes'

import { useSessionTileDelegate } from './use-session-tile-delegate'

vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  getLatestSessionMessages: vi.fn(async () => ({ messages: [], session_id: 'stored-workspace' }))
}))
vi.mock('@/store/gateway', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  requestGatewayForProfile: vi.fn()
}))
vi.mock('@/store/coding-status', () => ({
  registerRepoStatusCwd: () => () => {},
  repoStatusForCwd: () => atom(null),
  repoWorktreesForCwd: () => atom([])
}))

const binding: CodingWorkspaceBinding = {
  requestId: 'tile-workspace', projectId: 'project', sourcePath: '/repo',
  cwd: '/repo/.worktrees/task', repoRoot: '/repo', branch: 'task/tile'
}

afterEach(() => {
  cleanup()
  $sessionStates.set({})
  setSessions([])
  _resetSessionOwnerHintsForTests()
  $activeSessionId.set(null)
  $connection.set(null)
  vi.clearAllMocks()
})

it('hydrates the real tile cache and immutable summary from resume alone without publishing the foreground', async () => {
  setSessions([{ id: 'stored-workspace', profile: 'coder' }] as never)
  $activeSessionId.set('foreground-runtime')
  const foregroundMessages = $messages.get()
  const foregroundModel = $currentModel.get()
  const foregroundConnection = $connection.get()
  const setMessages = vi.fn()
  const setBusy = vi.fn()
  const setAwaitingResponse = vi.fn()
  const requestGateway = vi.fn()

  const { result } = renderHook(() => {
    const cache = useSessionStateCache({
      activeSessionId: 'foreground-runtime', selectedStoredSessionId: 'foreground-stored',
      busyRef: { current: false }, setMessages, setBusy, setAwaitingResponse
    })

    useSessionTileDelegate({
      ...cache, requestGateway, archiveSession: vi.fn(), branchStoredSession: vi.fn(),
      executeSlashCommand: vi.fn(), removeSession: vi.fn()
    })

    return cache
  })

  vi.mocked(requestGatewayForProfile).mockResolvedValueOnce({
    session_id: 'tile-runtime', info: { coding_workspace: binding, model: 'tile-model' }, messages: []
  })
  // No session.info event is emitted, and no cache binding is seeded.
  await act(async () => { await sessionTileDelegate()!.resumeTile('stored-workspace') })
  expect($sessionStates.get()['tile-runtime'].codingWorkspace).toEqual(binding)
  const onSwitchBranch = vi.fn()
  render(<CodingStatusRow onBranchOff={vi.fn()} onSwitchBranch={onSwitchBranch} sessionId="tile-runtime" />)
  fireEvent.pointerDown(screen.getByRole('button', { name: 'repo · Worktree · task/tile' }), { button: 0, ctrlKey: false, pointerType: 'mouse' })
  await screen.findByRole('menuitem', { name: 'Copy path' })
  expect(document.querySelector('[data-slot="coding-workspace-path"]')?.textContent).toBe(binding.cwd)
  expect(screen.queryByRole('menuitem', { name: /Switch to|New branch/ })).toBeNull()
  expect(onSwitchBranch).not.toHaveBeenCalled()

  // An older response that omits the field must retain the known binding;
  // explicit null is authoritative and must clear it on the same runtime.
  for (const [info, expected] of [[{}, binding], [{ coding_workspace: null }, null]] as const) {
    vi.mocked(requestGatewayForProfile).mockResolvedValueOnce({ session_id: 'tile-runtime', info, messages: [] })
    await act(async () => { await sessionTileDelegate()!.resumeTile('stored-workspace') })
    expect(result.current.sessionStateByRuntimeIdRef.current.get('tile-runtime')?.codingWorkspace).toEqual(expected)
    expect($sessionStates.get()['tile-runtime'].codingWorkspace).toEqual(expected)
  }

  expect($activeSessionId.get()).toBe('foreground-runtime')
  expect($messages.get()).toBe(foregroundMessages)
  expect($currentModel.get()).toBe(foregroundModel)
  expect($connection.get()).toBe(foregroundConnection)
  expect(setMessages).not.toHaveBeenCalled()
  expect(setBusy).not.toHaveBeenCalled()
  expect(setAwaitingResponse).not.toHaveBeenCalled()
  expect(requestGateway).not.toHaveBeenCalled()
})
