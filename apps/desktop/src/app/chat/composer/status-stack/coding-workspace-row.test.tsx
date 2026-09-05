import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, expect, it, vi } from 'vitest'

import type { ClientSessionState } from '@/app/types'
import { ensureGatewayForProfile, setPrimaryGateway, setPrimaryGatewayConnectionId } from '@/store/gateway'
import { $notifications, clearNotifications } from '@/store/notifications'
import * as profile from '@/store/profile'
import { $connection, _resetSessionOwnerHintsForTests, setSessionOwnerHint } from '@/store/session'
import { $sessionStates } from '@/store/session-states'
import type { CodingWorkspaceBinding } from '@/types/hermes'

const probe = vi.hoisted(() => vi.fn())
vi.mock('@/store/coding-status', () => ({
  registerRepoStatusCwd: (cwd?: string) => { probe(cwd) },
  repoStatusForCwd: (cwd?: string) => atom(cwd ? { added: 12, removed: 3, ahead: 0, behind: 0, untracked: 0, branch: 'task/a', defaultBranch: 'main', detached: false } : null),
  repoWorktreesForCwd: () => atom([])
}))
const { CodingStatusRow } = await import('./coding-row')

const binding: CodingWorkspaceBinding = { requestId: 'a', projectId: 'project-a', sourcePath: '/repo', cwd: '/repo/.worktrees/a', repoRoot: '/repo', branch: 'task/a' }

async function bind(workspace: CodingWorkspaceBinding, mode: 'local' | 'remote' = 'local') {
  setPrimaryGateway({ connectionState: 'open' } as never, 'coder')
  await ensureGatewayForProfile('coder')
  setPrimaryGatewayConnectionId(mode)
  $sessionStates.set({ a: { storedSessionId: 'stored-a', codingWorkspace: workspace } as ClientSessionState })
  setSessionOwnerHint('stored-a', { connectionId: mode, profile: 'coder', mode })
  $connection.set({ connectionId: mode, profile: 'coder', mode } as never)
}

afterEach(() => { cleanup(); $sessionStates.set({}); _resetSessionOwnerHintsForTests(); $connection.set(null); vi.clearAllMocks() })

it('starts another workspace through the exact owner new-chat flow, with selectors but no old binding edits', async () => {
  await bind(binding, 'remote')
  // Deliberately leave a different local source in the foreground.
  $connection.set({ connectionId: 'local', profile: 'other', mode: 'local' } as never)
  const newChat = vi.spyOn(profile, 'newSessionInAgent').mockImplementation(() => undefined)
  const old = $sessionStates.get().a
  render(<CodingStatusRow repoPath="/WRONG" sessionId="a" />)
  fireEvent.pointerDown(screen.getByRole('button', { name: 'repo · Worktree · task/a' }), { button: 0, ctrlKey: false, pointerType: 'mouse' })
  const action = await screen.findByRole('menuitem', { name: 'New chat in another workspace…' })
  expect(screen.queryByRole('menuitem', { name: 'Open folder' })).toBeNull()
  expect(probe).toHaveBeenCalledWith(undefined)
  expect(probe).not.toHaveBeenCalledWith(binding.cwd)
  fireEvent.click(action)
  expect(newChat).toHaveBeenCalledWith({ connectionId: 'remote', profile: 'coder', mode: 'remote' }, { codingWorkspaceControls: true, workspaceTarget: null })
  expect($sessionStates.get().a).toBe(old)
  newChat.mockRestore()
})

it('surfaces failed native reveal instead of losing the rejection', async () => {
  await bind(binding)
  clearNotifications()
  window.hermesDesktop = { ...window.hermesDesktop, revealPath: vi.fn().mockRejectedValue(new Error('unreadable folder')) } as never
  render(<CodingStatusRow sessionId="a" />)
  fireEvent.pointerDown(screen.getByRole('button', { name: 'repo · Worktree · task/a' }), { button: 0, ctrlKey: false, pointerType: 'mouse' })
  fireEvent.click(await screen.findByRole('menuitem', { name: 'Open folder' }))
  await waitFor(() => expect($notifications.get().some(item => JSON.stringify(item).includes('unreadable folder'))).toBe(true))
})

it.each([
  [binding, 'repo · Worktree · task/a'],
  [{ ...binding, cwd: '/repo' }, 'repo · Current checkout · task/a'],
  [{ ...binding, cwd: '/notes', sourcePath: '/notes', repoRoot: null, branch: null }, 'notes · Folder'],
  [{ ...binding, projectName: 'Named Project', mode: 'existing' as const }, 'Named Project · Worktree · task/a']
])('discloses an immutable persisted workspace with no draft/global fallback: %s', async (workspace, label) => {
  await bind(workspace)
  const writeText = vi.fn().mockResolvedValue(undefined)
  const revealPath = vi.fn().mockResolvedValue(true)
  Object.defineProperty(navigator, 'clipboard', { configurable: true, value: { writeText } })
  window.hermesDesktop = { ...window.hermesDesktop, revealPath } as never
  const onOpen = vi.fn()
  const onSwitchBranch = vi.fn()
  const view = render(<CodingStatusRow onBranchOff={vi.fn()} onOpen={onOpen} onSwitchBranch={onSwitchBranch} repoPath="/WRONG" sessionId="a" />)
  const summary = screen.getByRole('button', { name: label })
  expect(view.container.querySelectorAll('.coding-status-bar')).toHaveLength(1)
  fireEvent.pointerDown(summary, { button: 0, ctrlKey: false, pointerType: 'mouse' })
  await waitFor(() => expect(screen.getByText(workspace.cwd)).toBeTruthy())
  expect(document.querySelector('[data-slot="coding-workspace-path"]')?.textContent).toBe(workspace.cwd)
  expect(screen.queryByRole('menuitem', { name: /Switch to/ })).toBeNull()
  expect(screen.queryByRole('combobox')).toBeNull()
  fireEvent.click(screen.getByRole('menuitem', { name: 'Copy path' }))
  await waitFor(() => expect(writeText).toHaveBeenCalledWith(workspace.cwd))
  // Reopen after copy (the menu primitive may retain it for inline feedback).
  fireEvent.keyDown(document.activeElement!, { key: 'Escape' })
  fireEvent.pointerDown(summary, { button: 0, ctrlKey: false, pointerType: 'mouse' })
  await waitFor(() => expect(screen.getByRole('menuitem', { name: 'Open folder' })).toBeTruthy())
  fireEvent.click(screen.getByRole('menuitem', { name: 'Open folder' }))
  await waitFor(() => expect(revealPath).toHaveBeenCalledWith(workspace.cwd))
  expect(onSwitchBranch).not.toHaveBeenCalled()

  if (workspace.repoRoot) {
    fireEvent.click(screen.getByText('12'))
    expect(onOpen).toHaveBeenCalledOnce()
  }

  view.rerender(<CodingStatusRow repoPath={undefined} sessionId="unknown" />)
  expect(view.container.querySelector('.coding-status-bar')).toBeNull()
})
