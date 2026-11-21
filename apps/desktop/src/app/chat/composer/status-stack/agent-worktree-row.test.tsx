import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, expect, it, vi } from 'vitest'

import type { ClientSessionState } from '@/app/types'
import { ensureGatewayForProfile, setPrimaryGateway, setPrimaryGatewayConnectionId } from '@/store/gateway'
import { $connection, _resetSessionOwnerHintsForTests, setSessionOwnerHint } from '@/store/session'
import { $sessionStates } from '@/store/session-states'
import type { AgentWorktree, CodingWorkspaceBinding } from '@/types/hermes'

import { workspaceRowClassName } from '../workspace-row'

const probe = vi.hoisted(() => vi.fn())
vi.mock('@/store/coding-status', () => ({
  registerRepoStatusCwd: (cwd?: string) => {
    probe(cwd)
  },
  repoStatusForCwd: (cwd?: string) =>
    atom(
      cwd?.endsWith('/.worktrees/task-a')
        ? {
            added: 7,
            removed: 2,
            ahead: 1,
            behind: 0,
            untracked: 0,
            branch: 'task/a',
            defaultBranch: 'main',
            detached: false
          }
        : null
    ),
  repoWorktreesForCwd: () => atom([])
}))
const { CodingStatusRow } = await import('./coding-row')

const worktree: AgentWorktree = {
  cwd: '/repo/.worktrees/task-a',
  branch: 'task/a',
  repoRoot: '/repo',
  projectName: 'repo'
}

const binding: CodingWorkspaceBinding = {
  requestId: 'a',
  projectId: 'project-a',
  sourcePath: '/repo',
  cwd: '/repo/.worktrees/task-a',
  repoRoot: '/repo',
  branch: 'task/a'
}

async function bind(state: Partial<ClientSessionState>, mode: 'local' | 'remote' = 'local') {
  setPrimaryGateway({ connectionState: 'open' } as never, 'coder')
  await ensureGatewayForProfile('coder')
  setPrimaryGatewayConnectionId(mode)
  $sessionStates.set({ a: { storedSessionId: 'stored-a', ...state } as ClientSessionState })
  setSessionOwnerHint('stored-a', { connectionId: mode, profile: 'coder', mode })
  $connection.set({ connectionId: mode, profile: 'coder', mode } as never)
}

afterEach(() => {
  cleanup()
  $sessionStates.set({})
  _resetSessionOwnerHintsForTests()
  $connection.set(null)
  vi.clearAllMocks()
})

it('shows the agent-made worktree for a chat whose own workspace is not a repo', async () => {
  // The reported shape: desktop chat in ~/projects (no git), agent worked in <repo>/.worktrees/task-a via workdir=.
  await bind({ agentWorktree: worktree, cwd: '/home/person/projects' })
  const onOpen = vi.fn()
  const view = render(<CodingStatusRow onOpen={onOpen} repoPath="" sessionId="a" />)
  const summary = screen.getByRole('button', { name: 'repo · Agent worktree · task/a' })
  expect(view.container.querySelectorAll('.coding-status-bar')).toHaveLength(1)

  // Same slot + chrome as every other state of this strip.
  for (const token of workspaceRowClassName.split(' ')) {
    expect(summary.closest('.coding-status-bar')?.classList.contains(token)).toBe(true)
  }

  // Distinct from a user-chosen workspace: the leading glyph is the agent-worktree marker, not the plain branch glyph.
  expect(view.container.querySelector('[data-slot="agent-worktree-glyph"]')).toBeTruthy()
  // Status (±, ahead) is probed for THAT worktree, not the session's non-git cwd.
  expect(probe).toHaveBeenCalledWith(worktree.cwd)
  expect(screen.getByText('7')).toBeTruthy()
  fireEvent.click(screen.getByText('7'))
  expect(onOpen).toHaveBeenCalledOnce()
})

it('discloses the path and offers folder, copy and a new chat anchored in that worktree', async () => {
  await bind({ agentWorktree: worktree, cwd: '/home/person/projects' })
  const writeText = vi.fn().mockResolvedValue(undefined)
  const revealPath = vi.fn().mockResolvedValue(true)
  Object.defineProperty(navigator, 'clipboard', { configurable: true, value: { writeText } })
  window.hermesDesktop = { ...window.hermesDesktop, revealPath } as never
  const onOpenWorktree = vi.fn()
  render(<CodingStatusRow onOpenWorktree={onOpenWorktree} repoPath="" sessionId="a" />)
  const summary = screen.getByRole('button', { name: 'repo · Agent worktree · task/a' })
  fireEvent.pointerDown(summary, { button: 0, ctrlKey: false, pointerType: 'mouse' })
  await waitFor(() => expect(window.document.querySelector('[data-slot="coding-workspace-path"]')).toBeTruthy())
  expect(window.document.querySelector('[data-slot="coding-workspace-path"]')?.getAttribute('title')).toBe(worktree.cwd)
  // The explanation is one quiet line, not a heading.
  expect(screen.getByText('The agent created this worktree during the chat.')).toBeTruthy()
  fireEvent.click(screen.getByRole('menuitem', { name: 'Copy path' }))
  await waitFor(() => expect(writeText).toHaveBeenCalledWith(worktree.cwd))
  fireEvent.keyDown(window.document.activeElement!, { key: 'Escape' })
  fireEvent.pointerDown(summary, { button: 0, ctrlKey: false, pointerType: 'mouse' })
  await waitFor(() => expect(screen.getByRole('menuitem', { name: 'Open folder' })).toBeTruthy())
  fireEvent.click(screen.getByRole('menuitem', { name: 'Open folder' }))
  await waitFor(() => expect(revealPath).toHaveBeenCalledWith(worktree.cwd))
  fireEvent.keyDown(window.document.activeElement!, { key: 'Escape' })
  fireEvent.pointerDown(summary, { button: 0, ctrlKey: false, pointerType: 'mouse' })
  fireEvent.click(await screen.findByRole('menuitem', { name: 'New chat in this worktree' }))
  expect(onOpenWorktree).toHaveBeenCalledWith(worktree.cwd)
})

it('never shows the native folder action for a remote owner', async () => {
  await bind({ agentWorktree: worktree, cwd: '/srv/projects' }, 'remote')
  render(<CodingStatusRow repoPath="" sessionId="a" />)
  fireEvent.pointerDown(screen.getByRole('button', { name: 'repo · Agent worktree · task/a' }), {
    button: 0,
    ctrlKey: false,
    pointerType: 'mouse'
  })
  await screen.findByRole('menuitem', { name: 'Copy path' })
  expect(screen.queryByRole('menuitem', { name: 'Open folder' })).toBeNull()
  expect(probe).not.toHaveBeenCalledWith(worktree.cwd)
})

it('a user-chosen workspace binding always wins over the agent badge', async () => {
  await bind({
    codingWorkspace: binding,
    agentWorktree: { ...worktree, cwd: '/repo/.worktrees/other', branch: 'other' }
  })
  render(<CodingStatusRow repoPath="" sessionId="a" />)
  expect(screen.getByRole('button', { name: 'repo · Worktree · task/a' })).toBeTruthy()
  expect(screen.queryByRole('button', { name: /Agent worktree/ })).toBeNull()
})

it('falls back to the plain branch strip once the badge is cleared', async () => {
  await bind({ agentWorktree: worktree, cwd: '/home/person/projects' })
  const view = render(<CodingStatusRow repoPath="" sessionId="a" />)
  expect(screen.getByRole('button', { name: 'repo · Agent worktree · task/a' })).toBeTruthy()
  $sessionStates.set({
    a: { storedSessionId: 'stored-a', agentWorktree: null, cwd: '/home/person/projects' } as ClientSessionState
  })
  await waitFor(() => expect(view.container.querySelector('.coding-status-bar')).toBeNull())
})
