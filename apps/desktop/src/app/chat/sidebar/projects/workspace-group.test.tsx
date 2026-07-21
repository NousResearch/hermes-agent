import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { SessionInfo } from '@/hermes'
import { I18nProvider } from '@/i18n'
import { notifyError } from '@/store/notifications'
import { switchBranchInRepo } from '@/store/projects'

import { EnteredProjectContent } from './entered-content'
import { SidebarWorkspaceGroup } from './workspace-group'
import type { SidebarProjectTree, SidebarSessionGroup } from './workspace-groups'

vi.mock('@/store/notifications', () => ({ notifyError: vi.fn() }))
vi.mock('@/store/profile', () => ({ newSessionInProfile: vi.fn() }))
vi.mock('@/store/projects', async importOriginal => {
  const actual = await importOriginal<Record<string, unknown>>()

  return { ...actual, switchBranchInRepo: vi.fn() }
})

const lane = (git: boolean, overrides: Partial<SidebarSessionGroup> = {}): SidebarSessionGroup => ({
  id: git ? '/repo::branch::main' : '/work/notes',
  isMain: true,
  label: git ? 'main' : 'notes',
  path: git ? '/repo' : '/work/notes',
  sessions: [],
  ...overrides
})

function renderGroup(
  group: SidebarSessionGroup,
  onNewSession: (path: null | string) => void,
  gitKind?: 'directory' | 'git'
) {
  return render(
    <I18nProvider configClient={null}>
      <SidebarWorkspaceGroup gitKind={gitKind} group={group} onNewSession={onNewSession} renderRows={() => null} />
    </I18nProvider>
  )
}

const liveSession = (overrides: Partial<SessionInfo> = {}): SessionInfo => ({
  archived: false,
  cwd: '/repo',
  ended_at: null,
  id: 'fresh',
  input_tokens: 0,
  is_active: true,
  last_active: 1_000,
  message_count: 0,
  model: 'claude',
  output_tokens: 0,
  preview: null,
  source: 'cli',
  started_at: 1_000,
  title: null,
  tool_call_count: 0,
  ...overrides
})

describe('SidebarWorkspaceGroup new session behavior', () => {
  beforeEach(() => {
    vi.mocked(switchBranchInRepo).mockReset().mockResolvedValue()
    vi.mocked(notifyError).mockReset()
  })

  afterEach(cleanup)

  it('creates directly in a non-git directory lane without switching a branch', async () => {
    const onNewSession = vi.fn()

    renderGroup(lane(false), onNewSession, 'directory')
    fireEvent.click(screen.getByRole('button', { name: 'New session in notes' }))

    await waitFor(() => expect(onNewSession).toHaveBeenCalledWith('/work/notes'))
    expect(switchBranchInRepo).not.toHaveBeenCalled()
  })

  it('switches a real branch before creating the session', async () => {
    const calls: string[] = []
    const onNewSession = vi.fn(() => calls.push('create'))

    vi.mocked(switchBranchInRepo).mockImplementation(async () => {
      calls.push('switch')
    })

    renderGroup(lane(true), onNewSession, 'git')
    fireEvent.click(screen.getByRole('button', { name: 'New session in main' }))

    await waitFor(() => expect(onNewSession).toHaveBeenCalledWith('/repo'))
    expect(switchBranchInRepo).toHaveBeenCalledWith('/repo', 'main')
    expect(calls).toEqual(['switch', 'create'])
  })

  it('propagates optimistic git capability through entered project rendering before creating', async () => {
    const calls: string[] = []
    const onNewSession = vi.fn(() => calls.push('create'))

    const project: SidebarProjectTree = {
      id: '/repo',
      label: 'repo',
      path: '/repo',
      repos: [
        {
          id: '/repo',
          gitKind: 'directory',
          groups: [],
          label: 'repo',
          path: '/repo',
          sessionCount: 0
        }
      ],
      sessionCount: 0
    }

    vi.mocked(switchBranchInRepo).mockImplementation(async () => {
      calls.push('switch')
    })

    render(
      <I18nProvider configClient={null}>
        <EnteredProjectContent
          liveSessions={[liveSession({ git_branch: 'release', git_repo_root: '/repo' })]}
          onNewSession={onNewSession}
          project={project}
          renderRows={() => null}
        />
      </I18nProvider>
    )
    fireEvent.click(screen.getByRole('button', { name: 'New session in release' }))

    await waitFor(() => expect(onNewSession).toHaveBeenCalledWith('/repo'))
    expect(switchBranchInRepo).toHaveBeenCalledWith('/repo', 'release')
    expect(calls).toEqual(['switch', 'create'])
  })

  it('reports a real branch-switch failure and does not create a session', async () => {
    const failure = new Error('branch is checked out elsewhere')
    const onNewSession = vi.fn()

    vi.mocked(switchBranchInRepo).mockRejectedValue(failure)

    renderGroup(lane(true), onNewSession, 'git')
    fireEvent.click(screen.getByRole('button', { name: 'New session in main' }))

    await waitFor(() => expect(notifyError).toHaveBeenCalledWith(failure, 'Could not switch to main'))
    expect(onNewSession).not.toHaveBeenCalled()
  })

  it('keeps switching canonical branch lanes from an older backend', async () => {
    const onNewSession = vi.fn()

    renderGroup(lane(true, { id: '/repo::branch::release', label: 'release' }), onNewSession)
    fireEvent.click(screen.getByRole('button', { name: 'New session in release' }))

    await waitFor(() => expect(onNewSession).toHaveBeenCalledWith('/repo'))
    expect(switchBranchInRepo).toHaveBeenCalledWith('/repo', 'release')
  })

  it('does not switch a path-backed lane from an older backend', async () => {
    const onNewSession = vi.fn()

    renderGroup(lane(false), onNewSession)
    fireEvent.click(screen.getByRole('button', { name: 'New session in notes' }))

    await waitFor(() => expect(onNewSession).toHaveBeenCalledWith('/work/notes'))
    expect(switchBranchInRepo).not.toHaveBeenCalled()
  })

  it('does not switch a legacy lane whose branch id and label disagree', async () => {
    const onNewSession = vi.fn()

    renderGroup(lane(true, { id: '/repo::branch::release', label: 'main' }), onNewSession)
    fireEvent.click(screen.getByRole('button', { name: 'New session in main' }))

    await waitFor(() => expect(onNewSession).toHaveBeenCalledWith('/repo'))
    expect(switchBranchInRepo).not.toHaveBeenCalled()
  })

  it('does not switch a path-backed fallback lane inside a repo promoted to git', async () => {
    const onNewSession = vi.fn()

    renderGroup(lane(false), onNewSession, 'git')
    fireEvent.click(screen.getByRole('button', { name: 'New session in notes' }))

    await waitFor(() => expect(onNewSession).toHaveBeenCalledWith('/work/notes'))
    expect(switchBranchInRepo).not.toHaveBeenCalled()
  })
})
