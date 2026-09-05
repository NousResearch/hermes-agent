import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest'

import { CodingWorkspaceControls } from './coding-workspace-controls'

const mocks = vi.hoisted(() => ({ inspect: vi.fn(), set: vi.fn(), request: vi.fn(), register: vi.fn(), config: vi.fn((..._args: unknown[]) => ({ data: {} })) }))
vi.mock('@/app/hooks/use-config-record', () => ({ useHermesConfigRecord: (...a: unknown[]) => mocks.config(...a) }))
vi.mock('@/store/coding-workspaces', async () => {
  const { atom } = await import('nanostores')

  return { $codingWorkspaceDrafts: atom({}), codingWorkspaceKey: (o: unknown) => JSON.stringify(o),
    setCodingWorkspaceIntent: (...a: unknown[]) => mocks.set(...a), inspectCodingWorkspace: (...a: unknown[]) => mocks.inspect(...a),
    registerCodingWorkspaceFolder: (...a: unknown[]) => mocks.register(...a),
    listCodingWorkspaceProjects: (owner: { connectionId: string; profile: string }) => mocks.request(owner.connectionId, owner.profile, 'projects.list', { profile: owner.profile }).then((r: { projects: unknown[] }) => r.projects) }
})
vi.mock('@/store/gateway', () => ({ requestGatewayForAgent: (...a: unknown[]) => mocks.request(...a) }))
vi.mock('@/i18n', async () => { const { en } = await import('@/i18n/en');

 return { useI18n: () => ({ t: en }) } })

beforeAll(() => {
  vi.stubGlobal('ResizeObserver', class { observe() {} unobserve() {} disconnect() {} })
  Element.prototype.scrollIntoView = vi.fn()
  Element.prototype.hasPointerCapture = vi.fn(() => false)
  Element.prototype.releasePointerCapture = vi.fn()
})
afterEach(() => { cleanup(); vi.clearAllMocks() })
const owner = { connectionId: 'source-a', profile: 'coder', draftKey: 'draft:one' }
const project = { id: 'p_one', name: 'One project', primary_path: '/repo', archived: false, folders: [] }

function mount(draft?: any) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })

  return render(<QueryClientProvider client={client}><CodingWorkspaceControls draft={draft} onSelectFolder={vi.fn()} owner={owner} /></QueryClientProvider>)
}

describe('coding workspace controls', () => {
  it('shows the actual branch and lets new worktrees choose a base without creating anything', async () => {
    mount({ owner, requestId: 'r', status: 'ready', intent: { path: '/repo', mode: 'worktree' },
      inspection: { path: '/repo', repoRoot: '/repo', branch: 'feature/actual', branches: ['main', 'feature/actual'], dirty: true, worktrees: [] } })
    expect(screen.getByText('Uncommitted source changes are not copied into the new worktree.')).toBeTruthy()
    const workIn = screen.getByRole('combobox', { name: 'Work in' })
    const base = screen.getByRole('combobox', { name: 'From branch' })
    expect(workIn.className).toContain('w-auto')
    expect(base.className).toContain('w-auto')
    expect(workIn.closest('[data-slot="coding-workspace-main"]')).toBe(base.closest('[data-slot="coding-workspace-main"]'))
    expect(screen.queryByRole('button', { name: 'Cancel' })).toBeNull()
    fireEvent.keyDown(screen.getByRole('combobox', { name: 'From branch' }), { key: 'Enter' })
    fireEvent.click(await screen.findByRole('option', { name: 'main' }))
    expect(mocks.set).toHaveBeenCalledWith(owner, { path: '/repo', mode: 'worktree', base: 'main' })
  })

  it('offers existing checkout path, branch, dirty state and warns about sharing', async () => {
    mount({ owner, requestId: 'r', status: 'ready', intent: { path: '/repo', mode: 'existing' },
      inspection: { path: '/repo', repoRoot: '/repo', branch: 'main', dirty: false,
        worktrees: [{ path: '/repo/.worktrees/task', branch: 'task/real', dirty: true, sharedSessions: 2 }] } })
    expect(screen.getByText('This checkout may be shared with other chats. Changes are not isolated.')).toBeTruthy()
    fireEvent.keyDown(screen.getByRole('combobox', { name: 'Choose checkout' }), { key: 'Enter' })
    const option = await screen.findByRole('option', { name: /task\/real.*Uncommitted changes/ })
    expect(option.textContent).toContain('/repo/.worktrees/task')
    fireEvent.click(option)
    expect(mocks.set).toHaveBeenCalledWith(owner, { path: '/repo', mode: 'existing', existingPath: '/repo/.worktrees/task' })
  })

  it('describes the main checkout when Current is selected from a linked worktree', () => {
    mount({ owner, requestId: 'r', status: 'ready', intent: { path: '/repo/.worktrees/task', mode: 'current' },
      inspection: { path: '/repo/.worktrees/task', repoRoot: '/repo/.worktrees/task', branch: 'feature', dirty: false,
        worktrees: [
          { path: '/repo/.worktrees/task', branch: 'feature', dirty: false, activeSessionCount: 1 },
          { path: '/repo', branch: 'main', dirty: true, isMain: true, activeSessionCount: 3 }
        ] } })
    expect(screen.getByText('main · /repo · Uncommitted changes')).toBeTruthy()
    expect(screen.getByText('In use · 3')).toBeTruthy()
    expect(screen.queryByText('In use · 1')).toBeNull()
  })

  it('identifies a selected checkout already used by active chats', () => {
    mount({ owner, requestId: 'r', status: 'ready', intent: { path: '/repo', mode: 'existing', existingPath: '/repo/.worktrees/task' },
      inspection: { path: '/repo', repoRoot: '/repo', branch: 'main', dirty: false,
        worktrees: [{ path: '/repo/.worktrees/task', branch: 'task/real', dirty: true, activeSessionCount: 2 }] } })
    expect(screen.getByText('In use · 2')).toBeTruthy()
  })

  it('separates the project label from its selected value', () => {
    mount()
    expect(screen.getByRole('button', { name: /Project/ }).textContent).toBe('Project: No project')
  })

  it('chooses a project with read-only inspection and never navigates or prepares a checkout', async () => {
    mocks.request.mockResolvedValue({ projects: [project], active_id: project.id })
    mocks.inspect.mockResolvedValue({ path: '/repo', repoRoot: '/repo', branch: 'feature/actual', dirty: true, worktrees: [] })
    mount()
    fireEvent.click(screen.getByRole('button', { name: /Project/ }))
    fireEvent.click(await screen.findByRole('button', { name: /One project/ }))
    await waitFor(() => expect(mocks.set).toHaveBeenCalledWith(owner, { projectId: 'p_one', path: '/repo', mode: 'worktree' }))
    expect(mocks.inspect).toHaveBeenCalledWith(owner)
    expect(mocks.request).toHaveBeenCalledWith('source-a', 'coder', 'projects.list', { profile: 'coder' })
  })

  it.each(['current', 'worktree', undefined])('honors the owner-scoped default checkout %s in the project picker', async mode => {
    mocks.config.mockReturnValue({ data: { desktop: { coding: { default_checkout: mode } } } })
    mocks.request.mockResolvedValue({ projects: [project] })
    mocks.inspect.mockResolvedValue({ path: '/repo', repoRoot: '/repo', worktrees: [] })
    mount()
    fireEvent.click(screen.getByRole('button', { name: /Project/ }))
    fireEvent.click(await screen.findByRole('button', { name: /One project/ }))
    await waitFor(() => expect(mocks.set).toHaveBeenCalledWith(owner, { projectId: project.id, path: project.primary_path, mode: mode ?? 'worktree' }))
    expect(mocks.config).toHaveBeenCalledWith({ connectionId: owner.connectionId, profile: owner.profile })
  })
})
