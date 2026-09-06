import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest'

import { StatusRow } from '@/components/chat/status-row'
import type { CodingWorkspaceDraft } from '@/store/coding-workspaces'

import { CodingWorkspaceControls } from './coding-workspace-controls'
import { workspaceRowClassName } from './workspace-row'

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

function mount(draft?: any, onSelectFolder = vi.fn()) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })

  return render(<CodingWorkspaceControls draft={draft} onSelectFolder={onSelectFolder} owner={owner} />, {
    wrapper: ({ children }) => <QueryClientProvider client={client}>{children}</QueryClientProvider>
  })
}

describe('coding workspace controls', () => {
  it('shares bound StatusRow chrome and a fixed leading slot through inspection and preparation', () => {
    const reference = render(<StatusRow className={workspaceRowClassName} leading={<span />}>Bound summary</StatusRow>)
    const chrome = reference.container.firstElementChild!

    const draft: CodingWorkspaceDraft = { owner, requestId: 'r', status: 'ready', intent: { path: '/repo', mode: 'worktree' },
      inspection: { path: '/repo', repoRoot: '/repo', branch: 'main', dirty: false, worktrees: [] } }

    const view = mount(draft)
    const row = () => view.container.querySelector('.coding-status-bar')!
    expect(row()).toBeTruthy()
    expect(row().className).toBe(chrome.className)
    const leading = row().firstElementChild!
    expect(leading.className).toBe(chrome.firstElementChild!.className)
    expect(leading.querySelector('.codicon-git-branch')?.classList.contains('text-(--ui-text-tertiary)')).toBe(true)

    for (const status of ['inspecting', 'preparing'] as const) {
      view.rerender(<CodingWorkspaceControls draft={{ ...draft, status }} onSelectFolder={vi.fn()} owner={owner} />)
      expect(row().className).toBe(chrome.className)
      expect(row().firstElementChild).toBe(leading)
      expect(leading.querySelector('.codicon')).toBeNull()
      expect(leading.querySelector('[role="status"]')?.classList.contains('size-3.5')).toBe(true)
      expect(view.container.querySelectorAll('[role="status"]')).toHaveLength(1)

      for (const trigger of view.container.querySelectorAll('button')) {
        expect(trigger.getAttribute('data-size')).toBe('inline')
      }
    }

    view.rerender(<CodingWorkspaceControls onSelectFolder={vi.fn()} owner={owner} />)
    expect(leading.querySelector('.codicon-folder')).toBeTruthy()
  })

  it.each([
    [{ mode: 'existing', existingPath: '/repo/.worktrees/task' }, 'task/real', 'task/real', 'Existing worktree · task/real'],
    [{ mode: 'existing', existingPath: '/repo/.worktrees/task' }, null, 'task', 'Existing worktree · task'],
    [{ mode: 'worktree', base: 'release' }, 'main', 'New worktree · release', 'New worktree · release'],
    [{ mode: 'worktree', base: 'main' }, 'main', 'New worktree', 'New worktree']
  ])('reflects the selected checkout/base in the visible and accessible summary: %s', (choice, branch, visible, accessible) => {
    mount({ owner, requestId: 'r', status: 'ready', intent: { path: '/repo', ...choice },
      inspection: { path: '/repo', repoRoot: '/repo', branch: 'main', dirty: false,
        worktrees: [{ path: '/repo/.worktrees/task', branch }] } })
    const trigger = screen.getByRole('button', { name: `Work in: ${accessible}` })
    expect(trigger.textContent).toBe(visible)
    expect(trigger.querySelector('.codicon-chevron-down')).toBeTruthy()
  })

  it('keeps a prepared-but-unsent workspace compact while retaining its branch, exact path and failure', () => {
    const cwd = '/home/person/.hermes/profiles/coder/cache/project/.worktrees/prepared-task'
    mount({ owner, requestId: 'r', status: 'error', intent: { path: '/repo', mode: 'worktree' },
      inspection: { path: '/repo', repoRoot: '/repo', branch: 'main', dirty: false, worktrees: [] },
      prepared: { cwd, branch: 'task/prepared' }, error: 'Send failed; retry the draft' })
    const trigger = screen.getByRole('button', { name: 'Work in: New worktree · task/prepared' })
    expect(trigger.textContent).toBe('New worktree · task/prepared')
    expect(trigger.querySelector('span')?.getAttribute('title')).toBe(cwd)
    expect(trigger.hasAttribute('disabled')).toBe(true)
    expect(screen.queryByText(cwd, { exact: false })).toBeNull()
    expect(screen.getByRole('alert').textContent).toContain('Send failed; retry the draft')
  })

  it('shows the actual branch and lets new worktrees choose a base without creating anything', async () => {
    mount({ owner, requestId: 'r', status: 'ready', intent: { path: '/repo', mode: 'worktree' },
      inspection: { path: '/repo', repoRoot: '/repo', branch: 'feature/actual', branches: ['main', 'feature/actual'], dirty: true, worktrees: [] } })
    expect(screen.queryByRole('combobox', { name: 'From branch' })).toBeNull()
    expect(screen.queryByText('Created on first Send, not now.')).toBeNull()
    const workIn = screen.getByRole('button', { name: 'Work in: New worktree' })
    expect(workIn.textContent).toBe('New worktree')
    expect(workIn.getAttribute('data-variant')).toBe('text')
    fireEvent.keyDown(workIn, { key: 'Enter' })
    expect(screen.getByText('Uncommitted source changes are not copied into the new worktree.')).toBeTruthy()
    expect(screen.getByRole('menuitemradio', { name: 'New worktree' }).getAttribute('aria-checked')).toBe('true')
    expect(screen.queryByRole('combobox')).toBeNull()
    const base = screen.getByRole('menuitem', { name: 'From branch feature/actual' })
    expect(base.closest('[data-slot="coding-workspace-main"]')).toBeNull()
    fireEvent.keyDown(base, { key: 'ArrowRight' })
    fireEvent.click(await screen.findByRole('menuitemradio', { name: 'main' }))
    expect(mocks.set).toHaveBeenCalledWith(owner, { path: '/repo', mode: 'worktree', base: 'main' })
  })

  it('offers existing checkout path, branch, dirty state and warns about sharing', async () => {
    mount({ owner, requestId: 'r', status: 'ready', intent: { path: '/repo', mode: 'existing' },
      inspection: { path: '/repo', repoRoot: '/repo', branch: 'main', dirty: false,
        worktrees: [{ path: '/repo/.worktrees/task', branch: 'task/real', dirty: true, sharedSessions: 2 }] } })
    fireEvent.keyDown(screen.getByRole('button', { name: /^Work in: Existing worktree/ }), { key: 'Enter' })
    expect(screen.getByText('This checkout may be shared with other chats. Changes are not isolated.')).toBeTruthy()
    fireEvent.keyDown(screen.getByRole('menuitem', { name: 'Choose checkout' }), { key: 'ArrowRight' })
    const option = await screen.findByRole('menuitemradio', { name: /task\/real.*Uncommitted changes/ })
    expect(option.textContent).toContain('/repo/.worktrees/task')
    fireEvent.click(option)
    expect(mocks.set).toHaveBeenCalledWith(owner, { path: '/repo', mode: 'existing', existingPath: '/repo/.worktrees/task' })
  })

  it.each(['pointer', 'keyboard'])('preserves the selected existing checkout on %s mode reselection', async input => {
    const intent = { path: '/repo', mode: 'existing', existingPath: '/repo/.worktrees/task' }

    const inspection = { path: '/repo', repoRoot: '/repo', branch: 'main', dirty: false,
      worktrees: [{ path: intent.existingPath, branch: 'task/real', dirty: false }] }

    mocks.inspect.mockResolvedValue(inspection)
    mount({ owner, requestId: 'r', status: 'ready', intent, inspection })
    fireEvent.keyDown(screen.getByRole('button', { name: /^Work in: Existing worktree/ }), { key: 'Enter' })
    const checked = screen.getByRole('menuitemradio', { name: 'Existing worktree' })
    expect(checked.getAttribute('aria-checked')).toBe('true')

    const activate = (item: HTMLElement) => input === 'pointer'
      ? fireEvent.click(item) : fireEvent.keyDown(item, { key: 'Enter' })

    activate(checked)
    expect(mocks.set).not.toHaveBeenCalled()
    expect(mocks.inspect).not.toHaveBeenCalled()
    fireEvent.keyDown(screen.getByRole('menuitem', { name: 'Choose checkout' }), { key: 'ArrowRight' })
    const checkout = await screen.findByRole('menuitemradio', { name: /task\/real/ })
    expect(checkout.textContent).toContain(intent.existingPath)
    expect(checkout.getAttribute('aria-checked')).toBe('true')
    fireEvent.keyDown(checkout, { key: 'ArrowLeft' })

    activate(screen.getByRole('menuitemradio', { name: 'New worktree' }))
    expect(mocks.set).toHaveBeenCalledExactlyOnceWith(owner, { ...intent, mode: 'worktree', existingPath: undefined })
    await waitFor(() => expect(mocks.inspect).toHaveBeenCalledExactlyOnceWith(owner))
  })

  it.each([
    ['worktree', 'New worktree', 'From branch main', 'main'],
    ['existing', 'Existing worktree', 'Choose checkout', 'task/real · /repo/.worktrees/task']
  ] as const)('disables already-open %s submenu choices when preparation locks the draft', async (mode, label, submenu, option) => {
    const draft: CodingWorkspaceDraft = { owner, requestId: 'r', status: 'ready',
      intent: { path: '/repo', mode, existingPath: '/repo/.worktrees/task' },
      inspection: { path: '/repo', repoRoot: '/repo', branch: 'main', branches: ['main'], dirty: false,
        worktrees: [{ path: '/repo/.worktrees/task', branch: 'task/real' }] } }

    const view = mount(draft)
    fireEvent.keyDown(screen.getByRole('button', { name: new RegExp(`^Work in: ${label}`) }), { key: 'Enter' })
    fireEvent.keyDown(screen.getByRole('menuitem', { name: submenu }), { key: 'ArrowRight' })
    expect((await screen.findByRole('menuitemradio', { name: option })).getAttribute('aria-disabled')).not.toBe('true')

    view.rerender(<CodingWorkspaceControls draft={{ ...draft, status: 'preparing' }} onSelectFolder={vi.fn()} owner={owner} />)
    const choice = screen.getByRole('menuitemradio', { name: option })
    expect(choice.getAttribute('aria-disabled')).toBe('true')
    expect(choice.hasAttribute('data-disabled')).toBe(true)
    fireEvent.click(choice)
    fireEvent.keyDown(choice, { key: 'Enter' })
    expect(mocks.set).not.toHaveBeenCalled()
    expect(mocks.inspect).not.toHaveBeenCalled()
  })

  it('describes the main checkout when Current is selected from a linked worktree', () => {
    mount({ owner, requestId: 'r', status: 'ready', intent: { path: '/repo/.worktrees/task', mode: 'current' },
      inspection: { path: '/repo/.worktrees/task', repoRoot: '/repo/.worktrees/task', branch: 'feature', dirty: false,
        worktrees: [
          { path: '/repo/.worktrees/task', branch: 'feature', dirty: false, activeSessionCount: 1 },
          { path: '/repo', branch: 'main', dirty: true, isMain: true, activeSessionCount: 3 }
        ] } })
    fireEvent.keyDown(screen.getByRole('button', { name: 'Work in: Current checkout' }), { key: 'Enter' })
    expect(screen.getByText('main · /repo · Uncommitted changes')).toBeTruthy()
    expect(screen.getByText('In use · 3')).toBeTruthy()
    expect(screen.queryByText('In use · 1')).toBeNull()
  })

  it('identifies a selected checkout already used by active chats', () => {
    mount({ owner, requestId: 'r', status: 'ready', intent: { path: '/repo', mode: 'existing', existingPath: '/repo/.worktrees/task' },
      inspection: { path: '/repo', repoRoot: '/repo', branch: 'main', dirty: false,
        worktrees: [{ path: '/repo/.worktrees/task', branch: 'task/real', dirty: true, activeSessionCount: 2 }] } })
    fireEvent.keyDown(screen.getByRole('button', { name: /^Work in: Existing worktree/ }), { key: 'Enter' })
    expect(screen.getByText('In use · 2')).toBeTruthy()
  })

  it('keeps the project name unboxed without a visible field prefix', () => {
    mount({ intent: { path: '/home/person/projects/demo-app', mode: 'worktree' }, status: 'ready' })
    const project = screen.getByRole('button', { name: 'Project: demo-app' })
    expect(project.textContent).toBe('demo-app')
    expect(project.getAttribute('data-variant')).toBe('text')
  })

  it('filters the native project radio menu and navigates with arrows, Enter and Escape without resetting a reselected draft', async () => {
    mocks.request.mockResolvedValue({ projects: [project,
      { ...project, id: 'two', name: 'Other project', primary_path: '/other' },
      { ...project, id: 'archived', name: 'Archived', archived: true },
      { ...project, id: 'empty', name: 'No folder', primary_path: null }] })
    const intent = { projectId: project.id, path: '/repo', mode: 'existing', existingPath: '/repo/task', base: 'custom' }
    mount({ owner, intent, status: 'ready' })
    const trigger = screen.getByRole('button', { name: 'Project: repo' })
    fireEvent.keyDown(trigger, { key: 'Enter' })
    const search = await screen.findByRole('textbox', { name: 'Search projects' })
    const chosen = await screen.findByRole('menuitemradio', { name: /One project/ })
    expect(chosen.getAttribute('aria-checked')).toBe('true')
    expect(screen.queryByRole('menuitemradio', { name: /Archived|No folder/ })).toBeNull()
    fireEvent.change(search, { target: { value: '/repo' } })
    expect(screen.queryByRole('menuitemradio', { name: /Other project/ })).toBeNull()
    search.focus()
    fireEvent.keyDown(search, { key: 'ArrowDown' })
    await waitFor(() => expect(window.document.activeElement).toBe(screen.getByRole('menuitemradio', { name: 'No project' })))
    fireEvent.keyDown(window.document.activeElement!, { key: 'ArrowDown' })
    await waitFor(() => expect(window.document.activeElement).toBe(chosen))
    fireEvent.keyDown(window.document.activeElement!, { key: 'Enter' })
    await waitFor(() => expect(screen.queryByRole('menu')).toBeNull())
    expect(mocks.set).not.toHaveBeenCalled()
    expect(mocks.inspect).not.toHaveBeenCalled()
    fireEvent.keyDown(trigger, { key: 'Enter' })
    fireEvent.keyDown(await screen.findByRole('menuitemradio', { name: 'No project' }), { key: 'Enter' })
    expect(mocks.set).toHaveBeenCalledExactlyOnceWith(owner, null)
    expect(mocks.inspect).not.toHaveBeenCalled()
    fireEvent.keyDown(trigger, { key: 'Enter' })
    fireEvent.keyDown(await screen.findByRole('textbox'), { key: 'Escape' })
    await waitFor(() => expect(screen.queryByRole('menu')).toBeNull())
    await waitFor(() => expect(window.document.activeElement).toBe(trigger))
  })

  it('keeps Browse separate from filtered radio choices and lets ArrowUp reach it', async () => {
    mocks.request.mockResolvedValue({ projects: [project] })
    const browse = vi.fn()
    mount(undefined, browse)
    fireEvent.keyDown(screen.getByRole('button', { name: /Project/ }), { key: 'Enter' })
    const search = await screen.findByRole('textbox')
    fireEvent.change(search, { target: { value: 'no match' } })
    expect(screen.queryByRole('menuitemradio', { name: /One project/ })).toBeNull()
    expect(screen.getByRole('separator')).toBeTruthy()
    search.focus()
    fireEvent.keyDown(search, { key: 'ArrowUp' })
    await waitFor(() => expect(window.document.activeElement).toBe(screen.getByRole('menuitem', { name: 'Browse…' })))
    fireEvent.keyDown(window.document.activeElement!, { key: 'Enter' })
    expect(browse).toHaveBeenCalledExactlyOnceWith()
    expect(mocks.set).not.toHaveBeenCalled()
    expect(mocks.inspect).not.toHaveBeenCalled()
    await waitFor(() => expect(screen.queryByRole('menu')).toBeNull())
  })

  it('disables already-open project choices and Browse when preparation locks the draft', async () => {
    mocks.request.mockResolvedValue({ projects: [project] })
    const browse = vi.fn()
    const draft: CodingWorkspaceDraft = { owner, requestId: 'r', status: 'ready', intent: { path: '/other', mode: 'worktree' } }
    const view = mount(draft, browse)
    fireEvent.keyDown(screen.getByRole('button', { name: /Project/ }), { key: 'Enter' })
    await screen.findByRole('menuitemradio', { name: /One project/ })
    view.rerender(<CodingWorkspaceControls draft={{ ...draft, status: 'preparing' }} onSelectFolder={browse} owner={owner} />)

    for (const item of [...screen.getAllByRole('menuitemradio'), screen.getByRole('menuitem', { name: 'Browse…' })]) {
      expect(item.getAttribute('aria-disabled')).toBe('true')
      fireEvent.click(item)
      fireEvent.keyDown(item, { key: 'Enter' })
    }

    expect(browse).not.toHaveBeenCalled()
    expect(mocks.set).not.toHaveBeenCalled()
    expect(mocks.inspect).not.toHaveBeenCalled()
  })

  it('chooses a project with read-only inspection and never navigates or prepares a checkout', async () => {
    mocks.request.mockResolvedValue({ projects: [project], active_id: project.id })
    mocks.inspect.mockResolvedValue({ path: '/repo', repoRoot: '/repo', branch: 'feature/actual', dirty: true, worktrees: [] })
    mount()
    fireEvent.pointerDown(screen.getByRole('button', { name: /Project/ }), { button: 0, ctrlKey: false, pointerType: 'mouse' })
    fireEvent.click(await screen.findByRole('menuitemradio', { name: /One project/ }))
    await waitFor(() => expect(mocks.set).toHaveBeenCalledWith(owner, { projectId: 'p_one', path: '/repo', mode: 'worktree' }))
    expect(mocks.inspect).toHaveBeenCalledWith(owner)
    expect(mocks.request).toHaveBeenCalledWith('source-a', 'coder', 'projects.list', { profile: 'coder' })
  })

  it.each(['current', 'worktree', undefined])('honors the owner-scoped default checkout %s in the project picker', async mode => {
    mocks.config.mockReturnValue({ data: { desktop: { coding: { default_checkout: mode } } } })
    mocks.request.mockResolvedValue({ projects: [project] })
    mocks.inspect.mockResolvedValue({ path: '/repo', repoRoot: '/repo', worktrees: [] })
    mount()
    fireEvent.pointerDown(screen.getByRole('button', { name: /Project/ }), { button: 0, ctrlKey: false, pointerType: 'mouse' })
    fireEvent.click(await screen.findByRole('menuitemradio', { name: /One project/ }))
    await waitFor(() => expect(mocks.set).toHaveBeenCalledWith(owner, { projectId: project.id, path: project.primary_path, mode: mode ?? 'worktree' }))
    expect(mocks.config).toHaveBeenCalledWith({ connectionId: owner.connectionId, profile: owner.profile })
  })
})
