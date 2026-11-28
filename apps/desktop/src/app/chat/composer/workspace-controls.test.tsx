import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $approvalModes } from '@/store/approval-mode'
import { $projectTree } from '@/store/projects'
import { $currentCwd, $gatewayState } from '@/store/session'
const setApprovalModeForProfile = vi.hoisted(() => vi.fn(async () => {}))
const openProjectRename = vi.hoisted(() => vi.fn())
const openProjectAddFolder = vi.hoisted(() => vi.fn())
const setActiveProject = vi.hoisted(() => vi.fn(async () => {}))
const refreshProjectTree = vi.hoisted(() => vi.fn(async () => {}))

vi.mock('@/store/approval-mode', async importOriginal => {
  const actual = await importOriginal<Record<string, unknown>>()

  return {
    ...actual,
    setApprovalModeForProfile
  }
})

vi.mock('@/store/projects', async importOriginal => {
  const actual = await importOriginal<Record<string, unknown>>()

  return {
    ...actual,
    openProjectAddFolder,
    openProjectRename,
    refreshProjectTree,
    setActiveProject
  }
})

import { ComposerWorkspaceControls } from './workspace-controls'

describe('ComposerWorkspaceControls', () => {
  beforeEach(() => {
    setApprovalModeForProfile.mockClear()
    openProjectRename.mockClear()
    openProjectAddFolder.mockClear()
    setActiveProject.mockClear()
    refreshProjectTree.mockClear()
    $currentCwd.set('/home/user/hermes-desktop')
    $gatewayState.set('open')
    $approvalModes.set({ default: 'smart' })
    $projectTree.set([
      {
        id: 'proj-1',
        label: 'Hermes Desktop',
        path: '/home/user/hermes-desktop',
        repos: [],
        sessionCount: 3
      },
      {
        id: 'proj-2',
        label: 'Docs Site',
        path: '/home/user/docs-site',
        repos: [],
        sessionCount: 1
      }
    ])
  })

  afterEach(() => {
    cleanup()
    $projectTree.set([])
    $currentCwd.set('')
    $approvalModes.set({})
  })

  it('shows active project name and lets the user switch to another project via onChangeCwd', async () => {
    const onChangeCwd = vi.fn(async () => {})

    render(
      <ComposerWorkspaceControls
        busy={false}
        cwd="/home/user/hermes-desktop"
        disabled={false}
        onChangeCwd={onChangeCwd}
        profile="default"
        sessionId="sess-1"
      />
    )

    const trigger = screen.getByTestId('composer-project-pill')
    expect(trigger.textContent).toContain('Hermes Desktop')

    fireEvent.pointerDown(trigger, { button: 0, ctrlKey: false })
    fireEvent.click(await screen.findByRole('menuitem', { name: 'Docs Site' }))

    await waitFor(() => expect(onChangeCwd).toHaveBeenCalledWith('/home/user/docs-site', 'sess-1'))
  })

  it('shows current approval mode and changes it for the active profile', async () => {
    $approvalModes.set({ ops: 'manual' })
    render(<ComposerWorkspaceControls busy={false} cwd="/home/user/hermes-desktop" disabled={false} profile="ops" />)

    const trigger = screen.getByTestId('composer-approval-pill')
    expect(trigger.textContent).toContain('Manual')

    fireEvent.pointerDown(trigger, { button: 0, ctrlKey: false })
    fireEvent.click(await screen.findByRole('menuitemradio', { name: /Smart/i }))

    await waitFor(() => expect(setApprovalModeForProfile).toHaveBeenCalledWith(expect.any(Function), 'ops', 'smart'))
  })

  it('offers Rename and Add folder for the active explicit project', async () => {
    render(<ComposerWorkspaceControls busy={false} cwd="/home/user/hermes-desktop" disabled={false} />)

    fireEvent.pointerDown(screen.getByTestId('composer-project-pill'), { button: 0, ctrlKey: false })
    fireEvent.click(await screen.findByRole('menuitem', { name: 'Rename…' }))
    expect(openProjectRename).toHaveBeenCalledWith({ id: 'proj-1', name: 'Hermes Desktop' })

    fireEvent.pointerDown(screen.getByTestId('composer-project-pill'), { button: 0, ctrlKey: false })
    fireEvent.click(await screen.findByRole('menuitem', { name: 'Add folder' }))
    expect(openProjectAddFolder).toHaveBeenCalledWith({ id: 'proj-1', name: 'Hermes Desktop' })
  })

  it('lazily refreshes the project tree on open when empty', () => {
    $projectTree.set([])
    render(<ComposerWorkspaceControls busy={false} cwd="/home/user/hermes-desktop" disabled={false} />)

    fireEvent.pointerDown(screen.getByTestId('composer-project-pill'), { button: 0, ctrlKey: false })
    expect(refreshProjectTree).toHaveBeenCalledTimes(1)
  })
})
