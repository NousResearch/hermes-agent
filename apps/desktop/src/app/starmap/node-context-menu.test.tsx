import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type * as HermesApi from '@/hermes'

import { NodeContextMenu, type NodeMenuTarget } from './node-context-menu'

vi.mock('@/app/learning/archive-skill-confirm-dialog', () => ({
  ArchiveSkillConfirmDialog: () => null,
  fireOptimistic: vi.fn()
}))
vi.mock('@/components/chat/code-editor', () => ({ CodeEditor: () => null }))

const hermesMocks = vi.hoisted(() => ({
  deleteLearningNode: vi.fn(),
  editLearningNode: vi.fn(),
  getLearningNode: vi.fn()
}))

vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<typeof HermesApi>()),
  ...hermesMocks
}))
vi.mock('@/store/notifications', () => ({ notifyError: vi.fn() }))
const starmapMocks = vi.hoisted(() => ({ evictStarmapNode: vi.fn(), loadStarmapGraph: vi.fn() }))
vi.mock('@/store/starmap', () => starmapMocks)

const { setApiRequestConnection, setApiRequestProfile } = await import('@/hermes')

const target: NodeMenuTarget = { id: 'memory-1', kind: 'memory', label: 'Test memory', x: 1000, y: 750 }

beforeEach(() => {
  vi.clearAllMocks()
  setApiRequestProfile('shared')
  setApiRequestConnection('connection-a')
})

afterEach(() => {
  cleanup()
  setApiRequestConnection(null)
  setApiRequestProfile(null)
})

describe('NodeContextMenu', () => {
  it('keeps the destructive row functional through the shared menu', async () => {
    const onClose = vi.fn()

    hermesMocks.deleteLearningNode.mockResolvedValue({ message: 'deleted', ok: true })

    render(<NodeContextMenu onClose={onClose} onNodeRemoved={vi.fn()} target={target} />)

    const row = await screen.findByRole('menuitem', { name: 'Delete memory' })

    // Radix selects on pointer-up (or Enter); the confirm dialog must replace the menu.
    fireEvent.keyDown(row, { key: 'Enter' })

    expect(await screen.findByText('Delete Test memory?')).toBeTruthy()
    expect(screen.queryByRole('menu')).toBeNull()

    fireEvent.click(screen.getByRole('button', { name: 'Delete' }))
    expect(hermesMocks.deleteLearningNode).toHaveBeenCalledWith(
      'memory-1',
      expect.objectContaining({ connectionId: 'connection-a', profile: 'shared' })
    )
  })

  it('refreshes a current edit but ignores its completion after the owner changes', async () => {
    let resolveEdit!: (value: { message: string; ok: boolean }) => void

    hermesMocks.getLearningNode.mockResolvedValue({
      content: 'original',
      kind: 'memory',
      label: target.label,
      ok: true
    })
    hermesMocks.editLearningNode.mockResolvedValueOnce({ message: 'saved', ok: true }).mockReturnValueOnce(
      new Promise(resolve => {
        resolveEdit = resolve
      })
    )

    render(<NodeContextMenu onClose={vi.fn()} onNodeRemoved={vi.fn()} target={target} />)

    fireEvent.keyDown(await screen.findByRole('menuitem', { name: 'Edit memory…' }), { key: 'Enter' })
    await screen.findByText('Edit Test memory')
    fireEvent.click(screen.getByRole('button', { name: 'Save' }))

    await waitFor(() => expect(starmapMocks.loadStarmapGraph).toHaveBeenCalledWith(true))
    fireEvent.keyDown(await screen.findByRole('menuitem', { name: 'Edit memory…' }), { key: 'Enter' })
    await screen.findByText('Edit Test memory')
    fireEvent.click(screen.getByRole('button', { name: 'Save' }))

    expect(hermesMocks.editLearningNode).toHaveBeenLastCalledWith(
      'memory-1',
      'original',
      expect.objectContaining({ connectionId: 'connection-a', profile: 'shared' })
    )

    act(() => setApiRequestConnection('connection-b'))
    await act(async () => resolveEdit({ message: 'saved', ok: true }))

    expect(screen.queryByText('Edit Test memory')).toBeNull()
    expect(starmapMocks.loadStarmapGraph).toHaveBeenCalledTimes(1)
  })
})
