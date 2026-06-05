import type * as DndSortable from '@dnd-kit/sortable'
import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import { ProfileSquare } from './profile-switcher'

vi.mock('@dnd-kit/sortable', async () => {
  const actual = await vi.importActual<typeof DndSortable>('@dnd-kit/sortable')

  return {
    ...actual,
    useSortable: () => ({
      attributes: {},
      isDragging: false,
      listeners: {},
      setNodeRef: vi.fn(),
      transform: null,
      transition: undefined
    })
  }
})

describe('ProfileSquare', () => {
  it('keeps the color picker visible after selecting Color from the context menu', async () => {
    render(
      <ProfileSquare
        active={false}
        color={null}
        label="work"
        onDelete={vi.fn()}
        onRecolor={vi.fn()}
        onRename={vi.fn()}
        onSelect={vi.fn()}
      />
    )

    fireEvent.contextMenu(screen.getByRole('button', { name: 'work' }))
    fireEvent.click(await screen.findByText('Color…'))

    await waitFor(() => {
      expect(screen.getByRole('dialog', { name: 'Color for work' })).toBeTruthy()
    })
    expect(screen.getAllByRole('button', { name: /^Set color / })).toHaveLength(12)
  })
})
