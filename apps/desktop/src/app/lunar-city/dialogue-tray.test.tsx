import { render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import { DialogueTray } from './dialogue-tray'
import type { WorldActionRunner } from './world-actions'

describe('DialogueTray', () => {
  it('shows source truth and only allowed actions for a blocked task', () => {
    render(
      <DialogueTray
        onAction={vi.fn<WorldActionRunner['run']>(async () => ({ kind: 'completed', ok: true }))}
        subject={{
          condition: {
            active: true,
            facts: {},
            id: 'condition-7',
            kind: 'task.blocked',
            scope: 'task',
            severity: 'warning',
            source: 'kanban',
            sourceRef: { taskId: 'task-7' },
            title: 'Fix auth'
          },
          detail: 'dependency failed',
          title: 'Fix auth'
        }}
      />
    )

    expect(screen.getByText('task.blocked')).toBeTruthy()
    expect(screen.getByText('dependency failed')).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Inspect blocker' })).toBeTruthy()
    expect(screen.queryByRole('button', { name: 'Approve review' })).toBeNull()
  })
})
