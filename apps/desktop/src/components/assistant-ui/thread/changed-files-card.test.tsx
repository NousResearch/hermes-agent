import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

const review = vi.hoisted(() => ({
  openReviewForPath: vi.fn(),
  revealReview: vi.fn()
}))

vi.mock('@nanostores/react', () => ({ useStore: (store: { get: () => unknown }) => store.get() }))
vi.mock('@/app/chat/composer/scope', () => ({ useComposerScope: () => ({ target: 'main' }) }))
vi.mock('@/app/chat/session-view', () => ({
  useSessionView: () => ({ $cwd: { get: () => '/session' }, kind: 'primary' })
}))
vi.mock('@/store/review', () => review)

import { ChangedFilesCard } from './changed-files-card'

const diff = '--- a/note.md\n+++ b/note.md\n@@ -1 +1 @@\n-old\n+new'

const parts = [
  {
    args: { path: '/workspace/note.md' },
    result: { diff },
    toolName: 'patch',
    type: 'tool-call'
  }
]

describe('ChangedFilesCard review fallback', () => {
  it('passes the complete tool-diff snapshot from both review entry points', () => {
    render(<ChangedFilesCard parts={parts} />)

    const fallback = [
      { added: 1, diff, name: 'note.md', path: '/workspace/note.md', removed: 1 }
    ]

    const buttons = screen.getAllByRole('button')

    fireEvent.click(buttons[0]!)
    expect(review.revealReview).toHaveBeenCalledWith(null, 'main', fallback)

    fireEvent.click(screen.getByText('note.md').closest('button')!)
    expect(review.openReviewForPath).toHaveBeenCalledWith('/workspace/note.md', null, 'main', fallback)
  })
})
