import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { HermesReviewFile } from '@/global'
import { I18nProvider } from '@/i18n'
import {
  $reviewDiff,
  $reviewFiles,
  $reviewIsRepo,
  $reviewLoading,
  $reviewReadOnly,
  $reviewSelectedPath
} from '@/store/review'

vi.mock('@/components/chat/diff-lines', () => ({
  FileDiffPanel: ({ diff }: { diff: string }) => <div data-testid="diff">{diff}</div>
}))
vi.mock('./file-tree', () => ({ ReviewFileTree: () => <div data-testid="review-tree">tree</div> }))
vi.mock('./ship-bar', () => ({ ReviewShipBar: () => <div data-testid="ship-bar">ship</div> }))

import { ReviewPane } from './index'

const fallbackFile: HermesReviewFile = {
  added: 1,
  path: '/workspace/note.md',
  removed: 0,
  staged: false,
  status: 'M'
}

function renderPane() {
  return render(
    <I18nProvider configClient={null} initialLocale="en">
      <ReviewPane />
    </I18nProvider>
  )
}

describe('ReviewPane tool-diff fallback', () => {
  beforeEach(() => {
    $reviewDiff.set('+hello')
    $reviewFiles.set([fallbackFile])
    $reviewIsRepo.set(false)
    $reviewLoading.set(false)
    $reviewReadOnly.set(true)
    $reviewSelectedPath.set(fallbackFile.path)
  })

  afterEach(() => {
    cleanup()
    $reviewDiff.set('')
    $reviewFiles.set([])
    $reviewIsRepo.set(true)
    $reviewReadOnly.set(false)
    $reviewSelectedPath.set(null)
  })

  it('renders review content but no git mutation or ship actions', () => {
    renderPane()

    expect(screen.getByTestId('review-tree')).toBeTruthy()
    expect(screen.getByTestId('diff').textContent).toBe('+hello')
    expect(screen.queryByRole('button', { name: 'Stage all' })).toBeNull()
    expect(screen.queryByRole('button', { name: 'Revert all' })).toBeNull()
    expect(screen.queryByRole('button', { name: 'Stage' })).toBeNull()
    expect(screen.queryByTestId('ship-bar')).toBeNull()
  })
})
