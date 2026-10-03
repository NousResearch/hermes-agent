import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import type { HermesReviewFile } from '@/global'
import { I18nProvider } from '@/i18n'
import { $panesFlipped } from '@/store/layout'
import {
  $reviewDiff,
  $reviewDiffLoading,
  $reviewFiles,
  $reviewIsRepo,
  $reviewLoading,
  $reviewScope,
  $reviewSelectedPath
} from '@/store/review'

import { ReviewPane } from './index'

const file = (path: string): HermesReviewFile => ({ added: 1, path, removed: 0, staged: false, status: 'M' })

function renderPane() {
  return render(
    <I18nProvider configClient={null} initialLocale="en">
      <ReviewPane />
    </I18nProvider>
  )
}

describe('ReviewPane header gating', () => {
  beforeEach(() => {
    $panesFlipped.set(false)
    $reviewFiles.set([file('a.ts')])
    $reviewIsRepo.set(true)
    $reviewLoading.set(false)
    $reviewDiff.set(null)
    $reviewDiffLoading.set(false)
    $reviewSelectedPath.set(null)
    $reviewScope.set('uncommitted')
  })

  afterEach(() => {
    cleanup()
    $reviewFiles.set([])
    $reviewScope.set('uncommitted')
  })

  it('enables stage-all / revert-all under the uncommitted scope', () => {
    renderPane()

    expect((screen.getByLabelText('Stage all') as HTMLButtonElement).disabled).toBe(false)
    expect((screen.getByLabelText('Revert all') as HTMLButtonElement).disabled).toBe(false)
  })

  it('disables stage-all / revert-all under the branch scope (read-only)', () => {
    $reviewScope.set('branch')

    renderPane()

    expect((screen.getByLabelText('Stage all') as HTMLButtonElement).disabled).toBe(true)
    expect((screen.getByLabelText('Revert all') as HTMLButtonElement).disabled).toBe(true)
  })

  it('renders the three scope options and switches scope on selection', () => {
    renderPane()

    // Icon-only segments: a fixed-size track cannot wrap or collide with the
    // header icons at narrow pane widths, so the name lives in the
    // accessible name (and the tooltip) instead of rendered text.
    expect(screen.getByLabelText('Uncommitted')).toBeTruthy()
    expect(screen.getByLabelText('Branch')).toBeTruthy()
    expect(screen.getByLabelText('Last turn')).toBeTruthy()
    expect(screen.queryByText('Uncommitted')).toBeNull()
    expect(screen.queryByText('Branch')).toBeNull()
    expect(screen.queryByText('Last turn')).toBeNull()

    fireEvent.click(screen.getByLabelText('Branch'))

    expect($reviewScope.get()).toBe('branch')
  })
})
