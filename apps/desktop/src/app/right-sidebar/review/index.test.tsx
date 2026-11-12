import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeAll, beforeEach, describe, expect, it } from 'vitest'

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

// Radix menus use pointer capture and scrollIntoView; jsdom has neither.
beforeAll(() => {
  Element.prototype.hasPointerCapture ??= () => false
  Element.prototype.releasePointerCapture ??= () => undefined
  Element.prototype.scrollIntoView ??= () => undefined
})

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

  it('renders the three scope tabs and switches scope on selection', () => {
    renderPane()

    expect(screen.getByRole('button', { name: 'Uncommitted', pressed: true })).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Last turn', pressed: false })).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: 'Branch', pressed: false }))

    expect($reviewScope.get()).toBe('branch')
  })

  it('narrow-pane dropdown names the active scope and switches it', () => {
    $reviewScope.set('lastTurn')
    renderPane()

    const trigger = screen.getByRole('button', { name: 'Last turn', expanded: false })
    fireEvent.keyDown(trigger, { key: 'Enter' })
    fireEvent.click(screen.getByRole('menuitem', { name: 'Uncommitted' }))

    expect($reviewScope.get()).toBe('uncommitted')
  })
})
