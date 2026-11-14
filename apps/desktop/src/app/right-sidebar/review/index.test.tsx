import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

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
import { $currentCwd } from '@/store/session'

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

const TWO_HUNKS = [
  'diff --git a/a.ts b/a.ts',
  '--- a/a.ts',
  '+++ b/a.ts',
  '@@ -1,3 +1,3 @@',
  ' one',
  '-two',
  '+TWO',
  ' three',
  '@@ -20,3 +20,3 @@',
  ' twenty',
  '-twentyone',
  '+TWENTYONE',
  ' twentytwo',
  ''
].join('\n')

describe('ReviewPane per-change actions', () => {
  afterEach(() => {
    cleanup()
    delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop
    $reviewFiles.set([])
  })

  it('reverts only the chosen change after the confirm', async () => {
    const hunk = vi.fn(async () => ({ ok: true }))

    const revert = vi.fn(async () => ({ ok: true }))

    ;(window as unknown as { hermesDesktop?: unknown }).hermesDesktop = {
      git: { review: { diff: async () => TWO_HUNKS, hunk, list: async () => ({ files: [file('a.ts')] }), revert } }
    }
    $currentCwd.set('/repo')
    $reviewScope.set('uncommitted')
    $reviewFiles.set([file('a.ts')])
    $reviewSelectedPath.set('a.ts')
    $reviewDiff.set(TWO_HUNKS)
    $reviewDiffLoading.set(false)

    renderPane()

    expect(screen.getAllByLabelText('Stage this change')).toHaveLength(2)
    fireEvent.click(screen.getAllByLabelText('Revert this change')[1])
    fireEvent.click(await screen.findByRole('button', { name: 'Revert this change' }))

    await waitFor(() => expect(hunk).toHaveBeenCalledTimes(1))
    expect(hunk).toHaveBeenCalledWith('/repo', 'a.ts', TWO_HUNKS.split('\n').slice(8, 13).join('\n'), 'revert')
    expect(revert).not.toHaveBeenCalled()
  })
})
