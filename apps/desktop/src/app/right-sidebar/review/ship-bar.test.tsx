import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import type { HermesReviewFile } from '@/global'
import { I18nProvider } from '@/i18n'
import { $reviewFiles, $reviewScope, $reviewShipInfo } from '@/store/review'

import { ReviewShipBar } from './ship-bar'

const file = (path: string): HermesReviewFile => ({
  added: 1,
  path,
  removed: 0,
  staged: false,
  status: 'M'
})

function renderBar() {
  return render(
    <I18nProvider configClient={null} initialLocale="en">
      <ReviewShipBar />
    </I18nProvider>
  )
}

describe('ReviewShipBar', () => {
  beforeEach(() => {
    $reviewFiles.set([file('a.ts')])
    $reviewShipInfo.set({ ghReady: false, pr: null })
    $reviewScope.set('uncommitted')
  })

  afterEach(() => {
    cleanup()
    $reviewFiles.set([])
    $reviewScope.set('uncommitted')
  })

  it('renders the commit bar for the uncommitted scope', () => {
    renderBar()

    expect(screen.getByText('Commit')).toBeTruthy()
  })

  it('renders nothing for the branch scope (committed work has no working-tree actions)', () => {
    $reviewScope.set('branch')

    const { container } = renderBar()

    expect(container.firstChild).toBeNull()
  })
})
