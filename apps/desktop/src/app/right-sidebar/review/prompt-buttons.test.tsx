import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { HermesReviewFile } from '@/global'
import { I18nProvider } from '@/i18n'
import { $composerCommandsCatalogs, rememberComposerCommandsCatalog } from '@/lib/review-prompt-buttons'
import { $reviewFiles, $reviewScopeTarget } from '@/store/review'

import { ReviewPromptButtons } from './prompt-buttons'

const submit = vi.hoisted(() => vi.fn(() => true))

vi.mock('@/app/chat/composer/focus', () => ({ requestComposerSubmit: submit }))

const file: HermesReviewFile = { added: 1, path: 'a.ts', removed: 0, staged: false, status: 'M' }

const mainCatalog = {
  commands: {
    '/model': { argument_mode: 'options' as const },
    // A saved prompt on a name the desktop runs itself (`/new` opens a chat): no button.
    '/new': { argument_mode: 'text' as const, review_button: 'Fresh start' },
    '/notes': { argument_mode: 'text' as const },
    '/rn': { argument_mode: 'text' as const, review_button: 'Release notes' }
  }
}

const tileCatalog = { commands: { '/audit': { argument_mode: 'text' as const, review_button: 'Audit' } } }

function renderButtons() {
  render(
    <I18nProvider configClient={null} initialLocale="en">
      <ReviewPromptButtons />
    </I18nProvider>
  )
}

describe('ReviewPromptButtons', () => {
  beforeEach(() => {
    $reviewFiles.set([file])
  })

  afterEach(() => {
    cleanup()
    submit.mockClear()
    $composerCommandsCatalogs.set({})
    $reviewScopeTarget.set('main')
    $reviewFiles.set([])
  })

  it('renders only review_button prompts and sends the slash command to the owning chat', () => {
    rememberComposerCommandsCatalog('main', mainCatalog)
    renderButtons()

    expect(screen.queryAllByRole('button')).toHaveLength(1)
    fireEvent.click(screen.getByText('Release notes'))
    expect(submit).toHaveBeenCalledWith('/rn', { target: 'main' })
  })

  it.each([
    ['owner first', ['main', 'tile:b']],
    ['other profile first', ['tile:b', 'main']]
  ])("reads the owning chat's catalog, never another profile's (%s)", (_label, order) => {
    renderButtons()

    act(() => {
      for (const target of order) {
        rememberComposerCommandsCatalog(target, target === 'main' ? mainCatalog : tileCatalog)
      }
    })
    expect(screen.getByText('Release notes')).toBeTruthy()
    expect(screen.queryByText('Audit')).toBeNull()

    act(() => $reviewScopeTarget.set('tile:b'))
    expect(screen.getByText('Audit')).toBeTruthy()
    expect(screen.queryByText('Release notes')).toBeNull()
    fireEvent.click(screen.getByText('Audit'))
    expect(submit).toHaveBeenCalledWith('/audit', { target: 'tile:b' })
  })
})
