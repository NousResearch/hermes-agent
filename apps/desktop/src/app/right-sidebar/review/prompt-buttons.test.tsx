import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { HermesReviewFile } from '@/global'
import { I18nProvider } from '@/i18n'
import { rememberDesktopCommandsCatalog } from '@/lib/desktop-slash-commands'
import { $reviewFiles } from '@/store/review'

import { ReviewPromptButtons } from './prompt-buttons'

const submit = vi.hoisted(() => vi.fn(() => true))

vi.mock('@/app/chat/composer/focus', () => ({ requestComposerSubmit: submit }))

const file: HermesReviewFile = { added: 1, path: 'a.ts', removed: 0, staged: false, status: 'M' }

describe('ReviewPromptButtons', () => {
  beforeEach(() => {
    $reviewFiles.set([file])
    rememberDesktopCommandsCatalog({
      commands: {
        '/model': { argument_mode: 'options' },
        '/notes': { argument_mode: 'text' },
        '/rn': { argument_mode: 'text', review_button: 'Release notes' }
      }
    })
  })

  afterEach(() => {
    cleanup()
    submit.mockClear()
    rememberDesktopCommandsCatalog(undefined)
    $reviewFiles.set([])
  })

  it('renders only review_button prompts and sends the slash command to the owning chat', () => {
    render(
      <I18nProvider configClient={null} initialLocale="en">
        <ReviewPromptButtons />
      </I18nProvider>
    )

    expect(screen.queryAllByRole('button')).toHaveLength(1)
    fireEvent.click(screen.getByText('Release notes'))
    expect(submit).toHaveBeenCalledWith('/rn', { target: 'main' })
  })
})
