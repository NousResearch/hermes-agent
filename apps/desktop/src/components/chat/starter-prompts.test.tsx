import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { onComposerDraftRequests, onComposerSubmitRequest } from '@/app/chat/composer/focus'
import { I18nProvider } from '@/i18n'
import { freshDraftScope } from '@/store/composer'
import { setInterfaceMode } from '@/store/interface-mode'

import { Intro } from './intro'

const STARTERS = '[data-slot="aui_starter-prompts"] button'

function renderIntro() {
  return render(
    <I18nProvider configClient={null} initialLocale="en">
      <Intro />
    </I18nProvider>
  )
}

afterEach(() => {
  cleanup()
  setInterfaceMode('advanced')
})

it('offers starter cards on the empty chat only in Simple mode', () => {
  const { container, unmount } = renderIntro()
  expect(container.querySelectorAll(STARTERS)).toHaveLength(0)
  unmount()

  setInterfaceMode('simple')
  const simple = renderIntro()
  expect(simple.container.querySelectorAll(STARTERS).length).toBeGreaterThan(0)
})

it('fills the new-chat draft from a card without sending it', () => {
  setInterfaceMode('simple')
  // Stand in for the mounted new-chat composer: it answers for the fresh-draft key.
  const write = vi.fn((_text: string) => true)
  const submit = vi.fn()

  const offDraft = onComposerDraftRequests(
    { getIds: () => [freshDraftScope()], isActive: () => false },
    { read: () => '', write }
  )

  const offSubmit = onComposerSubmitRequest(submit)

  try {
    renderIntro()
    fireEvent.click(screen.getByRole('button', { name: /Plan this week's dinners/ }))

    expect(write).toHaveBeenCalledTimes(1)
    expect(write).toHaveBeenCalledWith(expect.stringContaining('dinners'))
    expect(submit).not.toHaveBeenCalled()
  } finally {
    offDraft()
    offSubmit()
  }
})
