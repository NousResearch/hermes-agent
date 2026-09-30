import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { beforeEach, expect, it, vi } from 'vitest'

import { ackComposerInsert, onComposerInsertRequest } from '@/app/chat/composer/focus'
import { I18nProvider } from '@/i18n/context'

import { LensPanel } from './panel'
import { pinLensCapture, setLensScope } from './store'

beforeEach(() => {
  localStorage.clear()
  setLensScope('research')
})

it('hands only selected evidence to a reviewable chat draft and removes a card', async () => {
  const insert = vi.fn()
  const unsubscribe = onComposerInsertRequest(detail => {
    insert(detail.text)
    ackComposerInsert(detail.token, true)
  })

  for (const title of ['Selected source', 'Private unselected source']) {
    pinLensCapture(
      {
        title,
        url: 'https://example.com/' + encodeURIComponent(title),
        text: title + ' details',
        selector: 'article',
        tag: 'ARTICLE',
        truncated: false
      },
      'research'
    )
  }

  const onError = vi.fn()
  render(
    <I18nProvider initialLocale="en">
      <LensPanel busy={false} error="" onClose={() => {}} onError={onError} onPin={() => {}} onRefresh={() => {}} />
    </I18nProvider>
  )
  expect(screen.getByRole('button', { name: 'Ask Hermes' }).hasAttribute('disabled')).toBe(true)
  fireEvent.click(screen.getByRole('checkbox', { name: 'Selected source' }))
  fireEvent.change(screen.getByRole('textbox', { name: 'What should Hermes investigate?' }), {
    target: { value: 'Compare delivery costs' }
  })
  fireEvent.click(screen.getByRole('button', { name: 'Ask Hermes' }))
  await waitFor(() => expect(insert).toHaveBeenCalledOnce())
  expect(insert.mock.calls[0][0]).toContain('Compare delivery costs')
  expect(insert.mock.calls[0][0]).toContain('Selected source details')
  expect(insert.mock.calls[0][0]).not.toContain('Private unselected source')
  await screen.findByText('Added to your chat draft')
  fireEvent.click(screen.getAllByRole('button', { name: 'Remove' })[0])
  expect(screen.getAllByRole('checkbox')).toHaveLength(1)
  expect(onError).not.toHaveBeenCalled()
  unsubscribe()
})
