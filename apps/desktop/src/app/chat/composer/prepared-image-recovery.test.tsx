import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, expect, test, vi } from 'vitest'

import { preparedSubmissionKey, writePreparedSubmission } from '@/app/session/hooks/use-prompt-actions/prepared-submissions'
import { captureSubmissionDestination } from '@/app/session/hooks/use-prompt-actions/submission-destination'

import { PreparedImageRecovery } from './prepared-image-recovery'

afterEach(() => { localStorage.clear() })

test('reopened composer offers exact image occurrence only for its original destination', async () => {
  const request = vi.fn()
  const destination = captureSubmissionDestination('original', request)
  const attachments = [{ id: 'image', occurrenceId: 'occurrence', kind: 'image' as const, label: 'image.png', path: '/cache/images/exact.png', mime: 'image/png' }]
  const key = preparedSubmissionKey('original', destination, '  retained image  ', attachments)
  await writePreparedSubmission(key, { id: 'submission', owner: destination.owner, text: 'expanded wire', attachments, params: { session_id: 'original' } })
  const restore = vi.fn()
  const view = render(<PreparedImageRecovery occupied={false} onRestore={restore} request={request} sessionKey="elsewhere" />)
  await waitFor(() => expect(request).not.toHaveBeenCalled())
  expect(screen.queryByRole('button')).toBeNull()
  view.rerender(<PreparedImageRecovery occupied={false} onRestore={restore} request={request} sessionKey="original" />)
  fireEvent.click(await screen.findByRole('button', { name: 'Restore image draft' }))
  expect(restore).toHaveBeenCalledWith('  retained image  ', attachments)
  expect(request).not.toHaveBeenCalled()
})

test('recovery never overwrites a newer draft or offers an ambiguous legacy send', async () => {
  const request = vi.fn()
  const destination = captureSubmissionDestination('original', request)
  const attachments = [{ id: 'image', kind: 'image' as const, label: 'image.png' }]

  for (const legacyAttempted of [false, true]) {
    await writePreparedSubmission(preparedSubmissionKey('original', destination, String(legacyAttempted), attachments), {
      id: String(legacyAttempted), owner: destination.owner, text: String(legacyAttempted), attachments, params: {}, legacyAttempted
    })
  }

  const restore = vi.fn()
  render(<PreparedImageRecovery occupied onRestore={restore} request={request} sessionKey="original" />)
  const button = await screen.findByRole('button', { name: 'Restore image draft' })
  expect(screen.getAllByRole('button')).toHaveLength(1)
  expect((button as HTMLButtonElement).disabled).toBe(true)
  fireEvent.click(button)
  expect(restore).not.toHaveBeenCalled()
})
