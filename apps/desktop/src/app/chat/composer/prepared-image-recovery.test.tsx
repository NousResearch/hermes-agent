import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, expect, test, vi } from 'vitest'

import { preparedSubmissionKey, writePreparedSubmission } from '@/app/session/hooks/use-prompt-actions/prepared-submissions'
import { captureSubmissionDestination } from '@/app/session/hooks/use-prompt-actions/submission-destination'
import { $notifications, clearNotifications } from '@/store/notifications'

import { PreparedImageRecovery } from './prepared-image-recovery'

afterEach(() => { localStorage.clear(); vi.unstubAllGlobals() })

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
  const button = await screen.findByRole('button', { name: 'Restore draft' })
  await act(async () => {fireEvent.click(button)})
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
  const button = await screen.findByRole('button', { name: 'Restore draft' })
  expect(screen.getAllByRole('button')).toHaveLength(1)
  expect((button as HTMLButtonElement).disabled).toBe(true)
  fireEvent.click(button)
  expect(restore).not.toHaveBeenCalled()
})

test.each([
  { action: 'switch', fails: false }, { action: 'unmount', fails: false },
  { action: 'switch', fails: true }, { action: 'unmount', fails: true }
])('late atomic recovery cannot restore or report errors into a disposed composer: %j', async ({ action, fails }) => {
  clearNotifications()
  const request = vi.fn()
  const destination = captureSubmissionDestination('original', request)
  const key = preparedSubmissionKey('original', destination, 'retained text', [])
  await writePreparedSubmission(key, { id: 'late-recovery', owner: destination.owner, text: 'retained text', attachments: [], params: { session_id: 'original' } })
  const serialized = localStorage.getItem('hermes.desktop.preparedSubmissions.v1')!
  let release!: (value: boolean) => void
  let reject!: (error: Error) => void
  const compareAndSet = vi.fn(() => new Promise<boolean>((resolve, fail) => {release = resolve; reject = fail}))
  vi.stubGlobal('hermesDesktop', { preparedSubmissions: {
    owner: async () => 'new-window', read: async () => serialized, update: vi.fn(), compareAndSet
  } })
  const restore = vi.fn()
  const view = render(<PreparedImageRecovery occupied={false} onRestore={restore} request={request} sessionKey="original" />)
  fireEvent.click(await screen.findByRole('button', { name: 'Restore draft' }))
  await waitFor(() => expect(compareAndSet).toHaveBeenCalledOnce())

  if (action === 'unmount') {view.unmount()}
  else {view.rerender(<PreparedImageRecovery occupied={false} onRestore={restore} request={request} sessionKey="elsewhere" />)}

  await act(async () => {
    if (fails) {reject(new Error('fixture stale claim failure'))}
    else {release(true)}
  })
  expect(restore).not.toHaveBeenCalled()
  expect(request).not.toHaveBeenCalled()
  expect($notifications.get()).toEqual([])
})
