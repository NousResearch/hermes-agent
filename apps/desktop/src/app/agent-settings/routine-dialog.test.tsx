import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import * as client from '@/api/client'

import { RoutineDialog } from './routine-dialog'

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

it('saves a new routine on the captured owner and leaves failures editable', async () => {
  const request = vi
    .spyOn(client, 'hermesApi')
    .mockRejectedValueOnce(new Error('Invalid schedule'))
    .mockResolvedValueOnce({ id: 'saved' })

  const close = vi.fn()
  const saved = vi.fn()
  render(<RoutineDialog onClose={close} onSaved={saved} owner={{ connectionId: 'remote', profile: 'research' }} />)
  fireEvent.change(screen.getByLabelText('Name'), { target: { value: 'Daily review' } })
  fireEvent.change(screen.getByLabelText('Schedule'), { target: { value: '0 9 * * *' } })
  fireEvent.change(screen.getByLabelText('Prompt'), { target: { value: 'Review my notes' } })
  fireEvent.click(screen.getByRole('button', { name: 'Save' }))
  expect((await screen.findByRole('alert')).textContent).toContain('Invalid schedule')
  expect(close).not.toHaveBeenCalled()
  fireEvent.click(screen.getByRole('button', { name: 'Save' }))
  await waitFor(() => expect(saved).toHaveBeenCalledOnce())
  expect(request).toHaveBeenLastCalledWith(
    expect.objectContaining({
      connectionId: 'remote',
      profile: 'research',
      path: '/api/cron/jobs?profile=research',
      method: 'POST',
      body: { name: 'Daily review', schedule: '0 9 * * *', prompt: 'Review my notes', deliver: 'local' }
    })
  )
  expect(close).toHaveBeenCalledOnce()
})

it('edits script routine metadata without overwriting its script or resending a display-only schedule', async () => {
  const request = vi.spyOn(client, 'hermesApi').mockResolvedValue({})
  const saved = vi.fn()
  render(
    <RoutineDialog
      job={{ id: 'script-job', enabled: true, script: '/scripts/job.py', schedule_display: 'Every hour' }}
      onClose={() => {}}
      onSaved={saved}
      owner={{ connectionId: 'alpha', profile: 'research' }}
    />
  )
  fireEvent.change(screen.getByLabelText('Name'), { target: { value: 'Renamed' } })
  fireEvent.click(screen.getByRole('button', { name: 'Save' }))
  await waitFor(() => expect(saved).toHaveBeenCalledOnce())
  expect(request).toHaveBeenCalledWith(
    expect.objectContaining({ body: { updates: { name: 'Renamed', enabled: true } } })
  )
})

it('keeps an in-flight save visible and restores editing after failure', async () => {
  let reject!: (reason: Error) => void
  vi.spyOn(client, 'hermesApi').mockImplementation(
    () =>
      new Promise((_resolve, fail) => {
        reject = fail
      })
  )
  const close = vi.fn()
  const saved = vi.fn()
  render(<RoutineDialog onClose={close} onSaved={saved} owner={{ connectionId: 'remote', profile: 'research' }} />)
  fireEvent.change(screen.getByLabelText('Schedule'), { target: { value: '0 9 * * *' } })
  fireEvent.change(screen.getByLabelText('Prompt'), { target: { value: 'Review my notes' } })
  fireEvent.click(screen.getByRole('button', { name: 'Save' }))
  expect(screen.getByRole('button', { name: 'Cancel' }).hasAttribute('disabled')).toBe(true)
  expect(screen.getByLabelText('Prompt').closest('fieldset')?.disabled).toBe(true)
  expect(screen.queryByRole('button', { name: 'Close' })).toBeNull()
  fireEvent.click(screen.getByRole('button', { name: 'Cancel' }))
  fireEvent.keyDown(window.document, { key: 'Escape' })
  expect(close).not.toHaveBeenCalled()
  await act(async () => {
    reject(new Error('Save failed'))
  })
  expect(screen.getByRole('alert').textContent).toContain('Save failed')
  expect(screen.getByRole('button', { name: 'Cancel' }).hasAttribute('disabled')).toBe(false)
  expect(screen.getByLabelText('Prompt').closest('fieldset')?.disabled).toBe(false)
  expect(saved).not.toHaveBeenCalled()
  fireEvent.click(screen.getByRole('button', { name: 'Cancel' }))
  expect(close).toHaveBeenCalledOnce()
})
