import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import * as client from '@/api/client'
import { en } from '@/i18n/en'

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
  fireEvent.change(screen.getByLabelText(en.cron.nameLabel), { target: { value: 'Daily review' } })
  fireEvent.change(screen.getByPlaceholderText('0 9 * * *'), { target: { value: '0 9 * * *' } })
  fireEvent.change(screen.getByLabelText(en.cron.promptLabel), { target: { value: 'Review my notes' } })
  fireEvent.click(screen.getByRole('button', { name: en.common.save }))
  expect((await screen.findByRole('alert')).textContent).toContain('Invalid schedule')
  expect(close).not.toHaveBeenCalled()
  fireEvent.click(screen.getByRole('button', { name: en.common.save }))
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
  fireEvent.change(screen.getByLabelText(en.cron.nameLabel), { target: { value: 'Renamed' } })
  fireEvent.click(screen.getByRole('button', { name: en.common.save }))
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
  fireEvent.change(screen.getByPlaceholderText('0 9 * * *'), { target: { value: '0 9 * * *' } })
  fireEvent.change(screen.getByLabelText(en.cron.promptLabel), { target: { value: 'Review my notes' } })
  fireEvent.click(screen.getByRole('button', { name: en.common.save }))
  expect(screen.getByRole('button', { name: en.common.cancel }).hasAttribute('disabled')).toBe(true)
  expect(screen.getByLabelText(en.cron.promptLabel).closest('fieldset')?.disabled).toBe(true)
  expect(screen.queryByRole('button', { name: en.common.close })).toBeNull()
  fireEvent.click(screen.getByRole('button', { name: en.common.cancel }))
  fireEvent.keyDown(window.document, { key: 'Escape' })
  expect(close).not.toHaveBeenCalled()
  await act(async () => {
    reject(new Error('Save failed'))
  })
  expect(screen.getByRole('alert').textContent).toContain('Save failed')
  expect(screen.getByRole('button', { name: en.common.cancel }).hasAttribute('disabled')).toBe(false)
  expect(screen.getByLabelText(en.cron.promptLabel).closest('fieldset')?.disabled).toBe(false)
  expect(saved).not.toHaveBeenCalled()
  fireEvent.click(screen.getByRole('button', { name: en.common.cancel }))
  expect(close).toHaveBeenCalledOnce()
})

it.each([{ skills: ['daily-review'] }, { skill: 'daily-review' }, { skills: [], skill: 'daily-review' }])(
  'saves metadata for a skill-only routine without inventing a prompt (%j)',
  async skills => {
    const request = vi.spyOn(client, 'hermesApi').mockResolvedValue({})
    const saved = vi.fn()
    render(
      <RoutineDialog
        job={{ id: 'skill-job', enabled: true, prompt: '', schedule: { expr: '0 9 * * *' }, ...skills }}
        onClose={() => {}}
        onSaved={saved}
        owner={{ connectionId: 'beta', profile: 'research' }}
      />
    )
    fireEvent.change(screen.getByLabelText(en.cron.nameLabel), { target: { value: 'Renamed review' } })
    const prompt = screen.getByLabelText(en.cron.promptLabel) as HTMLTextAreaElement

    expect(prompt.value).toBe('')
    expect(prompt.validity.valueMissing).toBe(false)
    expect(prompt.form?.checkValidity()).toBe(true)
    fireEvent.click(screen.getByRole('button', { name: en.common.save }))
    await waitFor(() => expect(saved).toHaveBeenCalledOnce())
    expect(request).toHaveBeenCalledWith(
      expect.objectContaining({
        connectionId: 'beta',
        profile: 'research',
        method: 'PUT',
        path: '/api/cron/jobs/skill-job?profile=research',
        body: { updates: { name: 'Renamed review', enabled: true, prompt: '' } }
      })
    )
  }
)

it.each([undefined, { skills: [] }, { skills: [' '], skill: ' ' }])(
  'requires a prompt when neither an existing skill nor a script supplies the task (%j)',
  async skills => {
    const request = vi.spyOn(client, 'hermesApi').mockResolvedValue({})
    render(
      <RoutineDialog
        job={skills ? { id: 'empty-job', enabled: true, schedule: { expr: '0 9 * * *' }, ...skills } : undefined}
        onClose={() => {}}
        onSaved={() => {}}
        owner={{ connectionId: 'beta', profile: 'research' }}
      />
    )
    fireEvent.change(screen.getByPlaceholderText('0 9 * * *'), { target: { value: '0 9 * * *' } })
    const prompt = screen.getByLabelText(en.cron.promptLabel) as HTMLTextAreaElement

    expect(prompt.validity.valueMissing).toBe(true)
    expect(prompt.form?.checkValidity()).toBe(false)
    fireEvent.click(screen.getByRole('button', { name: en.common.save }))
    expect(request).not.toHaveBeenCalled()
  }
)
