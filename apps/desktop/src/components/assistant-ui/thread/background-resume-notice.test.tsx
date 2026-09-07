import { act, cleanup, render, screen } from '@testing-library/react'
import { afterEach, expect, it } from 'vitest'

import { en } from '@/i18n/en'
import { $activeSessionId, $busy } from '@/store/session'
import { $subagentsBySession, upsertSubagent } from '@/store/subagents'

import { BackgroundResumeNotice } from './status'

afterEach(() => {
  cleanup()
  $activeSessionId.set(null)
  $busy.set(false)
  $subagentsBySession.set({})
})

it('identifies parked background work instead of presenting child thinking as parent activity', () => {
  $activeSessionId.set('parent')
  $busy.set(false)
  upsertSubagent('parent', { subagent_id: 'first', status: 'running', text: 'Ruminating…' }, true, 'subagent.thinking')
  upsertSubagent('parent', { subagent_id: 'second', status: 'queued' }, true, 'subagent.start')

  render(<BackgroundResumeNotice />)
  expect(screen.getByRole('status').textContent).toBe(en.assistant.thread.resumeWhenBackgroundDone(2))
  expect(screen.getByTitle('Ruminating…')).toBeTruthy()

  act(() => $activeSessionId.set('elsewhere'))
  expect(screen.queryByRole('status')).toBeNull()
  act(() => $activeSessionId.set('parent'))
  expect(screen.getByRole('status').textContent).toBe(en.assistant.thread.resumeWhenBackgroundDone(2))

  act(() => upsertSubagent('parent', { subagent_id: 'first', status: 'completed' }, true, 'subagent.complete'))
  expect(screen.getByRole('status').textContent).toBe(en.assistant.thread.resumeWhenBackgroundDone(1))
  act(() => $busy.set(true))
  expect(screen.queryByRole('status')).toBeNull()
  act(() => {
    upsertSubagent('parent', { subagent_id: 'second', status: 'failed' }, true, 'subagent.complete')
    $busy.set(false)
  })
  expect(screen.queryByRole('status')).toBeNull()
})
