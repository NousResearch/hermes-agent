import { act, cleanup, render } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, expect, it } from 'vitest'

import { PRIMARY_SESSION_VIEW, SessionViewProvider } from '@/app/chat/session-view'
import { $activeSessionId, $busy } from '@/store/session'
import { $subagentsBySession, upsertSubagent } from '@/store/subagents'

import { BackgroundResumeNotice } from './status'

afterEach(() => {
  cleanup()
  $activeSessionId.set(null)
  $busy.set(false)
  $subagentsBySession.set({})
})

it('keeps background waiting local to its idle thread without echoing child thinking chrome', () => {
  $activeSessionId.set('other')
  $busy.set(true)
  upsertSubagent('owner', { subagent_id: 'child', goal: 'Check relevance', status: 'running' })
  upsertSubagent('owner', { subagent_id: 'child', text: '(°□°) pondering...' }, false, 'subagent.thinking')
  const $ownerBusy = atom(false)
  const view = { ...PRIMARY_SESSION_VIEW, $runtimeId: atom<string | null>('owner'), $busy: $ownerBusy }

  const { container } = render(
    <SessionViewProvider value={view}>
      <BackgroundResumeNotice />
    </SessionViewProvider>
  )

  expect(container.querySelector('[role="status"]')).toBeTruthy()
  expect(container.textContent).not.toContain('pondering')
  expect(container.querySelector('.shimmer')).toBeNull()

  act(() => $ownerBusy.set(true))
  expect(container.querySelector('[role="status"]')).toBeNull()
  act(() => $ownerBusy.set(false))
  expect(container.querySelector('[role="status"]')).toBeTruthy()
  act(() => upsertSubagent('owner', { subagent_id: 'child', status: 'completed' }, true, 'subagent.complete'))
  expect(container.querySelector('[role="status"]')).toBeNull()
})
