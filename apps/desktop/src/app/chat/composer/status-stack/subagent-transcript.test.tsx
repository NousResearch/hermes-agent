import { act, cleanup, render, screen } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import * as gateway from '@/store/gateway'
import { _resetSessionOwnerHintsForTests, forgetSessionOwnerHintsForSession, setSessionOwnerHint } from '@/store/session'

import { SubagentTranscript } from './subagent-transcript'

afterEach(() => {
  cleanup()
  _resetSessionOwnerHintsForTests()
  vi.useRealTimers()
  vi.restoreAllMocks()
})

it('rejects an old route reply and refreshes from the new owner without remounting', async () => {
  vi.useFakeTimers()
  let finishOld!: (value: unknown) => void
  const request = vi.spyOn(gateway, 'requestGatewayForAgent')
  request.mockImplementationOnce(
    async () =>
      (await new Promise<unknown>(resolve => {
        finishOld = resolve
      })) as never
  )
  request.mockResolvedValue({ available: true, text: 'Current owner reply', truncated: false } as never)
  setSessionOwnerHint('parent', { connectionId: 'old', profile: 'research' })
  render(<SubagentTranscript sessionId="parent" subagentId="worker" />)
  forgetSessionOwnerHintsForSession('parent')
  setSessionOwnerHint('parent', { connectionId: 'new', profile: 'research' })
  await act(async () => finishOld({ available: true, text: 'Old route reply', truncated: false }))
  expect(screen.queryByText('Old route reply')).toBeNull()
  await act(async () => vi.advanceTimersByTimeAsync(2000))
  expect(request.mock.calls.map(call => call[0])).toEqual(['old', 'new'])
  expect(screen.getByText('Current owner reply')).toBeTruthy()
  expect(request).toHaveBeenLastCalledWith('new', 'research', 'subagent.tail', {
    session_id: 'parent',
    subagent_id: 'worker'
  })
})
