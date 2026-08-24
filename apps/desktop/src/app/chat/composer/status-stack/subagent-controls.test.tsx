import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import * as gateway from '@/store/gateway'
import { _resetSessionOwnerHintsForTests, setSessionOwnerHint } from '@/store/session'
import { $subagentsBySession, upsertSubagent } from '@/store/subagents'

import { SubagentSection } from './subagent-section'

vi.stubGlobal(
  'ResizeObserver',
  class {
    disconnect() {}
    observe() {}
    unobserve() {}
  }
)
Element.prototype.animate = vi.fn(() => ({ cancel() {} }) as Animation)
afterEach(() => {
  cleanup()
  $subagentsBySession.set({})
  _resetSessionOwnerHintsForTests()
  vi.restoreAllMocks()
})

it('sends steer and stop to the child parent owner, never the active gateway or child transcript', async () => {
  const request = vi.spyOn(gateway, 'requestGatewayForAgent').mockResolvedValue({ status: 'queued', found: true })
  setSessionOwnerHint('parent', { connectionId: 'remote-owner', profile: 'research' })
  upsertSubagent('parent', { subagent_id: 'worker', child_session_id: 'child-transcript', goal: 'Owned work' })
  render(<SubagentSection sessionId="parent" />)
  fireEvent.click(screen.getByRole('button', { name: /Owned work/ }))
  fireEvent.change(screen.getByRole('textbox'), { target: { value: 'Check the negative control' } })
  fireEvent.click(screen.getByRole('button', { name: 'Steer' }))
  await waitFor(() =>
    expect(request).toHaveBeenCalledWith('remote-owner', 'research', 'subagent.steer', {
      session_id: 'parent',
      subagent_id: 'worker',
      text: 'Check the negative control'
    })
  )
  expect(screen.getByText('Queued for the next checkpoint')).toBeTruthy()
  fireEvent.click(screen.getByRole('button', { name: 'Stop' }))
  await waitFor(() =>
    expect(request).toHaveBeenCalledWith('remote-owner', 'research', 'subagent.interrupt', {
      session_id: 'parent',
      subagent_id: 'worker'
    })
  )
  expect($subagentsBySession.get().parent?.[0]?.status).toBe('running')
})

it('keeps rejected steer text and does not retarget when the owner is unknown', async () => {
  const request = vi.spyOn(gateway, 'requestGatewayForAgent').mockResolvedValue({ status: 'rejected' })
  upsertSubagent('unknown', { subagent_id: 'worker', goal: 'Unbound work' })
  render(<SubagentSection sessionId="unknown" />)
  fireEvent.click(screen.getByRole('button', { name: /Unbound work/ }))
  fireEvent.change(screen.getByRole('textbox'), { target: { value: 'Keep this instruction' } })
  fireEvent.click(screen.getByRole('button', { name: 'Steer' }))
  await waitFor(() => expect(screen.getByRole('alert')).toBeTruthy())
  expect(request).not.toHaveBeenCalled()
  expect((screen.getByRole('textbox') as HTMLInputElement).value).toBe('Keep this instruction')
})
