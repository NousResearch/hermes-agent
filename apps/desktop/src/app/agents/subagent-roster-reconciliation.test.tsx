import { cleanup, render, screen, waitFor } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import * as gateway from '@/store/gateway'
import { _resetSessionOwnerHintsForTests, setSessionOwnerHint } from '@/store/session'
import { $subagentsBySession, upsertSubagent } from '@/store/subagents'

import { AgentsView } from './index'

Element.prototype.animate = vi.fn(() => ({ cancel() {} }) as Animation)

afterEach(() => {
  cleanup()
  $subagentsBySession.set({})
  _resetSessionOwnerHintsForTests()
  vi.restoreAllMocks()
})

it('drops stale running rows without removing work active on another owner backend', async () => {
  const request = vi
    .spyOn(gateway, 'requestGatewayForAgent')
    .mockImplementation(async (connectionId, _profile, method) => {
      expect(method).toBe('delegation.status')

      return {
        active: connectionId === 'remote-live' ? [{ subagent_id: 'live-worker', status: 'running' }] : []
      } as never
    })

  setSessionOwnerHint('stale-parent', { connectionId: 'remote-stale', profile: 'research' })
  setSessionOwnerHint('live-parent', { connectionId: 'remote-live', profile: 'research' })
  upsertSubagent('stale-parent', { subagent_id: 'stale-worker', goal: 'Stale work', status: 'running' })
  upsertSubagent('live-parent', { subagent_id: 'live-worker', goal: 'Live work', status: 'running' })

  render(<AgentsView onClose={vi.fn()} />)

  expect(screen.getByText('Stale work')).toBeTruthy()
  expect(screen.getByText('Live work')).toBeTruthy()

  await waitFor(() => expect(screen.queryByText('Stale work')).toBeNull())
  expect(screen.getByText('Live work')).toBeTruthy()
  expect(request).toHaveBeenCalledWith('remote-stale', 'research', 'delegation.status', {})
  expect(request).toHaveBeenCalledWith('remote-live', 'research', 'delegation.status', {})
})
