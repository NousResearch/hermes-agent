import { afterEach, expect, it, vi } from 'vitest'

import * as client from '@/api/client'

import { getAgentMessaging, getAgentRecentSessions, getAgentRoutines, saveAgentRoutine } from './data'

afterEach(() => vi.restoreAllMocks())

it('pins every read and routine write to the named gateway and profile', async () => {
  const request = vi.spyOn(client, 'hermesApi').mockResolvedValue({})

  for (const connectionId of ['alpha', 'beta']) {
    const owner = { connectionId, profile: 'research' }
    await getAgentRoutines(owner)
    await getAgentMessaging(owner)
    await getAgentRecentSessions(owner)
    await saveAgentRoutine(owner, { prompt: 'Review notes', schedule: '0 9 * * *', deliver: 'local' })
    await saveAgentRoutine(owner, { name: 'Updated' }, 'same-job')
  }

  expect(request).toHaveBeenCalledTimes(10)

  for (const [index, [call]] of request.mock.calls.entries()) {
    expect(call.connectionId).toBe(index < 5 ? 'alpha' : 'beta')
    expect(call.profile).toBe('research')
    expect(call.path).toContain('profile=research')
  }

  expect(request.mock.calls[4][0]).toEqual(
    expect.objectContaining({ method: 'PUT', body: { updates: { name: 'Updated' } } })
  )
})
