import { describe, expect, it, vi } from 'vitest'

import { archiveSessionWithOwner } from './archive-owner'

describe('archiveSessionWithOwner', () => {
  it('archives a registry session through its captured connection owner', async () => {
    const setArchived = vi.fn().mockResolvedValue({ ok: true })

    await archiveSessionWithOwner(
      'shared-session',
      { _desktop_route_binding: 'binding-a', connection_id: 'source-a', id: 'shared-session', profile: 'worker' },
      'ambient-profile',
      setArchived
    )

    expect(setArchived).toHaveBeenCalledWith('shared-session', true, {
      connectionId: 'source-a',
      profile: 'worker',
      sessionRouteBinding: 'binding-a',
      sessionRouteId: 'shared-session'
    })
  })

  it('archives a legacy remote row through its opaque route binding', async () => {
    const setArchived = vi.fn().mockResolvedValue({ ok: true })

    await archiveSessionWithOwner(
      'shared-session',
      { _desktop_route_binding: 'binding-a', id: 'shared-session', profile: 'worker' },
      'ambient-profile',
      setArchived
    )

    expect(setArchived).toHaveBeenCalledWith('shared-session', true, {
      connectionId: null,
      profile: 'worker',
      sessionRouteBinding: 'binding-a',
      sessionRouteId: 'shared-session'
    })
  })
})
