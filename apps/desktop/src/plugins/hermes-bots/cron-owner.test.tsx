/**
 * A cron mutation names the owner that rendered it, never the ambient profile.
 *
 * The Routines pane follows the focused chat, so the live gateway profile can
 * already be on another bot by the time a row's switch is flipped or a create
 * dialog is submitted. Both the RPC scope and the cache eviction therefore
 * ride the captured owner — and the eviction is `exact`, so flipping one bot's
 * job cannot invalidate every other bot's list.
 */

import type * as HermesSdk from '@hermes/plugin-sdk'
import { queryClient } from '@hermes/plugin-sdk'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const { request } = vi.hoisted(() => ({ request: vi.fn(async () => ({})) }))

vi.mock('@hermes/plugin-sdk', async importOriginal => {
  const sdk = await importOriginal<typeof HermesSdk>()

  return { ...sdk, host: { ...sdk.host, request } }
})

const { invalidateRoutineOwner, RoutineRow, routineCreateTarget } = await import('./cron')

const invalidateQueries = vi.spyOn(queryClient, 'invalidateQueries').mockResolvedValue(undefined)

beforeEach(() => {
  vi.clearAllMocks()
})

afterEach(() => {
  cleanup()
})

describe('the create target is captured, not re-read', () => {
  it('keeps the owner the dialog was opened for while another bot becomes active', () => {
    expect(routineCreateTarget('ops', 'ops')).toBe('ops')
    expect(routineCreateTarget('ops', 'default')).toBe('ops')
    // Nothing captured yet: the pane's resolved bot is the target.
    expect(routineCreateTarget(null, 'default')).toBe('default')
  })
})

describe('cache eviction is scoped to one owner', () => {
  it('invalidates that owner\u2019s list and nothing else', async () => {
    await invalidateRoutineOwner('ops')

    expect(invalidateQueries).toHaveBeenCalledWith({ exact: true, queryKey: ['hermes-bots', 'routines', 'ops'] })
  })
})

describe('a row mutation addresses the owner that rendered it', () => {
  it('scopes the pause RPC and the eviction to that owner', async () => {
    render(
      <RoutineRow
        job={{ enabled: true, job_id: 'digest', name: '[bot:ops] Digest', schedule: 'every 1h' }}
        onOpen={() => undefined}
        owner={{ name: 'ops' }}
      />
    )

    fireEvent.click(screen.getByRole('switch'))

    await waitFor(() => expect(invalidateQueries).toHaveBeenCalled())

    expect(request).toHaveBeenCalledWith('cron.manage', { action: 'pause', name: 'digest', profile: 'ops' })
    expect(invalidateQueries).toHaveBeenCalledWith({ exact: true, queryKey: ['hermes-bots', 'routines', 'ops'] })
  })
})
