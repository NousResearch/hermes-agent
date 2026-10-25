/**
 * The group capability gate must read the activation epoch LIVE on every render.
 *
 * `gatewayActivationEpoch()` is an impure read of the gateway store, fenced
 * against later activations by `groupCreationSource`. The React Compiler
 * assumes plain calls are pure and cached the read on the hook's other inputs,
 * so a dialog mounted before the primary route activated kept comparing every
 * `groups.capabilities` answer to a stale epoch: the answer said `driver: true`,
 * `settle()` dropped it, and the mode stayed `checking` forever (#144272:
 * "Create" disabled, "Group driver unavailable"). The `'use no memo'` directive
 * in `useGroupCapability` keeps the read un-memoized; this test runs through the
 * compiler (vitest extends vite.config.ts) and fails without the directive.
 */

import { renderHook, waitFor } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

const sdk = vi.hoisted(() => ({
  epoch: 0,
  host: {
    state: {
      connectionId: { get: () => 'local', listen: () => () => undefined },
      gateway: { get: () => 'open', listen: () => () => undefined },
      profile: { get: () => 'default', listen: () => () => undefined }
    }
  },
  request: vi.fn()
}))

vi.mock('@hermes/plugin-sdk', () => ({
  gatewayActivationEpoch: () => sdk.epoch,
  host: sdk.host,
  useValue: <T,>(store: { get: () => T }) => store.get()
}))

vi.mock('./canonical-groups', () => ({
  canonicalGroupRequest: sdk.request
}))

describe('useGroupCapability', () => {
  it('reads the activation epoch live, so a dialog mounted before the primary route activated still settles', async () => {
    sdk.request.mockResolvedValue({ driver: true, protocol_version: 2 })
    const { useGroupCapability } = await import('./canonical-group-capabilities')

    // Mounted (inactive) while the window was still on epoch 0...
    sdk.epoch = 0

    const { rerender, result } = renderHook(({ active }) => useGroupCapability(active), {
      initialProps: { active: false }
    })

    expect(result.current.mode).toBe('checking')

    // ...then the primary route activates (epoch 1) and the dialog opens. The
    // connection, profile and socket are unchanged, so only the epoch moved.
    sdk.epoch = 1
    rerender({ active: true })

    await waitFor(() => expect(result.current.mode).toBe('canonical'))
    expect(result.current.capability?.activationEpoch).toBe(1)
    expect(result.current.source).toBe(JSON.stringify(['local', 'default', 'open', 1]))
    expect(sdk.request).toHaveBeenCalledWith({ connectionId: 'local', profile: 'default' }, 'groups.capabilities')
  })
})
