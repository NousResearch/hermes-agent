import { describe, expect, it, vi } from 'vitest'

import { HostedRoomObservations } from './hosted-room-observations'
import type { GroupChat } from './types'

function room(roomId = 'classic', remoteSource = true): GroupChat {
  return { roomId, log: [], watermarks: {}, members: [{ name: 'writer', connectionId: 'local', remoteSource }] }
}

describe('hosted observation lifetimes', () => {
  it('orders capability probes without invalidating warm history and inventory reads', async () => {
    const observations = new HostedRoomObservations()
    const warm = observations.capture('local')
    observations.publish(warm, new Set(['hosted']), true)
    const oldProbe = observations.captureCapability('local')
    const latestProbe = observations.captureCapability('local')
    observations.captureCapability('other')

    expect(observations.current(oldProbe)).toBe(false)
    expect(observations.current(latestProbe)).toBe(true)
    expect(observations.current(warm)).toBe(true)
    expect(observations.classicReady(room())).toBe(true)
    expect(observations.classicReady(room('hosted', false))).toBe(false)
    await expect(observations.read(warm, async () => 'history')).resolves.toBe('history')
    const staleRequest = vi.fn(async () => 'stale')
    await expect(observations.read(oldProbe, staleRequest)).rejects.toThrow('Stale')
    expect(staleRequest).not.toHaveBeenCalled()

    observations.invalidateAll()
    expect(observations.current(latestProbe)).toBe(false)
    expect(observations.current(warm)).toBe(false)
    expect(observations.classicReady(room('hosted', false))).toBe(false)
  })

  it('preserves warm absence while positive IDs override all member provenance', () => {
    const observations = new HostedRoomObservations()
    const ticket = observations.capture('local')
    observations.publish(ticket, new Set(), true)
    expect(observations.classicReady(room())).toBe(true)
    observations.observe(ticket, new Set(['hosted']))
    expect(observations.classicReady(room())).toBe(true)
    expect(observations.classicReady(room('hosted', false))).toBe(false)
    observations.invalidate('local')
    expect(observations.classicReady(room())).toBe(false)
    expect(observations.classicReady(room('hosted', false))).toBe(false)
    expect(observations.classicReady(room('fresh-legacy', false))).toBe(true)
    const recovered = observations.capture('local')
    observations.publish(recovered, new Set(), true)
    expect(observations.classicReady(room())).toBe(true)
    expect(observations.classicReady(room('hosted', false))).toBe(false)
  })

  it.each(['invalidate', 'restart', 'prune'] as const)('never reuses a token after %s', async operation => {
    const observations = new HostedRoomObservations()
    const ticket = observations.capture('local')
    let release!: (value: string) => void

    const request = observations.read(
      ticket,
      () =>
        new Promise<string>(resolve => {
          release = resolve
        })
    )

    if (operation === 'invalidate') {
      observations.invalidate('local')
      observations.invalidate('local')
    } else if (operation === 'restart') {
      observations.invalidateAll()
    } else {
      observations.retain([], [])
    }

    const next = observations.capture('local')
    expect(observations.current(ticket)).toBe(false)
    expect(observations.current(next)).toBe(true)
    expect(observations.observe(ticket, new Set(['stale']))).toBe(false)
    expect(observations.publish(ticket, new Set(), true)).toBe(false)
    expect(observations.classicReady(room())).toBe(false)
    const rejected = expect(request).rejects.toThrow('Stale Group Chat observation')
    release('late')
    await rejected
    const neverCalled = vi.fn(async () => 'unused')
    await expect(observations.read(ticket, neverCalled)).rejects.toThrow('Stale')
    expect(neverCalled).not.toHaveBeenCalled()
  })

  it('bounds retained IDs and saturates rather than forgetting positive ownership', () => {
    const observations = new HostedRoomObservations()
    const ticket = observations.capture('local')
    observations.publish(ticket, new Set(Array.from({ length: 2000 }, (_, i) => `hosted-${i}`)), true)
    expect(observations.classicReady(room())).toBe(true)
    observations.observe(ticket, new Set(['overflow']))
    expect(observations.classicReady(room('hosted-0', false))).toBe(false)
    expect(observations.classicReady(room('overflow', false))).toBe(false)
    expect(observations.classicReady(room())).toBe(false)
    observations.invalidateAll()
    observations.publish(observations.capture('local'), new Set(), true)
    expect(observations.classicReady(room())).toBe(false)
    expect(observations.classicReady({ ...room(), members: [{ name: 'other', connectionId: 'other' }] })).toBe(true)
    // Retention follows current sources and local room references, not every historical connection.
    observations.retain([], [room()])
    expect(observations.classicReady(room('hosted-0', false))).toBe(false)
    observations.retain([], [])
    expect(observations.current(ticket)).toBe(false)
    expect(observations.classicReady(room('fresh', false))).toBe(true)
  })

  it('uses the explicit route connection for positive ownership', () => {
    const observations = new HostedRoomObservations()
    observations.observe(observations.capture('remote'), new Set(['classic']))
    const legacy = room('classic', false)
    legacy.members![0].route = { connectionId: 'remote', mode: 'remote', profile: 'writer', targetProfile: 'writer' }
    expect(observations.classicReady(legacy)).toBe(false)
    observations.invalidate('remote')
    expect(observations.classicReady(legacy)).toBe(false)
  })

  it('partial scans retain positives but withdraw absence', () => {
    const observations = new HostedRoomObservations()
    const ticket = observations.capture('local')
    observations.publish(ticket, new Set(), true)
    observations.publish(ticket, new Set(['hosted']), false)
    expect(observations.classicReady(room())).toBe(false)
    expect(observations.classicReady(room('hosted', false))).toBe(false)
  })
})
