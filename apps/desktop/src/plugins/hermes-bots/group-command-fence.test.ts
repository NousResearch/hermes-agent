import { afterEach, describe, expect, it } from 'vitest'

import {
  beginGroupCommandFence,
  bindGroupCommandFence,
  cancelGroupCommandFence,
  groupCommandFenceLive,
  releaseGroupCommandFence
} from './group-command-fence'
import type { GroupCommandFence } from './group-command-fence'

const fences: GroupCommandFence[] = []

const begin = (room = 'room', command = 'command') => {
  const fence = beginGroupCommandFence(room, command)
  fences.push(fence)

  return fence
}

afterEach(() => fences.splice(0).forEach(releaseGroupCommandFence))

describe('bounded command cancellation generations', () => {
  it('keeps a cancelled generation dead after reclamation and old teardown', () => {
    const old = begin()
    bindGroupCommandFence(old, 'thread', 1)
    cancelGroupCommandFence(old)
    const fresh = begin()
    bindGroupCommandFence(fresh, 'thread', 2)
    releaseGroupCommandFence(old)
    expect(groupCommandFenceLive(old)).toBe(false)
    expect(groupCommandFenceLive(fresh)).toBe(true)
    expect(fresh.generation).not.toBe(old.generation)
  })

  it('does not affect ordinary drives, other commands or recreated rooms', () => {
    const cancelled = begin()
    const otherThread = begin('room', 'other-command')
    const recreated = begin('other-room')
    releaseGroupCommandFence(cancelled)
    expect(groupCommandFenceLive()).toBe(true)
    expect(groupCommandFenceLive(otherThread)).toBe(true)
    expect(groupCommandFenceLive(recreated)).toBe(true)
  })

  it('bounds active generations without evicting and reviving an old turn', () => {
    for (let index = 0; index < 128; index += 1) {
      begin('room', `command-${index}`)
    }

    expect(() => begin('room', 'overflow')).toThrow('Too many')
    expect(fences.every(fence => groupCommandFenceLive(fence))).toBe(true)
    releaseGroupCommandFence(fences[0])
    expect(groupCommandFenceLive(begin('room', 'new'))).toBe(true)
    expect(groupCommandFenceLive(fences[0])).toBe(false)
  })

  it('keeps an invalidated room generation retired even if its record returns', () => {
    let valid = true
    const fence = beginGroupCommandFence('room', 'retired-command', () => true, () => valid)
    fences.push(fence)
    expect(groupCommandFenceLive(fence)).toBe(true)
    valid = false
    expect(groupCommandFenceLive(fence)).toBe(false)
    valid = true
    expect(groupCommandFenceLive(fence)).toBe(false)
  })
})
