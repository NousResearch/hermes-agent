import { afterEach, describe, expect, it, vi } from 'vitest'

import { $commentaryLane, setCommentaryLane, syncCommentaryLane } from './commentary-lane'

afterEach(() => {
  $commentaryLane.set(false)
  vi.restoreAllMocks()
})

describe('commentary-lane store', () => {
  it('hydrates the atom from config.get commentary_lane', async () => {
    const requestGateway = vi.fn(async () => ({ value: true }))

    const enabled = await syncCommentaryLane(requestGateway)

    expect(requestGateway).toHaveBeenCalledWith('config.get', { key: 'commentary_lane' })
    expect(enabled).toBe(true)
    expect($commentaryLane.get()).toBe(true)
  })

  it('writes config.set commentary_lane and reconciles with the echoed value', async () => {
    const requestGateway = vi.fn(async () => ({ value: true }))

    const result = await setCommentaryLane(requestGateway, true)

    expect(requestGateway).toHaveBeenCalledWith('config.set', { key: 'commentary_lane', value: true })
    expect(result).toBe(true)
    expect($commentaryLane.get()).toBe(true)
  })

  it('coerces non-boolean config values', async () => {
    const requestGateway = vi.fn(async () => ({ value: 'true' }))

    await syncCommentaryLane(requestGateway)

    expect($commentaryLane.get()).toBe(true)
  })
})
