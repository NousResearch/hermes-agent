import { describe, expect, it, vi } from 'vitest'

import { achievementsKey, bindApi, fetchAchievements, rescanAchievements } from './api'

describe('achievements api', () => {
  it('rejects until the rest door is bound', async () => {
    const unsub = bindApi(vi.fn())
    unsub()
    await expect(fetchAchievements()).rejects.toThrow('not bound')
  })

  it('fetches the achievements payload through the bound door', async () => {
    const rest = vi.fn().mockResolvedValue({ unlocked_count: 36, total_count: 60 })
    bindApi(rest)

    const data = await fetchAchievements()

    expect(rest).toHaveBeenCalledWith('/achievements')
    expect(data.unlocked_count).toBe(36)
    expect(data.total_count).toBe(60)
  })

  it('posts a rescan through the bound door', async () => {
    const rest = vi.fn().mockResolvedValue({ ok: true })
    bindApi(rest)

    await rescanAchievements()

    expect(rest).toHaveBeenCalledWith('/rescan', { method: 'POST' })
  })

  it('uses a stable query key', () => {
    expect(achievementsKey()).toEqual(['hermes-achievements', 'all'])
  })
})
