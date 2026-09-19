import { afterEach, describe, expect, it, vi } from 'vitest'

vi.mock('@/store/confirm')
vi.mock('@/hermes')
vi.mock('@/store/profile')
vi.mock('@/store/projects')
vi.mock('@/store/notifications')

import { confirm } from '@/store/confirm'

import { applyWhenConfirmed } from './config-settings'

describe('config-settings toolsets wipe confirm', () => {
  afterEach(() => {
    vi.clearAllMocks()
  })

  it('applies config when confirm resolves true', async () => {
    vi.mocked(confirm).mockResolvedValue(true)
    const apply = vi.fn()
    await applyWhenConfirmed(() => confirm({ destructive: true, title: 'confirm?' }), apply)
    expect(apply).toHaveBeenCalled()
  })

  it('does not apply config when confirm resolves false', async () => {
    vi.mocked(confirm).mockResolvedValue(false)
    const apply = vi.fn()
    await applyWhenConfirmed(() => confirm({ destructive: true, title: 'confirm?' }), apply)
    expect(apply).not.toHaveBeenCalled()
  })

  it('does not apply config when confirm rejects', async () => {
    vi.mocked(confirm).mockRejectedValue(new Error('dialog closed'))
    const apply = vi.fn()
    await applyWhenConfirmed(() => confirm({ destructive: true, title: 'confirm?' }), apply)
    expect(apply).not.toHaveBeenCalled()
  })
})
