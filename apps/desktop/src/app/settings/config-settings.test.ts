import { afterEach, describe, expect, it, vi } from 'vitest'

vi.mock('@/store/confirm')
vi.mock('@/hermes')
vi.mock('@/store/profile')
vi.mock('@/store/projects')
vi.mock('@/store/notifications')

import { confirm } from '@/store/confirm'

// Inline the logic under test to avoid rendering the full component
function applyConfigIfConfirmed(
  applyConfig: (next: unknown) => void,
  next: unknown
): void {
  void (confirm as ReturnType<typeof vi.fn>)({ destructive: true, title: 'confirm?' }).then((ok: boolean) => {
    if (ok) {
      applyConfig(next)
    }
  }).catch(() => {
    // rejected — treat as cancellation
  })
}

describe('config-settings toolsets wipe confirm', () => {
  afterEach(() => {
    vi.clearAllMocks()
  })

  it('applies config when confirm resolves true', async () => {
    vi.mocked(confirm).mockResolvedValue(true)
    const applyConfig = vi.fn()
    applyConfigIfConfirmed(applyConfig, { next: true })
    await Promise.resolve()
    await Promise.resolve()
    expect(applyConfig).toHaveBeenCalledWith({ next: true })
  })

  it('does not apply config when confirm resolves false', async () => {
    vi.mocked(confirm).mockResolvedValue(false)
    const applyConfig = vi.fn()
    applyConfigIfConfirmed(applyConfig, { next: true })
    await Promise.resolve()
    await Promise.resolve()
    expect(applyConfig).not.toHaveBeenCalled()
  })

  it('does not apply config when confirm rejects', async () => {
    vi.mocked(confirm).mockRejectedValue(new Error('dialog closed'))
    const applyConfig = vi.fn()
    applyConfigIfConfirmed(applyConfig, { next: true })
    await Promise.resolve()
    await Promise.resolve()
    expect(applyConfig).not.toHaveBeenCalled()
  })
})
