import { afterEach, describe, expect, it, vi } from 'vitest'

const mocks = vi.hoisted(() => ({
  getAllSessionMessages: vi.fn()
}))

vi.mock('@/hermes', () => ({
  getAllSessionMessages: mocks.getAllSessionMessages
}))

vi.mock('@/i18n', () => ({
  translateNow: vi.fn((key: string) => key)
}))

vi.mock('@/store/notifications', () => ({
  notify: vi.fn(),
  notifyError: vi.fn()
}))

import { exportSession } from './session-export'

afterEach(() => {
  vi.useRealTimers()
  vi.restoreAllMocks()
})

describe('exportSession', () => {
  it('keeps the JSON download URL alive until deferred cleanup', async () => {
    vi.useFakeTimers()
    mocks.getAllSessionMessages.mockResolvedValue({ messages: [] })
    const downloadUrl = 'blob:session-export'
    vi.spyOn(URL, 'createObjectURL').mockReturnValue(downloadUrl)
    const revokeObjectURL = vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => undefined)
    const click = vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => undefined)

    await exportSession('session-123', { title: 'Example Session' })

    expect(click).toHaveBeenCalledOnce()
    expect(revokeObjectURL).not.toHaveBeenCalled()

    await vi.advanceTimersByTimeAsync(30_000)

    expect(revokeObjectURL).toHaveBeenCalledOnce()
    expect(revokeObjectURL).toHaveBeenCalledWith(downloadUrl)
  })
})
