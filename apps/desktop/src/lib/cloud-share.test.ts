import { afterEach, describe, expect, it, vi } from 'vitest'

vi.mock('@/components/ui/copy-button', () => ({
  writeClipboardText: vi.fn().mockResolvedValue(undefined)
}))

vi.mock('@/store/gateway', () => ({
  activeGateway: vi.fn()
}))

vi.mock('@/store/notifications', () => ({
  notify: vi.fn(),
  notifyError: vi.fn()
}))

const gatewayStore = vi.mocked(await import('@/store/gateway'))
const notifications = vi.mocked(await import('@/store/notifications'))
const cloudShare = await import('./cloud-share')

afterEach(() => {
  vi.clearAllMocks()
})

describe('ensureCloudActionReady', () => {
  it('shows a setup notice instead of an error when cloud sharing is unconfigured', async () => {
    gatewayStore.activeGateway.mockReturnValue({
      request: vi.fn().mockResolvedValue({ configured: false, shared: false })
    } as never)

    const result = await cloudShare.ensureCloudActionReady('session-123', { title: 'Share to cloud' })

    expect(result).toBeNull()
    expect(notifications.notify).toHaveBeenCalledWith({
      durationMs: 7_000,
      kind: 'info',
      title: "Cloud sharing isn't set up",
      message: 'Add HERMES_CLOUD_TOKEN where the gateway runs, then try again.'
    })
    expect(notifications.notifyError).not.toHaveBeenCalled()
  })

  it('blocks shared-only actions until the session is shared', async () => {
    gatewayStore.activeGateway.mockReturnValue({
      request: vi.fn().mockResolvedValue({ configured: true, shared: false })
    } as never)

    const result = await cloudShare.ensureCloudActionReady('session-123', {
      requireShared: true,
      title: 'Invite to cloud'
    })

    expect(result).toBeNull()
    expect(notifications.notify).toHaveBeenCalledWith({
      kind: 'warning',
      title: 'Invite to cloud',
      message: 'Share this chat to the cloud first.'
    })
  })
})
