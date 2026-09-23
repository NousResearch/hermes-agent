import { describe, expect, it, vi } from 'vitest'

import { createDesktopPluginCompatNoticeRuntime } from './desktop-plugin-compat-notice-runtime'

describe('desktop plugin compatibility notice', () => {
  it('shows one notice and resolves the late protocol only when opening plugins', async () => {
    let protocol = 'hermes-dev'
    const handleDeepLink = vi.fn()

    const pendingPluginCompatNotice = vi.fn(() => ({
      detail: 'One plugin imports an old path.',
      key: 'report-1',
      message: 'Plugin compatibility',
      title: 'Plugin compatibility'
    }))

    const showMessageBox = vi.fn(async () => ({ response: 0 }))
    const recordPluginCompatDismissed = vi.fn()

    const runtime = createDesktopPluginCompatNoticeRuntime({
      HERMES_HOME: 'test-home',
      app: { getPath: () => 'test-user-data' },
      dialog: { showMessageBox },
      getHermesProtocol: () => protocol,
      getMainWindow: () => ({ isDestroyed: () => false }) as any,
      handleDeepLink,
      pendingPluginCompatNotice,
      recordPluginCompatDismissed,
      rememberLog: vi.fn()
    })

    protocol = 'hermes'
    await runtime.showPluginCompatNoticeOnce()
    await runtime.showPluginCompatNoticeOnce()

    expect(showMessageBox).toHaveBeenCalledOnce()
    expect(handleDeepLink).toHaveBeenCalledWith('hermes://open/capabilities?tab=plugins')
    expect(recordPluginCompatDismissed).toHaveBeenCalledWith('test-user-data', 'report-1')
  })
})
