import { LOCAL_CONNECTION_ID } from '@hermes/shared'
import { act, cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { installBrowserDesktopBridge } from '@/lib/browser-desktop-bridge'
import { clearWebappSession, seedWebappSession } from '@/test/webapp-session'

import { useLocalDeviceSwitch } from './local-device-switch'

afterEach(() => {
  cleanup()
  clearWebappSession()
  Reflect.deleteProperty(window, 'hermesDesktop')
  delete globalThis.document.documentElement.dataset.hermesDesktopHost
  vi.unstubAllGlobals()
})

describe('useLocalDeviceSwitch in the browser-hosted Webapp', () => {
  it('switches to This device with no install prompt', async () => {
    seedWebappSession('served-token')
    vi.stubGlobal('fetch', vi.fn())
    expect(installBrowserDesktopBridge()).toBe(true)

    let request: ReturnType<typeof useLocalDeviceSwitch>['request'] = async () => false

    function Harness() {
      const localDevice = useLocalDeviceSwitch()
      request = localDevice.request

      return localDevice.dialog
    }

    render(<Harness />)
    const settled = vi.fn()

    await act(async () => {
      void request({ connectionId: LOCAL_CONNECTION_ID, label: 'This device' }).then(settled)
    })

    // Wait for whichever comes first: the switch settles, or the install prompt opens.
    await vi.waitFor(() => expect(settled.mock.calls.length > 0 || screen.queryByRole('dialog') !== null).toBe(true))
    expect(screen.queryByRole('dialog')).toBeNull()
    expect(settled).toHaveBeenCalledWith(true)
  })
})
