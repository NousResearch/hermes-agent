import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { LinuxWindowControlsSetting } from './linux-window-controls-setting'

const relaunchApp = vi.fn(async () => undefined)

function installBridge() {
  const statuses = { auto: false, hide: true, show: false }

  Object.defineProperty(window, 'hermesDesktop', {
    configurable: true,
    value: {
      relaunchApp,
      windowControls: {
        custom: false,
        getLinuxMode: async () => ({ mode: 'auto', restartNeeded: false }),
        setLinuxMode: async (mode: keyof typeof statuses) => ({ mode, restartNeeded: statuses[mode] })
      }
    }
  })
  vi.spyOn(navigator, 'platform', 'get').mockReturnValue('Linux x86_64')
}

describe('LinuxWindowControlsSetting', () => {
  beforeEach(installBridge)

  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
    relaunchApp.mockClear()
    Reflect.deleteProperty(window, 'hermesDesktop')
  })

  it('offers a restart only while the choice differs from this launch', async () => {
    render(<LinuxWindowControlsSetting />)

    fireEvent.click(await screen.findByRole('button', { name: 'Hide' }))
    fireEvent.click(await screen.findByRole('button', { name: 'Restart now' }))
    expect(relaunchApp).toHaveBeenCalledOnce()

    fireEvent.click(screen.getByRole('button', { name: 'Auto' }))
    await vi.waitFor(() => expect(screen.queryByRole('alert')).toBeNull())
  })
})
