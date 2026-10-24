import { EventEmitter } from 'node:events'

import { describe, expect, it, vi } from 'vitest'

import { trackGuestBounds } from './float-guest-bounds'

describe('trackGuestBounds', () => {
  it('fits to parent layout changes without a BrowserWindow resize and detaches cleanly', () => {
    let bounds = { width: 947, height: 505 }

    const contentView = Object.assign(new EventEmitter(), {
      getBounds: () => bounds
    })

    const guest = { setBounds: vi.fn() }
    const tracking = trackGuestBounds(contentView, guest)

    expect(guest.setBounds).toHaveBeenLastCalledWith({ x: 0, y: 0, width: 947, height: 505 })

    // Native menu/client layout reduces the root view height without changing
    // BrowserWindow's outer resize event.
    bounds = { width: 947, height: 479 }
    contentView.emit('bounds-changed')
    expect(guest.setBounds).toHaveBeenLastCalledWith({ x: 0, y: 0, width: 947, height: 479 })

    tracking.dispose()
    bounds = { width: 947, height: 460 }
    contentView.emit('bounds-changed')
    expect(guest.setBounds).toHaveBeenCalledTimes(2)
  })
})
