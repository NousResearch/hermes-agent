interface ContentViewBounds {
  width: number
  height: number
}

interface ContentView {
  getBounds(): ContentViewBounds
  on(event: 'bounds-changed', listener: () => void): unknown
  off(event: 'bounds-changed', listener: () => void): unknown
}

interface GuestBoundsTarget {
  setBounds(bounds: { x: number; y: number; width: number; height: number }): void
}

export interface GuestBoundsTracking {
  resize(): void
  dispose(): void
}

/** Keep a child view fitted to its native parent, including layout-only changes. */
export function trackGuestBounds(contentView: ContentView, guest: GuestBoundsTarget): GuestBoundsTracking {
  const resize = () => {
    const { width, height } = contentView.getBounds()
    guest.setBounds({ x: 0, y: 0, width, height })
  }

  contentView.on('bounds-changed', resize)
  resize()

  return {
    resize,
    dispose: () => contentView.off('bounds-changed', resize)
  }
}
