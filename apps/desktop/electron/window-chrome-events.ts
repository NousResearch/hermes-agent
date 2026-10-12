interface WindowChromeEvents {
  on(event: string, listener: () => void): unknown
}

/** Every window publishes its own chrome state, never the primary's. */
export function bindWindowChromeEvents<T extends WindowChromeEvents>(
  win: T,
  publish: (isFullscreen: boolean | undefined, target: T) => void
): void {
  for (const event of ['will-enter-full-screen', 'enter-full-screen']) {
    win.on(event, () => publish(true, win))
  }

  for (const event of ['will-leave-full-screen', 'leave-full-screen']) {
    win.on(event, () => publish(false, win))
  }

  for (const event of ['maximize', 'unmaximize', 'minimize', 'restore', 'hide', 'show']) {
    win.on(event, () => publish(undefined, win))
  }
}
