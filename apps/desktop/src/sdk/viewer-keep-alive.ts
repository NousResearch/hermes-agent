interface ViewerKeepAliveOptions {
  isOpen: () => boolean | Promise<boolean>
  onKeepAlive: () => Promise<void>
  onStop?: () => void
}

/** Renderer-owned lease: never overlap renewals or revive a disposed viewer.
 * Callback/probe failures get three attempts, then leave the existing lease finite.
 */
export function startViewerKeepAlive({ isOpen, onKeepAlive, onStop }: ViewerKeepAliveOptions): () => void {
  let stopped = false
  let failures = 0
  let timer: ReturnType<typeof setTimeout> | undefined

  const stop = () => {
    if (stopped) {
      return
    }

    stopped = true
    clearTimeout(timer)
    onStop?.()
  }

  const schedule = (delay: number) => {
    if (!stopped) {
      timer = setTimeout(() => void renew(), delay)
    }
  }

  const renew = async () => {
    try {
      const open = await isOpen()

      if (stopped || !open) {
        stop()

        return
      }

      await onKeepAlive()

      if (stopped) {
        return
      }

      // Close/navigation can happen while the callback is awaiting its backend.
      if (!(await isOpen())) {
        stop()

        return
      }

      failures = 0
      schedule(60_000)
    } catch {
      failures += 1

      if (failures >= 3) {
        stop()
      } else {
        schedule(5_000)
      }
    }
  }

  schedule(0)

  return stop
}
