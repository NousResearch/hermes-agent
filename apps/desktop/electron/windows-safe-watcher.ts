import fs from 'node:fs'

const WINDOWS_WATCH_INTERVAL_MS = 2_000

type PathWatchListener = (...args: any[]) => void

type PathWatchFs = {
  unwatchFile(path: string, listener: PathWatchListener): void
  watch(path: string, listener: PathWatchListener): { close(): void }
  watchFile(path: string, options: { interval: number }, listener: PathWatchListener): void
}

type WatchPathOptions = {
  fsApi?: PathWatchFs
  isWindows?: boolean
}

/**
 * Windows ReadDirectoryChangesW can repeatedly surface the same directory
 * notification through fs.watch. Polling is deliberately used there so a
 * noisy directory cannot monopolize Electron's main thread. Other platforms
 * retain fs.watch's immediate event delivery.
 */
export function watchPath(path: string, listener: PathWatchListener, options: WatchPathOptions = {}) {
  const fsApi = options.fsApi ?? fs
  const isWindows = options.isWindows ?? process.platform === 'win32'

  if (isWindows) {
    // fs.watchFile passes current/previous Stats objects, unlike fs.watch's
    // eventType/filename pair. Deliberately normalize the polling callback so
    // callers do not mistake Stats for a changed filename.
    const pollingListener = () => listener()

    fsApi.watchFile(path, { interval: WINDOWS_WATCH_INTERVAL_MS }, pollingListener)

    return {
      close: () => fsApi.unwatchFile(path, pollingListener)
    }
  }

  return fsApi.watch(path, listener)
}

export function createDebouncedCallback(callback: () => void, delayMs: number) {
  let timer: ReturnType<typeof setTimeout> | null = null

  return {
    cancel() {
      if (timer) {
        clearTimeout(timer)
        timer = null
      }
    },
    invoke() {
      if (timer) {
        clearTimeout(timer)
      }

      timer = setTimeout(() => {
        timer = null
        callback()
      }, delayMs)
    }
  }
}
