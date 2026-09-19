type CloseEvent = { preventDefault: () => void }

type WindowForTray = {
  focus?: () => void
  hide: () => void
  isDestroyed: () => boolean
  show?: () => void
}

/** Read the opt-in without accepting a similarly named top-level setting. */
export function readCloseToTraySetting(config: string): boolean {
  let inDesktop = false

  for (const line of config.split(/\r?\n/)) {
    if (/^desktop:\s*(?:#.*)?$/.test(line)) {
      inDesktop = true
      continue
    }

    if (/^[^\s#][^:]*:/.test(line)) {
      inDesktop = false
    }

    if (inDesktop && /^\s{2,}close_to_tray:\s*true\s*(?:#.*)?$/.test(line)) {
      return true
    }
  }

  return false
}

export function createTrayLifecycle({ enabled, isWindows }: { enabled: boolean; isWindows: boolean }) {
  let quitRequested = false

  return {
    handleWindowClose(event: CloseEvent, window: WindowForTray): boolean {
      if (!enabled || !isWindows || quitRequested || window.isDestroyed()) {
        return false
      }

      event.preventDefault()
      window.hide()
      return true
    },
    requestQuit() {
      quitRequested = true
    },
    restoreWindow(window: WindowForTray | null) {
      if (!window || window.isDestroyed()) {
        return false
      }

      window.show?.()
      window.focus?.()
      return true
    }
  }
}
