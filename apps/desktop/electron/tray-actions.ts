type RestorableWindow = {
  isDestroyed(): boolean
}

type RestoreOptions<T extends RestorableWindow> = {
  window: T | null
  createWindow(): void
  focusWindow(window: T): void
}

type QuitOptions = {
  markQuitting(): void
  quit(): void
}

type DestroyableTray = {
  destroy(): void
  isDestroyed(): boolean
}

type DestroyTrayOptions<T extends DestroyableTray> = {
  tray: T | null
  clear(): void
}

export function restoreMainWindow<T extends RestorableWindow>({
  window,
  createWindow,
  focusWindow
}: RestoreOptions<T>) {
  if (!window || window.isDestroyed()) {
    createWindow()

    return
  }

  focusWindow(window)
}

export function quitFromTray({ markQuitting, quit }: QuitOptions) {
  markQuitting()
  quit()
}

export function destroyTray<T extends DestroyableTray>({ tray, clear }: DestroyTrayOptions<T>) {
  if (tray && !tray.isDestroyed()) {
    tray.destroy()
  }

  clear()
}
