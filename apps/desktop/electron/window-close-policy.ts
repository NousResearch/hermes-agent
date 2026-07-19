type ClosePolicy = {
  isWindows: boolean
  trayAvailable: boolean
  isQuitting: boolean
  isQuittingForHandoff: boolean
}

type WindowAllClosedPolicy = {
  isWindows: boolean
  isMac: boolean
  trayAvailable: boolean
  isQuitting: boolean
  isQuittingForHandoff: boolean
}

type StartupVisibilityPolicy = {
  isWindows: boolean
  trayAvailable: boolean
  startInTray: boolean
  isInitialWindow?: boolean
}

export function shouldHideMainWindowToTray({
  isWindows,
  trayAvailable,
  isQuitting,
  isQuittingForHandoff
}: ClosePolicy) {
  return isWindows && trayAvailable && !isQuitting && !isQuittingForHandoff
}

export function shouldShowMainWindowOnStartup({
  isWindows,
  trayAvailable,
  startInTray,
  isInitialWindow = true
}: StartupVisibilityPolicy) {
  return !isInitialWindow || !isWindows || !trayAvailable || !startInTray
}

export function shouldQuitAfterAllWindowsClose({
  isWindows,
  isMac,
  trayAvailable,
  isQuitting,
  isQuittingForHandoff
}: WindowAllClosedPolicy) {
  if (isWindows) {
    return !trayAvailable || isQuitting || isQuittingForHandoff
  }

  return !isMac || isQuittingForHandoff
}
