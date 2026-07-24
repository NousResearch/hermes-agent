type ClosePolicy = {
  isWindows: boolean
  trayAvailable: boolean
  closeToTray: boolean
  isQuitting: boolean
  isQuittingForHandoff: boolean
}

type WindowAllClosedPolicy = {
  isWindows: boolean
  isMac: boolean
  trayAvailable: boolean
  closeToTray: boolean
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
  closeToTray,
  isQuitting,
  isQuittingForHandoff
}: ClosePolicy) {
  return (
    isWindows &&
    trayAvailable &&
    closeToTray &&
    !isQuitting &&
    !isQuittingForHandoff
  )
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
  closeToTray,
  isQuitting,
  isQuittingForHandoff
}: WindowAllClosedPolicy) {
  if (isWindows) {
    return !trayAvailable || !closeToTray || isQuitting || isQuittingForHandoff
  }

  return !isMac || isQuittingForHandoff
}
