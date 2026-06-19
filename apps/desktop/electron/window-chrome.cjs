// BrowserWindow titlebar/chrome policy kept pure so platform-specific choices
// can be tested without booting Electron.

function usesNativeSystemTitleBar(platform = process.platform) {
  return platform === 'linux'
}

function chatWindowChromeOptions({
  platform = process.platform,
  isMac = platform === 'darwin',
  titleBarOverlay,
  trafficLightPosition
} = {}) {
  if (usesNativeSystemTitleBar(platform)) {
    return {}
  }

  return {
    titleBarStyle: 'hidden',
    titleBarOverlay,
    trafficLightPosition: isMac ? trafficLightPosition : undefined
  }
}

function nativeOverlayWidthForPlatform({
  platform = process.platform,
  isMac = platform === 'darwin',
  nativeOverlayButtonWidth
} = {}) {
  if (isMac || usesNativeSystemTitleBar(platform)) {
    return 0
  }

  return nativeOverlayButtonWidth
}

module.exports = {
  chatWindowChromeOptions,
  nativeOverlayWidthForPlatform,
  usesNativeSystemTitleBar
}
