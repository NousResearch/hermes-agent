const assert = require('node:assert/strict')
const test = require('node:test')

const {
  chatWindowChromeOptions,
  nativeOverlayWidthForPlatform,
  usesNativeSystemTitleBar
} = require('./window-chrome.cjs')

const overlay = { color: '#111111', height: 34, symbolColor: '#f7f7f7' }
const trafficLightPosition = { x: 24, y: 10 }

test('Linux keeps the system titlebar and avoids Electron titlebar overlay', () => {
  const options = chatWindowChromeOptions({
    platform: 'linux',
    titleBarOverlay: overlay,
    trafficLightPosition
  })

  assert.deepEqual(options, {})
  assert.equal(usesNativeSystemTitleBar('linux'), true)
})

test('macOS keeps the hidden titlebar and traffic-light placement', () => {
  const options = chatWindowChromeOptions({
    platform: 'darwin',
    isMac: true,
    titleBarOverlay: { height: 34 },
    trafficLightPosition
  })

  assert.deepEqual(options, {
    titleBarStyle: 'hidden',
    titleBarOverlay: { height: 34 },
    trafficLightPosition
  })
})

test('Windows keeps hidden titlebar overlay without macOS traffic-light options', () => {
  const options = chatWindowChromeOptions({
    platform: 'win32',
    isMac: false,
    titleBarOverlay: overlay,
    trafficLightPosition
  })

  assert.deepEqual(options, {
    titleBarStyle: 'hidden',
    titleBarOverlay: overlay,
    trafficLightPosition: undefined
  })
})

test('native overlay width is reserved only for non-Linux non-macOS overlay chrome', () => {
  const width = 144

  assert.equal(nativeOverlayWidthForPlatform({ platform: 'darwin', isMac: true, nativeOverlayButtonWidth: width }), 0)
  assert.equal(nativeOverlayWidthForPlatform({ platform: 'linux', isMac: false, nativeOverlayButtonWidth: width }), 0)
  assert.equal(
    nativeOverlayWidthForPlatform({ platform: 'win32', isMac: false, nativeOverlayButtonWidth: width }),
    width
  )
})
