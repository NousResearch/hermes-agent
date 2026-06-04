import { describe, expect, it } from 'vitest'

import { isDesktopComposerImagePath, isRemoteGatewayConnection } from './gateway-connection'
describe('isRemoteGatewayConnection', () => {
  it('treats explicit remote mode as remote', () => {
    expect(
      isRemoteGatewayConnection({
        baseUrl: 'http://127.0.0.1:8787',
        mode: 'remote',
        token: 't',
        wsUrl: 'ws://127.0.0.1:8787/api/ws?token=t',
        isFullscreen: false,
        nativeOverlayWidth: 0,
        logs: [],
        windowButtonPosition: null
      })
    ).toBe(true)
  })

  it('detects remote hosts even when mode is unset', () => {
    expect(
      isRemoteGatewayConnection({
        baseUrl: 'https://hermes.example.com',
        token: 't',
        wsUrl: 'wss://hermes.example.com/api/ws?token=t',
        isFullscreen: false,
        nativeOverlayWidth: 0,
        logs: [],
        windowButtonPosition: null
      })
    ).toBe(true)
  })

  it('treats loopback hosts as local', () => {
    expect(
      isRemoteGatewayConnection({
        baseUrl: 'http://127.0.0.1:8787',
        mode: 'local',
        token: 't',
        wsUrl: 'ws://127.0.0.1:8787/api/ws?token=t',
        isFullscreen: false,
        nativeOverlayWidth: 0,
        logs: [],
        windowButtonPosition: null
      })
    ).toBe(false)
  })
})

describe('isDesktopComposerImagePath', () => {
  it('matches desktop composer cache paths', () => {
    expect(
      isDesktopComposerImagePath(
        'C:\\Users\\danny\\AppData\\Roaming\\Hermes\\composer-images\\composer_2026-06-04_16-35-59-305_f434ea.png'
      )
    ).toBe(true)
  })

  it('ignores unrelated image paths', () => {
    expect(isDesktopComposerImagePath('/tmp/screenshot.png')).toBe(false)
  })
})
