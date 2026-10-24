import { describe, expect, it } from 'vitest'

import { PLAYER_SERVICES } from '../../src/lib/player-services'

import {
  clampPlayerBounds,
  type FixtureOrigin,
  isAllowedPlayerFrame,
  isAllowedPlayerMediaKeyHost,
  isAllowedPlayerNavigation,
  isAllowedPlayerPopup,
  safePlayerOrigin
} from './policy'

const fixture: FixtureOrigin = { hostname: '127.0.0.1', port: '43127' }

describe('player guest URL policy', () => {
  it('allows the YouTube service and required Google sign-in navigation over HTTPS only', () => {
    expect(isAllowedPlayerNavigation('https://www.youtube.com/watch?v=example')).toBe(true)
    expect(isAllowedPlayerNavigation('https://tv.youtube.com/', undefined, 'tv')).toBe(true)
    expect(isAllowedPlayerNavigation('https://www.youtube.com/', undefined, 'tv')).toBe(true)
    expect(isAllowedPlayerNavigation('https://tv.youtube.com/')).toBe(true)
    expect(isAllowedPlayerNavigation('https://accounts.google.com/')).toBe(true)
    expect(isAllowedPlayerNavigation('http://www.youtube.com/')).toBe(false)
    expect(isAllowedPlayerNavigation('https://youtube.evil.example/')).toBe(false)
    expect(isAllowedPlayerNavigation('file:///etc/passwd')).toBe(false)
  })

  it('permits only the exact injected loopback fixture authority', () => {
    expect(isAllowedPlayerNavigation('http://127.0.0.1:43127/fixture.webm', fixture)).toBe(true)
    expect(isAllowedPlayerNavigation('http://localhost:43127/', fixture)).toBe(false)
    expect(isAllowedPlayerNavigation('http://127.0.0.1:43128/', fixture)).toBe(false)
    expect(isAllowedPlayerNavigation('http://127.0.0.1:43127/', undefined)).toBe(false)
    expect(safePlayerOrigin('http://127.0.0.1:43127/x', fixture)).toBe('http://127.0.0.1:43127')
    expect(safePlayerOrigin('https://accounts.google.com/signin')).toBeNull()
  })

  it('keeps frame policy scheme-bounded and limits popups to service/auth origins', () => {
    expect(isAllowedPlayerFrame('https://youtubei.googleapis.com/embed')).toBe(true)
    expect(isAllowedPlayerFrame('about:blank')).toBe(true)
    expect(isAllowedPlayerFrame('data:text/html,evil')).toBe(false)
    expect(isAllowedPlayerFrame('blob:http://127.0.0.1:43127/id', fixture)).toBe(true)
    expect(isAllowedPlayerPopup('https://accounts.google.com/')).toBe(true)
    expect(isAllowedPlayerPopup('https://example.com/')).toBe(false)
    expect(isAllowedPlayerPopup('http://127.0.0.1:43127/')).toBe(false)
  })

  it('scopes experimental provider navigation, popups, and reported origin to the active service', () => {
    expect(isAllowedPlayerNavigation('https://www.netflix.com/', undefined, 'netflix')).toBe(true)
    expect(isAllowedPlayerNavigation('https://netflix.com/login', undefined, 'netflix')).toBe(true)
    expect(isAllowedPlayerNavigation('https://www.primevideo.com/', undefined, 'netflix')).toBe(false)
    expect(isAllowedPlayerNavigation('https://accounts.google.com/', undefined, 'netflix')).toBe(false)
    expect(isAllowedPlayerPopup('https://www.netflix.com/login', 'netflix')).toBe(true)
    expect(isAllowedPlayerPopup('https://netflix.com/login', 'netflix')).toBe(true)
    expect(isAllowedPlayerPopup('https://www.primevideo.com/', 'netflix')).toBe(false)
    expect(safePlayerOrigin('https://www.netflix.com/title/1', undefined, 'netflix')).toBe('https://www.netflix.com')
    expect(safePlayerOrigin('https://netflix.com/login', undefined, 'netflix')).toBeNull()
    expect(safePlayerOrigin('https://www.primevideo.com/', undefined, 'netflix')).toBeNull()
    expect(isAllowedPlayerNavigation('https://auth.hulu.com/web/login', undefined, 'hulu')).toBe(true)
    expect(isAllowedPlayerPopup('https://auth.hulu.com/web/login', 'hulu')).toBe(true)
    expect(isAllowedPlayerNavigation('https://auth.hulu.com/web/login', undefined, 'netflix')).toBe(false)
  })

  it('keeps experimental navigation limited to each service exact documented host list', () => {
    const experimental = PLAYER_SERVICES.filter(service => service.validation === 'experimental')
    expect(experimental).toHaveLength(8)

    for (const service of experimental) {
      const host = new URL(service.homeUrl).hostname
      expect(service.serviceHosts).toContain(host)
      expect(service.requiresWidevine).toBe(true)
      expect(isAllowedPlayerNavigation(service.homeUrl, undefined, service.id)).toBe(true)
      expect(isAllowedPlayerNavigation('https://accounts.google.com/', undefined, service.id)).toBe(false)

      for (const serviceHost of service.serviceHosts) {
        expect(isAllowedPlayerNavigation(`https://${serviceHost}/`, undefined, service.id)).toBe(true)
      }

      for (const other of experimental) {
        if (other.id === service.id) {
          continue
        }

        for (const serviceHost of other.serviceHosts) {
          if (!(service.serviceHosts as readonly string[]).includes(serviceHost)) {
            expect(isAllowedPlayerNavigation(`https://${serviceHost}/`, undefined, service.id)).toBe(false)
          }
        }
      }

      for (const authHost of service.authHosts) {
        expect(isAllowedPlayerNavigation(`https://${authHost}/`, undefined, service.id)).toBe(true)
      }

      for (const mediaKeyHost of service.mediaKeyHosts) {
        expect(isAllowedPlayerMediaKeyHost(mediaKeyHost, service.id)).toBe(true)
      }
    }

    expect(isAllowedPlayerMediaKeyHost('music.youtube.com', 'youtube')).toBe(false)
    expect(isAllowedPlayerMediaKeyHost('music.youtube.com', 'tv')).toBe(false)
  })

  it('permits Prime Video Amazon account sign-in without granting Amazon playback authority', () => {
    const signIn = 'https://www.amazon.com/ap/signin'
    expect(isAllowedPlayerNavigation(signIn, undefined, 'prime-video')).toBe(true)
    expect(isAllowedPlayerPopup(signIn, 'prime-video')).toBe(true)
    expect(isAllowedPlayerFrame(signIn, undefined, 'prime-video')).toBe(true)
    expect(safePlayerOrigin(signIn, undefined, 'prime-video')).toBeNull()
    expect(isAllowedPlayerMediaKeyHost('www.amazon.com', 'prime-video')).toBe(false)

    for (const url of [
      'http://www.amazon.com/ap/signin',
      'https://amazon.com/ap/signin',
      'https://signin.amazon.com/',
      'https://www.amazon.com.evil.example/'
    ]) {
      expect(isAllowedPlayerNavigation(url, undefined, 'prime-video')).toBe(false)
      expect(isAllowedPlayerPopup(url, 'prime-video')).toBe(false)
    }

    expect(isAllowedPlayerNavigation(signIn, undefined, 'netflix')).toBe(false)
  })

  it('permits Hulu reCAPTCHA frames only on the documented HTTPS paths', () => {
    for (const host of ['www.google.com', 'recaptcha.google.com']) {
      const url = `https://${host}/recaptcha/api2/anchor?example=fixture`
      expect(isAllowedPlayerFrame(url, undefined, 'hulu')).toBe(true)
      expect(isAllowedPlayerNavigation(url, undefined, 'hulu')).toBe(false)
      expect(isAllowedPlayerPopup(url, 'hulu')).toBe(false)
      expect(safePlayerOrigin(url, undefined, 'hulu')).toBeNull()
      expect(isAllowedPlayerMediaKeyHost(host, 'hulu')).toBe(false)
      expect(isAllowedPlayerFrame(url, undefined, 'netflix')).toBe(false)
    }

    for (const url of [
      'http://www.google.com/recaptcha/api2/anchor',
      'https://www.google.com/search',
      'https://www.google.com/recaptcha-lookalike/anchor',
      'https://www.google.com/recaptcha/../search',
      'https://www.google.com.evil.example/recaptcha/anchor',
      'https://www.google.com:444/recaptcha/anchor',
      'https://user@www.google.com/recaptcha/anchor'
    ]) {
      expect(isAllowedPlayerFrame(url, undefined, 'hulu')).toBe(false)
    }
  })
})

describe('clampPlayerBounds', () => {
  it('clips negative origins and dimensions that exceed the host content area', () => {
    expect(clampPlayerBounds({ x: -9, y: 12.6, width: 900, height: 400 }, { width: 800, height: 300 })).toEqual({
      x: 0,
      y: 13,
      width: 800,
      height: 287
    })
  })

  it('intersects an offscreen pane with the content area instead of shifting it over chat', () => {
    expect(clampPlayerBounds({ x: 740.7, y: 280, width: 90, height: 50 }, { width: 800, height: 300 })).toEqual({
      x: 741,
      y: 280,
      width: 59,
      height: 20
    })
  })
})
