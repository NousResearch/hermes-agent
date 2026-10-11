import assert from 'node:assert/strict'

import { test } from 'vitest'

import {
  buildOauthLoginTimeoutPage,
  loadOauthLoginTimeoutPage,
  shouldSurfaceInteractiveLoginFailure
} from './oauth-login-window-guard'

test('deadline page names the hung gateway and points at the backend, not sign-out', () => {
  const html = buildOauthLoginTimeoutPage({
    gatewayUrl: 'http://devbox.local:9119',
    timedOut: true
  })

  // #80733's second half: the user chased auth ("you are signed out") while
  // the server was hung — the page must say it is a gateway problem.
  assert.match(html, /gateway process is hung/)
  assert.match(html, /not a sign-out/)
  assert.match(html, /http:\/\/devbox\.local:9119/)
  assert.match(html, /Retry/)
})

test('load-failure page surfaces the error code', () => {
  const html = buildOauthLoginTimeoutPage({
    gatewayUrl: 'http://devbox.local:9119',
    errorCode: -102
  })

  assert.match(html, /could not be reached \(-102\)/)
  assert.match(html, /Retry/)
})

test('retry button navigates back to the gateway /login, not the data: page', () => {
  const html = buildOauthLoginTimeoutPage({ gatewayUrl: 'http://devbox.local:9119/' })

  // A data: page cannot recover with location.reload() — the button must go
  // back to the real interactive login URL.
  assert.match(html, /location\.replace\("http:\/\/devbox\.local:9119\/login"\)/)
  assert.doesNotMatch(html, /location\.reload\(\)/)
})

test('gatewayUrl cannot break out of the inline script block', () => {
  const html = buildOauthLoginTimeoutPage({
    gatewayUrl: 'http://devbox.local:9119</script><script>alert("pwned")</script>'
  })

  assert.doesNotMatch(html, /<\/script><script>/)
  assert.doesNotMatch(html, /alert\("pwned"\)/)
  assert.match(html, /\\u003c\/script/)
})

test('only real load failures surface — ERR_ABORTED navigation churn does not', () => {
  // OAuth sign-in is a navigation storm: provider redirects and the callback
  // bounce arrive as ERR_ABORTED (-3) and must not blank a working login.
  assert.equal(shouldSurfaceInteractiveLoginFailure({ code: -3, message: 'ERR_ABORTED -3' }), false)
  assert.equal(shouldSurfaceInteractiveLoginFailure(new Error('net::ERR_ABORTED (-3)')), false)

  // A hung/unreachable gateway is a real failure: DNS, connection reset,
  // timeout, TLS, or a loadURL rejection with no code at all.
  assert.equal(shouldSurfaceInteractiveLoginFailure({ code: -105, message: 'ERR_NAME_NOT_RESOLVED' }), true)
  assert.equal(shouldSurfaceInteractiveLoginFailure({ code: -7, message: 'ERR_TIMED_OUT' }), true)
  assert.equal(shouldSurfaceInteractiveLoginFailure(new Error('ERR_CONNECTION_REFUSED')), true)
  assert.equal(shouldSurfaceInteractiveLoginFailure('weird non-error'), true)
  assert.equal(shouldSurfaceInteractiveLoginFailure(null), true)
})

test('loadOauthLoginTimeoutPage loads a data: URL and swallows loadURL rejections', async () => {
  const loads: string[] = []

  const win = {
    loadURL: async (url: string) => {
      loads.push(url)
    }
  }

  await loadOauthLoginTimeoutPage(win, { gatewayUrl: 'http://box:9119', timedOut: true })

  assert.equal(loads.length, 1)
  assert.ok(loads[0].startsWith('data:text/html;charset=utf-8,'))

  // The error page itself must never become a second blank window.
  const failing = {
    loadURL: async () => {
      throw new Error('cannot load')
    }
  }

  await loadOauthLoginTimeoutPage(failing, {})
})
