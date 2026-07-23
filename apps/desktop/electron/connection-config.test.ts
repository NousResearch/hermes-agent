/**
 * Tests for electron/connection-config.ts.
 *
 * Run with: node --test electron/connection-config.test.ts
 * (Wire into npm test:desktop:platforms in package.json.)
 *
 * These are the pure helpers behind the remote-gateway connection settings:
 * URL normalization, WS-URL construction (token vs OAuth ticket), auth-mode
 * classification from /api/status, the coerce-time auth-mode resolution rules,
 * and the OAuth session-cookie detector.
 */

import assert from 'node:assert/strict'

import { test } from 'vitest'

import {
  AT_COOKIE_VARIANTS,
  authModeFromStatus,
  buildGatewayWsUrl,
  buildGatewayWsUrlWithTicket,
  connectionScopeKey,
  cookiesHaveLiveSession,
  cookiesHavePrivySession,
  cookiesHaveSession,
  gatewayTicketFailure,
  gatewayWsUrlIpcResult,
  isGatewayAuthRejection,
  localProfileEntry,
  modeIsRemoteLike,
  normalizeRemoteBaseUrl,
  normalizeSshConfig,
  normAuthMode,
  pathWithGlobalRemoteProfile,
  profileHasRemoteConnection,
  profileRemoteOverride,
  profileSshOverride,
  resolveAuthMode,
  resolveTestWsUrl,
  RT_COOKIE_VARIANTS,
  savedProfileSsh,
  tokenPreview
} from './connection-config'

// --- connectionScopeKey / normAuthMode ---

test('connectionScopeKey trims to a name or null for the global scope', () => {
  assert.equal(connectionScopeKey('  coder '), 'coder')
  assert.equal(connectionScopeKey(''), null)
  assert.equal(connectionScopeKey(null), null)
  assert.equal(connectionScopeKey(undefined), null)
})

test('normAuthMode coerces to token unless explicitly oauth', () => {
  assert.equal(normAuthMode('oauth'), 'oauth')
  assert.equal(normAuthMode('token'), 'token')
  assert.equal(normAuthMode(undefined), 'token')
  assert.equal(normAuthMode('weird'), 'token')
})

// --- modeIsRemoteLike ---

test('modeIsRemoteLike is true for remote and cloud, false otherwise', () => {
  // cloud resolves to a remote backend under the hood (Q6), so every resolution
  // site treats it like remote.
  assert.equal(modeIsRemoteLike('remote'), true)
  assert.equal(modeIsRemoteLike('cloud'), true)
  assert.equal(modeIsRemoteLike('local'), false)
  assert.equal(modeIsRemoteLike(undefined), false)
  assert.equal(modeIsRemoteLike(null), false)
  assert.equal(modeIsRemoteLike('weird'), false)
})

// --- profileRemoteOverride ---

test('profileRemoteOverride returns null when no profile is given', () => {
  const config = { profiles: { coder: { mode: 'remote', url: 'https://x' } } }
  assert.equal(profileRemoteOverride(config, ''), null)
  assert.equal(profileRemoteOverride(config, null), null)
  assert.equal(profileRemoteOverride(config, undefined), null)
})

test('profileRemoteOverride returns null when the profile has no entry', () => {
  const config = { profiles: { coder: { mode: 'remote', url: 'https://x' } } }
  assert.equal(profileRemoteOverride(config, 'writer'), null)
})

test('profileRemoteOverride ignores local or url-less profile entries', () => {
  assert.equal(profileRemoteOverride({ profiles: { p: { mode: 'local', url: 'https://x' } } }, 'p'), null)
  assert.equal(profileRemoteOverride({ profiles: { p: { mode: 'remote', url: '' } } }, 'p'), null)
  assert.equal(profileRemoteOverride({ profiles: { p: { mode: 'remote' } } }, 'p'), null)
})

test('profileRemoteOverride returns the per-profile remote with defaulted auth mode', () => {
  const config = {
    profiles: {
      coder: { mode: 'remote', url: '  https://coder.example.com/hermes  ', token: { value: 'sek' } }
    }
  }

  assert.deepEqual(profileRemoteOverride(config, 'coder'), {
    url: 'https://coder.example.com/hermes',
    authMode: 'token',
    token: { value: 'sek' }
  })
})

test('profileRemoteOverride preserves an explicit oauth auth mode', () => {
  const config = { profiles: { coder: { mode: 'remote', url: 'https://x', authMode: 'oauth' } } }
  assert.equal(profileRemoteOverride(config, 'coder').authMode, 'oauth')
})

test('profileRemoteOverride treats a cloud entry as a remote override', () => {
  // A 'cloud' per-profile entry resolves to the same remote backend a 'remote'
  // entry would (Q6) — the override must be returned, not dropped.
  const config = {
    profiles: {
      coder: { mode: 'cloud', url: 'https://agent-1.agents.nousresearch.com', authMode: 'oauth' }
    }
  }

  assert.deepEqual(profileRemoteOverride(config, 'coder'), {
    url: 'https://agent-1.agents.nousresearch.com',
    authMode: 'oauth',
    token: undefined
  })
})

test('profileRemoteOverride tolerates a missing/!object profiles map', () => {
  assert.equal(profileRemoteOverride({}, 'coder'), null)
  assert.equal(profileRemoteOverride({ profiles: null }, 'coder'), null)
  assert.equal(profileRemoteOverride(null, 'coder'), null)
})

test('SSH remains separate from URL-shaped remote modes', () => {
  assert.equal(modeIsRemoteLike('ssh'), false)
  const config = { profiles: { coder: { mode: 'ssh', host: 'alice@box:2222', keyPath: '/key' } } }
  assert.equal(profileRemoteOverride(config, 'coder'), null)
  assert.deepEqual(profileSshOverride(config, 'coder'), {
    mode: 'ssh',
    host: 'box',
    user: 'alice',
    port: 2222,
    keyPath: '/key'
  })
})

test('normalizeSshConfig handles IPv6 and strict port bounds', () => {
  assert.deepEqual(normalizeSshConfig({ mode: 'ssh', host: '::1', port: 22 }), {
    mode: 'ssh',
    host: '::1'
  })
  assert.deepEqual(normalizeSshConfig({ mode: 'ssh', host: '[::1]:2222' }), {
    mode: 'ssh',
    host: '::1',
    port: 2222
  })
  assert.deepEqual(normalizeSshConfig({ mode: 'ssh', host: 'box', port: '2222junk' }), {
    mode: 'ssh',
    host: 'box'
  })
  assert.deepEqual(normalizeSshConfig({ mode: 'ssh', host: 'box', port: 65536 }), {
    mode: 'ssh',
    host: 'box'
  })
})

test('localProfileEntry preserves inactive SSH drafts but drops Cloud state', () => {
  const ssh = { mode: 'ssh', host: 'box', user: 'alice', remoteHermesPath: '/hermes' }
  assert.deepEqual(localProfileEntry(ssh), { mode: 'local', savedSsh: ssh })
  assert.deepEqual(localProfileEntry({ mode: 'local', savedSsh: ssh }), {
    mode: 'local',
    savedSsh: ssh
  })
  assert.equal(localProfileEntry({ mode: 'cloud', url: 'https://agent' }), null)
})

test('saved SSH drafts are inactive and explicit overrides take precedence', () => {
  const saved = { mode: 'ssh', host: 'saved' }
  const config: any = { profiles: { coder: { mode: 'local', savedSsh: saved } } }
  assert.deepEqual(savedProfileSsh(config, 'coder'), saved)
  assert.equal(profileSshOverride(config, 'coder'), null)
  assert.equal(profileHasRemoteConnection(config, 'coder'), false)

  config.profiles.coder = { mode: 'ssh', host: 'active' }
  assert.deepEqual(profileSshOverride(config, 'coder'), { mode: 'ssh', host: 'active' })
  assert.equal(profileHasRemoteConnection(config, 'coder'), true)
})

// --- pathWithGlobalRemoteProfile ---

test('pathWithGlobalRemoteProfile appends profile in global remote mode', () => {
  assert.equal(
    pathWithGlobalRemoteProfile('/api/model/info', 'iris', {
      globalRemote: true,
      profileRemoteOverride: false
    }),
    '/api/model/info?profile=iris'
  )
})

test('pathWithGlobalRemoteProfile preserves existing query params', () => {
  assert.equal(
    pathWithGlobalRemoteProfile('/api/model/options?force=1', 'iris', {
      globalRemote: true,
      profileRemoteOverride: false
    }),
    '/api/model/options?force=1&profile=iris'
  )
})

test('pathWithGlobalRemoteProfile does not replace an explicit profile query', () => {
  assert.equal(
    pathWithGlobalRemoteProfile('/api/model/info?profile=default', 'iris', {
      globalRemote: true,
      profileRemoteOverride: false
    }),
    '/api/model/info?profile=default'
  )
})

test('pathWithGlobalRemoteProfile skips local and per-profile remote override paths', () => {
  assert.equal(
    pathWithGlobalRemoteProfile('/api/model/info', 'iris', {
      globalRemote: false,
      profileRemoteOverride: false
    }),
    '/api/model/info'
  )
  assert.equal(
    pathWithGlobalRemoteProfile('/api/model/info', 'iris', {
      globalRemote: true,
      profileRemoteOverride: true
    }),
    '/api/model/info'
  )
})

test('pathWithGlobalRemoteProfile skips empty profile/path safely', () => {
  assert.equal(
    pathWithGlobalRemoteProfile('/api/model/info', '', {
      globalRemote: true,
      profileRemoteOverride: false
    }),
    '/api/model/info'
  )
  assert.equal(
    pathWithGlobalRemoteProfile('', 'iris', {
      globalRemote: true,
      profileRemoteOverride: false
    }),
    ''
  )
})

// --- normalizeRemoteBaseUrl ---

test('normalizeRemoteBaseUrl strips trailing slashes, hash, and query', () => {
  assert.equal(normalizeRemoteBaseUrl('https://gw.example.com/'), 'https://gw.example.com')
  assert.equal(normalizeRemoteBaseUrl('https://gw.example.com/hermes/'), 'https://gw.example.com/hermes')
  assert.equal(normalizeRemoteBaseUrl('https://gw.example.com/hermes?x=1#frag'), 'https://gw.example.com/hermes')
})

test('normalizeRemoteBaseUrl preserves a path prefix', () => {
  assert.equal(normalizeRemoteBaseUrl('https://host/hermes'), 'https://host/hermes')
})

test('normalizeRemoteBaseUrl rejects empty input', () => {
  assert.throws(() => normalizeRemoteBaseUrl(''), /required/)
  assert.throws(() => normalizeRemoteBaseUrl('   '), /required/)
})

test('normalizeRemoteBaseUrl rejects non-http(s) protocols', () => {
  assert.throws(() => normalizeRemoteBaseUrl('ftp://host'), /http:\/\/ or https:\/\//)
  assert.throws(() => normalizeRemoteBaseUrl('file:///etc/passwd'), /http:\/\/ or https:\/\//)
})

test('normalizeRemoteBaseUrl rejects garbage', () => {
  assert.throws(() => normalizeRemoteBaseUrl('not a url'), /not valid/)
})

// --- buildGatewayWsUrl (token) ---

test('buildGatewayWsUrl uses wss for https and bakes the token', () => {
  assert.equal(buildGatewayWsUrl('https://gw.example.com', 'tok123'), 'wss://gw.example.com/api/ws?token=tok123')
})

test('buildGatewayWsUrl uses ws for http', () => {
  assert.equal(buildGatewayWsUrl('http://127.0.0.1:9119', 'abc'), 'ws://127.0.0.1:9119/api/ws?token=abc')
})

test('buildGatewayWsUrl honors a path prefix', () => {
  assert.equal(buildGatewayWsUrl('https://host/hermes', 't'), 'wss://host/hermes/api/ws?token=t')
})

test('buildGatewayWsUrl url-encodes the token', () => {
  assert.equal(buildGatewayWsUrl('https://host', 'a/b c+d'), 'wss://host/api/ws?token=a%2Fb%20c%2Bd')
})

// --- buildGatewayWsUrlWithTicket (oauth) ---

test('buildGatewayWsUrlWithTicket uses ?ticket= not ?token=', () => {
  const url = buildGatewayWsUrlWithTicket('https://gw.example.com/hermes', 'tkt-9')
  assert.equal(url, 'wss://gw.example.com/hermes/api/ws?ticket=tkt-9')
  assert.ok(!url.includes('token='))
})

test('buildGatewayWsUrlWithTicket url-encodes the ticket', () => {
  assert.equal(buildGatewayWsUrlWithTicket('https://host', 'a+b/c'), 'wss://host/api/ws?ticket=a%2Bb%2Fc')
})

// --- authModeFromStatus ---

test('authModeFromStatus returns oauth when auth_required is true', () => {
  assert.equal(authModeFromStatus({ auth_required: true, auth_providers: ['nous'] }), 'oauth')
})

test('authModeFromStatus returns token when auth_required is false/missing', () => {
  assert.equal(authModeFromStatus({ auth_required: false }), 'token')
  assert.equal(authModeFromStatus({}), 'token')
  assert.equal(authModeFromStatus(null), 'token')
  assert.equal(authModeFromStatus(undefined), 'token')
})

// --- resolveAuthMode ---

test('resolveAuthMode: explicit input wins over existing', () => {
  assert.equal(resolveAuthMode('oauth', 'token'), 'oauth')
  assert.equal(resolveAuthMode('token', 'oauth'), 'token')
})

test('resolveAuthMode: falls back to existing when input absent', () => {
  assert.equal(resolveAuthMode(undefined, 'oauth'), 'oauth')
  assert.equal(resolveAuthMode(undefined, 'token'), 'token')
  assert.equal(resolveAuthMode('', 'oauth'), 'oauth')
})

test('resolveAuthMode: defaults to token when nothing is set', () => {
  assert.equal(resolveAuthMode(undefined, undefined), 'token')
  assert.equal(resolveAuthMode(null, null), 'token')
})

test('resolveAuthMode: ignores unknown values, defaults to token', () => {
  assert.equal(resolveAuthMode('bogus', 'also-bogus'), 'token')
})

// --- cookiesHaveSession ---

test('cookiesHaveSession detects the bare access-token cookie', () => {
  assert.equal(cookiesHaveSession([{ name: 'hermes_session_at', value: 'x' }]), true)
})

test('cookiesHaveSession detects the __Host- and __Secure- prefixed variants', () => {
  assert.equal(cookiesHaveSession([{ name: '__Host-hermes_session_at', value: 'x' }]), true)
  assert.equal(cookiesHaveSession([{ name: '__Secure-hermes_session_at', value: 'x' }]), true)
})

test('cookiesHaveSession is false for an empty value', () => {
  assert.equal(cookiesHaveSession([{ name: 'hermes_session_at', value: '' }]), false)
})

test('cookiesHaveSession ignores unrelated cookies (AT-only by design)', () => {
  // cookiesHaveSession is deliberately access-token-only — a lone RT cookie
  // is NOT an access token, so this returns false. Connectivity callers must
  // use cookiesHaveLiveSession instead (see below).
  assert.equal(cookiesHaveSession([{ name: 'hermes_session_rt', value: 'x' }]), false)
  assert.equal(cookiesHaveSession([{ name: 'other', value: 'x' }]), false)
})

test('cookiesHaveSession handles non-arrays', () => {
  assert.equal(cookiesHaveSession(null), false)
  assert.equal(cookiesHaveSession(undefined), false)
  assert.equal(cookiesHaveSession([]), false)
})

test('AT_COOKIE_VARIANTS covers all three deploy shapes', () => {
  assert.deepEqual(AT_COOKIE_VARIANTS, ['__Host-hermes_session_at', '__Secure-hermes_session_at', 'hermes_session_at'])
})

test('RT_COOKIE_VARIANTS covers all three deploy shapes', () => {
  assert.deepEqual(RT_COOKIE_VARIANTS, ['__Host-hermes_session_rt', '__Secure-hermes_session_rt', 'hermes_session_rt'])
})

// --- cookiesHaveLiveSession (AT or RT — the connectivity check) ---

test('cookiesHaveLiveSession is true for a live access-token cookie', () => {
  assert.equal(cookiesHaveLiveSession([{ name: 'hermes_session_at', value: 'x' }]), true)
  assert.equal(cookiesHaveLiveSession([{ name: '__Host-hermes_session_at', value: 'x' }]), true)
  assert.equal(cookiesHaveLiveSession([{ name: '__Secure-hermes_session_at', value: 'x' }]), true)
})

test('cookiesHaveLiveSession is true for an RT cookie even with NO access-token cookie', () => {
  // This is the bug-fix case: the AT cookie has lapsed (dropped from the jar)
  // but the 24h RT cookie is still alive. The session is still connectable —
  // the gateway rotates a fresh AT from the RT on the next request.
  assert.equal(cookiesHaveLiveSession([{ name: 'hermes_session_rt', value: 'x' }]), true)
  assert.equal(cookiesHaveLiveSession([{ name: '__Host-hermes_session_rt', value: 'x' }]), true)
  assert.equal(cookiesHaveLiveSession([{ name: '__Secure-hermes_session_rt', value: 'x' }]), true)
})

test('cookiesHaveLiveSession is true when both AT and RT are present', () => {
  assert.equal(
    cookiesHaveLiveSession([
      { name: 'hermes_session_at', value: 'a' },
      { name: 'hermes_session_rt', value: 'r' }
    ]),
    true
  )
})

test('cookiesHaveLiveSession is false for empty values', () => {
  assert.equal(cookiesHaveLiveSession([{ name: 'hermes_session_at', value: '' }]), false)
  assert.equal(cookiesHaveLiveSession([{ name: 'hermes_session_rt', value: '' }]), false)
  assert.equal(
    cookiesHaveLiveSession([
      { name: 'hermes_session_at', value: '' },
      { name: 'hermes_session_rt', value: '' }
    ]),
    false
  )
})

test('cookiesHaveLiveSession is false for unrelated cookies and non-arrays', () => {
  assert.equal(cookiesHaveLiveSession([{ name: 'other', value: 'x' }]), false)
  assert.equal(cookiesHaveLiveSession(null), false)
  assert.equal(cookiesHaveLiveSession(undefined), false)
  assert.equal(cookiesHaveLiveSession([]), false)
})

// --- cookiesHavePrivySession (Nous portal / Privy auth, NOT gateway cookies) ---

test('cookiesHavePrivySession detects the privy-token access cookie', () => {
  assert.equal(cookiesHavePrivySession([{ name: 'privy-token', value: 'jwt' }]), true)
})

test('cookiesHavePrivySession detects __Host-/__Secure- prefixes and the legacy privy-session name', () => {
  assert.equal(cookiesHavePrivySession([{ name: '__Host-privy-token', value: 'x' }]), true)
  assert.equal(cookiesHavePrivySession([{ name: '__Secure-privy-token', value: 'x' }]), true)
  assert.equal(cookiesHavePrivySession([{ name: 'privy-session', value: 'x' }]), true)
})

test('cookiesHavePrivySession is false for an empty value', () => {
  assert.equal(cookiesHavePrivySession([{ name: 'privy-token', value: '' }]), false)
})

test('cookiesHavePrivySession does NOT treat hermes gateway cookies as a portal session', () => {
  // The whole point of Q7: a gateway session cookie is NOT a portal sign-in.
  assert.equal(cookiesHavePrivySession([{ name: 'hermes_session_at', value: 'x' }]), false)
  assert.equal(cookiesHavePrivySession([{ name: '__Host-hermes_session_rt', value: 'x' }]), false)
})

test('cookiesHavePrivySession is false for unrelated cookies and non-arrays', () => {
  assert.equal(cookiesHavePrivySession([{ name: 'other', value: 'x' }]), false)
  assert.equal(cookiesHavePrivySession(null), false)
  assert.equal(cookiesHavePrivySession(undefined), false)
  assert.equal(cookiesHavePrivySession([]), false)
})

// --- tokenPreview ---

test('tokenPreview returns null for empty', () => {
  assert.equal(tokenPreview(''), null)
  assert.equal(tokenPreview(null), null)
})

test('tokenPreview returns set for short tokens', () => {
  assert.equal(tokenPreview('12345678'), 'set')
})

test('tokenPreview returns a masked suffix for long tokens', () => {
  assert.equal(tokenPreview('abcdefghijklmnop'), '...klmnop')
})

// --- resolveTestWsUrl ---
//
// The "Test remote" button must exercise the same WS transport the app uses,
// and must FAIL (not skip) when an OAuth session can't mint a ws-ticket — that
// is the exact false-positive PR #39098 set out to eliminate.

test('resolveTestWsUrl (token mode) builds a ?token= URL the WS probe can use', async () => {
  const url = await resolveTestWsUrl('https://gw.example.com', 'token', 'tok123')
  assert.equal(url, 'wss://gw.example.com/api/ws?token=tok123')
})

test('resolveTestWsUrl (token mode, no token) returns null — genuine skip', async () => {
  assert.equal(await resolveTestWsUrl('https://gw.example.com', 'token', null), null)
})

test('resolveTestWsUrl (oauth, mint ok) builds a ?ticket= URL', async () => {
  const url = await resolveTestWsUrl('https://gw.example.com', 'oauth', null, {
    mintTicket: async () => 'tkt-9'
  })

  assert.equal(url, 'wss://gw.example.com/api/ws?ticket=tkt-9')
})

test('resolveTestWsUrl (oauth, auth rejected) requests sign-in and does not skip WS validation', async () => {
  const cause = Object.assign(new Error('ticket mint failed'), { statusCode: 401 })

  await assert.rejects(
    () =>
      resolveTestWsUrl('https://gw.example.com', 'oauth', null, {
        mintTicket: async () => {
          throw cause
        }
      }),
    (err: any) => {
      // Actionable, points the user at re-auth, and preserves the cause + flag
      // the boot overlay uses to offer a sign-in prompt.
      assert.match(err.message, /WebSocket ticket/i)
      assert.match(err.message, /sign in again/i)
      assert.equal(err.needsOauthLogin, true)
      assert.ok(err.cause instanceof Error)

      return true
    }
  )
})

test('resolveTestWsUrl (oauth, transport failure) remains a retryable connection error', async () => {
  const cause = new Error('socket timed out')

  await assert.rejects(
    () =>
      resolveTestWsUrl('https://gw.example.com', 'oauth', null, {
        mintTicket: async () => {
          throw cause
        }
      }),
    (err: any) => {
      assert.match(err.message, /could not mint a WebSocket ticket/i)
      assert.equal(err.needsOauthLogin, undefined)
      assert.equal(err.cause, cause)

      return true
    }
  )
})

test('gateway ticket failures classify only explicit auth rejection statuses as reauth', () => {
  assert.equal(isGatewayAuthRejection({ statusCode: 401 }), true)
  assert.equal(isGatewayAuthRejection({ statusCode: 403 }), true)
  assert.equal(isGatewayAuthRejection({ needsOauthLogin: true }), true)
  assert.equal(isGatewayAuthRejection({ statusCode: 500 }), false)
  assert.equal(isGatewayAuthRejection(new Error('network timeout')), false)

  const serverFailure = gatewayTicketFailure(new Error('network timeout'), 'sign in', 'retry connection') as any
  assert.equal(serverFailure.message, 'retry connection')
  assert.equal(serverFailure.needsOauthLogin, undefined)
})

test('gateway WS URL IPC result serializes success and the auth-vs-transport matrix', async () => {
  assert.deepEqual(await gatewayWsUrlIpcResult(async () => 'wss://gateway.example.com/api/ws?ticket=fresh'), {
    ok: true,
    wsUrl: 'wss://gateway.example.com/api/ws?ticket=fresh'
  })

  for (const statusCode of [401, 403]) {
    const error = Object.assign(new Error(`${statusCode}: rejected`), { statusCode })

    assert.deepEqual(await gatewayWsUrlIpcResult(async () => Promise.reject(error)), {
      error: `${statusCode}: rejected`,
      needsOauthLogin: true,
      ok: false
    })
  }

  for (const error of [
    Object.assign(new Error('500: unavailable'), { statusCode: 500 }),
    new Error('Timed out connecting to Hermes backend after 8000ms'),
    Object.assign(new Error('socket reset'), { code: 'ECONNRESET' })
  ]) {
    assert.deepEqual(await gatewayWsUrlIpcResult(async () => Promise.reject(error)), {
      error: error.message,
      ok: false
    })
  }
})

test('resolveTestWsUrl (oauth) requires a mintTicket function', async () => {
  await assert.rejects(
    () => resolveTestWsUrl('https://gw.example.com', 'oauth', null),
    /mintTicket function is required/
  )
})
