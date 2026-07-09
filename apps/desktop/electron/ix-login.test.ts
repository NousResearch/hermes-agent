/**
 * Tests for electron/ix-login.ts — the native (no-webview) admin portal OTP
 * login: request-code and verify-code against /api/auth/otp/*.
 *
 * Run with: node --test electron/ix-login.test.ts
 *
 * Why this matters: the verify step is what mints the httpOnly portal
 * session cookie that unlocks the native chat / MCP surfaces, so it must
 * fail loudly (portal error text, not generic messages) and must never
 * report success unless the portal explicitly confirmed {ok:true}.
 */

import assert from 'node:assert/strict'
import test from 'node:test'

import { ixLoginSendOtp, ixLoginVerifyOtp } from './ix-login'

const PORTAL = 'https://admin.example.test'

function fakeFetch(handler: (url: string, init?: RequestInit) => { body?: unknown; status?: number }): typeof fetch {
  return (async (input: RequestInfo | URL, init?: RequestInit) => {
    const { body = {}, status = 200 } = handler(String(input), init)

    return new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } })
  }) as typeof fetch
}

/* ── send ─────────────────────────────────────────────────────────────────── */

test('sendOtp posts the email and returns the challenge', async () => {
  let seenUrl = ''
  let seenBody = ''

  const result = await ixLoginSendOtp(
    PORTAL,
    ' user@intelli-verse-x.ai ',
    fakeFetch((url, init) => {
      seenUrl = url
      seenBody = String(init?.body)

      return { body: { challenge: 'ch-1', expiresAt: 123 } }
    })
  )

  assert.equal(seenUrl, `${PORTAL}/api/auth/otp/send`)
  assert.deepEqual(JSON.parse(seenBody), { email: 'user@intelli-verse-x.ai' })
  assert.equal(result.challenge, 'ch-1')
})

test('sendOtp rejects invalid emails without any network call', async () => {
  await assert.rejects(
    ixLoginSendOtp(PORTAL, 'not-an-email', () => {
      throw new Error('must not be called')
    }),
    /valid email/i
  )
})

test('sendOtp surfaces the portal error text (cooldown, etc.)', async () => {
  await assert.rejects(
    ixLoginSendOtp(
      PORTAL,
      'user@intelli-verse-x.ai',
      fakeFetch(() => ({ status: 429, body: { error: 'Please wait 25s before requesting another code' } }))
    ),
    /wait 25s/
  )
})

test('sendOtp fails when the portal returns no challenge', async () => {
  await assert.rejects(
    ixLoginSendOtp(PORTAL, 'user@intelli-verse-x.ai', fakeFetch(() => ({ body: {} }))),
    /no OTP challenge/
  )
})

/* ── verify ───────────────────────────────────────────────────────────────── */

test('verifyOtp posts email+code+challenge with credentials include', async () => {
  let seenInit: RequestInit | undefined

  await ixLoginVerifyOtp(
    PORTAL,
    { email: 'user@intelli-verse-x.ai', code: ' 123456 ', challenge: 'ch-1' },
    fakeFetch((url, init) => {
      assert.equal(url, `${PORTAL}/api/auth/otp/verify`)
      seenInit = init

      return { body: { ok: true } }
    })
  )

  assert.equal(seenInit?.credentials, 'include')
  assert.deepEqual(JSON.parse(String(seenInit?.body)), {
    email: 'user@intelli-verse-x.ai',
    code: '123456',
    challenge: 'ch-1'
  })
})

test('verifyOtp rejects non-6-digit codes without any network call', async () => {
  await assert.rejects(
    ixLoginVerifyOtp(PORTAL, { email: 'a@b.co', code: '12345', challenge: 'ch' }, () => {
      throw new Error('must not be called')
    }),
    /6 digits/
  )
})

test('verifyOtp surfaces the portal rejection (wrong code, no access)', async () => {
  await assert.rejects(
    ixLoginVerifyOtp(
      PORTAL,
      { email: 'a@b.co', code: '123456', challenge: 'ch' },
      fakeFetch(() => ({ status: 403, body: { error: 'No portal access is configured for this account' } }))
    ),
    /No portal access/
  )
})

test('verifyOtp never succeeds without an explicit {ok:true}', async () => {
  await assert.rejects(
    ixLoginVerifyOtp(PORTAL, { email: 'a@b.co', code: '123456', challenge: 'ch' }, fakeFetch(() => ({ body: {} }))),
    /did not confirm/
  )
})
