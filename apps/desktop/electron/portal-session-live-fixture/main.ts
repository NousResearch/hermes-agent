import assert from 'node:assert/strict'
import { createServer } from 'node:http'
import type { AddressInfo } from 'node:net'
import { join } from 'node:path'

import { app, BrowserWindow, session } from 'electron'

import { createPortalSession } from '../portal-session'

// Real Chromium cookies, HTTP redirects and renderer requests. All credentials
// are synthetic and the portal is loopback-only; no existing Electron jar is used.
app.setPath('userData', join(process.argv[2], 'user-data'))
app.on('window-all-closed', () => {})

async function run() {
  await app.whenReady()
  const jar = session.fromPartition('persist:portal-test')
  let provider: 'privy' | 'nas' = 'privy'
  let mode: 'login' | 'renew' | 'reject' = 'login'
  let version = 0
  let refreshes = 0

  let completeRequested = () => {}

  let releaseLogin = () => {}
  let loginGate = Promise.resolve()
  const accessName = () => (provider === 'nas' ? 'nas-session' : 'privy-token')
  const refreshName = () => (provider === 'nas' ? 'nas-refresh' : 'privy-refresh-token')
  const cookie = (name: string, value: string) => `${name}=${value}; Path=/; HttpOnly; SameSite=Lax; Max-Age=3600`

  const accessCookies = () => [
    cookie(accessName(), `access-${++version}`),
    cookie(refreshName(), `refresh-${version}`),
    `${provider === 'nas' ? 'privy-token' : 'nas-session'}=; Path=/; Max-Age=0`,
    `${provider === 'nas' ? 'privy-refresh-token' : 'nas-refresh'}=; Path=/; Max-Age=0`
  ]

  const server = createServer(async (req, res) => {
    if (req.url === '/api/agents') {
      res.statusCode = req.headers.cookie?.includes(`${accessName()}=access-${version}`) ? 200 : 401
      res.end(JSON.stringify({ agents: [] }))

      return
    }

    if (req.url === '/complete') {
      completeRequested()
      await loginGate
      res.setHeader('Set-Cookie', accessCookies())
      res.end('ok')

      return
    }

    if (req.url === '/refresh') {
      refreshes++
      res.setHeader('Set-Cookie', accessCookies())
      res.writeHead(302, { Location: '/ready' }).end()

      return
    }

    if (req.url === '/' && mode === 'renew') {
      res.writeHead(302, { Location: '/refresh' }).end()

      return
    }

    res.setHeader('Content-Type', 'text/html')

    if (req.url === '/' && mode === 'login') {
      res.setHeader('Set-Cookie', cookie(refreshName(), 'pending'))
      res.end('<!doctype html><script>fetch("/complete")</script>')
    } else {
      res.end('<!doctype html><p>Portal</p>')
    }
  })

  await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve))
  const base = `http://127.0.0.1:${(server.address() as AddressInfo).port}`

  const portal = createPortalSession({
    isReady: () => app.isReady(),
    getOauthSession: () => jar,
    resolvePortalBaseUrl: () => base,
    warmOauthCookieStore: () => jar.cookies.flushStore(),
    createWindow: options => new BrowserWindow({ ...options, show: false }),
    rememberLog: message => console.log(message)
  })

  try {
    // Real interactive completion must wait past refresh-only state, for both
    // providers, and NAS login must survive removal of all Privy cookies.
    for (const next of ['privy', 'nas'] as const) {
      provider = next
      mode = 'login'
      await jar.clearStorageData()
      loginGate = new Promise(resolve => {
        releaseLogin = resolve
      })

      const requested = new Promise<void>(resolve => {
        completeRequested = resolve
      })

      let completed = false

      const login = portal.openPortalLoginWindow().then(() => {
        completed = true
      })

      await requested
      assert.equal(await portal.hasLivePortalSession(), true)
      assert.equal(await portal.hasPortalAccessToken(), false)
      assert.equal(completed, false)
      releaseLogin()
      await login
      assert.equal(await portal.hasPortalAccessToken(), true)
      assert.equal((await jar.fetch(`${base}/api/agents`)).status, 200)

      // Expired access + durable refresh, including concurrent discovery/cascade.
      await jar.cookies.remove(base, accessName())
      mode = 'renew'
      const before = refreshes

      const [first, second] = await Promise.all([
        portal.renewPortalAccessSilently(),
        portal.renewPortalAccessSilently()
      ])

      assert.equal(first, true)
      assert.equal(second, true)
      assert.equal(refreshes, before + 1)
      assert.equal((await jar.fetch(`${base}/api/agents`)).status, 200)
    }

    // Dual-mint, followed by both provider transitions, with no cached identity.
    await jar.cookies.set({ url: base, name: 'privy-token', value: 'old-privy' })
    assert.equal(await portal.hasPortalAccessToken(), true)

    for (const next of ['privy', 'nas', 'privy'] as const) {
      provider = next
      mode = 'renew'
      assert.equal(await portal.renewPortalAccessSilently({ force: true }), true)
      assert.equal((await jar.fetch(`${base}/api/agents`)).status, 200)
    }

    // A server-rejected cookie must not short-circuit renewal as successful.
    mode = 'reject'
    await jar.cookies.set({ url: base, name: accessName(), value: 'rejected', httpOnly: true })
    assert.equal((await jar.fetch(`${base}/api/agents`)).status, 401)
    assert.equal(await portal.renewPortalAccessSilently({ force: true }), false)
    assert.equal(BrowserWindow.getAllWindows().length, 0)

    // The same rejected cookie must not close the interactive login before the
    // portal issues a new one; a forced renewal requested while an unforced one
    // is short-circuiting must still drive the portal.
    mode = 'login'
    loginGate = new Promise(resolve => {
      releaseLogin = resolve
    })

    const requestedAgain = new Promise<void>(resolve => {
      completeRequested = resolve
    })

    let completedAgain = false

    const loginAgain = portal.openPortalLoginWindow().then(() => {
      completedAgain = true
    })

    // A premature completion destroys the window before its page can reach
    // /complete, so race the two instead of waiting on the request alone.
    await Promise.race([requestedAgain, loginAgain])
    assert.equal(await portal.hasPortalAccessToken(), true)
    assert.equal(completedAgain, false)
    releaseLogin()
    await loginAgain
    assert.equal((await jar.fetch(`${base}/api/agents`)).status, 200)

    mode = 'renew'
    const beforeForced = refreshes

    const [unforced, forced] = await Promise.all([
      portal.renewPortalAccessSilently(),
      portal.renewPortalAccessSilently({ force: true })
    ])

    assert.equal(unforced, true)
    assert.equal(forced, true)
    assert.equal(refreshes, beforeForced + 1)

    await jar.clearStorageData()
    await jar.cookies.set({ url: base, name: 'next-auth-provider', value: 'workos' })
    assert.equal(await portal.hasLivePortalSession(), false)
    assert.equal(await portal.renewPortalAccessSilently(), false)
    console.log('PORTAL_SESSION_LIVE_OK')
  } finally {
    for (const window of BrowserWindow.getAllWindows()) {
      window.destroy()
    }

    server.closeAllConnections()
    await new Promise<void>(resolve => server.close(() => resolve()))
  }
}

void run().then(
  () => app.exit(0),
  error => {
    console.error(error)
    app.exit(1)
  }
)
