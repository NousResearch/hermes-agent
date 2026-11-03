import * as fs from 'node:fs'
import * as http from 'node:http'
import type { AddressInfo } from 'node:net'
import * as path from 'node:path'

import { buildAppEnv, createSandbox, launchDesktop } from './fixtures'
import { allowErrorBanners, expect, test } from './test'

type DesktopWindow = Window & {
  hermesDesktop: {
    getBootProgress: () => Promise<{ running: boolean; retryable?: boolean; statusCode?: number; error?: string }>
    getConnection: () => Promise<unknown>
  }
}

// Real Electron/preload/renderer against a loopback gateway whose session was
// lost during restart. No real credentials or installed user state are used.
for (const status of [401, 403]) {
  test(`unsigned gateway ${status} leaves recovery settings usable`, async () => {
    allowErrorBanners()
    const sandbox = createSandbox(`oauth-recovery-${status}`)
    let mints = 0
    let signedIn = false
    const server = http.createServer((req, res) => {
      req.resume()
      req.on('end', () => {
        const pathname = new URL(req.url ?? '/', 'http://localhost').pathname
        res.setHeader('Content-Type', 'application/json')
        if (pathname === '/api/status') {
          res.end(JSON.stringify({ auth_required: true, auth_flows: [], version: 'test' }))
        } else if (pathname === '/api/auth/providers') {
          res.end(JSON.stringify({ providers: [{ name: 'portal', supports_password: false }] }))
        } else if (pathname === '/login') {
          signedIn = true
          res.setHeader('Set-Cookie', 'hermes_session_at=fixture-session; Path=/; HttpOnly; SameSite=Lax')
          res.end('{}')
        } else if (signedIn && pathname === '/api/auth/ws-ticket') {
          mints += 1
          res.end(JSON.stringify({ ticket: `fixture-ticket-${mints}` }))
        } else if (signedIn && pathname === '/api/health') {
          res.end(JSON.stringify({ ok: true, status: 'ok' }))
        } else {
          if (pathname === '/api/auth/ws-ticket') {
            mints += 1
            if (mints === 1) {
              // A NAS restart can truncate a response after headers. Exercise
              // Electron's real IncomingMessage error, then recover on retry.
              res.setHeader('Content-Length', '1024')
              res.write('{"partial":')
              setTimeout(() => res.destroy(), 20)
              return
            }
          }
          res.statusCode = status
          res.end(JSON.stringify({ error: 'unauthenticated', reason: 'no_cookie' }))
        }
      })
    })
    await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve))
    const url = `http://127.0.0.1:${(server.address() as AddressInfo).port}`
    fs.writeFileSync(path.join(sandbox.userDataDir, 'connection.json'), JSON.stringify({
      mode: 'remote', remote: { url, authMode: 'oauth' }, profiles: {}
    }))
    let app: Awaited<ReturnType<typeof launchDesktop>>['app'] | undefined
    try {
      const launched = await launchDesktop(buildAppEnv(sandbox, {
        HERMES_DESKTOP_DEV_SERVER: ''
      }))
      app = launched.app
      const page = launched.page
      await expect(page.getByRole('button', { name: /gateway settings/i })).toBeVisible({ timeout: 60_000 })
      const snapshot = await page.evaluate(() => (window as unknown as DesktopWindow).hermesDesktop.getBootProgress())
      expect(snapshot).toMatchObject({ running: false, retryable: false, statusCode: status })
      expect(snapshot.error).toMatch(/not signed in/)
      expect(mints).toBeGreaterThan(0)
      const settledMints = mints
      await page.getByRole('button', { name: /gateway settings/i }).click()
      const back = page.getByRole('button', { name: /^back$/i })
      await expect(back).toBeVisible()
      const gatewayUrl = page.getByPlaceholder('https://gateway.example.com/hermes')
      await expect(gatewayUrl).toHaveValue(url)
      // Concurrent IPC readers must reuse the terminal failure, not republish
      // startup progress and unmount the settings form.
      await page.evaluate(async () => {
        await Promise.allSettled(Array.from({ length: 20 }, () => (window as unknown as DesktopWindow).hermesDesktop.getConnection()))
      })
      await expect(back).toBeVisible()
      await expect(gatewayUrl).toHaveValue(url)
      expect(mints).toBe(settledMints)
      await page.screenshot({ path: test.info().outputPath(`gateway-settings-${status}.png`) })
      await back.click()
      const signIn = page.getByRole('button', { name: /sign in/i })
      await expect(signIn).toBeEnabled()
      await signIn.click()
      await expect.poll(() => signedIn).toBe(true)
      await expect.poll(async () => {
        try {
          return await page.evaluate(() => (window as unknown as DesktopWindow).hermesDesktop.getConnection())
        } catch {
          return null
        }
      }).toMatchObject({ mode: 'remote', baseUrl: url })
      expect(mints).toBeGreaterThan(settledMints)
    } finally {
      await app?.close()
      server.closeAllConnections()
      await new Promise<void>(resolve => server.close(() => resolve()))
      sandbox.cleanup()
    }
  })
}
