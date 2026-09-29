// Real Chromium transport witness, launched with a disposable userData path.
// No windows, provider requests, persistent cookie jars, or user credentials.
import assert from 'node:assert/strict'
import http from 'node:http'
import type { AddressInfo } from 'node:net'

import { app, net, session } from 'electron'

import { resetKeepaliveTransports } from './api-transport'
import { attachPowerResumeRemoteRevalidation } from './remote-liveness'

app.setPath('userData', process.argv[2])
app.setPath('sessionData', process.argv[2])

async function run() {
  const jar = session.fromPartition('wake-gateway-witness')
  let staleSocket: http.IncomingMessage['socket']

  const server = http.createServer((req, res) => {
    if (req.socket === staleSocket) {
      console.log('STALE_POOLED_SOCKET_REUSED')

      return
    }

    staleSocket ??= req.socket
    res.setHeader('Content-Type', 'application/json')
    res.end(JSON.stringify({ cookie: req.headers.cookie, port: req.socket.remotePort }))
  })

  await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve))
  const base = `http://127.0.0.1:${(server.address() as AddressInfo).port}`

  const get = () =>
    new Promise<{ cookie: string; port: number }>((resolve, reject) => {
      const request = net.request({ url: base + '/api/health', session: jar, useSessionCookies: true })

      const timer = setTimeout(() => {
        reject(new Error('Request on stale pooled socket timed out'))
        request.abort()
      }, 5000)

      request.on('error', error => {
        clearTimeout(timer)
        reject(error)
      })
      request.on('response', response => {
        let body = ''
        response.on('error', error => {
          clearTimeout(timer)
          reject(error)
        })
        response.on('data', chunk => {
          body += chunk
        })
        response.on('end', () => {
          clearTimeout(timer)
          resolve(JSON.parse(body))
        })
      })
      request.end()
    })

  try {
    await jar.cookies.set({ url: base, name: 'gateway_login', value: 'preserved', httpOnly: true })
    const prime = await get()
    let notifications = 0

    const trigger = attachPowerResumeRemoteRevalidation({
      log: console.log,
      notifyResume: () => {
        notifications += 1
      },
      powerMonitor: { on: () => undefined },
      resetTransports: async () => {
        if (process.argv[3] !== 'baseline') {
          await resetKeepaliveTransports([jar])
        }
      },
      revalidate: async () => undefined
    })

    await trigger('resume')
    const recovered = await get()
    assert.notEqual(recovered.port, prime.port, 'Recovery must open a fresh socket')
    assert.equal(recovered.cookie, prime.cookie)
    assert.equal(recovered.cookie, 'gateway_login=preserved')
    assert.equal(notifications, 1)
    console.log(
      'WAKE_NATIVE_RESULT ' +
        JSON.stringify({
          ok: true,
          electron: process.versions.electron,
          processType: process.type,
          freshSocket: true,
          cookiePreserved: true
        })
    )
  } finally {
    server.closeAllConnections()
    server.close()
  }
}

app
  .whenReady()
  .then(run)
  .then(() => app.exit(0))
  .catch(error => {
    console.error(error)
    app.exit(1)
  })
