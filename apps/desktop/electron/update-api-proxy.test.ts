import assert from 'node:assert/strict'
import https from 'node:https'
import net from 'node:net'

import { afterEach, test } from 'vitest'

import { updateCheckAgent } from './update-api-proxy'

const keys = [
  'HTTPS_PROXY',
  'https_proxy',
  'HTTP_PROXY',
  'http_proxy',
  'ALL_PROXY',
  'all_proxy',
  'NO_PROXY',
  'no_proxy'
] as const

const saved = Object.fromEntries(keys.map(key => [key, process.env[key]]))

afterEach(() => {
  for (const key of keys) {
    const value = saved[key]

    if (value === undefined) {
      delete process.env[key]
    } else {
      process.env[key] = value
    }
  }
})

function clearProxyEnv() {
  for (const key of keys) {
    delete process.env[key]
  }
}

test('update API uses an HTTPS proxy only when the environment selects one', () => {
  clearProxyEnv()
  const url = 'https://api.github.com/repos/NousResearch/hermes-agent/commits/main'

  assert.equal(updateCheckAgent(url), undefined)
  process.env.HTTPS_PROXY = 'http://127.0.0.1:8080'
  assert.ok(updateCheckAgent(url))
  process.env.NO_PROXY = 'api.github.com'
  assert.equal(updateCheckAgent(url), undefined)
})

test('update API request reaches the configured HTTP CONNECT proxy', async () => {
  clearProxyEnv()
  let connectRequest = ''

  const server = net.createServer(socket => {
    socket.once('data', data => {
      connectRequest = data.toString()
      socket.end('HTTP/1.1 502 Bad Gateway\r\nContent-Length: 0\r\n\r\n')
    })
  })

  await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve))

  try {
    const address = server.address()
    assert.ok(address && typeof address !== 'string')
    process.env.HTTPS_PROXY = `http://127.0.0.1:${address.port}`

    const status = await new Promise<number | undefined>((resolve, reject) => {
      https
        .get(
          'https://api.github.com/repos/NousResearch/hermes-agent/commits/main',
          {
            agent: updateCheckAgent('https://api.github.com/repos/NousResearch/hermes-agent/commits/main')
          },
          res => resolve(res.statusCode)
        )
        .on('error', reject)
    })

    assert.equal(status, 502)
    assert.match(connectRequest, /^CONNECT api\.github\.com:443 HTTP\/1\.1\r\n/)
  } finally {
    server.close()
  }
})
