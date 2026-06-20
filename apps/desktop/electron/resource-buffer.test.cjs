const assert = require('node:assert/strict')
const http = require('node:http')
const test = require('node:test')

const { resourceBufferFromUrl } = require('./resource-buffer.cjs')

function listen(handler) {
  return new Promise(resolve => {
    const server = http.createServer(handler)
    server.listen(0, '127.0.0.1', () => resolve(server))
  })
}

function serverUrl(server, path = '/') {
  const { port } = server.address()
  return `http://127.0.0.1:${port}${path}`
}

test('resourceBufferFromUrl rejects remote responses whose content-length exceeds the byte cap', async t => {
  const server = await listen((_req, res) => {
    res.writeHead(200, {
      'content-type': 'image/png',
      'content-length': '6'
    })
    res.end('abcdef')
  })
  t.after(() => server.close())

  await assert.rejects(
    resourceBufferFromUrl(serverUrl(server), {
      isBlockedUrl: () => false,
      maxRemoteBytes: 5
    }),
    /too large/i
  )
})

test('resourceBufferFromUrl stops buffering chunked remote responses after the byte cap is exceeded', async t => {
  const server = await listen((_req, res) => {
    res.writeHead(200, { 'content-type': 'image/png' })
    res.write('abc')
    setImmediate(() => res.end('def'))
  })
  t.after(() => server.close())

  await assert.rejects(
    resourceBufferFromUrl(serverUrl(server), {
      isBlockedUrl: () => false,
      maxRemoteBytes: 5
    }),
    /too large/i
  )
})

test('resourceBufferFromUrl returns remote buffers within the byte cap', async t => {
  const server = await listen((_req, res) => {
    res.writeHead(200, { 'content-type': 'image/png' })
    res.end('abcde')
  })
  t.after(() => server.close())

  const result = await resourceBufferFromUrl(serverUrl(server), {
    isBlockedUrl: () => false,
    maxRemoteBytes: 5
  })

  assert.equal(result.buffer.toString('utf8'), 'abcde')
  assert.equal(result.mimeType, 'image/png')
})
