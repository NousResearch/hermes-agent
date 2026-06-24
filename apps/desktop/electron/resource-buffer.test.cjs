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
      allowPrivateNetwork: true,
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
      allowPrivateNetwork: true,
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
    allowPrivateNetwork: true,
    maxRemoteBytes: 5
  })

  assert.equal(result.buffer.toString('utf8'), 'abcde')
  assert.equal(result.mimeType, 'image/png')
})

test('resourceBufferFromUrl rejects hostnames that resolve to loopback before connecting', async t => {
  let requested = false
  const server = await listen((_req, res) => {
    requested = true
    res.writeHead(200, { 'content-type': 'text/plain' })
    res.end('LOCAL_SECRET')
  })
  t.after(() => server.close())

  const lookup = (_hostname, options, callback) => {
    if (options?.all) return callback(null, [{ address: '127.0.0.1', family: 4 }])
    callback(null, '127.0.0.1', 4)
  }

  await assert.rejects(
    resourceBufferFromUrl(`http://localtest.example:${server.address().port}/secret`, { lookup }),
    /private URL/i
  )
  assert.equal(requested, false)
})

test('resourceBufferFromUrl rejects redirects to private-network targets', async t => {
  const server = await listen((_req, res) => {
    res.writeHead(302, { location: 'http://127.0.0.1/secret' })
    res.end()
  })
  t.after(() => server.close())

  await assert.rejects(
    resourceBufferFromUrl(serverUrl(server, '/redirect'), { allowPrivateNetwork: true }),
    /private URL/i
  )
})
