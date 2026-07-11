const assert = require('node:assert/strict')
const http = require('node:http')
const test = require('node:test')

const {
  MEDIA_PROTOCOL_PRIVILEGES,
  buildGatewayMediaHeaders,
  parseMediaProtocolUrl,
  proxyGatewayMediaRequest
} = require('./media-protocol.cjs')

test('media protocol is CORS-enabled for renderer video and fetch requests', () => {
  assert.equal(MEDIA_PROTOCOL_PRIVILEGES.corsEnabled, true)
})

test('buildGatewayMediaHeaders preserves Range and authenticates local and API-server backends', () => {
  const headers = buildGatewayMediaHeaders(new Headers({ Range: 'bytes=0-511' }), 'session-token')

  assert.equal(headers.get('Range'), 'bytes=0-511')
  assert.equal(headers.get('Authorization'), 'Bearer session-token')
  assert.equal(headers.get('X-Hermes-Session-Token'), 'session-token')
})

test('parseMediaProtocolUrl accepts authenticated MoneyPrinter gateway media paths', () => {
  assert.deepEqual(
    parseMediaProtocolUrl(
      'hermes-media://gateway/%2Fapi%2Fcapabilities%2Fmoneyprinter%2Fstream%2Ftask-1%2Ffinal-1.mp4'
    ),
    {
      apiPath: '/api/capabilities/moneyprinter/stream/task-1/final-1.mp4',
      kind: 'gateway'
    }
  )
})

test('parseMediaProtocolUrl rejects unrelated gateway API paths', () => {
  assert.throws(
    () => parseMediaProtocolUrl('hermes-media://gateway/%2Fapi%2Fsessions'),
    /Unsupported gateway media path/
  )
})

test('parseMediaProtocolUrl preserves encoded absolute local file paths', () => {
  assert.deepEqual(
    parseMediaProtocolUrl('hermes-media://local/%2FUsers%2Ftest%2Fclip.mp4'),
    {
      filePath: '/Users/test/clip.mp4',
      kind: 'file'
    }
  )
})

test('proxyGatewayMediaRequest streams authenticated range responses without waiting for the full body', async t => {
  let capturedHeaders = null
  let releaseResponse = null
  let responseEnded = false
  const server = http.createServer((request, response) => {
    capturedHeaders = request.headers
    response.writeHead(206, {
      'Accept-Ranges': 'bytes',
      'Content-Length': '6',
      'Content-Range': 'bytes 0-5/6',
      'Content-Type': 'video/mp4'
    })
    response.write('abc')
    releaseResponse = () => {
      responseEnded = true
      response.end('def')
    }
  })
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve))
  t.after(() => new Promise(resolve => server.close(resolve)))

  const address = server.address()
  assert.ok(address && typeof address === 'object')
  const response = await proxyGatewayMediaRequest(`http://127.0.0.1:${address.port}/video.mp4`, {
    headers: new Headers({
      Authorization: 'Bearer session-token',
      Range: 'bytes=0-5'
    }),
    method: 'GET'
  })

  assert.equal(responseEnded, false)
  assert.equal(response.status, 206)
  assert.equal(response.headers.get('accept-ranges'), 'bytes')
  assert.equal(response.headers.get('content-range'), 'bytes 0-5/6')
  assert.equal(capturedHeaders?.authorization, 'Bearer session-token')
  assert.equal(capturedHeaders?.range, 'bytes=0-5')

  assert.ok(releaseResponse)
  releaseResponse()
  assert.equal(await response.text(), 'abcdef')
})
