const assert = require('node:assert/strict')
const test = require('node:test')

const { isBlockedUrl, resolveSafeHttpUrl, validateRedirectUrl } = require('./url-guard.cjs')

test('isBlockedUrl blocks private and loopback IPv6 hosts including bracketed URL hostnames', () => {
  const blocked = [
    'http://[::1]/',
    'https://[::]/image.png',
    'http://[fc00::1]/',
    'http://[fd12:3456::1]/',
    'http://[fe80::1]/',
    'http://[::ffff:127.0.0.1]/',
    'http://[::ffff:10.0.0.5]/',
    'http://[::ffff:192.168.1.7]/',
    'http://[::ffff:172.16.0.2]/'
  ]

  for (const url of blocked) {
    assert.equal(isBlockedUrl(url), true, `${url} should be blocked`)
  }
})

test('isBlockedUrl allows public IPv6 HTTP(S) hosts', () => {
  assert.equal(isBlockedUrl('https://[2606:4700:4700::1111]/'), false)
})

test('resolveSafeHttpUrl rejects hostnames that resolve to loopback addresses', async () => {
  const lookup = (_hostname, options, callback) => {
    if (options?.all) return callback(null, [{ address: '127.0.0.1', family: 4 }])
    callback(null, '127.0.0.1', 4)
  }

  await assert.rejects(
    resolveSafeHttpUrl('http://localtest.example/secret', { lookup }),
    /private URL/i
  )
})

test('resolveSafeHttpUrl rejects hostnames with mixed public and private DNS answers', async () => {
  const lookup = (_hostname, options, callback) => {
    if (options?.all) {
      return callback(null, [
        { address: '93.184.216.34', family: 4 },
        { address: '10.0.0.7', family: 4 }
      ])
    }
    callback(null, '93.184.216.34', 4)
  }

  await assert.rejects(
    resolveSafeHttpUrl('http://mixed.example/image.png', { lookup }),
    /private URL/i
  )
})

test('resolveSafeHttpUrl returns a pinned lookup using the validated address', async () => {
  let lookupCalls = 0
  const lookup = (_hostname, options, callback) => {
    lookupCalls += 1
    if (options?.all) return callback(null, [{ address: '93.184.216.34', family: 4 }])
    callback(null, '127.0.0.1', 4)
  }

  const result = await resolveSafeHttpUrl('http://example.com/image.png', { lookup })
  const pinned = await new Promise((resolve, reject) => {
    result.lookup('example.com', {}, (error, address, family) => {
      if (error) reject(error)
      else resolve({ address, family })
    })
  })

  assert.deepEqual(pinned, { address: '93.184.216.34', family: 4 })
  assert.equal(lookupCalls, 1)
})

test('validateRedirectUrl rejects public-to-private redirect targets', () => {
  assert.throws(
    () => validateRedirectUrl('https://example.com/start', 'http://127.0.0.1/secret'),
    /private URL/i
  )
})
