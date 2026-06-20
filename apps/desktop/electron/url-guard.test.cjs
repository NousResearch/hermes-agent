const assert = require('node:assert/strict')
const test = require('node:test')

const { isBlockedUrl } = require('./url-guard.cjs')

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
