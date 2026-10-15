import assert from 'node:assert/strict'

import { afterEach, describe, test } from 'vitest'

import { isSafeTitleFetchTarget, literalTitleBlockReason, sensitiveTitleQueryParam } from './link-title-guard'

// #126885: the title prefetch fires for links tool payloads can inject, so
// the admission must refuse the same destination classes tools/url_safety.py
// blocks for agent fetches — including the inet_aton spellings curl honors
// (127.1, 0x7f.1, integer form) and the IPv6 wrappers resolvers hand out.

describe('literalTitleBlockReason', () => {
  test('refuses RFC1918, loopback, link-local, CGNAT and metadata literals', () => {
    for (const host of [
      '10.0.0.5',
      '10.255.255.255',
      '172.16.0.1',
      '172.31.255.254',
      '192.168.1.1',
      '127.0.0.1',
      '169.254.169.254',
      '169.254.170.2',
      '100.100.100.200',
      '100.64.0.1',
      '100.127.255.254',
      '0.0.0.0',
      '255.255.255.255',
      '240.0.0.1'
    ]) {
      assert.ok(literalTitleBlockReason(host), host)
    }
  })

  test('refuses the inet_aton spellings curl would dial as private', () => {
    assert.equal(literalTitleBlockReason('127.1'), 'loopback')
    assert.equal(literalTitleBlockReason('127.000000001'), 'loopback')
    assert.equal(literalTitleBlockReason('0x7f.1'), 'loopback')
    assert.equal(literalTitleBlockReason('0x7f000001'), 'loopback')
    assert.equal(literalTitleBlockReason('2130706433'), 'loopback')
    assert.equal(literalTitleBlockReason('0177.0.0.1'), 'loopback')
    assert.equal(literalTitleBlockReason('0xA.0.0.1'), 'private')
    assert.equal(literalTitleBlockReason('0xac100001'), 'private')
    assert.equal(literalTitleBlockReason('3232235521'), 'private') // 192.168.0.1
    assert.equal(literalTitleBlockReason('169.0xfe0000'), 'link-local') // 169.254.0.0, wide tail part
  })

  test('refuses private/reserved IPv6 literals, including v4 wrappers', () => {
    for (const host of [
      '::',
      '::1',
      '[::1]',
      'fe80::1',
      'febf::ffff',
      'fc00::1',
      'fd12:3456:789a::1',
      'fec0::1',
      '::ffff:10.0.0.1',
      '::ffff:127.0.0.1',
      '::ffff:0:10.0.0.1',
      'fd00:ec2::254',
      'fe80::1%en0'
    ]) {
      assert.ok(literalTitleBlockReason(host), host)
    }
  })

  test('admits public literals and ordinary hostnames', () => {
    for (const host of [
      'example.com',
      '8.8.8.8',
      '1.1.1.1',
      '172.32.0.1',
      '100.128.0.1',
      '2606:4700::1111',
      'Example.COM.'
    ]) {
      assert.equal(literalTitleBlockReason(host), null, host)
    }
  })

  test('refuses cloud metadata hostnames and mDNS names outright', () => {
    assert.equal(literalTitleBlockReason('metadata.google.internal'), 'metadata-host')
    assert.equal(literalTitleBlockReason('Metadata.Goog.'), 'metadata-host')
    assert.equal(literalTitleBlockReason('printer.local'), 'mdns')
  })

  test('refuses the loopback hostname and its subdomains without DNS', () => {
    assert.equal(literalTitleBlockReason('localhost'), 'loopback-host')
    assert.equal(literalTitleBlockReason('LOCALHOST.'), 'loopback-host')
    assert.equal(literalTitleBlockReason('app.localhost'), 'loopback-host')
  })
})

describe('sensitiveTitleQueryParam', () => {
  test('flags unambiguous credential-bearing params', () => {
    assert.equal(sensitiveTitleQueryParam('https://a.test/login?token=abc'), 'token')
    assert.equal(sensitiveTitleQueryParam('https://a.test/x?ACCESS_TOKEN=abc'), 'ACCESS_TOKEN')
    assert.equal(sensitiveTitleQueryParam('https://a.test/x?client_secret=k&x=1'), 'client_secret')
    assert.equal(sensitiveTitleQueryParam('https://a.test/x?signature=s'), 'signature')
  })

  test('flags one-time-link params by their _token suffix (#126885)', () => {
    assert.equal(sensitiveTitleQueryParam('https://a.test/u/confirm?confirmation_token=abc'), 'confirmation_token')
    assert.equal(sensitiveTitleQueryParam('https://a.test/r?RESET_PASSWORD_TOKEN=x'), 'RESET_PASSWORD_TOKEN')
    assert.equal(sensitiveTitleQueryParam('https://a.test/m?magic_token=y&z=1'), 'magic_token')
  })

  test('keeps ambiguous facet params prefetchable, aligned with tools/url_safety.py', () => {
    for (const url of [
      'https://a.test/docs?code=123',
      'https://a.test/p?key=color',
      'https://a.test/p?auth=github',
      'https://a.test/p?token='
    ]) {
      assert.equal(sensitiveTitleQueryParam(url), null, url)
    }
  })
})

const fakeLookup = (addresses: string[] | Error) => async () => {
  if (addresses instanceof Error) {
    throw addresses
  }

  return addresses.map(address => ({ address, family: address.includes(':') ? 6 : 4 }))
}

const PROXY_VARS = ['HTTPS_PROXY', 'https_proxy', 'HTTP_PROXY', 'http_proxy', 'ALL_PROXY', 'all_proxy'] as const

afterEach(() => {
  for (const name of PROXY_VARS) {
    delete process.env[name]
  }
})

describe('isSafeTitleFetchTarget', () => {
  test('admits a public name resolving to public space', async () => {
    assert.equal(await isSafeTitleFetchTarget('https://example.com/docs', fakeLookup(['93.184.216.34'])), true)
  })

  test('refuses literal private destinations without any DNS call', async () => {
    let lookupRan = false

    const tracking = async (host: string) => {
      lookupRan = true

      return [{ address: '93.184.216.34', family: 4 }]
    }

    for (const url of ['http://192.168.1.1/', 'http://169.254.169.254/latest/meta-data/', 'http://[fd00::1]/']) {
      lookupRan = false
      assert.equal(await isSafeTitleFetchTarget(url, tracking), false, url)
      assert.equal(lookupRan, false, url)
    }
  })

  test('refuses a public name that resolves into private space, fail-closed on any answer', async () => {
    assert.equal(await isSafeTitleFetchTarget('https://rebind.test/', fakeLookup(['93.184.216.34', '10.0.0.7'])), false)
    assert.equal(await isSafeTitleFetchTarget('https://rebind.test/', fakeLookup(['192.168.0.20'])), false)
    assert.equal(await isSafeTitleFetchTarget('https://v6.test/', fakeLookup(['::ffff:127.0.0.1'])), false)
  })

  test('refuses on resolver failure or empty answers', async () => {
    assert.equal(await isSafeTitleFetchTarget('https://nx.test/', fakeLookup(new Error('ENOTFOUND'))), false)
    assert.equal(await isSafeTitleFetchTarget('https://empty.test/', fakeLookup([])), false)
  })

  test('still resolves and vets a name when a proxy owns the dial', async () => {
    process.env.HTTPS_PROXY = 'http://127.0.0.1:7890'
    process.env.HTTP_PROXY = 'http://127.0.0.1:7890'

    assert.equal(await isSafeTitleFetchTarget('https://example.com/', fakeLookup(['93.184.216.34'])), true)
    // A public-looking name whose answer is cloud metadata or the LAN: the proxy
    // would resolve it the same way, so it is refused before the proxy sees it.
    assert.equal(await isSafeTitleFetchTarget('https://evil.example.com/', fakeLookup(['169.254.169.254'])), false)
    assert.equal(await isSafeTitleFetchTarget('http://nas.lan/', fakeLookup(['192.168.1.20'])), false)
    assert.equal(await isSafeTitleFetchTarget('https://nx.test/', fakeLookup(new Error('ENOTFOUND'))), false)
    // Literal checks need no DNS.
    assert.equal(await isSafeTitleFetchTarget('https://192.168.0.10/', fakeLookup(['93.184.216.34'])), false)
    assert.equal(await isSafeTitleFetchTarget('http://localhost:8080/', fakeLookup(['93.184.216.34'])), false)
  })

  test('refuses non-http schemes and credential-bearing queries', async () => {
    assert.equal(await isSafeTitleFetchTarget('file:///etc/passwd', fakeLookup([])), false)
    assert.equal(await isSafeTitleFetchTarget('not a url', fakeLookup([])), false)
    assert.equal(
      await isSafeTitleFetchTarget('https://magic.test/verify?token=xyz', fakeLookup(['93.184.216.34'])),
      false
    )
  })
})
