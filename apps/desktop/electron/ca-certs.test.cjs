const test = require('node:test')
const assert = require('node:assert/strict')
const https = require('node:https')

const {
  splitPemCertificates,
  collectMacKeychainCerts,
  collectNodeExtraCaCerts,
  buildCaBundle,
  installNodeCaTrust
} = require('./ca-certs.cjs')

const CERT_A =
  '-----BEGIN CERTIFICATE-----\nAAAAcertificateAAAA\n-----END CERTIFICATE-----'
const CERT_B =
  '-----BEGIN CERTIFICATE-----\nBBBBcertificateBBBB\n-----END CERTIFICATE-----'
const DEFAULT_ROOT = '-----BEGIN CERTIFICATE-----\nMOZILLAROOT\n-----END CERTIFICATE-----'

test('splitPemCertificates extracts every concatenated cert and normalizes trailing newline', () => {
  const blob = `noise\n${CERT_A}\nmiddle\n${CERT_B}\ntrailer`
  const certs = splitPemCertificates(blob)
  assert.equal(certs.length, 2)
  assert.ok(certs[0].startsWith('-----BEGIN CERTIFICATE-----'))
  assert.ok(certs[0].endsWith('-----END CERTIFICATE-----\n'))
  assert.ok(certs[1].includes('BBBBcertificateBBBB'))
})

test('splitPemCertificates returns empty for empty/no-match input', () => {
  assert.deepEqual(splitPemCertificates(''), [])
  assert.deepEqual(splitPemCertificates('not a cert'), [])
  assert.deepEqual(splitPemCertificates(null), [])
})

test('collectMacKeychainCerts queries login and System keychains', () => {
  const calls = []
  const runSecurity = args => {
    calls.push(args)
    return args.length === 3 ? CERT_A : CERT_B // login (no path) vs System.keychain
  }
  const certs = collectMacKeychainCerts({ runSecurity })
  assert.equal(certs.length, 2)
  // First query has no keychain path, second targets System.keychain.
  assert.deepEqual(calls[0], ['find-certificate', '-a', '-p'])
  assert.equal(calls[1].at(-1), '/Library/Keychains/System.keychain')
})

test('collectMacKeychainCerts swallows security failures per keychain', () => {
  const runSecurity = args => {
    if (args.length === 3) throw new Error('security exploded')
    return CERT_B
  }
  const certs = collectMacKeychainCerts({ runSecurity })
  assert.equal(certs.length, 1)
  assert.ok(certs[0].includes('BBBB'))
})

test('collectNodeExtraCaCerts reads the env-referenced file, tolerating absence', () => {
  assert.deepEqual(collectNodeExtraCaCerts({ env: {}, readFile: () => CERT_A }), [])
  const certs = collectNodeExtraCaCerts({
    env: { NODE_EXTRA_CA_CERTS: '/tmp/extra.pem' },
    readFile: p => {
      assert.equal(p, '/tmp/extra.pem')
      return CERT_A
    }
  })
  assert.equal(certs.length, 1)
})

test('collectNodeExtraCaCerts returns empty when the file is unreadable', () => {
  const certs = collectNodeExtraCaCerts({
    env: { NODE_EXTRA_CA_CERTS: '/nope' },
    readFile: () => {
      throw new Error('ENOENT')
    }
  })
  assert.deepEqual(certs, [])
})

test('buildCaBundle merges default roots + keychain + extra, deduped, roots first', () => {
  const bundle = buildCaBundle({
    platform: 'darwin',
    env: { NODE_EXTRA_CA_CERTS: '/tmp/extra.pem' },
    runSecurity: () => CERT_A,
    readFile: () => CERT_B,
    defaultRoots: [DEFAULT_ROOT]
  })
  // Default root first, then unique additions; CERT_A returned by both keychain
  // queries collapses to one entry.
  assert.equal(bundle[0], DEFAULT_ROOT)
  const joined = bundle.join('|')
  assert.ok(joined.includes('MOZILLAROOT'))
  assert.ok(joined.includes('AAAAcertificate'))
  assert.ok(joined.includes('BBBBcertificate'))
  // No duplicate of CERT_A even though both keychains returned it.
  assert.equal(bundle.filter(c => c.includes('AAAAcertificate')).length, 1)
})

test('buildCaBundle returns null when there is nothing extra to add', () => {
  const bundle = buildCaBundle({
    platform: 'darwin',
    env: {},
    runSecurity: () => '',
    readFile: () => '',
    defaultRoots: [DEFAULT_ROOT]
  })
  assert.equal(bundle, null)
})

test('buildCaBundle skips keychain harvesting off macOS', () => {
  const bundle = buildCaBundle({
    platform: 'linux',
    env: {},
    runSecurity: () => CERT_A,
    readFile: () => '',
    defaultRoots: [DEFAULT_ROOT]
  })
  assert.equal(bundle, null)
})

test('installNodeCaTrust sets globalAgent ca and returns the count', () => {
  const https = { globalAgent: { options: {} } }
  const count = installNodeCaTrust(https, {
    platform: 'darwin',
    env: {},
    runSecurity: () => CERT_A,
    readFile: () => '',
    defaultRoots: [DEFAULT_ROOT]
  })
  assert.equal(count, 2)
  assert.equal(https.globalAgent.options.ca.length, 2)
})

test('installNodeCaTrust leaves the agent untouched when nothing to add', () => {
  const https = { globalAgent: { options: {} } }
  const count = installNodeCaTrust(https, {
    platform: 'linux',
    env: {},
    runSecurity: () => '',
    readFile: () => '',
    defaultRoots: [DEFAULT_ROOT]
  })
  assert.equal(count, 0)
  assert.equal(https.globalAgent.options.ca, undefined)
})

test('installNodeCaTrust never throws on failure', () => {
  const https = { globalAgent: { options: {} } }
  const count = installNodeCaTrust(https, {
    platform: 'darwin',
    env: {},
    defaultRoots: [DEFAULT_ROOT],
    runSecurity: () => {
      throw new Error('boom')
    },
    readFile: () => {
      throw new Error('boom')
    }
  })
  assert.equal(count, 0)
})

// --- Integration: prove the fix against a real TLS handshake ---------------
//
// The bug: Node's `https` stack ignores the macOS keychain, so a backend
// fronted by a private CA the user trusts in Keychain Access fails the boot
// probe with `unable to get local issuer certificate`. These tests stand up a
// real in-process TLS server presenting a cert Node does NOT trust by default,
// then show that feeding `buildCaBundle`'s output (with the CA sourced the same
// way the keychain harvest delivers it) into the client's `ca` makes the
// otherwise-rejected handshake succeed — the exact mechanism the fix relies on.
//
// Static self-signed fixture (CN=hermes-ca-certs-test.localhost, SAN
// localhost/127.0.0.1, valid until 2126) so the test is hermetic — no openssl,
// no network, no clock dependence.
const TEST_CERT = `-----BEGIN CERTIFICATE-----
MIIC9TCCAd2gAwIBAgIJAMrEBjLk0gcWMA0GCSqGSIb3DQEBCwUAMCkxJzAlBgNV
BAMMHmhlcm1lcy1jYS1jZXJ0cy10ZXN0LmxvY2FsaG9zdDAgFw0yNjA3MDIxNjQ2
MzBaGA8yMTI2MDYwODE2NDYzMFowKTEnMCUGA1UEAwweaGVybWVzLWNhLWNlcnRz
LXRlc3QubG9jYWxob3N0MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEA
oF0/hpMKFDkLzsiTTYnQJVc+3ib7sIFQwiuQpJJBw4GCZw0DlA6arb/TVR3ZP/Aj
MLfhkjNM89A6N9N9EFRLrkdiefcn/P912XDmSgX0IohvweBbzLGH4p0RmbL2nC0J
gjUrAImaATtWZ6dxJfO1faPEiHhV8ZTNquOLs8xWOQGYcr1wl5Iii9rBMDFTOxwq
EkWVBGRkjI4ReEwJ4BAIqJe/ganWCFTw1sru/xfQSMTGXZ4xfxp+mIz7Fiy7iSlC
ou7EduNL+KdnBLVIbDud5IcaoIfaKsb68a3r42pz9FTm7ZuSu7ql5tDE+vT/s76m
KSQLpC9U7KiBVFEYuSLHjwIDAQABox4wHDAaBgNVHREEEzARgglsb2NhbGhvc3SH
BH8AAAEwDQYJKoZIhvcNAQELBQADggEBAG7rEy5L/49edaVhxncwYeMcAJxSlNuL
/qn6houKfBNdq1dD3KSW4PBf1RT1cazMOwxULea+YkjAz9iDfd74p4vMwlEUNNqM
KYGq3rRgOB7M0UpbjnNFvUySufgk/Kr+ZHZCtnaqaziS4/CDRzsh5ii2n8NK2Ka/
wfi26ruyZ62Y8t3qwMGUiazciMmEkW3sCetdBSmCCCfLAk1+67xmuN9mMtvEaSwb
XTUZzQCAUViNyna10oRu+CPWPlMti+Yl3gHqM5C16VP/4l3m6BpgVASm4PXvjazv
WpXsqDz0EdrKGlEywz4zZwGxGxJZnmqNirReUFTS/crQqQ06t4aca9U=
-----END CERTIFICATE-----
`

const TEST_KEY = `-----BEGIN PRIVATE KEY-----
MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQCgXT+GkwoUOQvO
yJNNidAlVz7eJvuwgVDCK5CkkkHDgYJnDQOUDpqtv9NVHdk/8CMwt+GSM0zz0Do3
030QVEuuR2J59yf8/3XZcOZKBfQiiG/B4FvMsYfinRGZsvacLQmCNSsAiZoBO1Zn
p3El87V9o8SIeFXxlM2q44uzzFY5AZhyvXCXkiKL2sEwMVM7HCoSRZUEZGSMjhF4
TAngEAiol7+BqdYIVPDWyu7/F9BIxMZdnjF/Gn6YjPsWLLuJKUKi7sR240v4p2cE
tUhsO53khxqgh9oqxvrxrevjanP0VObtm5K7uqXm0MT69P+zvqYpJAukL1TsqIFU
URi5IsePAgMBAAECggEABpWtnRX+jPQGsxfzTHTDMjVR5IdPQGEe8sodJ1TGFIQF
2btkETlESqwcvlr6Z7CxkTeLsJggqcY8DbzGXtxHO0G9HmrynmXS7fm/yvxRmPVn
OVvACTik7r4qUdhSXflPcaRDPsNnqnNISftLHxbRtfX6hOwEA2Zrx3+OZfSW+cIx
XoOHkFunawk/qR1WwfaPvpZZlcNnoIWWvpQHBB38D30MFbtiYLmHqWTUH+yherC0
4++WC1kfJG3fWte8mSidNskN5aFRrn08ID//aARmTudZjO1iWALMoi1Ia5AP9gpK
XtvkyugohiCsHOuPjl6wKY0BlASLNhKF0Vs+ZwJtoQKBgQDPRnnzdt0Zv26N4We/
0aSR8UQdlT+7y6VGGTmZg/eySTHJXliHIctqcS0zUWjSHqaxjQ0JacWJv43IYzN8
u7Tb/CKsqQx2KUgofE6f83GZgon/OoNTJ2ELNUxBp1Z2Kvf7ZPaAZhEmY9pJP8sb
Xj8rv4Pu4vliKC4ONckXINS4EQKBgQDGD7vSwdcEjAzP3D+lja6butM3TfFw6whB
/CgSi8GPaCPiqCwZ5tTVLGUs6LBJScdcY09sG+FTEit0dl2ppUnvyzUNwAZGv/f3
tWL7C4BCSOgKJzyhIxBBpY7LvF16fqkU5uU5QzYP5QZxTixqSUw/H2gRPdJrJVkR
ntu+ycolnwKBgQC1SYbQEl4/btda8JK1ir7Nhp904Fzl+6+KJ/Xg9zNlk+8fmI0F
Y+FuL57BC6sKXBSfpiaI3SIQ4KE5aspVhjchUN1i9lgX4PNjtZVvAJWTFkFsIdlK
mV6fVvZjVeChaeOK1TtkAeFuGleJSWpzfXLy6IaUIaDM4Sem9hPzTpu+gQKBgGvR
TNGiK8aR5reQkiUxR4gG38wPZguuJkSlW7sc0TWb300Xd0pyWhHhpQIZeT2sKBan
CSk01ChAj99KQBqFnAYpfKwLiF8jSX1TBJrc2+k5fvdn/J1LVSInWeCWndx87tYu
C0Js0BU++47am1sQo60JD8GzAcTKA/6Pl9f4SU7JAoGABea1abmhGlAIzm2NaZSG
XTp/4BJxOm68kVP7HetGMOmC6/VKE0rU3GdHTALcGZ722cYK8OEZxqSzsT1A816b
PITKno//19xfZepjLSSxajruuLmfk329qaX6NjBnVm4+q+TNJI56iBWVxNkPA1Pw
GHuiAQdCpBnurhUyHL5Hzt0=
-----END PRIVATE KEY-----
`

function startTlsServer() {
  return new Promise((resolve, reject) => {
    const server = https.createServer({ cert: TEST_CERT, key: TEST_KEY }, (_req, res) => {
      res.writeHead(200, { 'Content-Type': 'application/json' })
      res.end('{"ok":true}')
    })
    server.on('error', reject)
    server.listen(0, '127.0.0.1', () => resolve({ server, port: server.address().port }))
  })
}

function getWithCa(port, ca) {
  return new Promise((resolve, reject) => {
    const req = https.get({ host: '127.0.0.1', port, path: '/', ca, servername: 'localhost' }, res => {
      res.resume()
      res.on('end', () => resolve(res.statusCode))
    })
    req.on('error', reject)
    req.setTimeout(5000, () => req.destroy(new Error('timeout')))
  })
}

test('a TLS client rejects the private cert by default (reproduces the bug)', async t => {
  const { server, port } = await startTlsServer()
  t.after(() => server.close())
  await assert.rejects(
    () => getWithCa(port, undefined),
    err => /self-signed|self signed|unable to (get|verify)|leaf/i.test(err.message) || /_CERT|_SIGNATURE/.test(err.code || ''),
    'default trust store must not accept the private cert'
  )
})

test('buildCaBundle output makes the same handshake succeed (proves the fix)', async t => {
  const { server, port } = await startTlsServer()
  t.after(() => server.close())

  // Source the CA exactly as the macOS keychain harvest would, and keep the
  // bundle minimal so the test asserts on our additions, not the Mozilla roots.
  const bundle = buildCaBundle({
    platform: 'darwin',
    env: {},
    runSecurity: () => TEST_CERT,
    readFile: () => '',
    defaultRoots: []
  })
  assert.ok(bundle && bundle.length === 1, 'bundle should carry the harvested CA')

  const status = await getWithCa(port, bundle)
  assert.equal(status, 200)
})

test('NODE_EXTRA_CA_CERTS is honored end-to-end in the merged bundle', async t => {
  const { server, port } = await startTlsServer()
  t.after(() => server.close())

  // Non-darwin: keychain harvest is skipped, so trust must come solely from
  // NODE_EXTRA_CA_CERTS — the second half of the fix's contract.
  const bundle = buildCaBundle({
    platform: 'linux',
    env: { NODE_EXTRA_CA_CERTS: '/etc/hermes/extra.pem' },
    runSecurity: () => TEST_CERT,
    readFile: () => TEST_CERT,
    defaultRoots: []
  })
  assert.ok(bundle && bundle.length === 1, 'bundle should carry the NODE_EXTRA_CA_CERTS cert')

  const status = await getWithCa(port, bundle)
  assert.equal(status, 200)
})
