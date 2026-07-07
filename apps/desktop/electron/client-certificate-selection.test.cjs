/**
 * Tests for TLS client-certificate selection helpers.
 *
 * Run with: node --test electron/client-certificate-selection.test.cjs
 */

const test = require('node:test')
const assert = require('node:assert/strict')

const {
  certificateMatchesFilters,
  chooseClientCertificate,
  filtersFromEnv,
  normalizeSerial
} = require('./client-certificate-selection.cjs')

const cloudflareCert = {
  issuerName: 'Cloudflare Managed CA',
  serialNumber: '64:44:82:AE',
  subjectName: 'C=US, CN=Cloudflare'
}

const otherCert = {
  issuerName: 'Example CA',
  serialNumber: 'AA:BB',
  subjectName: 'CN=Example Client'
}

test('normalizeSerial ignores punctuation and case', () => {
  assert.equal(normalizeSerial('64:44:82:ae'), '644482ae')
})

test('chooseClientCertificate auto-selects a single candidate', () => {
  assert.equal(chooseClientCertificate([cloudflareCert]), cloudflareCert)
})

test('chooseClientCertificate declines to guess between multiple candidates', () => {
  assert.equal(chooseClientCertificate([cloudflareCert, otherCert]), null)
})

test('chooseClientCertificate can filter by subject', () => {
  assert.equal(
    chooseClientCertificate([cloudflareCert, otherCert], {
      autoSelect: true,
      subject: 'cloudflare'
    }),
    cloudflareCert
  )
})

test('chooseClientCertificate can filter by issuer and serial', () => {
  assert.equal(
    chooseClientCertificate([cloudflareCert, otherCert], {
      autoSelect: true,
      issuer: 'managed ca',
      serial: '644482ae'
    }),
    cloudflareCert
  )
})

test('chooseClientCertificate respects auto-select opt out', () => {
  assert.equal(chooseClientCertificate([cloudflareCert], { autoSelect: false }), null)
})

test('filtersFromEnv parses env filters', () => {
  assert.deepEqual(
    filtersFromEnv({
      HERMES_DESKTOP_CLIENT_CERT_AUTO_SELECT: '1',
      HERMES_DESKTOP_CLIENT_CERT_ISSUER: ' Managed CA ',
      HERMES_DESKTOP_CLIENT_CERT_SERIAL: '64:44:82:AE',
      HERMES_DESKTOP_CLIENT_CERT_SUBJECT: ' Cloudflare '
    }),
    {
      autoSelect: true,
      issuer: 'managed ca',
      serial: '644482ae',
      subject: 'cloudflare'
    }
  )
})

test('certificateMatchesFilters applies all provided filters', () => {
  assert.equal(
    certificateMatchesFilters(cloudflareCert, {
      issuer: 'managed',
      serial: '644482ae',
      subject: 'cloudflare'
    }),
    true
  )
  assert.equal(
    certificateMatchesFilters(cloudflareCert, {
      issuer: 'managed',
      serial: 'deadbeef',
      subject: 'cloudflare'
    }),
    false
  )
})
