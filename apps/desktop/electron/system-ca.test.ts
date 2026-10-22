import assert from 'node:assert/strict'
import { X509Certificate } from 'node:crypto'

import { afterEach, test, vi } from 'vitest'

import { bundledRoot, expiredRoot, privateRoot } from './fixtures/system-ca'
import { installSystemCaTrust, type NodeTlsCaApi } from './system-ca'

afterEach(() => vi.restoreAllMocks())

test('excludes expired roots and deduplicates real certificates with defaults first', () => {
  vi.spyOn(Date, 'now').mockReturnValue(new Date('2026-01-01T00:00:00Z').getTime())

  const tlsApi = fakeTlsApi(
    [expiredRoot, bundledRoot, bundledRoot, 'unparseable-default'],
    [expiredRoot, bundledRoot.replaceAll('\n', '\r\n'), privateRoot, privateRoot, 'unparseable-system']
  )

  const result = installSystemCaTrust(tlsApi, 'win32')

  assert.deepEqual(tlsApi.installed, [[bundledRoot, 'unparseable-default', privateRoot, 'unparseable-system']])
  assert.deepEqual(result, { applied: true, systemCertificateCount: 2, totalCertificateCount: 4 })
})

test('excludes a root at its exact expiry while retaining valid defaults', () => {
  vi.spyOn(Date, 'now').mockReturnValue(new X509Certificate(expiredRoot).validToDate.getTime())
  const tlsApi = fakeTlsApi([bundledRoot], [expiredRoot])

  const result = installSystemCaTrust(tlsApi, 'win32')

  assert.deepEqual(tlsApi.installed, [[bundledRoot]])
  assert.deepEqual(result, { applied: true, systemCertificateCount: 0, totalCertificateCount: 1 })
})

function fakeTlsApi(
  defaults: string[] = ['bundled-ca', 'extra-ca'],
  system: string[] = ['windows-root-ca']
): NodeTlsCaApi & { installed: string[][] } {
  const installed: string[][] = []

  return {
    installed,
    getCACertificates(type = 'default') {
      return type === 'system' ? [...system] : [...defaults]
    },
    setDefaultCACertificates(certificates) {
      installed.push([...certificates])
    }
  }
}

test('installs Windows system CAs without dropping existing defaults', () => {
  const tlsApi = fakeTlsApi(['mozilla-root', 'extra-ca'], ['machine-root', 'user-root'])

  const result = installSystemCaTrust(tlsApi, 'win32')

  assert.deepEqual(tlsApi.installed, [['mozilla-root', 'extra-ca', 'machine-root', 'user-root']])
  assert.deepEqual(result, {
    applied: true,
    systemCertificateCount: 2,
    totalCertificateCount: 4
  })
})

test('installs macOS keychain CAs (the renderer already trusts them; Node https did not)', () => {
  // #57241: a remote gateway fronted by a private/homelab CA trusted in Keychain Access
  // loads in the Chromium renderer but fails every main-process Node https call with
  // `unable to get local issuer certificate`. On darwin the same installer must fold the
  // keychain's user + System roots into the default trust store.
  const tlsApi = fakeTlsApi(['mozilla-root'], ['homelab-root', 'corp-root'])

  const result = installSystemCaTrust(tlsApi, 'darwin')

  assert.deepEqual(tlsApi.installed, [['mozilla-root', 'homelab-root', 'corp-root']])
  assert.deepEqual(result, {
    applied: true,
    systemCertificateCount: 2,
    totalCertificateCount: 3
  })
})

test('darwin without keychain trust additions leaves the defaults untouched', () => {
  // A machine with no user-installed anchors enumerates zero 'system' certs; the default
  // trust store must stay untouched (also the pre-keychain-reader Node behavior).
  const tlsApi = fakeTlsApi(['mozilla-root'], [])

  const result = installSystemCaTrust(tlsApi, 'darwin')

  assert.deepEqual(tlsApi.installed, [])
  assert.deepEqual(result, {
    applied: false,
    systemCertificateCount: 0,
    totalCertificateCount: 1
  })
})

test('does not inspect or replace CAs on linux', () => {
  let reads = 0

  const tlsApi: NodeTlsCaApi = {
    getCACertificates() {
      reads += 1

      return []
    },
    setDefaultCACertificates() {
      throw new Error('should not install')
    }
  }

  const result = installSystemCaTrust(tlsApi, 'linux')

  assert.equal(reads, 0)
  assert.deepEqual(result, {
    applied: false,
    systemCertificateCount: 0,
    totalCertificateCount: 0
  })
})

test('leaves the existing defaults untouched when the OS store has no system CAs', () => {
  const tlsApi = fakeTlsApi(['mozilla-root'], [])

  const result = installSystemCaTrust(tlsApi, 'win32')

  assert.deepEqual(tlsApi.installed, [])
  assert.deepEqual(result, {
    applied: false,
    systemCertificateCount: 0,
    totalCertificateCount: 1
  })
})

test('fails open when the runtime cannot load the OS certificate store', () => {
  const tlsApi: NodeTlsCaApi = {
    getCACertificates(type = 'default') {
      if (type === 'system') {
        throw new Error('certificate store unavailable')
      }

      return ['mozilla-root']
    },
    setDefaultCACertificates() {
      throw new Error('should not install')
    }
  }

  const result = installSystemCaTrust(tlsApi, 'win32')

  assert.deepEqual(result, {
    applied: false,
    systemCertificateCount: 0,
    totalCertificateCount: 0,
    error: 'certificate store unavailable'
  })
})
