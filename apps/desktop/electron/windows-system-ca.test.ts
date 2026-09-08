import assert from 'node:assert/strict'

import { test, vi } from 'vitest'

vi.mock('node:crypto', () => ({
  X509Certificate: class {
    fingerprint256: string
    validToDate: Date
    validTo = 'not a parseable display date'

    constructor(pem: string) {
      if (!pem.startsWith('cert:')) throw new Error('unparseable certificate')
      const [, fingerprint, expiry] = pem.split(':')
      this.fingerprint256 = fingerprint
      this.validToDate = new Date(Number(expiry))
    }
  }
}))

test('excludes expired system roots and deduplicates by fingerprint with defaults first', () => {
  const future = Date.now() + 86_400_000
  const past = Date.now() - 86_400_000
  const bundled = `cert:shared:${future}`
  const privateRoot = `cert:private:${future}`
  const tlsApi = fakeTlsApi(
    [bundled, 'unparseable-default'],
    [`cert:expired:${past}`, `${bundled}:alternate-pem`, privateRoot, privateRoot, 'unparseable-system']
  )
  const result = installWindowsSystemCaTrust(tlsApi, 'win32')
  assert.deepEqual(tlsApi.installed, [[bundled, 'unparseable-default', privateRoot, 'unparseable-system']])
  assert.equal(result.systemCertificateCount, 2)
  assert.equal(result.totalCertificateCount, 4)
})

import { installWindowsSystemCaTrust, type NodeTlsCaApi } from './windows-system-ca'

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

  const result = installWindowsSystemCaTrust(tlsApi, 'win32')

  assert.deepEqual(tlsApi.installed, [['mozilla-root', 'extra-ca', 'machine-root', 'user-root']])
  assert.deepEqual(result, {
    applied: true,
    systemCertificateCount: 2,
    totalCertificateCount: 4
  })
})

test('does not inspect or replace CAs outside Windows', () => {
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

  const result = installWindowsSystemCaTrust(tlsApi, 'darwin')

  assert.equal(reads, 0)
  assert.deepEqual(result, {
    applied: false,
    systemCertificateCount: 0,
    totalCertificateCount: 0
  })
})

test('leaves the existing defaults untouched when Windows has no system CAs', () => {
  const tlsApi = fakeTlsApi(['mozilla-root'], [])

  const result = installWindowsSystemCaTrust(tlsApi, 'win32')

  assert.deepEqual(tlsApi.installed, [])
  assert.deepEqual(result, {
    applied: false,
    systemCertificateCount: 0,
    totalCertificateCount: 1
  })
})

test('fails open when the runtime cannot load the Windows certificate store', () => {
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

  const result = installWindowsSystemCaTrust(tlsApi, 'win32')

  assert.deepEqual(result, {
    applied: false,
    systemCertificateCount: 0,
    totalCertificateCount: 0,
    error: 'certificate store unavailable'
  })
})
