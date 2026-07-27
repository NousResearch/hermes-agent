import assert from 'node:assert/strict'

import { test } from 'vitest'

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
