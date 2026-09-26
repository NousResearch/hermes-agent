import { X509Certificate } from 'node:crypto'

interface NodeTlsCaApi {
  getCACertificates(type?: 'default' | 'system'): string[]
  setDefaultCACertificates(certificates: string[]): void
}

interface SystemCaResult {
  applied: boolean
  systemCertificateCount: number
  totalCertificateCount: number
  error?: string
}

// Platforms whose OS trust store tls.getCACertificates('system') can enumerate: the Windows
// cert store and the macOS keychain (Node reads user + System keychains there, honoring the
// "Always Trust" SSL policy — Node ≥ 22.15). Linux is deliberately absent: its 'system' store
// is the OpenSSL directory scan, which the default trust already covers.
const SYSTEM_CA_PLATFORMS = new Set(['win32', 'darwin'])

function installSystemCaTrust(tlsApi: NodeTlsCaApi, platform = process.platform): SystemCaResult {
  if (!SYSTEM_CA_PLATFORMS.has(platform)) {
    return {
      applied: false,
      systemCertificateCount: 0,
      totalCertificateCount: 0
    }
  }

  try {
    const defaultCertificates = tlsApi.getCACertificates('default')
    const systemCertificates = tlsApi.getCACertificates('system')

    if (systemCertificates.length === 0) {
      return {
        applied: false,
        systemCertificateCount: 0,
        totalCertificateCount: defaultCertificates.length
      }
    }

    // Prefer existing defaults. Expired Windows roots can divert OpenSSL onto
    // an expired chain even when a valid bundled trust path exists.
    const seen = new Set<string>()
    const now = Date.now()

    const keepCertificate = (pem: string): boolean => {
      try {
        const certificate = new X509Certificate(pem)

        if (certificate.validToDate.getTime() <= now || seen.has(certificate.fingerprint256)) {
          return false
        }

        seen.add(certificate.fingerprint256)
      } catch {
        // Leave PEM acceptability to Node if its X.509 parser cannot inspect it.
      }

      return true
    }

    const filteredDefaults = defaultCertificates.filter(keepCertificate)
    const filteredSystem = systemCertificates.filter(keepCertificate)
    const certificates = [...filteredDefaults, ...filteredSystem]

    tlsApi.setDefaultCACertificates(certificates)

    return {
      applied: true,
      systemCertificateCount: filteredSystem.length,
      totalCertificateCount: certificates.length
    }
  } catch (error) {
    return {
      applied: false,
      systemCertificateCount: 0,
      totalCertificateCount: 0,
      error: error instanceof Error ? error.message : String(error)
    }
  }
}

export { installSystemCaTrust }
export type { NodeTlsCaApi, SystemCaResult }
