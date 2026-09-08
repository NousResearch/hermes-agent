import { X509Certificate } from 'node:crypto'

interface NodeTlsCaApi {
  getCACertificates(type?: 'default' | 'system'): string[]
  setDefaultCACertificates(certificates: string[]): void
}

interface WindowsSystemCaResult {
  applied: boolean
  systemCertificateCount: number
  totalCertificateCount: number
  error?: string
}

function installWindowsSystemCaTrust(tlsApi: NodeTlsCaApi, platform = process.platform): WindowsSystemCaResult {
  if (platform !== 'win32') {
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
        if (certificate.validToDate.getTime() <= now || seen.has(certificate.fingerprint256)) return false
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

export { installWindowsSystemCaTrust }
export type { NodeTlsCaApi, WindowsSystemCaResult }
