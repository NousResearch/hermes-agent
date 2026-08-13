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

    const certificates = [...defaultCertificates, ...systemCertificates]
    tlsApi.setDefaultCACertificates(certificates)

    return {
      applied: true,
      systemCertificateCount: systemCertificates.length,
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
