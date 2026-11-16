/** The connected process owns its runtime version, not a checkout on disk.
 *  `displayVersion` carries the distance past the release tag (`0.21.5+1913`);
 *  a backend that predates it only reports the bare release. */
export async function resolveGatewayVersion(request: (path: string) => Promise<unknown>): Promise<string> {
  try {
    const status: unknown = await request('/api/health')

    if (status && typeof status === 'object') {
      if ('displayVersion' in status && typeof status.displayVersion === 'string' && status.displayVersion) {
        return status.displayVersion
      }

      if ('version' in status && typeof status.version === 'string') {
        return status.version
      }
    }
  } catch {
    // An offline gateway has no known running version.
  }

  return ''
}
