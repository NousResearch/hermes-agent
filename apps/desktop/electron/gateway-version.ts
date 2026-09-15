/** The connected process owns its runtime version, not a checkout on disk. */
export async function resolveGatewayVersion(request: (path: string) => Promise<unknown>): Promise<string> {
  try {
    const status = await request('/api/health')

    if (status && typeof status === 'object' && 'version' in status && typeof status.version === 'string') {
      return status.version
    }
  } catch {
    // An offline gateway has no known running version.
  }

  return ''
}
