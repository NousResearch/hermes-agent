const PLUGIN_MANIFEST_ID = /^[a-z0-9](?:[a-z0-9._-]*[a-z0-9])?$/
const URI_SCHEME = /^[a-z][a-z0-9+.-]*:/i

export function isValidPluginManifestId(value: string): boolean {
  return PLUGIN_MANIFEST_ID.test(value)
}

export function normalizePluginRelativePath(path: string): string {
  if (typeof path !== 'string' || !path || path.includes('#')) {
    throw new TypeError('Invalid plugin-relative API path')
  }

  if (URI_SCHEME.test(path) || path.startsWith('//')) {
    throw new TypeError('Unsafe plugin-relative API path')
  }

  const withLead = path.startsWith('/') ? path : `/${path}`
  const queryIndex = withLead.indexOf('?')
  const pathname = queryIndex === -1 ? withLead : withLead.slice(0, queryIndex)

  if (pathname === '/') {
    throw new TypeError('Invalid plugin-relative API path')
  }

  let decoded = pathname

  for (let pass = 0; pass < 2; pass += 1) {
    assertSafePathname(decoded)

    try {
      decoded = decodeURIComponent(decoded)
    } catch {
      throw new TypeError('Invalid percent-encoding in plugin-relative API path')
    }
  }

  assertSafePathname(decoded)

  return withLead
}

export function buildPluginApiPath(manifestId: string, pluginPath: string): string {
  if (typeof manifestId !== 'string' || !isValidPluginManifestId(manifestId)) {
    throw new TypeError('Invalid plugin manifest id')
  }

  return `/api/plugins/${manifestId}${normalizePluginRelativePath(pluginPath)}`
}

function assertSafePathname(pathname: string): void {
  if (
    URI_SCHEME.test(pathname) ||
    pathname.includes('\\') ||
    pathname.split('/').some(segment => segment === '.' || segment === '..')
  ) {
    throw new TypeError('Unsafe plugin-relative API path')
  }
}
