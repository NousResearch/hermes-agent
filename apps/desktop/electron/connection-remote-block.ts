import { normalizeRemoteBaseUrl, normalizeRemoteHeaders, normalizeRemoteProfileName } from './connection-config'

// Build + validate a `{ url, authMode, token }` remote block. OAuth gateways
// authenticate via the login-window session cookie (verified at connect time in
// resolveRemoteBackend), so only token-auth remotes require a saved token.
// `org` (optional) is the Hermes Cloud org slug/id the instance was discovered
// under — persisted so Settings can reopen into the same org; omitted from the
// block when empty so plain remote connections stay unchanged.
export function buildRemoteBlock(
  decrypt: (secret: unknown) => string,
  remoteUrl: string,
  authMode: string,
  token: any,
  org?: string,
  headers?: object,
  name?: string,
  remoteProfile?: string
) {
  if (authMode !== 'oauth' && !decrypt(token)) {
    throw new Error('Remote gateway session token is required.')
  }

  const block: {
    url: string
    authMode: string
    token: object
    headers?: object
    org?: string
    name?: string
    remoteProfile?: string
  } = {
    url: normalizeRemoteBaseUrl(remoteUrl),
    authMode,
    token
  }

  const remoteHeaders = normalizeRemoteHeaders(headers)

  if (Object.keys(remoteHeaders).length > 0) {
    block.headers = remoteHeaders
  }

  const nameValue = typeof name === 'string' ? name.trim() : ''

  if (nameValue) {
    block.name = nameValue
  }

  const orgValue = typeof org === 'string' ? org.trim() : ''

  if (orgValue) {
    block.org = orgValue
  }

  const remoteProfileName = normalizeRemoteProfileName(remoteProfile)

  if (remoteProfileName) {
    block.remoteProfile = remoteProfileName.remoteProfile
  }

  return block
}
