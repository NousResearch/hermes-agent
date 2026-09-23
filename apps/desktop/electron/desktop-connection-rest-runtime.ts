import { requestWithOauthFallback } from './oauth-rest-request'

export interface DesktopConnectionRestRuntimeDeps {
  ensureNativeAccessToken: any
  fetchJson: any
  fetchJsonViaOauthSession: any
}

export function createDesktopConnectionRestRuntime(deps: DesktopConnectionRestRuntimeDeps) {
  const { ensureNativeAccessToken, fetchJson, fetchJsonViaOauthSession } = deps

  // Convenience wrappers around the bearer-aware descriptor request path.
  // Native OAuth sessions are cookieless, so these must not bypass
  // fetchJsonForBackend and fall straight through to the cookie partition.
  async function postJsonForBackend(descriptor, path, body, opts: any = {}) {
    return fetchJsonForBackend(descriptor, path, { ...opts, body: body ?? {}, method: 'POST' })
  }

  // GET twin of postJsonForBackend.
  async function getJsonForBackend(descriptor, path, opts: any = {}) {
    return fetchJsonForBackend(descriptor, path, opts)
  }

  // Any-method REST call against a resolved backend descriptor — the descriptor
  // analogue of the hermes:api handler's own auth split: OAuth backends prefer a
  // native bearer (cookieless RFC 8252 flow) and fall back to the OAuth cookie
  // partition; token/local descriptors use the static session-token header.
  async function fetchJsonForBackend(
    descriptor,
    path,
    opts: { method?: string; body?: unknown; upload?: unknown; timeoutMs?: number } = {}
  ) {
    const url = `${descriptor.baseUrl}${path}`

    if (descriptor.authMode === 'oauth') {
      // The OAuth cookie path rides electron.net with JSON headers; multipart
      // isn't wired there. Fail loudly rather than corrupting the upload.
      if (opts.upload) {
        throw new Error('File uploads are not supported against OAuth-gated remote backends yet.')
      }

      const options = {
        method: opts.method,
        body: opts.body,
        timeoutMs: opts.timeoutMs,
        headers: descriptor.headers
      }

      return requestWithOauthFallback(descriptor.baseUrl, {
        ensureNativeAccessToken,
        requestWithBearer: bearer => fetchJson(url, null, { ...options, bearer }),
        requestWithCookie: () => fetchJsonViaOauthSession(url, options)
      })
    }

    return fetchJson(url, descriptor.token, {
      method: opts.method,
      body: opts.body,
      upload: opts.upload,
      timeoutMs: opts.timeoutMs,
      headers: descriptor.headers
    })
  }

  return { postJsonForBackend, getJsonForBackend, fetchJsonForBackend }
}
