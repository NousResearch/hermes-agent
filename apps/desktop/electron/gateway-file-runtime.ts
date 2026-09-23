import http from 'node:http'
import https from 'node:https'
import path from 'node:path'

import { downloadAgentFor, httpStatusError } from './api-transport'
import {
  pathForRegistryBackendRequest,
  pathWithGlobalRemoteProfile,
  type RegistryBackendRequestScope
} from './connection-config'
import {
  filenameFromContentDisposition,
  fsPumpDeps,
  gatewayFilePath,
  gatewayFileRequestPaths,
  isNotFoundError,
  parseDataUrlToBuffer,
  pumpStreamToFile,
  resolveGatewayFileBackend,
  writeBufferToFile
} from './gateway-file-download'
import { DEFAULT_FETCH_TIMEOUT_MS, resolveTimeoutMs } from './hardening'
import { requestWithOauthFallback } from './oauth-rest-request'

interface GatewayFileConnection extends RegistryBackendRequestScope {
  authMode?: 'oauth' | 'token'
  baseUrl: string
  token?: null | string
}

export interface GatewayFileSavePayload {
  sessionId?: string
  connectionId?: unknown
  path?: unknown
  profile?: unknown
  suggestedName?: unknown
}

interface GatewayFileRuntimeDeps {
  dialog: { showSaveDialog: (window: any, options: any) => Promise<{ canceled: boolean; filePath?: string }> }
  electronNet: { request: (options: any) => any }
  ensureBackend: (profile: null | string) => Promise<GatewayFileConnection>
  ensureRegistryBackend: (connectionId: string, profile: null | string) => Promise<GatewayFileConnection>
  ensureNativeAccessToken: (baseUrl: string) => Promise<null | string>
  fetchJsonForBackend: (connection: GatewayFileConnection, requestPath: string) => Promise<any>
  getMainWindow: () => any
  getOauthSessionForUrl: (url: string) => any
  profileRouteOptions: (profile: null | string) => any
}

export function createGatewayFileRuntime({
  dialog,
  electronNet,
  ensureBackend,
  ensureRegistryBackend,
  ensureNativeAccessToken,
  fetchJsonForBackend,
  getMainWindow,
  getOauthSessionForUrl,
  profileRouteOptions
}: GatewayFileRuntimeDeps) {
  // Token-auth download that streams the response body straight to a
  // user-selected destination (via finalizeGatewayDownload) instead of buffering
  // the whole file in memory. The connect timeout is cleared once headers arrive
  // so a slow save dialog or a large stream doesn't trip it. `options.bearer`
  // switches the header to Authorization (RFC 8252 native flow), matching fetchJson.
  function downloadViaTokenToFile(url, token, ctx, options: any = {}) {
    return new Promise((resolve, reject) => {
      let parsed

      try {
        parsed = new URL(url)
      } catch (error) {
        reject(new Error(`Invalid URL: ${error.message}`))

        return
      }

      if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') {
        reject(new Error(`Unsupported Hermes backend URL protocol: ${parsed.protocol}`))

        return
      }

      const client = parsed.protocol === 'https:' ? https : http
      const agent = downloadAgentFor(parsed.protocol)
      const timeoutMs = resolveTimeoutMs(options.timeoutMs, DEFAULT_FETCH_TIMEOUT_MS)

      const req = client.request(
        parsed,
        {
          agent,
          method: 'GET',
          headers: options.bearer ? { Authorization: `Bearer ${options.bearer}` } : { 'X-Hermes-Session-Token': token }
        },
        res => {
          // Headers arrived — the connection phase is done. Drop the idle timeout
          // so it can't abort mid-stream or while the save dialog is open.
          req.setTimeout(0)
          finalizeGatewayDownload(res, res.statusCode || 500, res.headers || {}, {
            ...ctx,
            abort: () => {
              try {
                req.destroy()
              } catch {
                // already finished
              }
            }
          }).then(resolve, reject)
        }
      )

      req.on('error', reject)
      req.setTimeout(timeoutMs, () => {
        req.destroy(new Error(`Timed out connecting to Hermes backend after ${timeoutMs}ms`))
      })
      req.end()
    })
  }

  // OAuth-session download that streams the response body straight to a
  // user-selected destination (via finalizeGatewayDownload). The connect timeout
  // is cleared once the response headers arrive.
  function downloadViaOauthSessionToFile(url, ctx, options: any = {}) {
    return new Promise((resolve, reject) => {
      const sess = getOauthSessionForUrl(url)

      if (!sess) {
        reject(new Error('OAuth session partition is unavailable.'))

        return
      }

      let parsed

      try {
        parsed = new URL(url)
      } catch (error) {
        reject(new Error(`Invalid URL: ${error.message}`))

        return
      }

      if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') {
        reject(new Error(`Unsupported Hermes backend URL protocol: ${parsed.protocol}`))

        return
      }

      const timeoutMs = resolveTimeoutMs(options.timeoutMs, DEFAULT_FETCH_TIMEOUT_MS)

      const request = electronNet.request({
        method: 'GET',
        url,
        session: sess,
        useSessionCookies: true,
        redirect: 'follow'
      } as any)

      let settled = false

      const timer = setTimeout(() => {
        if (settled) {
          return
        }

        settled = true

        try {
          request.abort()
        } catch {
          // already finished
        }

        reject(new Error(`Timed out connecting to Hermes backend after ${timeoutMs}ms`))
      }, timeoutMs)

      request.on('response', res => {
        if (settled) {
          return
        }

        // Response headers arrived — cancel the connect timeout so it can't abort
        // the stream while the save dialog is open or bytes are still flowing.
        settled = true
        clearTimeout(timer)
        finalizeGatewayDownload(res, res.statusCode || 500, res.headers || {}, {
          ...ctx,
          abort: () => {
            try {
              request.abort()
            } catch {
              // already finished
            }
          }
        }).then(resolve, reject)
      })
      request.on('error', error => {
        if (settled) {
          return
        }

        settled = true
        clearTimeout(timer)
        reject(error)
      })
      request.end()
    })
  }

  // Shared tail for both transports: validate status, pick a filename, prompt the
  // save dialog, then stream the (still-unconsumed) response body to the chosen
  // destination. On an HTTP error the status code is attached so saveGatewayFile
  // can trigger the 404-only compatibility fallback.
  async function finalizeGatewayDownload(res, statusCode, headers, ctx: any = {}) {
    if (statusCode >= 400) {
      throw httpStatusError(statusCode, await readGatewayErrorText(res))
    }

    const disposition = headers['content-disposition'] || headers['Content-Disposition']
    const filename = filenameFromContentDisposition(disposition) || ctx.suggested || ctx.fallbackName

    const result = await dialog.showSaveDialog(getMainWindow(), {
      defaultPath: filename,
      title: 'Save File'
    })

    if (result.canceled || !result.filePath) {
      ctx.abort?.()

      return { canceled: true, saved: false }
    }

    try {
      // Failure-atomic: exclusive temp create beside the destination, rename into
      // place only once the body is complete (#96597).
      await pumpStreamToFile(res, result.filePath, fsPumpDeps())
    } catch (error) {
      ctx.abort?.()
      throw error
    }

    return { path: result.filePath, saved: true }
  }

  // Read a bounded amount of an error response body for the thrown message.
  function readGatewayErrorText(res): Promise<string> {
    return new Promise(resolve => {
      const chunks = []
      let total = 0

      res.on('data', chunk => {
        if (total >= 500) {
          return
        }

        const buffer = Buffer.from(chunk)

        total += buffer.length
        chunks.push(buffer)
      })
      res.on('end', () => resolve(Buffer.concat(chunks).toString('utf8').slice(0, 500)))
      res.on('error', () => resolve(Buffer.concat(chunks).toString('utf8').slice(0, 500)))
    })
  }

  interface GatewayFileSaveContext {
    fallbackName: string
    suggested: string
  }

  function gatewayFileRequestPath(
    connection: GatewayFileConnection,
    connectionId: null | string,
    profile: null | string,
    requestPath: string
  ) {
    return connectionId
      ? pathForRegistryBackendRequest(requestPath, profile, connection)
      : pathWithGlobalRemoteProfile(requestPath, profile, profileRouteOptions(profile))
  }

  async function saveGatewayFile(payload: GatewayFileSavePayload = {}) {
    const filePath = gatewayFilePath(payload.path)

    if (!filePath) {
      throw new Error('Missing gateway file path')
    }

    const { connection, connectionId, profile } = await resolveGatewayFileBackend<GatewayFileConnection>(payload, {
      ensureLegacy: ensureBackend,
      ensureRegistry: ensureRegistryBackend
    })

    const suggested = String(payload.suggestedName || '').trim()
    const fallbackName = path.basename(filePath) || suggested || 'download'
    const ctx = { suggested, fallbackName }

    const requestPaths = gatewayFileRequestPaths(
      filePath,
      requestPath => gatewayFileRequestPath(connection, connectionId, profile, requestPath),
      payload.sessionId
    )

    const url = `${connection.baseUrl}${requestPaths.download}`

    try {
      if (connection.authMode === 'oauth') {
        return await requestWithOauthFallback(connection.baseUrl, {
          ensureNativeAccessToken,
          requestWithBearer: bearer => downloadViaTokenToFile(url, null, ctx, { bearer }),
          requestWithCookie: () => downloadViaOauthSessionToFile(url, ctx)
        })
      }

      return await downloadViaTokenToFile(url, connection.token, ctx)
    } catch (error) {
      // Desktop and the remote gateway update independently. A gateway predating
      // /api/fs/download 404s here; fall back (ONLY on 404) to the older capped
      // data-URL route so downloads keep working against older backends.
      if (isNotFoundError(error)) {
        return await saveGatewayFileViaDataUrl(connection, requestPaths.dataUrl, ctx)
      }

      throw error
    }
  }

  // Compatibility fallback: fetch the file through the capped
  // `/api/fs/read-data-url` route, decode it, and save. Bounded by the gateway's
  // data-URL cap, so it only serves smaller files — enough to keep older gateways
  // working until they gain the streaming route.
  async function saveGatewayFileViaDataUrl(
    connection: GatewayFileConnection,
    requestPath: string,
    ctx: GatewayFileSaveContext
  ) {
    const json = await fetchJsonForBackend(connection, requestPath)

    const dataUrl =
      json && typeof json === 'object' && 'dataUrl' in json && typeof json.dataUrl === 'string' ? json.dataUrl : ''

    if (!dataUrl) {
      throw new Error('Gateway returned no file data')
    }

    const buffer = parseDataUrlToBuffer(dataUrl)
    const filename = ctx.suggested || ctx.fallbackName

    const result = await dialog.showSaveDialog(getMainWindow(), {
      defaultPath: filename,
      title: 'Save File'
    })

    if (result.canceled || !result.filePath) {
      return { canceled: true, saved: false }
    }

    // Same failure-atomic contract as the streaming path: a direct writeFile
    // truncates an existing destination before the write completes (#96597).
    await writeBufferToFile(buffer, result.filePath, fsPumpDeps())

    return { path: result.filePath, saved: true }
  }

  return { saveGatewayFile }
}
