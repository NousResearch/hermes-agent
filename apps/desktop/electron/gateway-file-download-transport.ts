import http from 'node:http'
import https from 'node:https'

import { downloadAgentFor } from './api-transport'
import type {
  GatewayDownloadResponse,
  GatewayFileSaveContext,
  GatewayFileSaveDeps,
  GatewayFileSaveResult
} from './gateway-file-download'
import { finalizeGatewayDownload } from './gateway-file-download'
import { DEFAULT_FETCH_TIMEOUT_MS, resolveTimeoutMs } from './hardening'

export interface GatewayDownloadOptions {
  bearer?: string
  timeoutMs?: number
}

function downloadUrl(url: string): URL {
  let parsed: URL

  try {
    parsed = new URL(url)
  } catch (error) {
    throw new Error(`Invalid URL: ${error instanceof Error ? error.message : String(error)}`)
  }

  if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') {
    throw new Error(`Unsupported Hermes backend URL protocol: ${parsed.protocol}`)
  }

  return parsed
}

export function downloadViaTokenToFile(
  url: string,
  token: null | string,
  context: GatewayFileSaveContext,
  deps: GatewayFileSaveDeps,
  options: GatewayDownloadOptions = {}
): Promise<GatewayFileSaveResult> {
  return new Promise((resolve: (result: GatewayFileSaveResult) => void, reject: (error: Error) => void): void => {
    const parsed: URL = downloadUrl(url)
    const client: typeof http | typeof https = parsed.protocol === 'https:' ? https : http
    const timeoutMs: number = resolveTimeoutMs(options.timeoutMs, DEFAULT_FETCH_TIMEOUT_MS)

    const request: http.ClientRequest = client.request(
      parsed,
      {
        agent: downloadAgentFor(parsed.protocol),
        method: 'GET',
        headers: options.bearer
          ? { Authorization: `Bearer ${options.bearer}` }
          : { 'X-Hermes-Session-Token': token ?? '' }
      },
      (response: http.IncomingMessage): void => {
        // Headers end the connection deadline, not the user's save-dialog time.
        request.setTimeout(0)
        void finalizeGatewayDownload(
          response,
          context,
          (): void => {
            request.destroy()
          },
          deps
        ).then(resolve, reject)
      }
    )

    request.on('error', reject)
    request.setTimeout(timeoutMs, (): void => {
      request.destroy(new Error(`Timed out connecting to Hermes backend after ${timeoutMs}ms`))
    })
    request.end()
  })
}

export interface GatewayOauthDownloadRequest {
  on(event: 'response', listener: (response: GatewayDownloadResponse) => void): void
  on(event: 'error', listener: (error: Error) => void): void
  abort(): void
  end(): void
}

export interface GatewayOauthRequestOptions<S> {
  method: 'GET'
  url: string
  session: S
  useSessionCookies: true
  redirect: 'follow'
}

export interface GatewayOauthDownloadDeps<S> extends GatewayFileSaveDeps {
  getSession: (url: string) => S | null
  request: (options: GatewayOauthRequestOptions<S>) => GatewayOauthDownloadRequest
}

export function downloadViaOauthSessionToFile<S>(
  url: string,
  context: GatewayFileSaveContext,
  deps: GatewayOauthDownloadDeps<S>,
  options: GatewayDownloadOptions = {}
): Promise<GatewayFileSaveResult> {
  return new Promise((resolve: (result: GatewayFileSaveResult) => void, reject: (error: Error) => void): void => {
    const session: S | null = deps.getSession(url)

    if (!session) {
      throw new Error('OAuth session partition is unavailable.')
    }

    downloadUrl(url)
    const timeoutMs: number = resolveTimeoutMs(options.timeoutMs, DEFAULT_FETCH_TIMEOUT_MS)

    const request: GatewayOauthDownloadRequest = deps.request({
      method: 'GET',
      url,
      session,
      useSessionCookies: true,
      redirect: 'follow'
    })

    let settled: boolean = false

    const timer: ReturnType<typeof setTimeout> = setTimeout((): void => {
      if (settled) {
        return
      }

      settled = true
      request.abort()
      reject(new Error(`Timed out connecting to Hermes backend after ${timeoutMs}ms`))
    }, timeoutMs)

    request.on('response', (response: GatewayDownloadResponse): void => {
      if (settled) {
        return
      }

      settled = true
      clearTimeout(timer)
      void finalizeGatewayDownload(
        response,
        context,
        (): void => {
          request.abort()
        },
        deps
      ).then(resolve, reject)
    })
    request.on('error', (error: Error): void => {
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
