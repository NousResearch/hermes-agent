import { existsSync, statSync } from 'node:fs'
import path from 'node:path'
import { pathToFileURL } from 'node:url'

import { CLERK_RENDERER_HOST, CLERK_RENDERER_ORIGIN, CLERK_RENDERER_SCHEME } from './clerk-config'

export interface RendererLocation {
  customScheme: boolean
  devServer: string | null
  rendererIndexPath?: string
}

export function rendererBaseUrl({ customScheme, devServer, rendererIndexPath }: RendererLocation): string {
  if (devServer) {
    return `${devServer.replace(/\/+$/, '')}/`
  }

  if (customScheme) {
    return CLERK_RENDERER_ORIGIN
  }

  if (!rendererIndexPath) {
    throw new Error('rendererIndexPath is required when the custom renderer scheme is disabled')
  }

  return pathToFileURL(rendererIndexPath).toString()
}

export function isTrustedRendererUrl(
  url: string,
  { customScheme, devServer }: Pick<RendererLocation, 'customScheme' | 'devServer'>
): boolean {
  if (devServer) {
    const base = devServer.replace(/\/+$/, '')

    if (url === base || url.startsWith(`${base}/`)) {
      return true
    }
  }

  if (customScheme) {
    try {
      const parsed = new URL(url)

      return parsed.protocol === `${CLERK_RENDERER_SCHEME}:` && parsed.host === CLERK_RENDERER_HOST
    } catch {
      return false
    }
  }

  return url.startsWith('file:')
}

export function resolveRendererFileRequest(requestUrl: string, root: string): string | null {
  let parsed: URL

  try {
    parsed = new URL(requestUrl)
  } catch {
    return null
  }

  if (parsed.protocol !== `${CLERK_RENDERER_SCHEME}:` || parsed.host !== CLERK_RENDERER_HOST) {
    return null
  }

  let pathname: string

  try {
    pathname = decodeURIComponent(parsed.pathname)
  } catch {
    return null
  }

  const rootPath = path.resolve(root)
  const indexPath = path.join(rootPath, 'index.html')
  const relative = pathname.replace(/^\/+/, '')

  if (!relative) {
    return indexPath
  }

  const candidate = path.resolve(rootPath, relative)
  const relativeToRoot = path.relative(rootPath, candidate)

  if (relativeToRoot.startsWith('..') || path.isAbsolute(relativeToRoot)) {
    return null
  }

  if (existsSync(candidate) && statSync(candidate).isFile()) {
    return candidate
  }

  return path.extname(relative) ? null : indexPath
}
