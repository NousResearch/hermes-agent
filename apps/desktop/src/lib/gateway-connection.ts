import type { DesktopConnectionConfig, HermesConnection } from '@/global'

const LOCAL_GATEWAY_HOSTS = new Set(['127.0.0.1', 'localhost', '[::1]'])

/** True when the desktop shell talks to a gateway that is not on this machine. */
export function isRemoteGatewayConnection(conn: HermesConnection | null | undefined): boolean {
  if (!conn) {
    return false
  }

  if (conn.mode === 'remote') {
    return true
  }

  const candidates = [conn.baseUrl, conn.wsUrl].filter(Boolean)

  for (const raw of candidates) {
    try {
      const hostname = new URL(raw).hostname.toLowerCase()

      if (!LOCAL_GATEWAY_HOSTS.has(hostname)) {
        return true
      }
    } catch {
      // Ignore malformed URLs and fall through to other signals.
    }
  }

  return false
}

export function isRemoteGatewayConfig(config: DesktopConnectionConfig | null | undefined): boolean {
  return config?.mode === 'remote'
}

/** Paths written by the desktop composer's local image cache (never on a remote gateway). */
export function isDesktopComposerImagePath(filePath: string): boolean {
  const normalized = filePath.replace(/\\/g, '/').toLowerCase()

  return normalized.includes('/composer-images/') && /\/composer_[^/]+\.(png|jpe?g|gif|webp|bmp|tiff?|svg)$/i.test(normalized)
}

/** Resolve whether pasted/attached images must be inlined instead of image.attach. */
export async function shouldInlineImageAttachmentsForGateway(): Promise<boolean> {
  const desktop = window.hermesDesktop

  if (!desktop) {
    return false
  }

  // Lazy import avoids a circular dependency through the session store.
  const { $connection } = await import('@/store/session')
  const cached = $connection.get()

  if (isRemoteGatewayConnection(cached)) {
    return true
  }

  try {
    const live = await desktop.getConnection()

    if (isRemoteGatewayConnection(live)) {
      return true
    }
  } catch {
    // Fall through to saved settings.
  }

  try {
    const config = await desktop.getConnectionConfig()

    if (isRemoteGatewayConfig(config)) {
      return true
    }
  } catch {
    // Fall through.
  }

  return false
}