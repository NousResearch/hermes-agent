import { resolveGatewayWsUrl } from '@hermes/shared'
import QRCode from 'qrcode'
import { useCallback, useEffect, useRef, useState } from 'react'

import { useGatewayRequest } from '@/app/gateway/hooks/use-gateway-request'
import { Button } from '@/components/ui/button'
import { useI18n } from '@/i18n'
import { Loader2, Monitor, RefreshCw } from '@/lib/icons'

import { SettingsSection } from './primitives'

const PAIRING_CODE_TTL_MS = 2 * 60 * 1000

type PairingView =
  | { kind: 'hidden' }
  | { kind: 'loading' }
  | { dataUrl: string; kind: 'ready' }
  | { kind: 'error'; message: string }

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

export function readPublicGatewayUrl(response: unknown): string | null {
  if (!isRecord(response) || !isRecord(response.config) || !isRecord(response.config.dashboard)) {
    return null
  }

  const value = response.config.dashboard.public_url

  return typeof value === 'string' && value.trim() ? value.trim() : null
}

function normalizePublicGatewayUrl(value: string): string {
  let url: URL

  try {
    url = new URL(value)
  } catch {
    throw new Error('invalid-public-url')
  }

  const hostname = url.hostname.toLowerCase()

  const isLoopback =
    hostname === 'localhost' ||
    hostname.endsWith('.localhost') ||
    hostname === '0.0.0.0' ||
    hostname === '::1' ||
    hostname === '[::1]' ||
    hostname.startsWith('127.')

  if (url.protocol !== 'https:' || isLoopback || url.username || url.password) {
    throw new Error('invalid-public-url')
  }

  url.search = ''
  url.hash = ''
  url.pathname = url.pathname.replace(/\/+$/, '')

  return url.toString().replace(/\/$/, '')
}

export function buildMobilePairingPayload(publicGatewayUrl: string, websocketUrl: string): string {
  const server = normalizePublicGatewayUrl(publicGatewayUrl)
  let socket: URL

  try {
    socket = new URL(websocketUrl)
  } catch {
    throw new Error('missing-token')
  }

  const token = socket.searchParams.get('token')?.trim()

  if (!token || socket.searchParams.has('ticket')) {
    throw new Error('missing-token')
  }

  const query = new URLSearchParams({ server, token })

  return `hermes-agent://desktop-pair?${query.toString()}`
}

export function MobileCompanionPairing() {
  const { requestGateway } = useGatewayRequest()
  const { t } = useI18n()
  const copy = t.settings.mobilePairing
  const [view, setView] = useState<PairingView>({ kind: 'hidden' })
  const generationRef = useRef(0)

  const hide = useCallback(() => {
    generationRef.current += 1
    setView({ kind: 'hidden' })
  }, [])

  const generate = useCallback(async () => {
    const generation = generationRef.current + 1
    generationRef.current = generation
    setView({ kind: 'loading' })

    try {
      const desktop = window.hermesDesktop

      if (!desktop) {
        throw new Error('desktop-unavailable')
      }

      const [configResponse, connection] = await Promise.all([
        requestGateway<unknown>('config.get', { key: 'full' }),
        desktop.getConnection()
      ])

      const publicUrl = readPublicGatewayUrl(configResponse)

      if (!publicUrl) {
        throw new Error('missing-public-url')
      }

      if (connection.authMode === 'oauth') {
        throw new Error('oauth-not-shareable')
      }

      const websocketUrl = await resolveGatewayWsUrl(desktop, connection)
      const payload = buildMobilePairingPayload(publicUrl, websocketUrl)

      const dataUrl = await QRCode.toDataURL(payload, {
        color: { dark: '#111111', light: '#ffffff' },
        errorCorrectionLevel: 'M',
        margin: 2,
        width: 288
      })

      if (generationRef.current === generation) {
        setView({ dataUrl, kind: 'ready' })
      }
    } catch (error) {
      if (generationRef.current !== generation) {
        return
      }

      const code = error instanceof Error ? error.message : ''

      const message =
        code === 'missing-public-url' || code === 'invalid-public-url'
          ? copy.publicUrlRequired
          : code === 'oauth-not-shareable' || code === 'missing-token'
            ? copy.tokenRequired
            : copy.failed

      setView({ kind: 'error', message })
    }
  }, [copy.failed, copy.publicUrlRequired, copy.tokenRequired, requestGateway])

  useEffect(() => {
    if (view.kind !== 'ready') {
      return
    }

    const timeout = window.setTimeout(hide, PAIRING_CODE_TTL_MS)

    return () => window.clearTimeout(timeout)
  }, [hide, view.kind])

  return (
    <SettingsSection icon={Monitor} title={copy.title}>
      <div className="rounded-xl border border-border/70 bg-(--ui-bg-secondary) p-4">
        <div className="max-w-2xl text-[length:var(--conversation-caption-font-size)] leading-(--conversation-caption-line-height) text-(--ui-text-tertiary)">
          {copy.description}
        </div>

        {view.kind === 'ready' ? (
          <div className="mt-4 flex flex-col items-start gap-3 sm:flex-row sm:items-center">
            <div className="rounded-xl bg-white p-2 shadow-sm">
              <img alt={copy.qrAlt} className="size-52" src={view.dataUrl} />
            </div>
            <div className="max-w-sm text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
              <p className="font-medium text-foreground">{copy.scan}</p>
              <p className="mt-1">{copy.expires}</p>
              <div className="mt-3 flex flex-wrap gap-2">
                <Button onClick={() => void generate()} size="sm" variant="textStrong">
                  <RefreshCw />
                  {copy.refresh}
                </Button>
                <Button onClick={hide} size="sm" variant="text">
                  {copy.hide}
                </Button>
              </div>
            </div>
          </div>
        ) : (
          <div className="mt-4">
            {view.kind === 'error' ? (
              <p className="mb-3 max-w-2xl text-[length:var(--conversation-caption-font-size)] text-destructive">
                {view.message}
              </p>
            ) : null}
            <Button disabled={view.kind === 'loading'} onClick={() => void generate()} size="sm">
              {view.kind === 'loading' ? <Loader2 className="animate-spin" /> : null}
              {view.kind === 'error' ? copy.retry : copy.show}
            </Button>
          </div>
        )}
      </div>
    </SettingsSection>
  )
}
