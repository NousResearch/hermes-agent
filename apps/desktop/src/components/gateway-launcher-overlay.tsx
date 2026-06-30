import { useEffect, useMemo, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { LogView } from '@/components/ui/log-view'
import type { DesktopConnectionProbeResult } from '@/global'
import { AlertCircle, Check, FileText, Globe, Loader2, LogIn, RefreshCw, Settings2 } from '@/lib/icons'
import { cn } from '@/lib/utils'
import { notify, notifyError } from '@/store/notifications'

import type { GatewayStartupKind } from './gateway-launcher-state'

const COMMON_PORTS = ['9119', '8000', '3000']

function assetPath(path: string) {
  return `${import.meta.env.BASE_URL}${path.replace(/^\/+/, '')}`
}

function normalizePathPrefix(value: string) {
  const trimmed = value.trim()
  if (!trimmed) return ''

  return `/${trimmed.replace(/^\/+/, '').replace(/\/+$/, '')}`
}

function buildRemoteUrl({ host, pathPrefix, port, scheme }: { host: string; pathPrefix: string; port: string; scheme: string }) {
  const cleanHost = host
    .trim()
    .replace(/^https?:\/\//i, '')
    .replace(/\/.*$/, '')
  const cleanPort = port.trim()
  const portSuffix = cleanPort ? `:${cleanPort.replace(/^:+/, '')}` : ''

  if (!cleanHost) {
    return ''
  }

  return `${scheme}://${cleanHost}${portSuffix}${normalizePathPrefix(pathPrefix)}`
}

function isPlausibleUrl(value: string) {
  return /^https?:\/\/[^/\s:]+(?::\d+)?(?:\/\S*)?$/i.test(value.trim())
}

interface GatewaySetupPanelProps {
  onConfigured: () => void
}

export function GatewaySetupPanel({ onConfigured }: GatewaySetupPanelProps) {
  const [scheme, setScheme] = useState('http')
  const [host, setHost] = useState('')
  const [port, setPort] = useState('9119')
  const [pathPrefix, setPathPrefix] = useState('')
  const [remoteUrl, setRemoteUrl] = useState('')
  const [remoteToken, setRemoteToken] = useState('')
  const [probe, setProbe] = useState<DesktopConnectionProbeResult | null>(null)
  const [probing, setProbing] = useState(false)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const constructedUrl = useMemo(
    () => buildRemoteUrl({ host, pathPrefix, port, scheme }),
    [host, pathPrefix, port, scheme]
  )
  const trimmedUrl = remoteUrl.trim()
  const authMode = probe?.reachable && probe.authMode !== 'unknown' ? probe.authMode : 'oauth'
  const needsToken = authMode === 'token'

  useEffect(() => {
    if (!trimmedUrl || !isPlausibleUrl(trimmedUrl)) {
      setProbe(null)
      setProbing(false)

      return
    }

    const desktop = window.hermesDesktop
    if (!desktop?.probeConnectionConfig) {
      return
    }

    let cancelled = false
    setProbing(true)

    const timer = window.setTimeout(() => {
      desktop
        .probeConnectionConfig(trimmedUrl)
        .then(result => {
          if (!cancelled) {
            setProbe(result)
          }
        })
        .catch(() => {
          if (!cancelled) {
            setProbe(null)
          }
        })
        .finally(() => {
          if (!cancelled) {
            setProbing(false)
          }
        })
    }, 350)

    return () => {
      cancelled = true
      window.clearTimeout(timer)
    }
  }, [trimmedUrl])

  const save = async () => {
    const desktop = window.hermesDesktop
    if (!desktop?.saveConnectionConfig) {
      setError('Desktop gateway settings are unavailable.')

      return
    }

    if (!isPlausibleUrl(trimmedUrl)) {
      setError('Enter a full gateway URL, for example http://mercury2:9119.')

      return
    }

    if (needsToken && !remoteToken.trim()) {
      setError('This gateway expects a session token. Paste the token before saving.')

      return
    }

    setSaving(true)
    setError(null)

    try {
      await desktop.saveConnectionConfig({
        mode: 'remote',
        remoteAuthMode: authMode,
        remoteToken: needsToken ? remoteToken.trim() : undefined,
        remoteUrl: trimmedUrl
      })

      if (authMode === 'oauth' && probe?.reachable && desktop.oauthLoginConnectionConfig) {
        const result = await desktop.oauthLoginConnectionConfig(trimmedUrl)

        if (!result.connected) {
          notify({
            kind: 'warning',
            title: 'Gateway saved',
            message: 'The gateway URL was saved, but sign-in did not finish. You can sign in from the recovery screen.'
          })
          onConfigured()

          return
        }
      }

      notify({ kind: 'success', title: 'Gateway saved', message: 'Reconnecting to the configured gateway…' })
      onConfigured()
    } catch (err) {
      notifyError(err, 'Could not save gateway')
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="grid gap-5">
      <div className="flex items-center gap-4">
        <img alt="" className="size-16 rounded-2xl border border-(--ui-stroke-tertiary)" src={assetPath('reuben-icon.png')} />
        <div>
          <div className="text-[0.68rem] font-semibold uppercase tracking-[0.26em] text-primary">Reuben Agent</div>
          <h1 className="mt-1 text-xl font-semibold tracking-tight text-(--ui-text-primary)">Choose a gateway</h1>
          <p className="mt-1 max-w-[34rem] text-sm leading-6 text-(--ui-text-tertiary)">
            Reuben is a remote desktop client. Connect it to an already-running Reuben or Hermes gateway to begin.
          </p>
        </div>
      </div>

      <div className="grid gap-3 rounded-xl border border-(--ui-stroke-tertiary) bg-(--ui-bg-tertiary)/45 p-4">
        <div className="grid gap-3 sm:grid-cols-[1fr_7rem_8rem]">
          <label className="grid gap-1.5 text-xs font-medium text-(--ui-text-secondary)">
            Hostname
            <Input
              onChange={event => setHost(event.target.value)}
              onKeyDown={event => {
                if (event.key === 'Enter' && constructedUrl) {
                  setRemoteUrl(constructedUrl)
                }
              }}
              placeholder="mercury2"
              value={host}
            />
          </label>
          <label className="grid gap-1.5 text-xs font-medium text-(--ui-text-secondary)">
            Scheme
            <select
              className="h-8 rounded-[4px] border border-(--ui-stroke-secondary) bg-(--ui-bg-quaternary) px-2 text-sm text-(--ui-text-primary)"
              onChange={event => setScheme(event.target.value)}
              value={scheme}
            >
              <option value="http">http</option>
              <option value="https">https</option>
            </select>
          </label>
          <label className="grid gap-1.5 text-xs font-medium text-(--ui-text-secondary)">
            Port
            <Input onChange={event => setPort(event.target.value)} placeholder="9119" value={port} />
          </label>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          {COMMON_PORTS.map(value => (
            <Button key={value} onClick={() => setPort(value)} size="xs" type="button" variant={port === value ? 'secondary' : 'outline'}>
              {value}
            </Button>
          ))}
          <Input
            className="h-7 max-w-[13rem]"
            onChange={event => setPathPrefix(event.target.value)}
            placeholder="/hermes path prefix"
            value={pathPrefix}
          />
          <Button disabled={!constructedUrl} onClick={() => setRemoteUrl(constructedUrl)} size="xs" type="button">
            Use constructed URL
          </Button>
        </div>
      </div>

      <label className="grid gap-1.5 text-xs font-medium text-(--ui-text-secondary)">
        Full gateway URL
        <Input
          autoFocus
          className="font-mono"
          onChange={event => setRemoteUrl(event.target.value)}
          onKeyDown={event => {
            if (event.key === 'Enter') {
              void save()
            }
          }}
          placeholder="http://mercury2:9119"
          value={remoteUrl}
        />
      </label>

      {probing ? (
        <div className="flex items-center gap-2 text-xs text-(--ui-text-tertiary)">
          <Loader2 className="size-3.5 animate-spin" />
          Checking gateway…
        </div>
      ) : probe?.reachable ? (
        <div className="flex items-center gap-2 text-xs text-primary">
          <Check className="size-3.5" />
          Gateway reachable. Authentication: {authMode === 'oauth' ? 'sign-in session' : 'session token'}.
        </div>
      ) : probe?.error ? (
        <div className="flex items-start gap-2 text-xs text-(--ui-text-tertiary)">
          <AlertCircle className="mt-0.5 size-3.5 shrink-0" />
          {probe.error}
        </div>
      ) : null}

      {needsToken ? (
        <label className="grid gap-1.5 text-xs font-medium text-(--ui-text-secondary)">
          Session token
          <Input
            className="font-mono"
            onChange={event => setRemoteToken(event.target.value)}
            placeholder="Paste session token"
            type="password"
            value={remoteToken}
          />
        </label>
      ) : null}

      {error ? <div className="rounded-lg border border-destructive/30 bg-destructive/10 px-3 py-2 text-xs text-destructive">{error}</div> : null}

      <div className="flex flex-wrap items-center justify-between gap-3">
        <p className="max-w-[28rem] text-xs leading-5 text-(--ui-text-tertiary)">
          The saved gateway becomes the default backend for future launches. More backend choices can fit here later.
        </p>
        <Button disabled={saving || !trimmedUrl} onClick={() => void save()} type="button">
          {saving ? <Loader2 className="animate-spin" /> : <Globe />}
          {authMode === 'oauth' && probe?.reachable ? 'Save and sign in' : 'Configure gateway'}
        </Button>
      </div>
    </div>
  )
}

interface GatewayRecoveryPanelProps {
  busy: 'retry' | 'signin' | null
  diagnostic: string
  kind: Exclude<GatewayStartupKind, 'setup'>
  logs: string[]
  onOpenLogs: () => void
  onRetry: () => void
  onSettings: () => void
  onSignIn: () => void
  remoteUrl: string
  showLogs: boolean
  signInLabel: string
  toggleLogs: () => void
}

export function GatewayRecoveryPanel({
  busy,
  diagnostic,
  kind,
  logs,
  onOpenLogs,
  onRetry,
  onSettings,
  onSignIn,
  remoteUrl,
  showLogs,
  signInLabel,
  toggleLogs
}: GatewayRecoveryPanelProps) {
  const copy = {
    signin: {
      title: 'Sign in to configured gateway',
      body: 'The configured gateway is reachable, but Reuben needs a fresh gateway session before it can continue.',
      hint: 'The next step opens the remote gateway sign-in page. Reuben keeps the session in its own desktop cookie jar.'
    },
    unavailable: {
      title: 'Configured gateway unavailable',
      body: 'Reuben has a saved default backend and is trying to reconnect to it.',
      hint: 'Check the host, port, VPN, proxy, or remote service status, then retry.'
    },
    unexpected: {
      title: 'Configured gateway returned an unexpected response',
      body: 'The host responded, but it did not look like a compatible Reuben/Hermes gateway.',
      hint: 'Confirm the URL points at the gateway root and preserves any required path prefix.'
    }
  }[kind]

  return (
    <div className="grid gap-5">
      <div className="flex items-center gap-4">
        <img alt="" className="size-14 rounded-2xl border border-(--ui-stroke-tertiary)" src={assetPath('reuben-icon.png')} />
        <div>
          <div className="text-[0.68rem] font-semibold uppercase tracking-[0.26em] text-primary">Reuben Agent</div>
          <h1 className="mt-1 text-xl font-semibold tracking-tight text-(--ui-text-primary)">{copy.title}</h1>
          <p className="mt-1 max-w-[34rem] text-sm leading-6 text-(--ui-text-tertiary)">{copy.body}</p>
        </div>
      </div>

      <div className="grid gap-3 rounded-xl border border-(--ui-stroke-tertiary) bg-(--ui-bg-tertiary)/45 p-4">
        <div className="text-xs font-medium uppercase tracking-[0.18em] text-(--ui-text-tertiary)">Default backend</div>
        <div className="break-all font-mono text-sm text-(--ui-text-primary)">{remoteUrl}</div>
      </div>

      <div
        className={cn(
          'rounded-xl border px-4 py-3 text-xs leading-5',
          kind === 'signin'
            ? 'border-primary/30 bg-primary/10 text-(--ui-text-secondary)'
            : 'border-destructive/30 bg-destructive/10 text-destructive'
        )}
      >
        <div className="font-medium">{kind === 'signin' ? 'Authentication required' : 'Diagnostic'}</div>
        <div className="mt-1">{diagnostic}</div>
      </div>

      <div className="grid gap-2">
        <div className="flex flex-wrap gap-2">
          {kind === 'signin' ? (
            <Button disabled={Boolean(busy)} onClick={onSignIn}>
              {busy === 'signin' ? <Loader2 className="animate-spin" /> : <LogIn />}
              {signInLabel}
            </Button>
          ) : (
            <Button disabled={Boolean(busy)} onClick={onRetry}>
              {busy === 'retry' ? <Loader2 className="animate-spin" /> : <RefreshCw />}
              Retry
            </Button>
          )}
          <Button disabled={Boolean(busy)} onClick={onSettings} variant="secondary">
            <Settings2 />
            {kind === 'signin' ? 'Change gateway' : 'Configure gateway'}
          </Button>
          <Button onClick={onOpenLogs} variant="ghost">
            <FileText />
            Open logs
          </Button>
        </div>
        <p className="text-xs text-(--ui-text-tertiary)">{copy.hint}</p>
      </div>

      {logs.length > 0 ? (
        <div className="grid gap-2">
          <Button className="-ml-2 self-start font-medium" onClick={toggleLogs} size="xs" type="button" variant="text">
            {showLogs ? 'Hide recent logs' : 'Show recent logs'}
          </Button>
          {showLogs ? <LogView className="max-h-48">{logs.slice(-40).join('')}</LogView> : null}
        </div>
      ) : null}
    </div>
  )
}
