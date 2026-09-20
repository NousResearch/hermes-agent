import { type ReactNode, useEffect, useRef, useState } from 'react'

import type { OwnerScope } from '@/api/client'
import { Button } from '@/components/ui/button'
import { getMemoryProviderOAuthStatus, startMemoryProviderOAuth } from '@/hermes'
import { useI18n } from '@/i18n'
import { Check, ExternalLink, Loader2 } from '@/lib/icons'
import type { MemoryProviderOAuthStatus } from '@/types/hermes'

const POLL_MS = 1500
const POLL_TIMEOUT_MS = 120_000

const OAUTH_REQUEST = { start: startMemoryProviderOAuth, status: getMemoryProviderOAuthStatus }

type Busy = 'checking' | 'waiting'
type Failure = 'checkFailed' | 'failed' | 'timeout'

interface MemoryConnectProps {
  profile: OwnerScope
  provider: string
  /** Fires once when a check or wait moves this owner's provider from not connected to connected. */
  onConnected?: () => void
}

// One lifecycle per owner and provider, so a late response never lands on another view.
export function MemoryConnect({ profile, provider, onConnected }: MemoryConnectProps) {
  return (
    <MemoryConnectOwner
      key={JSON.stringify([profile.connectionId, profile.profile, provider])}
      onConnected={onConnected}
      profile={profile}
      provider={provider}
    />
  )
}

function MemoryConnectOwner({ profile, provider, onConnected }: MemoryConnectProps) {
  const { t } = useI18n()
  const c = t.memoryProviders.oauth
  const [status, setStatus] = useState<MemoryProviderOAuthStatus | null>(null)
  const [busy, setBusy] = useState<Busy | null>('checking')
  const [error, setError] = useState<Failure | null>(null)
  const run = useRef(0)

  // Retires the running sequence. The wait ends, but the request and any browser authorization continue.
  const retire = () => {
    run.current++
  }

  const finish = (outcome: MemoryProviderOAuthStatus | Failure | null) => {
    retire()
    setBusy(null)

    if (outcome && typeof outcome === 'object') {
      setStatus(outcome)
      setError(outcome.state === 'error' ? 'failed' : null)

      // `status` is the last known state when this sequence began; the initial check has none to transition from.
      if (outcome.connected && status?.connected === false) {
        onConnected?.()
      }
    } else {
      setError(outcome)
    }
  }

  // A check reads status once. A wait starts (or resumes) the flow and polls until the backend leaves `pending`;
  // a failed poll is retried, a failed start is not.
  const begin = async (kind: Busy, resume = false) => {
    const id = ++run.current
    const live = () => id === run.current
    setBusy(kind)
    setError(null)
    const deadline = setTimeout(() => live() && finish('timeout'), POLL_TIMEOUT_MS)
    let operation: keyof typeof OAUTH_REQUEST = kind === 'waiting' && !resume ? 'start' : 'status'

    try {
      while (live()) {
        try {
          const next = await OAUTH_REQUEST[operation](provider, profile)

          if (!live()) {
            return
          }

          setStatus(next)
          setError(null)

          if (kind === 'checking' || next.supported === false || next.state !== 'pending') {
            finish(next)

            return
          }
        } catch {
          if (!live()) {
            return
          }

          if (kind === 'checking' || operation === 'start') {
            finish(kind === 'checking' ? 'checkFailed' : 'failed')

            return
          }

          setError('checkFailed')
        }

        operation = 'status'
        await new Promise(resolve => setTimeout(resolve, POLL_MS))
      }
    } finally {
      clearTimeout(deadline)
    }
  }

  useEffect(() => {
    void begin('checking')

    return retire
    // eslint-disable-next-line react-hooks/exhaustive-deps -- provider and profile are fixed for this mount (keyed by MemoryConnect)
  }, [])

  if (status?.supported === false) {
    return null
  }

  const connected = status?.connected ?? false
  const apiKey = status?.auth === 'apikey'

  const spinner = (label: string) => (
    <span className="inline-flex items-center gap-1.5 text-muted-foreground" role="status">
      <Loader2 className="size-3 animate-spin" />
      {label}
    </span>
  )

  const action = (label: string, onClick: () => void, icon?: ReactNode) => (
    <Button onClick={onClick} size="inline" type="button" variant="link">
      {icon}
      {label}
    </Button>
  )

  const view =
    busy ?? (status?.state === 'pending' ? 'consent' : !status || error === 'checkFailed' ? 'recheck' : 'connect')

  const views: Record<typeof view, ReactNode> = {
    checking: spinner(c.checking),
    waiting: (
      <>
        {spinner(c.waiting)}
        {action(c.stopWaiting, () => finish(null))}
      </>
    ),
    consent: (
      <>
        <span className="text-muted-foreground">{c.waiting}</span>
        {action(c.retryCheck, () => void begin('waiting', true))}
      </>
    ),
    recheck: action(c.retryCheck, () => void begin('checking')),
    connect: action(
      connected ? (apiKey ? c.viaOAuth : c.reconnect) : c.connect,
      () => void begin('waiting'),
      <ExternalLink />
    )
  }

  return (
    <span className="inline-flex flex-wrap items-center gap-x-3 gap-y-1 text-xs">
      {connected && (
        <span className="inline-flex items-center gap-1 text-muted-foreground">
          <Check className="size-3" />
          {apiKey ? c.apiKeySet : c.connected}
        </span>
      )}
      {views[view]}
      {error && (
        <span className="text-destructive" role="alert">
          {c[error]}
        </span>
      )}
    </span>
  )
}
