import { useCallback, useEffect, useRef, useState } from 'react'

import type { ResolvedOwner } from '@/api/client'
import { Button } from '@/components/ui/button'
import { getMemoryProviderOAuthStatus, startMemoryProviderOAuth } from '@/hermes'
import { Check, ExternalLink, Loader2 } from '@/lib/icons'
import { notifyError } from '@/store/notifications'
import type { MemoryProviderOAuthStatus } from '@/types/hermes'

const POLL_MS = 1500
const POLL_TIMEOUT_MS = 120_000

// Small connect affordance rendered under the provider dropdown. Capability is
// backend-driven: the status route 404s for providers without an oauth_flow
// module, so non-OAuth providers render nothing.
export function MemoryConnect({
  profile,
  provider,
  owner,
  onConnected
}: {
  onConnected?: () => Promise<unknown>
  profile?: string
  provider: string
  owner?: ResolvedOwner
}) {
  const [capable, setCapable] = useState<'no' | 'unknown' | 'yes'>('unknown')
  const [connected, setConnected] = useState(false)
  const [auth, setAuth] = useState<MemoryProviderOAuthStatus['auth']>(null)
  const [phase, setPhase] = useState<'error' | 'idle' | 'pending'>('idle')
  const [detail, setDetail] = useState('')
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null)
  const deadline = useRef<ReturnType<typeof setTimeout> | null>(null)
  const generation = useRef(0)
  const waiting = useRef(false)

  const stop = useCallback(() => {
    generation.current += 1
    waiting.current = false

    if (deadline.current !== null) {
      clearTimeout(deadline.current)
      deadline.current = null
    }

    if (timer.current !== null) {
      clearTimeout(timer.current)
      timer.current = null
    }
  }, [])

  useEffect(() => {
    let active = true
    setCapable('unknown')
    setPhase('idle')
    setConnected(false)
    setAuth(null)
    setDetail('')
    getMemoryProviderOAuthStatus(provider, profile, owner)
      .then(s => {
        if (!active) {
          return
        }

        setCapable('yes')
        setConnected(s.connected)
        setAuth(s.auth)
      })
      .catch(() => {
        if (active) {
          setCapable('no')
        }
      })

    return () => {
      active = false
      stop()
    }
  }, [profile, provider, stop, owner])

  // An error message isn't sticky — it clears back to the steady state
  // (Connect link, plus the connected badge if a credential is stored).
  useEffect(() => {
    if (phase !== 'error') {
      return
    }

    const t = setTimeout(() => {
      setPhase('idle')
      setDetail('')
    }, 6000)

    return () => clearTimeout(t)
  }, [phase])

  const connect = useCallback(async () => {
    if (waiting.current) {
      return
    }

    stop()
    waiting.current = true
    const current = generation.current
    setPhase('pending')
    deadline.current = setTimeout(() => {
      stop()
      setPhase('error')
      setDetail('Stopped waiting — authorization may still be pending.')
    }, POLL_TIMEOUT_MS)

    try {
      await startMemoryProviderOAuth(provider, profile, owner)
    } catch (err) {
      if (current !== generation.current) {
        return
      }

      stop()
      setPhase('error')
      setDetail('Could not start the connection.')
      notifyError(err, 'Failed to start connection')

      return
    }

    if (current !== generation.current) {
      return
    }

    const poll = async () => {
      let next: MemoryProviderOAuthStatus

      try {
        next = await getMemoryProviderOAuthStatus(provider, profile, owner)
      } catch {
        // Only one poll at a time; the independent deadline also covers hangs.
        if (current === generation.current) {
          timer.current = setTimeout(() => void poll(), POLL_MS)
        }

        return
      }

      if (current !== generation.current) {
        return
      }

      if (next.state === 'pending') {
        timer.current = setTimeout(() => void poll(), POLL_MS)

        return
      }

      stop()
      const completed = generation.current
      setConnected(next.connected)
      setAuth(next.auth)

      if (next.state === 'error') {
        setPhase('error')
        setDetail(next.detail || 'Connection failed.')
      } else {
        setPhase('idle')

        if (next.connected) {
          try {
            await onConnected?.()
          } catch (err) {
            if (completed === generation.current) {
              notifyError(err, 'Failed to refresh memory provider status')
            }
          }
        }
      }
    }

    timer.current = setTimeout(() => void poll(), POLL_MS)
  }, [profile, provider, stop, owner, onConnected])

  const cancel = useCallback(() => {
    stop()
    setPhase('idle')
  }, [stop])

  if (capable !== 'yes') {
    return null
  }

  const connectLabel = connected ? (auth === 'apikey' ? 'Connect via OAuth' : 'Reconnect') : 'Connect'

  return (
    <span className="inline-flex flex-wrap items-center gap-x-3 gap-y-1 text-xs">
      {phase === 'idle' && connected && (
        <span className="inline-flex items-center gap-1 text-muted-foreground">
          <Check className="size-3" />
          {auth === 'apikey' ? 'api key set' : 'oauth set'}
        </span>
      )}
      {phase === 'pending' ? (
        <>
          <span className="inline-flex items-center gap-1.5 text-muted-foreground">
            <Loader2 className="size-3 animate-spin" />
            Waiting for browser consent…
          </span>
          <Button className="h-auto p-0 text-xs" onClick={cancel} size="sm" type="button" variant="link">
            Stop waiting
          </Button>
        </>
      ) : (
        <Button
          className="h-auto gap-1 p-0 text-xs"
          onClick={() => void connect()}
          size="sm"
          type="button"
          variant="link"
        >
          <ExternalLink className="size-3" />
          {connectLabel}
        </Button>
      )}
      {phase === 'error' && detail && <span className="text-destructive">{detail}</span>}
    </span>
  )
}
