import { useEffect, useMemo, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { AlertCircle, ChevronLeft, Loader2 } from '@/lib/icons'
import { cn } from '@/lib/utils'
import { notify, notifyError } from '@/store/notifications'

const CONTROL_TEXT = 'text-[0.8125rem]'

interface ConnectionFormState {
  envOverride: boolean
  remoteTokenPreview: string | null
  remoteTokenSet: boolean
  remoteUrl: string
}

const EMPTY_STATE: ConnectionFormState = {
  envOverride: false,
  remoteTokenPreview: null,
  remoteTokenSet: false,
  remoteUrl: ''
}

interface Feedback {
  kind: 'error' | 'success'
  message: string
}

// Toasts render at z-1050, below the boot-failure overlay's z-1400 backdrop, so
// they are invisible inside the recovery panel. Surface a readable message inline
// instead. Mirrors the IPC-wrapper unwrapping that notifyError does for toasts.
function errorText(err: unknown): string {
  const raw = err instanceof Error ? err.message : typeof err === 'string' ? err : 'Something went wrong.'

  return raw
    .replace(/^Error invoking remote method '[^']+': Error:\s*/, '')
    .replace(/^Error:\s*/, '')
    .trim()
}

export interface GatewayConnectionFormProps {
  /** Called after a successful save+reconnect — the window typically reloads before this fires. */
  onApplied?: () => void
  /** Back navigation to the recovery actions list. */
  onBack?: () => void
}

// Remote-gateway recovery form shown inside the boot-failure overlay. It edits,
// tests, and reconnects the remote connection using only pre-backend desktop IPC,
// so it works when the gateway is down. The full Settings → Gateway page keeps its
// own richer, OAuth-aware form; this is the minimal token-based escape hatch for
// when the app cannot start. (Follow-up: teach this panel about OAuth gateways.)
export function GatewayConnectionForm({ onApplied, onBack }: GatewayConnectionFormProps) {
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [testing, setTesting] = useState(false)
  const [state, setState] = useState<ConnectionFormState>(EMPTY_STATE)
  const [remoteToken, setRemoteToken] = useState('')
  const [feedback, setFeedback] = useState<Feedback | null>(null)

  useEffect(() => {
    let cancelled = false
    const desktop = window.hermesDesktop

    if (!desktop?.getConnectionConfig) {
      setLoading(false)

      return () => void (cancelled = true)
    }

    desktop
      .getConnectionConfig()
      .then(config => {
        if (cancelled) {
          return
        }

        setState(config)
      })
      .catch(err => notifyError(err, 'Gateway settings failed to load'))
      .finally(() => {
        if (!cancelled) {
          setLoading(false)
        }
      })

    return () => void (cancelled = true)
  }, [])

  const canUseRemote = useMemo(
    () => Boolean(state.remoteUrl.trim()) && (Boolean(remoteToken.trim()) || state.remoteTokenSet),
    [remoteToken, state.remoteTokenSet, state.remoteUrl]
  )

  // Editing either field invalidates a prior test/save result, so clear the inline
  // feedback to avoid a stale "Connected to …" line implying the new value was tested.
  const updateUrl = (value: string) => {
    setFeedback(null)
    setState(current => ({ ...current, remoteUrl: value }))
  }

  const updateToken = (value: string) => {
    setFeedback(null)
    setRemoteToken(value)
  }

  const payload = () => ({
    mode: 'remote' as const,
    remoteToken: remoteToken.trim() || undefined,
    remoteUrl: state.remoteUrl.trim()
  })

  const saveAndReconnect = async () => {
    if (!canUseRemote) {
      const message = 'Enter a remote URL and session token before reconnecting.'
      setFeedback({ kind: 'error', message })
      notify({ kind: 'warning', title: 'Remote gateway incomplete', message })

      return
    }

    setSaving(true)
    setFeedback(null)

    try {
      // applyConnectionConfig persists the config and reloads the window from the
      // main process, so any inline confirmation here is short-lived; onApplied
      // lets the overlay react if the reload is delayed.
      await window.hermesDesktop.applyConnectionConfig(payload())
      setRemoteToken('')
      onApplied?.()
    } catch (err) {
      setFeedback({ kind: 'error', message: errorText(err) })
      notifyError(err, 'Could not apply gateway settings')
    } finally {
      setSaving(false)
    }
  }

  const testRemote = async () => {
    if (!canUseRemote) {
      const message = 'Enter a remote URL and session token before testing.'
      setFeedback({ kind: 'error', message })
      notify({ kind: 'warning', title: 'Remote gateway incomplete', message })

      return
    }

    setTesting(true)
    setFeedback(null)

    try {
      const result = await window.hermesDesktop.testConnectionConfig({
        mode: 'remote',
        remoteToken: remoteToken.trim() || undefined,
        remoteUrl: state.remoteUrl.trim()
      })

      const message = `Connected to ${result.baseUrl}${result.version ? ` · Hermes ${result.version}` : ''}`
      setFeedback({ kind: 'success', message })
      notify({ kind: 'success', title: 'Remote gateway reachable', message })
    } catch (err) {
      setFeedback({ kind: 'error', message: errorText(err) })
      notifyError(err, 'Remote gateway test failed')
    } finally {
      setTesting(false)
    }
  }

  if (loading) {
    return (
      <div className="flex items-center gap-2 py-4 text-xs text-muted-foreground">
        <Loader2 className="size-4 animate-spin" />
        Loading gateway settings…
      </div>
    )
  }

  return (
    <div className="grid gap-3">
      {state.envOverride ? (
        <div className="flex items-start gap-2 rounded-xl border border-destructive/30 bg-destructive/10 px-3 py-2.5 text-[length:var(--conversation-caption-font-size)] text-destructive">
          <AlertCircle className="mt-0.5 size-4 shrink-0" />
          <div>
            <div className="font-medium">Environment variables are controlling this desktop session.</div>
            <div className="mt-1 leading-5">
              Unset <code>HERMES_DESKTOP_REMOTE_URL</code> and <code>HERMES_DESKTOP_REMOTE_TOKEN</code> to use the
              settings below. Saving here will not override the current launch.
            </div>
          </div>
        </div>
      ) : null}

      <div className="grid gap-2">
        <div>
          <label className="text-xs font-medium text-foreground">Remote URL</label>
          <Input
            className={cn('mt-1 h-8', CONTROL_TEXT)}
            disabled={state.envOverride}
            onChange={event => updateUrl(event.target.value)}
            placeholder="https://gateway.example.com/hermes"
            value={state.remoteUrl}
          />
        </div>
        <div>
          <label className="text-xs font-medium text-foreground">Session token</label>
          <Input
            autoComplete="off"
            className={cn('mt-1 h-8 font-mono', CONTROL_TEXT)}
            disabled={state.envOverride}
            onChange={event => updateToken(event.target.value)}
            placeholder={
              state.remoteTokenSet ? `Existing token ${state.remoteTokenPreview ?? 'saved'}` : 'Paste session token'
            }
            type="password"
            value={remoteToken}
          />
        </div>
      </div>

      {feedback ? (
        <div className={cn('text-xs', feedback.kind === 'error' ? 'text-destructive' : 'text-primary')} role="status">
          {feedback.message}
        </div>
      ) : null}

      <div className="flex items-center justify-between gap-2">
        {onBack ? (
          <Button onClick={onBack} size="sm" variant="ghost">
            <ChevronLeft className="size-4" />
            Back
          </Button>
        ) : null}
        <div className="flex gap-2">
          <Button
            disabled={state.envOverride || testing || !canUseRemote}
            onClick={() => void testRemote()}
            size="sm"
            variant="outline"
          >
            {testing ? <Loader2 className="size-4 animate-spin" /> : null}
            Test remote
          </Button>
          <Button
            disabled={state.envOverride || saving || !canUseRemote}
            onClick={() => void saveAndReconnect()}
            size="sm"
          >
            {saving ? <Loader2 className="size-4 animate-spin" /> : null}
            Save and reconnect
          </Button>
        </div>
      </div>
    </div>
  )
}
