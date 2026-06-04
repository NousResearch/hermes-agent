import { useEffect, useMemo, useRef, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import type { DesktopAuthProvider, DesktopConnectionProbeResult } from '@/global'
import { AlertCircle, Check, FileText, Globe, Loader2, LogIn, Monitor } from '@/lib/icons'
import { cn } from '@/lib/utils'
import { notify, notifyError } from '@/store/notifications'

import { CONTROL_TEXT } from './constants'
import { EmptyState, ListRow, LoadingState, Pill, SettingsContent } from './primitives'

type Mode = 'local' | 'remote'
type AuthMode = 'oauth' | 'token'
type ProbeStatus = 'idle' | 'probing' | 'done' | 'error'

interface GatewaySettingsState {
  envOverride: boolean
  mode: Mode
  remoteAuthMode: AuthMode
  remoteOauthConnected: boolean
  remoteTokenPreview: string | null
  remoteTokenSet: boolean
  remoteUrl: string
}

const EMPTY_STATE: GatewaySettingsState = {
  envOverride: false,
  mode: 'local',
  remoteAuthMode: 'token',
  remoteOauthConnected: false,
  remoteTokenPreview: null,
  remoteTokenSet: false,
  remoteUrl: ''
}

function ModeCard({
  active,
  description,
  disabled,
  icon: Icon,
  onSelect,
  title
}: {
  active: boolean
  description: string
  disabled?: boolean
  icon: typeof Monitor
  onSelect: () => void
  title: string
}) {
  return (
    <button
      className={cn(
        'rounded-xl border p-3 text-left transition',
        active
          ? 'border-(--ui-stroke-secondary) bg-(--ui-bg-tertiary)'
          : 'border-(--ui-stroke-tertiary) bg-(--ui-bg-quinary) hover:bg-(--chrome-action-hover)',
        disabled && 'cursor-not-allowed opacity-50'
      )}
      disabled={disabled}
      onClick={onSelect}
      type="button"
    >
      <div className="flex items-center gap-2 text-[length:var(--conversation-text-font-size)] font-medium">
        <Icon className="size-4 text-muted-foreground" />
        <span>{title}</span>
        {active ? <Check className="ml-auto size-4 text-primary" /> : null}
      </div>
      <p className="mt-1.5 text-[length:var(--conversation-caption-font-size)] leading-(--conversation-caption-line-height) text-(--ui-text-tertiary)">
        {description}
      </p>
    </button>
  )
}

export function GatewaySettings() {
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [testing, setTesting] = useState(false)
  const [signingIn, setSigningIn] = useState(false)
  const [state, setState] = useState<GatewaySettingsState>(EMPTY_STATE)
  const [remoteToken, setRemoteToken] = useState('')
  const [lastTest, setLastTest] = useState<null | string>(null)

  // Auth-mode probe: as the user types a remote URL we ask the gateway (via
  // its public /api/status) whether it gates with OAuth or a static session
  // token, so we can show the right control (login button vs token box).
  const [probeStatus, setProbeStatus] = useState<ProbeStatus>('idle')
  const [probe, setProbe] = useState<DesktopConnectionProbeResult | null>(null)
  const probeSeq = useRef(0)

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

  // Debounced probe of the entered remote URL. Only runs in remote mode with a
  // syntactically plausible URL. The probe result drives whether we render the
  // OAuth login button or the session-token entry box. The effective auth mode
  // prefers a fresh probe result over the saved value.
  const trimmedUrl = state.remoteUrl.trim()
  useEffect(() => {
    if (state.mode !== 'remote' || !trimmedUrl || !/^https?:\/\//i.test(trimmedUrl)) {
      setProbeStatus('idle')
      setProbe(null)

      return
    }

    const desktop = window.hermesDesktop

    if (!desktop?.probeConnectionConfig) {
      return
    }

    const seq = ++probeSeq.current
    setProbeStatus('probing')

    const timer = setTimeout(() => {
      desktop
        .probeConnectionConfig(trimmedUrl)
        .then(result => {
          if (seq !== probeSeq.current) {
            return
          }

          setProbe(result)
          setProbeStatus(result.reachable ? 'done' : 'error')
        })
        .catch(() => {
          if (seq !== probeSeq.current) {
            return
          }

          setProbe(null)
          setProbeStatus('error')
        })
    }, 500)

    return () => clearTimeout(timer)
  }, [state.mode, trimmedUrl])

  // Effective auth mode: a reachable probe wins; otherwise fall back to the
  // saved config's mode so a re-open of settings doesn't flicker.
  const authMode: AuthMode = useMemo(() => {
    if (probeStatus === 'done' && probe && probe.authMode !== 'unknown') {
      return probe.authMode
    }

    return state.remoteAuthMode
  }, [probe, probeStatus, state.remoteAuthMode])

  // Whether we actually KNOW how this gateway authenticates yet. Until we do,
  // neither the OAuth button nor the session-token box should render —
  // `authMode` defaults to 'token', so without this gate the token box flashes
  // for every gateway (including OAuth ones) during the idle/probing window
  // before the first probe lands. The scheme is known when either:
  //   * the live probe finished (probeStatus 'done'), or
  //   * we're idle but showing a previously-saved remote config (re-opening
  //     settings for a gateway already signed-in or with a saved token), so
  //     its control appears immediately with no flicker.
  // While probing (or after a probe error), the scheme is unknown and we show
  // the probe status row instead of a control.
  const hasSavedRemote = state.remoteTokenSet || state.remoteOauthConnected
  const authResolved = useMemo(() => {
    if (probeStatus === 'done') {
      return true
    }

    return probeStatus === 'idle' && hasSavedRemote
  }, [probeStatus, hasSavedRemote])

  const providerLabel = useMemo(() => {
    const providers: DesktopAuthProvider[] = probe?.providers ?? []

    if (providers.length === 1) {
      return providers[0].displayName || providers[0].name
    }

    if (providers.length > 1) {
      return providers.map(p => p.displayName || p.name).join(' / ')
    }

    return 'your identity provider'
  }, [probe])

  // A username/password gateway authenticates through a credential form on the
  // gateway's /login page (POST /auth/password-login) rather than an OAuth
  // redirect. Everything downstream — the session cookie, the ws-ticket mint,
  // the persistent partition — is identical, so the desktop drives it through
  // the same sign-in window; only the button copy changes. We treat the
  // gateway as password-style only when EVERY advertised provider supports
  // password, so a mixed deployment keeps the generic OAuth copy.
  const isPasswordProvider = useMemo(() => {
    const providers: DesktopAuthProvider[] = probe?.providers ?? []

    return providers.length > 0 && providers.every(p => p.supportsPassword)
  }, [probe])

  const oauthConnected = state.remoteOauthConnected

  const canUseRemote = useMemo(() => {
    if (!trimmedUrl) {
      return false
    }

    if (authMode === 'oauth') {
      return oauthConnected
    }

    return Boolean(remoteToken.trim()) || state.remoteTokenSet
  }, [authMode, oauthConnected, remoteToken, state.remoteTokenSet, trimmedUrl])

  const payload = () => ({
    mode: state.mode,
    remoteAuthMode: authMode,
    remoteToken: authMode === 'token' ? remoteToken.trim() || undefined : undefined,
    remoteUrl: trimmedUrl
  })

  const save = async (apply: boolean) => {
    if (state.mode === 'remote' && !canUseRemote) {
      notify({
        kind: 'warning',
        title: 'Remote gateway incomplete',
        message:
          authMode === 'oauth'
            ? 'Enter a remote URL and sign in before switching to remote.'
            : 'Enter a remote URL and session token before switching to remote.'
      })

      return
    }

    setSaving(true)

    try {
      const next = apply
        ? await window.hermesDesktop.applyConnectionConfig(payload())
        : await window.hermesDesktop.saveConnectionConfig(payload())

      setState(next)
      setRemoteToken('')
      notify({
        kind: 'success',
        title: apply ? 'Gateway connection restarting' : 'Gateway settings saved',
        message: apply ? 'Hermes Desktop will reconnect using the saved settings.' : 'Saved for the next restart.'
      })
    } catch (err) {
      notifyError(err, apply ? 'Could not apply gateway settings' : 'Could not save gateway settings')
    } finally {
      setSaving(false)
    }
  }

  // OAuth sign-in: persist the URL + oauth mode first (so the saved config has
  // the URL the login window needs), then open the gateway login window and
  // refresh the connection status from the saved config once it completes.
  const signIn = async () => {
    if (!trimmedUrl) {
      notify({ kind: 'warning', title: 'Remote gateway incomplete', message: 'Enter a remote URL first.' })

      return
    }

    setSigningIn(true)

    try {
      // Save (don't apply/restart) so the login window has a URL to use and the
      // oauth mode is persisted, without yet flipping the live connection.
      const saved = await window.hermesDesktop.saveConnectionConfig({
        mode: state.mode,
        remoteAuthMode: 'oauth',
        remoteUrl: trimmedUrl
      })

      setState(saved)

      const result = await window.hermesDesktop.oauthLoginConnectionConfig(trimmedUrl)

      if (result.connected) {
        const refreshed = await window.hermesDesktop.getConnectionConfig()
        setState(refreshed)
        notify({ kind: 'success', title: 'Signed in', message: `Connected to ${providerLabel}.` })
      } else {
        notify({
          kind: 'warning',
          title: 'Sign-in incomplete',
          message: 'The login window closed before authentication finished.'
        })
      }
    } catch (err) {
      notifyError(err, 'Sign-in failed')
    } finally {
      setSigningIn(false)
    }
  }

  const signOut = async () => {
    setSigningIn(true)

    try {
      await window.hermesDesktop.oauthLogoutConnectionConfig(trimmedUrl || undefined)
      const refreshed = await window.hermesDesktop.getConnectionConfig()
      setState(refreshed)
      notify({ kind: 'success', title: 'Signed out', message: 'Cleared the remote gateway session.' })
    } catch (err) {
      notifyError(err, 'Sign-out failed')
    } finally {
      setSigningIn(false)
    }
  }

  const testRemote = async () => {
    if (!canUseRemote) {
      notify({
        kind: 'warning',
        title: 'Remote gateway incomplete',
        message:
          authMode === 'oauth'
            ? 'Enter a remote URL and sign in before testing.'
            : 'Enter a remote URL and session token before testing.'
      })

      return
    }

    setTesting(true)
    setLastTest(null)

    try {
      const result = await window.hermesDesktop.testConnectionConfig({
        mode: 'remote',
        remoteAuthMode: authMode,
        remoteToken: authMode === 'token' ? remoteToken.trim() || undefined : undefined,
        remoteUrl: trimmedUrl
      })

      const message = `Connected to ${result.baseUrl}${result.version ? ` · Hermes ${result.version}` : ''}`
      setLastTest(message)
      notify({ kind: 'success', title: 'Remote gateway reachable', message })
    } catch (err) {
      notifyError(err, 'Remote gateway test failed')
    } finally {
      setTesting(false)
    }
  }

  if (loading) {
    return <LoadingState label="Loading gateway settings..." />
  }

  if (!window.hermesDesktop?.getConnectionConfig) {
    return (
      <EmptyState
        description="The desktop IPC bridge does not expose gateway settings."
        title="Gateway settings unavailable"
      />
    )
  }

  return (
    <SettingsContent>
      <div className="mb-5">
        <div className="flex items-center gap-2 text-[length:var(--conversation-text-font-size)] font-medium">
          <Globe className="size-4 text-muted-foreground" />
          Gateway Connection
          {state.envOverride ? <Pill tone="primary">env override</Pill> : null}
        </div>
        <p className="mt-2 max-w-2xl text-[length:var(--conversation-caption-font-size)] leading-(--conversation-caption-line-height) text-(--ui-text-tertiary)">
          Hermes Desktop starts its own local gateway by default. Use a remote gateway when you want this app to control
          an already-running Hermes backend on another machine or behind a trusted proxy.
        </p>
      </div>

      {state.envOverride ? (
        <div className="mb-5 flex items-start gap-2 rounded-xl border border-destructive/30 bg-destructive/10 px-3 py-2.5 text-[length:var(--conversation-caption-font-size)] text-destructive">
          <AlertCircle className="mt-0.5 size-4 shrink-0" />
          <div>
            <div className="font-medium">Environment variables are controlling this desktop session.</div>
            <div className="mt-1 leading-5">
              Unset <code>HERMES_DESKTOP_REMOTE_URL</code> and <code>HERMES_DESKTOP_REMOTE_TOKEN</code> to use the saved
              setting below.
            </div>
          </div>
        </div>
      ) : null}

      <div className="grid gap-3 sm:grid-cols-2">
        <ModeCard
          active={state.mode === 'local'}
          description="Start a private Hermes backend on localhost. This is the default and works offline."
          disabled={state.envOverride}
          icon={Monitor}
          onSelect={() => setState(current => ({ ...current, mode: 'local' }))}
          title="Local gateway"
        />
        <ModeCard
          active={state.mode === 'remote'}
          description="Connect this desktop shell to a remote Hermes backend. Hosted gateways use OAuth or a username and password; self-hosted ones may use a session token."
          disabled={state.envOverride}
          icon={Globe}
          onSelect={() => setState(current => ({ ...current, mode: 'remote' }))}
          title="Remote gateway"
        />
      </div>

      <div className="mt-5 grid gap-1">
        <ListRow
          action={
            <Input
              className={cn('h-8', CONTROL_TEXT)}
              disabled={state.envOverride}
              onChange={event => setState(current => ({ ...current, remoteUrl: event.target.value }))}
              placeholder="https://gateway.example.com/hermes"
              value={state.remoteUrl}
            />
          }
          description="Base URL for the remote dashboard backend. Path prefixes are supported, for example /hermes."
          title="Remote URL"
        />

        {state.mode === 'remote' && probeStatus === 'probing' ? (
          <div className="flex items-center gap-2 py-3 text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
            <Loader2 className="size-4 animate-spin" />
            Checking how this gateway authenticates…
          </div>
        ) : null}

        {state.mode === 'remote' && probeStatus === 'error' ? (
          <div className="flex items-start gap-2 py-3 text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
            <AlertCircle className="mt-0.5 size-4 shrink-0" />
            Could not reach this gateway yet. Check the URL — the auth method will appear once it responds.
          </div>
        ) : null}

        {/* OAuth / password gateways: present a sign-in button + connection status. */}
        {state.mode === 'remote' && authResolved && authMode === 'oauth' ? (
          <ListRow
            action={
              oauthConnected ? (
                <div className="flex items-center gap-2">
                  <Pill tone="primary">
                    <Check className="size-3" /> Signed in
                  </Pill>
                  <Button disabled={signingIn || state.envOverride} onClick={() => void signOut()} variant="outline">
                    {signingIn ? <Loader2 className="size-4 animate-spin" /> : null}
                    Sign out
                  </Button>
                </div>
              ) : (
                <Button disabled={signingIn || state.envOverride || !trimmedUrl} onClick={() => void signIn()}>
                  {signingIn ? <Loader2 className="size-4 animate-spin" /> : <LogIn className="size-4" />}
                  {isPasswordProvider ? 'Sign in' : `Sign in with ${providerLabel}`}
                </Button>
              )
            }
            description={
              oauthConnected
                ? isPasswordProvider
                  ? 'This gateway uses a username and password. You are signed in; the session refreshes automatically.'
                  : 'This gateway uses OAuth. You are signed in; the session refreshes automatically.'
                : isPasswordProvider
                  ? 'This gateway uses a username and password. Sign in to authorize this desktop app.'
                  : `This gateway uses OAuth. Sign in with ${providerLabel} to authorize this desktop app.`
            }
            title="Authentication"
          />
        ) : null}

        {/* Session-token gateways: keep the existing token entry box. */}
        {state.mode === 'remote' && authResolved && authMode === 'token' ? (
          <ListRow
            action={
              <Input
                autoComplete="off"
                className={cn('h-8 font-mono', CONTROL_TEXT)}
                disabled={state.envOverride}
                onChange={event => setRemoteToken(event.target.value)}
                placeholder={
                  state.remoteTokenSet ? `Existing token ${state.remoteTokenPreview ?? 'saved'}` : 'Paste session token'
                }
                type="password"
                value={remoteToken}
              />
            }
            description="The dashboard session token used for REST and WebSocket access. Leave blank to keep the saved token."
            title="Session token"
          />
        ) : null}
      </div>

      {lastTest ? <div className="mt-4 text-xs text-primary">{lastTest}</div> : null}

      <div className="mt-6 flex flex-wrap items-center justify-end gap-4">
        <Button
          className="mr-auto"
          disabled={state.envOverride || testing || !canUseRemote}
          onClick={() => void testRemote()}
          size="sm"
          variant="text"
        >
          {testing ? <Loader2 className="size-4 animate-spin" /> : null}
          Test remote
        </Button>
        <Button disabled={state.envOverride || saving} onClick={() => void save(false)} size="sm" variant="textStrong">
          Save for next restart
        </Button>
        <Button disabled={state.envOverride || saving} onClick={() => void save(true)} size="sm">
          {saving ? <Loader2 className="size-4 animate-spin" /> : null}
          Save and reconnect
        </Button>
      </div>

      <div className="mt-6 grid gap-1">
        <ListRow
          action={
            <Button onClick={() => void window.hermesDesktop?.revealLogs()} size="sm" variant="textStrong">
              <FileText className="size-4" />
              Open logs
            </Button>
          }
          description="Reveal desktop.log in your file manager — useful when the gateway fails to start."
          title="Diagnostics"
        />
      </div>
    </SettingsContent>
  )
}
