import { useEffect, useMemo, useRef, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import type { DesktopAuthProvider, DesktopConnectionProbeResult } from '@/global'
import { useTranslation } from '@/i18n'
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
  const t = useTranslation()
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
      .catch(err => notifyError(err, t('settings.gateway.loadError')))
      .finally(() => {
        if (!cancelled) {
          setLoading(false)
        }
      })

    return () => void (cancelled = true)
  }, [t])

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

    return t('settings.gateway.oauth.identityProvider')
  }, [probe, t])

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
        title: t('settings.gateway.remoteIncomplete.title'),
        message:
          authMode === 'oauth'
            ? t('settings.gateway.remoteIncomplete.oauthSwitchMessage')
            : t('settings.gateway.remoteIncomplete.switchMessage')
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
        title: apply ? t('settings.gateway.saved.restartingTitle') : t('settings.gateway.saved.title'),
        message: apply ? t('settings.gateway.saved.restartingMessage') : t('settings.gateway.saved.restartMessage')
      })
    } catch (err) {
      notifyError(err, apply ? t('settings.gateway.applyError') : t('settings.gateway.saveError'))
    } finally {
      setSaving(false)
    }
  }

  // OAuth sign-in: persist the URL + oauth mode first (so the saved config has
  // the URL the login window needs), then open the gateway login window and
  // refresh the connection status from the saved config once it completes.
  const signIn = async () => {
    if (!trimmedUrl) {
      notify({
        kind: 'warning',
        title: t('settings.gateway.remoteIncomplete.title'),
        message: t('settings.gateway.remoteIncomplete.urlFirst')
      })

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
        notify({
          kind: 'success',
          title: t('settings.gateway.oauth.signedIn'),
          message: t('settings.gateway.oauth.connectedToProvider', { provider: providerLabel })
        })
      } else {
        notify({
          kind: 'warning',
          title: t('settings.gateway.oauth.signInIncomplete'),
          message: t('settings.gateway.oauth.signInIncompleteMessage')
        })
      }
    } catch (err) {
      notifyError(err, t('settings.gateway.oauth.signInFailed'))
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
      notify({
        kind: 'success',
        title: t('settings.gateway.oauth.signedOut'),
        message: t('settings.gateway.oauth.signedOutMessage')
      })
    } catch (err) {
      notifyError(err, t('settings.gateway.oauth.signOutFailed'))
    } finally {
      setSigningIn(false)
    }
  }

  const testRemote = async () => {
    if (!canUseRemote) {
      notify({
        kind: 'warning',
        title: t('settings.gateway.remoteIncomplete.title'),
        message:
          authMode === 'oauth'
            ? t('settings.gateway.remoteIncomplete.oauthTestMessage')
            : t('settings.gateway.remoteIncomplete.testMessage')
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

      const message = t('settings.gateway.test.connected', {
        target: `${result.baseUrl}${result.version ? ` · Hermes ${result.version}` : ''}`
      })

      setLastTest(message)
      notify({ kind: 'success', title: t('settings.gateway.test.successTitle'), message })
    } catch (err) {
      notifyError(err, t('settings.gateway.test.error'))
    } finally {
      setTesting(false)
    }
  }

  if (loading) {
    return <LoadingState label={t('settings.gateway.loading')} />
  }

  if (!window.hermesDesktop?.getConnectionConfig) {
    return (
      <EmptyState
        description={t('settings.gateway.unavailable.description')}
        title={t('settings.gateway.unavailable.title')}
      />
    )
  }

  return (
    <SettingsContent>
      <div className="mb-5">
        <div className="flex items-center gap-2 text-[length:var(--conversation-text-font-size)] font-medium">
          <Globe className="size-4 text-muted-foreground" />
          {t('settings.gateway.title')}
          {state.envOverride ? <Pill tone="primary">{t('settings.gateway.envOverride')}</Pill> : null}
        </div>
        <p className="mt-2 max-w-2xl text-[length:var(--conversation-caption-font-size)] leading-(--conversation-caption-line-height) text-(--ui-text-tertiary)">
          {t('settings.gateway.description')}
        </p>
      </div>

      {state.envOverride ? (
        <div className="mb-5 flex items-start gap-2 rounded-xl border border-destructive/30 bg-destructive/10 px-3 py-2.5 text-[length:var(--conversation-caption-font-size)] text-destructive">
          <AlertCircle className="mt-0.5 size-4 shrink-0" />
          <div>
            <div className="font-medium">{t('settings.gateway.envWarning.title')}</div>
            <div className="mt-1 leading-5">
              {t('settings.gateway.envWarning.before')} <code>HERMES_DESKTOP_REMOTE_URL</code>{' '}
              {t('settings.gateway.envWarning.and')} <code>HERMES_DESKTOP_REMOTE_TOKEN</code>{' '}
              {t('settings.gateway.envWarning.after')}
            </div>
          </div>
        </div>
      ) : null}

      <div className="grid gap-3 sm:grid-cols-2">
        <ModeCard
          active={state.mode === 'local'}
          description={t('settings.gateway.local.description')}
          disabled={state.envOverride}
          icon={Monitor}
          onSelect={() => setState(current => ({ ...current, mode: 'local' }))}
          title={t('settings.gateway.local.title')}
        />
        <ModeCard
          active={state.mode === 'remote'}
          description={t('settings.gateway.remote.description')}
          disabled={state.envOverride}
          icon={Globe}
          onSelect={() => setState(current => ({ ...current, mode: 'remote' }))}
          title={t('settings.gateway.remote.title')}
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
          description={t('settings.gateway.remoteUrl.description')}
          title={t('settings.gateway.remoteUrl.title')}
        />

        {state.mode === 'remote' && probeStatus === 'probing' ? (
          <div className="flex items-center gap-2 py-3 text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
            <Loader2 className="size-4 animate-spin" />
            {t('settings.gateway.oauth.checkingAuth')}
          </div>
        ) : null}

        {state.mode === 'remote' && probeStatus === 'error' ? (
          <div className="flex items-start gap-2 py-3 text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
            <AlertCircle className="mt-0.5 size-4 shrink-0" />
            {t('settings.gateway.oauth.probeFailed')}
          </div>
        ) : null}

        {/* OAuth / password gateways: present a sign-in button + connection status. */}
        {state.mode === 'remote' && authResolved && authMode === 'oauth' ? (
          <ListRow
            action={
              oauthConnected ? (
                <div className="flex items-center gap-2">
                  <Pill tone="primary">
                    <Check className="size-3" /> {t('settings.gateway.oauth.signedIn')}
                  </Pill>
                  <Button disabled={signingIn || state.envOverride} onClick={() => void signOut()} variant="outline">
                    {signingIn ? <Loader2 className="size-4 animate-spin" /> : null}
                    {t('settings.gateway.oauth.signOut')}
                  </Button>
                </div>
              ) : (
                <Button disabled={signingIn || state.envOverride || !trimmedUrl} onClick={() => void signIn()}>
                  {signingIn ? <Loader2 className="size-4 animate-spin" /> : <LogIn className="size-4" />}
                  {isPasswordProvider
                    ? t('settings.gateway.oauth.signIn')
                    : t('settings.gateway.oauth.signInWith', { provider: providerLabel })}
                </Button>
              )
            }
            description={
              oauthConnected
                ? isPasswordProvider
                  ? t('settings.gateway.oauth.passwordConnectedDescription')
                  : t('settings.gateway.oauth.connectedDescription')
                : isPasswordProvider
                  ? t('settings.gateway.oauth.passwordSignInDescription')
                  : t('settings.gateway.oauth.signInDescription', { provider: providerLabel })
            }
            title={t('settings.gateway.oauth.authentication')}
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
                  state.remoteTokenSet
                    ? t('settings.gateway.sessionToken.existing', {
                        token: state.remoteTokenPreview ?? t('settings.gateway.sessionToken.saved')
                      })
                    : t('settings.gateway.sessionToken.placeholder')
                }
                type="password"
                value={remoteToken}
              />
            }
            description={t('settings.gateway.sessionToken.description')}
            title={t('settings.gateway.sessionToken.title')}
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
          {t('settings.gateway.actions.testRemote')}
        </Button>
        <Button disabled={state.envOverride || saving} onClick={() => void save(false)} size="sm" variant="textStrong">
          {t('settings.gateway.actions.saveForRestart')}
        </Button>
        <Button disabled={state.envOverride || saving} onClick={() => void save(true)} size="sm">
          {saving ? <Loader2 className="size-4 animate-spin" /> : null}
          {t('settings.gateway.actions.saveAndReconnect')}
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
