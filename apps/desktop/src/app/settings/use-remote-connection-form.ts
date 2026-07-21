import { type Dispatch, type SetStateAction, useEffect, useMemo, useRef, useState } from 'react'

import type { DesktopAuthProvider, DesktopConnectionConfig, DesktopConnectionProbeResult } from '@/global'
import { useI18n } from '@/i18n'
import { notify, notifyError } from '@/store/notifications'

type Mode = 'local' | 'remote' | 'cloud' | 'ssh'
type AuthMode = 'oauth' | 'token'
type ProbeStatus = 'idle' | 'probing' | 'done' | 'error'

// The renderer-facing connection state the form edits: the sanitized config
// minus its scope tag (the hook already knows its scope).
type GatewaySettingsState = Omit<DesktopConnectionConfig, 'profile'>

const EMPTY_STATE: GatewaySettingsState = {
  envOverride: false,
  mode: 'local',
  remoteAuthMode: 'token',
  remoteOauthConnected: false,
  remoteTokenPreview: null,
  remoteTokenSet: false,
  remoteUrl: '',
  cloudOrg: '',
  sshHost: '',
  sshUser: '',
  sshPort: null,
  sshKeyPath: '',
  sshRemoteHermesPath: ''
}

export interface UseRemoteConnectionFormOptions {
  // Connection scope: null = the global/default connection; a profile name =
  // that profile's per-profile remote override.
  scope: null | string
  // Lock the form to a fixed mode regardless of the loaded config's saved mode,
  // normalized ONCE when the config loads (and on the initial state). Used by
  // the first-run overlay, which passes 'remote' — a fresh machine's saved
  // config is 'local', which would otherwise disable the probe/save flow.
  // Omitted by Settings, which preserves the saved mode.
  lockedMode?: Mode
}

// The form API is exactly the hook's return shape (no explicit return-type
// annotation on the hook, or this would reference itself circularly).
export type RemoteConnectionForm = ReturnType<typeof useRemoteConnectionForm>

// Encapsulates the remote-connection form: config load, URL/token entry, the
// debounced auth-mode probe, the derived auth-resolution memos, and the
// save/sign-in/sign-out/test handlers. Shared verbatim between Settings →
// Gateway and the first-run choice overlay so both drive one source of truth.
export function useRemoteConnectionForm({ scope, lockedMode }: UseRemoteConnectionFormOptions) {
  const { t } = useI18n()
  const g = t.settings.gateway

  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [testing, setTesting] = useState(false)
  const [signingIn, setSigningIn] = useState(false)

  const [state, setState] = useState<GatewaySettingsState>(
    lockedMode ? { ...EMPTY_STATE, mode: lockedMode } : EMPTY_STATE
  )

  const [remoteToken, setRemoteToken] = useState('')
  const [lastTest, setLastTest] = useState<null | string>(null)
  // Last operation error surfaced inline (e.g. by the first-run overlay, whose
  // backdrop paints over the global toast region). Set alongside the existing
  // notifyError calls; cleared on a fresh attempt or a URL/token edit.
  const [lastError, setLastError] = useState<null | string>(null)

  // Auth-mode probe: as the user types a remote URL we ask the gateway (via
  // its public /api/status) whether it gates with OAuth or a static session
  // token, so we can show the right control (login button vs token box).
  const [probeStatus, setProbeStatus] = useState<ProbeStatus>('idle')
  const [probe, setProbe] = useState<DesktopConnectionProbeResult | null>(null)
  const probeSeq = useRef(0)
  const saveSeq = useRef(0)
  const signingSeq = useRef(0)
  const testSeq = useRef(0)

  // Flipped true once the user edits the form, so the async config load below
  // won't clobber a value they already typed — the overlay mounts the form
  // immediately, unlike Settings which waits out the load. Reset per scope load.
  const dirtyRef = useRef(false)

  // Public setters flag the form dirty; the config-load effect uses the raw
  // setters so its own writes don't count as user edits.
  const setStateDirty: Dispatch<SetStateAction<GatewaySettingsState>> = value => {
    dirtyRef.current = true
    setState(value)
  }

  const setRemoteTokenDirty = (value: string) => {
    dirtyRef.current = true
    setRemoteToken(value)
  }

  useEffect(() => {
    let cancelled = false
    const desktop = window.hermesDesktop

    if (!desktop?.getConnectionConfig) {
      setLoading(false)

      return () => void (cancelled = true)
    }

    setLoading(true)
    // New scope: forget prior edits and clear scope-local entry state so a token
    // from one scope can't leak into the next when switching profiles.
    dirtyRef.current = false
    setRemoteToken('')
    setLastTest(null)

    desktop
      .getConnectionConfig(scope)
      .then(config => {
        // Don't clobber a value the user already typed while the load was in
        // flight (the overlay renders the form immediately).
        if (cancelled || dirtyRef.current) {
          return
        }

        setState(lockedMode ? { ...config, mode: lockedMode } : config)
      })
      .catch(err => notifyError(err, g.failedLoad))
      .finally(() => {
        if (!cancelled) {
          setLoading(false)
        }
      })

    return () => void (cancelled = true)
    // eslint-disable-next-line react-hooks/exhaustive-deps -- reload on scope change only; copy + lockedMode are stable
  }, [scope])

  // Drop a stale inline error as soon as the user changes the URL or token.
  useEffect(() => {
    setLastError(null)
  }, [state.remoteUrl, remoteToken])

  useEffect(() => {
    saveSeq.current += 1
    signingSeq.current += 1
    testSeq.current += 1
    setLastTest(null)
  }, [scope, state.mode, state.sshHost, state.sshUser, state.sshPort, state.sshKeyPath, state.sshRemoteHermesPath])

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

    return t.boot.failure.identityProvider
  }, [probe, t.boot.failure.identityProvider])

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
    profile: scope ?? undefined,
    remoteAuthMode: authMode,
    remoteToken: authMode === 'token' ? remoteToken.trim() || undefined : undefined,
    remoteUrl: trimmedUrl,
    sshHost: state.sshHost.trim(),
    sshUser: state.sshUser.trim() || undefined,
    sshPort: state.sshPort,
    sshKeyPath: state.sshKeyPath.trim() || undefined,
    sshRemoteHermesPath: state.sshRemoteHermesPath.trim()
  })

  const save = async (apply: boolean) => {
    const seq = ++saveSeq.current

    if (state.mode === 'remote' && !canUseRemote) {
      notify({
        kind: 'warning',
        title: g.incompleteTitle,
        message: authMode === 'oauth' ? g.incompleteSignIn : g.incompleteToken
      })

      return
    }

    setSaving(true)
    setLastError(null)

    try {
      const next = apply
        ? await window.hermesDesktop.applyConnectionConfig(payload())
        : await window.hermesDesktop.saveConnectionConfig(payload())

      if (seq !== saveSeq.current) {
        return
      }

      setState(next)
      setRemoteToken('')

      // In the first-run context (lockedMode) an apply hands off to main's boot
      // transition + the overlay dismissal, which is the user's feedback — the
      // Settings "Restarting…" toast would be wrong copy there, so skip it.
      if (!lockedMode || !apply) {
        notify({
          kind: 'success',
          title: apply ? g.restartingTitle : g.savedTitle,
          message: apply ? g.restartingMessage : g.savedMessage
        })
      }
    } catch (err) {
      if (seq !== saveSeq.current) {
        return
      }

      const sshError = err && typeof err === 'object' && 'sshError' in err ? String(err.sshError) : ''

      const sshErrors = {
        'auth-failed': g.sshErrAuth,
        'hermes-not-found': g.sshErrNotInstalled,
        'host-key-changed': g.sshErrHostKey,
        timeout: g.sshErrTimeout,
        unreachable: g.sshErrUnreachable,
        'unsupported-platform': g.sshErrPlatform,
        'update-required': g.sshErrUpdateRequired
      }

      if (state.mode === 'ssh' && sshError) {
        const message = (sshErrors as Record<string, string>)[sshError] || g.sshErrUnknown
        notify({ kind: 'error', title: apply ? g.applyFailed : g.saveFailed, message })
        setLastError(message)
      } else {
        notifyError(err, apply ? g.applyFailed : g.saveFailed)
        setLastError(err instanceof Error ? err.message : String(err))
      }
    } finally {
      if (seq === saveSeq.current) {
        setSaving(false)
      }
    }
  }

  // OAuth sign-in: persist the URL + oauth mode first (so the saved config has
  // the URL the login window needs), then open the gateway login window and
  // refresh the connection status from the saved config once it completes.
  const signIn = async () => {
    const seq = ++signingSeq.current

    if (!trimmedUrl) {
      notify({ kind: 'warning', title: g.incompleteTitle, message: g.enterUrlFirst })

      return
    }

    setSigningIn(true)
    setLastError(null)

    try {
      // Save (don't apply/restart) so the login window has a URL to use and the
      // oauth mode is persisted, without yet flipping the live connection.
      const saved = await window.hermesDesktop.saveConnectionConfig({
        mode: state.mode,
        profile: scope ?? undefined,
        remoteAuthMode: 'oauth',
        remoteUrl: trimmedUrl
      })

      if (seq !== signingSeq.current) {
        return
      }

      setState(saved)

      const result = await window.hermesDesktop.oauthLoginConnectionConfig(trimmedUrl)

      if (seq !== signingSeq.current) {
        return
      }

      if (result.connected) {
        const refreshed = await window.hermesDesktop.getConnectionConfig(scope)
        setState(refreshed)
        notify({ kind: 'success', title: g.signedIn, message: g.connectedTo(providerLabel) })
      } else {
        notify({
          kind: 'warning',
          title: t.boot.failure.signInIncompleteTitle,
          message: t.boot.failure.signInIncompleteMessage
        })
      }
    } catch (err) {
      if (seq === signingSeq.current) {
        notifyError(err, g.signInFailed)
        setLastError(err instanceof Error ? err.message : String(err))
      }
    } finally {
      if (seq === signingSeq.current) {
        setSigningIn(false)
      }
    }
  }

  const signOut = async () => {
    const seq = ++signingSeq.current
    setSigningIn(true)

    try {
      await window.hermesDesktop.oauthLogoutConnectionConfig(trimmedUrl || undefined)
      const refreshed = await window.hermesDesktop.getConnectionConfig(scope)

      if (seq !== signingSeq.current) {
        return
      }

      setState(refreshed)
      notify({ kind: 'success', title: g.signedOutTitle, message: g.signedOutMessage })
    } catch (err) {
      if (seq === signingSeq.current) {
        notifyError(err, g.signOutFailed)
      }
    } finally {
      if (seq === signingSeq.current) {
        setSigningIn(false)
      }
    }
  }

  const testRemote = async () => {
    const seq = ++testSeq.current

    if (!canUseRemote) {
      notify({
        kind: 'warning',
        title: g.incompleteTitle,
        message: authMode === 'oauth' ? g.incompleteSignInTest : g.incompleteTokenTest
      })

      return
    }

    setTesting(true)
    setLastTest(null)
    setLastError(null)

    try {
      const result = await window.hermesDesktop.testConnectionConfig({
        mode: 'remote',
        profile: scope ?? undefined,
        remoteAuthMode: authMode,
        remoteToken: authMode === 'token' ? remoteToken.trim() || undefined : undefined,
        remoteUrl: trimmedUrl
      })

      if (seq !== testSeq.current) {
        return
      }

      const message = g.connectedTo(result.baseUrl || trimmedUrl, result.version ?? undefined)
      setLastTest(message)
      notify({ kind: 'success', title: g.reachableTitle, message })
    } catch (err) {
      if (seq === testSeq.current) {
        notifyError(err, g.testFailed)
        setLastError(err instanceof Error ? err.message : String(err))
      }
    } finally {
      if (seq === testSeq.current) {
        setTesting(false)
      }
    }
  }

  const testSsh = async () => {
    const seq = ++testSeq.current

    if (!state.sshHost.trim()) {
      notify({ kind: 'warning', title: g.incompleteTitle, message: g.sshIncompleteHost })

      return
    }

    setTesting(true)
    setLastTest(null)
    setLastError(null)

    try {
      const result = await window.hermesDesktop.testConnectionConfig(payload())

      if (seq !== testSeq.current) {
        return
      }

      if (!result.reachable) {
        const errors = {
          'auth-failed': g.sshErrAuth,
          'hermes-not-found': g.sshErrNotInstalled,
          'host-key-changed': g.sshErrHostKey,
          timeout: g.sshErrTimeout,
          unreachable: g.sshErrUnreachable,
          'unsupported-platform': g.sshErrPlatform,
          'update-required': g.sshErrUpdateRequired,
          unknown: g.sshErrUnknown
        }

        throw new Error(errors[result.sshError || 'unknown'] || result.error || g.sshErrUnknown)
      }

      const message = g.sshReachable(result.host || state.sshHost, result.remotePlatform || '?')
      setLastTest(message)
      notify({ kind: 'success', title: g.reachableTitle, message })
    } catch (err) {
      if (seq === testSeq.current) {
        notifyError(err, g.testFailed)
        setLastError(err instanceof Error ? err.message : String(err))
      }
    } finally {
      if (seq === testSeq.current) {
        setTesting(false)
      }
    }
  }

  return {
    state,
    setState: setStateDirty,
    loading,
    saving,
    testing,
    signingIn,
    remoteToken,
    setRemoteToken: setRemoteTokenDirty,
    lastTest,
    lastError,
    probeStatus,
    trimmedUrl,
    authMode,
    authResolved,
    providerLabel,
    isPasswordProvider,
    oauthConnected,
    canUseRemote,
    save,
    signIn,
    signOut,
    testRemote,
    testSsh
  }
}
