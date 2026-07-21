import { useStore } from '@nanostores/react'
import { useEffect, useMemo, useRef, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Tip } from '@/components/ui/tooltip'
import type {
  DesktopAuthProvider,
  DesktopCloudAgent,
  DesktopCloudOrg,
  DesktopConnectionProbeResult,
  DesktopSavedConnection
} from '@/global'
import { useI18n } from '@/i18n'
import { ExternalLink } from '@/lib/external-link'
import {
  AlertCircle,
  Check,
  Cloud,
  FileText,
  Globe,
  HelpCircle,
  Loader2,
  LogIn,
  Monitor,
  Pencil,
  Plus,
  RefreshCw,
  Trash2
} from '@/lib/icons'
import { selectableCardClass } from '@/lib/selectable-card'
import { cn } from '@/lib/utils'
import { notify, notifyError } from '@/store/notifications'
import { $profiles, refreshActiveProfile } from '@/store/profile'

import { CONTROL_TEXT } from './constants'
import { EmptyState, ListRow, LoadingState, Pill, SettingsContent } from './primitives'

type Mode = 'local' | 'remote' | 'cloud'
type AuthMode = 'oauth' | 'token'
type ProbeStatus = 'idle' | 'probing' | 'done' | 'error'
// Hermes Cloud discovery lifecycle for the cloud-mode panel.
type CloudDiscoverStatus = 'idle' | 'loading' | 'done' | 'error'
type GlobalConnectionPanel = 'cloud' | 'remote' | null

interface GatewaySettingsState {
  connections: DesktopSavedConnection[]
  envOverride: boolean
  mode: Mode
  remoteAuthMode: AuthMode
  remoteOauthConnected: boolean
  remoteTokenPreview: string | null
  remoteTokenSet: boolean
  remoteUrl: string
  selectedConnectionId: null | string
  selectedConnectionName: string
  cloudOrg: string
}

const EMPTY_STATE: GatewaySettingsState = {
  connections: [],
  envOverride: false,
  mode: 'local',
  remoteAuthMode: 'token',
  remoteOauthConnected: false,
  remoteTokenPreview: null,
  remoteTokenSet: false,
  remoteUrl: '',
  selectedConnectionId: null,
  selectedConnectionName: '',
  cloudOrg: ''
}

export function savedCloudConnectionUrl(config: Pick<GatewaySettingsState, 'mode' | 'remoteUrl'>): string {
  return config.mode === 'cloud' ? config.remoteUrl.trim().replace(/\/+$/, '').toLowerCase() : ''
}

export function stateForRemoteMode(current: GatewaySettingsState, scope: null | string): GatewaySettingsState {
  if (scope !== null) {
    return { ...current, mode: 'remote' }
  }

  const selected = current.connections.find(connection => connection.id === current.selectedConnectionId)

  if (!selected) {
    return { ...current, mode: 'remote' }
  }

  return {
    ...current,
    mode: 'remote',
    remoteAuthMode: selected.remoteAuthMode,
    remoteOauthConnected: selected.remoteOauthConnected,
    remoteTokenPreview: selected.remoteTokenPreview,
    remoteTokenSet: selected.remoteTokenSet,
    remoteUrl: selected.remoteUrl,
    selectedConnectionName: selected.name
  }
}

export function stateForNewRemote(current: GatewaySettingsState): GatewaySettingsState {
  return {
    ...current,
    mode: 'remote',
    remoteAuthMode: 'token',
    remoteOauthConnected: false,
    remoteTokenPreview: null,
    remoteTokenSet: false,
    remoteUrl: '',
    selectedConnectionId: null,
    selectedConnectionName: ''
  }
}

export function stateForSavedRemote(
  current: GatewaySettingsState,
  connection: DesktopSavedConnection
): GatewaySettingsState {
  return {
    ...current,
    mode: 'remote',
    remoteAuthMode: connection.remoteAuthMode,
    remoteOauthConnected: connection.remoteOauthConnected,
    remoteTokenPreview: connection.remoteTokenPreview,
    remoteTokenSet: connection.remoteTokenSet,
    remoteUrl: connection.remoteUrl,
    selectedConnectionId: connection.id,
    selectedConnectionName: connection.name
  }
}

export function connectionsWithActiveSnapshot(
  current: GatewaySettingsState,
  nextConnections: DesktopSavedConnection[]
): DesktopSavedConnection[] {
  if (current.mode !== 'remote' || !current.selectedConnectionId) {
    return nextConnections
  }

  const active = current.connections.find(entry => entry.id === current.selectedConnectionId)

  return active ? nextConnections.map(entry => (entry.id === active.id ? active : entry)) : nextConnections
}

function ModeCard({
  active,
  description,
  disabled,
  hint,
  icon: Icon,
  onSelect,
  title
}: {
  active: boolean
  description: string
  disabled?: boolean
  hint?: string
  icon: typeof Monitor
  onSelect: () => void
  title: string
}) {
  return (
    <button
      className={cn(
        'flex h-full min-h-0 w-full flex-col p-3 text-left disabled:cursor-not-allowed disabled:opacity-50',
        selectableCardClass({ active, prominent: true })
      )}
      disabled={disabled}
      onClick={onSelect}
      type="button"
    >
      <div className="flex items-center gap-1.5">
        <Icon className="size-3.5 shrink-0 text-muted-foreground" />
        <span className="min-w-0 text-[length:var(--conversation-text-font-size)] font-medium">{title}</span>
        {hint ? (
          <Tip label={hint}>
            <span
              className="grid size-3.5 shrink-0 cursor-help place-items-center text-(--ui-text-tertiary) hover:text-(--ui-text-secondary)"
              onClick={event => event.stopPropagation()}
            >
              <HelpCircle className="size-3.5" />
            </span>
          </Tip>
        ) : null}
        {active ? <Check className="ml-auto size-3.5 shrink-0 text-primary" /> : null}
      </div>
      <p className="mt-1.5 flex-1 text-[length:var(--conversation-caption-font-size)] leading-(--conversation-caption-line-height) text-(--ui-text-tertiary)">
        {description}
      </p>
    </button>
  )
}

function ScopeChip({ active, label, onSelect }: { active: boolean; label: string; onSelect: () => void }) {
  return (
    <button
      className={cn(
        'rounded-full border px-3 py-1 text-[length:var(--conversation-caption-font-size)] transition',
        active
          ? 'border-(--ui-stroke-secondary) bg-(--ui-bg-tertiary) text-(--ui-text-primary)'
          : 'border-(--ui-stroke-tertiary) bg-(--ui-bg-quinary) text-(--ui-text-tertiary) hover:bg-(--chrome-action-hover)'
      )}
      onClick={onSelect}
      type="button"
    >
      {label}
    </button>
  )
}

// `embedded` trims the page chrome for reuse inside the boot-failure recovery
// card: the outer title/intro, the "Save for next restart" action, and the
// Diagnostics row are redundant there (the card owns its header + a single
// reconnect action), so only the connection controls render.
export function GatewaySettings({ embedded = false }: { embedded?: boolean } = {}) {
  const { t } = useI18n()
  const g = t.settings.gateway
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [testing, setTesting] = useState(false)
  const [signingIn, setSigningIn] = useState(false)
  const [state, setState] = useState<GatewaySettingsState>(EMPTY_STATE)
  const [activeConfig, setActiveConfig] = useState<GatewaySettingsState>(EMPTY_STATE)
  const [globalPanel, setGlobalPanel] = useState<GlobalConnectionPanel>(null)
  const [connectionNameInvalid, setConnectionNameInvalid] = useState(false)
  const [remoteToken, setRemoteToken] = useState('')
  const [lastTest, setLastTest] = useState<null | string>(null)
  const [connectedCloudUrl, setConnectedCloudUrl] = useState('')
  const editorRef = useRef<HTMLDivElement>(null)
  const connectionNameInputRef = useRef<HTMLInputElement>(null)

  const acceptSavedConfig = (config: GatewaySettingsState) => {
    setState(config)
    setActiveConfig(config)
    setConnectedCloudUrl(savedCloudConnectionUrl(config))
  }

  const acceptPersistedConfig = (config: GatewaySettingsState) => {
    setState(config)
    // Saving a draft for a later restart must not repaint it as the live
    // connection. Only refresh the catalog; Connect/apply remains the action
    // that moves the active pointer.
    setActiveConfig(current => ({
      ...current,
      connections: connectionsWithActiveSnapshot(current, config.connections)
    }))
  }

  const acceptPersistedRemoteConfig = (
    config: GatewaySettingsState,
    connectionId: null | string,
    connectionName: string
  ) => {
    const foldedName = connectionName.trim().toLocaleLowerCase()

    const connection =
      config.connections.find(entry => Boolean(connectionId) && entry.id === connectionId) ??
      config.connections.find(entry => foldedName && entry.name.toLocaleLowerCase() === foldedName) ??
      null

    setActiveConfig(current => ({
      ...current,
      connections: connectionsWithActiveSnapshot(current, config.connections)
    }))
    setState(connection ? stateForSavedRemote(config, connection) : config)

    return connection
  }

  const revealGlobalEditor = (panel: Exclude<GlobalConnectionPanel, null>, focusName = false) => {
    setGlobalPanel(panel)
    requestAnimationFrame(() => {
      editorRef.current?.scrollIntoView({ behavior: 'smooth', block: 'nearest' })

      if (focusName) {
        connectionNameInputRef.current?.focus()
      }
    })
  }

  // --- Hermes Cloud (cloud mode) state ---
  // One portal session powers discovery + the silent per-agent cascade. These
  // track the cloud panel: whether we're signed in, the discovered agent list,
  // and which agent is mid-connect.
  const [cloudSignedIn, setCloudSignedIn] = useState(false)
  const [cloudSigningIn, setCloudSigningIn] = useState(false)
  const [cloudAgents, setCloudAgents] = useState<DesktopCloudAgent[]>([])
  const [cloudDiscover, setCloudDiscover] = useState<CloudDiscoverStatus>('idle')
  const [cloudConnectingId, setCloudConnectingId] = useState<null | string>(null)
  // Multi-org users: when discovery returns needsOrgSelection, we hold the org
  // list here and show a picker. `cloudOrg` is the chosen org slug/id (null =
  // not yet chosen / single-org user).
  const [cloudOrgs, setCloudOrgs] = useState<DesktopCloudOrg[]>([])
  const [cloudOrg, setCloudOrgState] = useState<null | string>(null)
  // Mirror the selected org into a ref so connect reads the CURRENT value, not a
  // value captured in a stale render closure. discoverCloud() resolves the org
  // asynchronously (from the NAS response) and a user can click Connect in the
  // same render tick; without the ref, connectCloudAgent could persist a null
  // org even though discovery just resolved one. Always set both together.
  const cloudOrgRef = useRef<null | string>(null)

  const setCloudOrg = (value: null | string) => {
    cloudOrgRef.current = value
    setCloudOrgState(value)
  }

  // Connection scope: null = the global/default connection (the original
  // behavior); a profile name = that profile's per-profile remote override, so
  // each profile can point at its own backend.
  const [scope, setScope] = useState<null | string>(null)
  const profiles = useStore($profiles)
  const remoteEditorOpen = scope === null ? globalPanel === 'remote' : state.mode === 'remote'
  const cloudPanelOpen = scope === null ? globalPanel === 'cloud' : state.mode === 'cloud'
  const bottomActionsVisible = scope === null ? remoteEditorOpen : state.mode !== 'cloud'

  useEffect(() => {
    void refreshActiveProfile()
  }, [])

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

    setLoading(true)
    setGlobalPanel(null)
    setConnectionNameInvalid(false)
    // Clear scope-local entry state so a token from one scope can't leak into
    // the next when switching profiles.
    setRemoteToken('')
    setLastTest(null)

    desktop
      .getConnectionConfig(scope)
      .then(config => {
        if (cancelled) {
          return
        }

        acceptSavedConfig(config)
      })
      .catch(err => notifyError(err, g.failedLoad))
      .finally(() => {
        if (!cancelled) {
          setLoading(false)
        }
      })

    return () => void (cancelled = true)
    // eslint-disable-next-line react-hooks/exhaustive-deps -- reload on scope change only; copy is stable
  }, [scope])

  // Debounced probe of the entered remote URL. Only runs in remote mode with a
  // syntactically plausible URL. The probe result drives whether we render the
  // OAuth login button or the session-token entry box. The effective auth mode
  // prefers a fresh probe result over the saved value.
  const trimmedUrl = state.remoteUrl.trim()

  // The dashboardUrl of the currently-connected cloud instance (the saved
  // cloud connection's remoteUrl), normalized for comparison against each
  // discovered agent's dashboardUrl so we can highlight the active one and hide
  // its Connect button. Empty unless the saved connection is a cloud one.
  // The saved cloud URL was stored via the main-side normalizeRemoteBaseUrl
  // (which lowercases the host through URL.toString()), but a discovered agent's
  // dashboardUrl arrives raw from NAS — so normalize both sides the same way
  // (trim, drop trailing slash, lowercase) or a host-casing difference would
  // silently break the connected-highlight.
  const normalizeCloudUrl = (url: string) => url.trim().replace(/\/+$/, '').toLowerCase()

  const isConnectedAgent = (agent: DesktopCloudAgent) =>
    Boolean(connectedCloudUrl && agent.dashboardUrl && normalizeCloudUrl(agent.dashboardUrl) === connectedCloudUrl)

  useEffect(() => {
    if (!remoteEditorOpen || !trimmedUrl || !/^https?:\/\//i.test(trimmedUrl)) {
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
  }, [remoteEditorOpen, trimmedUrl])

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

  // The 'default' profile uses the global ("All profiles") connection, so the
  // per-profile scopes are the named, non-default profiles.
  const namedProfiles = useMemo(() => profiles.filter(profile => profile.name !== 'default'), [profiles])

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

  const selectRemoteMode = () => {
    setRemoteToken('')
    setState(current => stateForRemoteMode(current, scope))
  }

  const startNewConnection = () => {
    setConnectionNameInvalid(false)
    setRemoteToken('')
    setLastTest(null)
    setState(stateForNewRemote(activeConfig))
    revealGlobalEditor('remote', true)
  }

  const editSavedConnection = (connection: DesktopSavedConnection) => {
    setConnectionNameInvalid(false)
    setRemoteToken('')
    setLastTest(null)
    setState(stateForSavedRemote(activeConfig, connection))
    revealGlobalEditor('remote', true)
  }

  const cancelGlobalEditor = () => {
    setConnectionNameInvalid(false)
    setRemoteToken('')
    setLastTest(null)
    setState(activeConfig)
    setGlobalPanel(null)
  }

  const configureCloud = () => {
    setRemoteToken('')
    setLastTest(null)
    setState(current => ({ ...current, mode: 'cloud' }))
    revealGlobalEditor('cloud')
  }

  const connectLocal = async () => {
    setSaving(true)

    try {
      const next = await window.hermesDesktop.selectConnectionConfig('local')
      acceptSavedConfig(next)
      setRemoteToken('')
      setGlobalPanel(null)
      notify({ kind: 'success', title: g.switchingTitle, message: g.switchingMessage })
    } catch (err) {
      notifyError(err, g.applyFailed)
    } finally {
      setSaving(false)
    }
  }

  const connectSavedConnection = async (id: string) => {
    setSaving(true)

    try {
      const next = await window.hermesDesktop.selectConnectionConfig(id)
      acceptSavedConfig(next)
      setRemoteToken('')
      setGlobalPanel(null)
      notify({ kind: 'success', title: g.switchingTitle, message: g.switchingMessage })
    } catch (err) {
      notifyError(err, g.applyFailed)
    } finally {
      setSaving(false)
    }
  }

  const deleteSavedConnection = async (connection: DesktopSavedConnection) => {
    if (!window.confirm(g.deleteConnectionConfirm(connection.name))) {
      return
    }

    setSaving(true)

    try {
      const next = await window.hermesDesktop.deleteConnectionConfig(connection.id)
      acceptSavedConfig(next)
      setRemoteToken('')
      setGlobalPanel(null)
      notify({ kind: 'success', title: g.deletedConnectionTitle, message: g.deletedConnection(connection.name) })
    } catch (err) {
      notifyError(err, g.deleteConnectionFailed)
    } finally {
      setSaving(false)
    }
  }

  const payload = () => ({
    connectionId: scope === null ? state.selectedConnectionId : undefined,
    connectionName: scope === null ? state.selectedConnectionName.trim() || undefined : undefined,
    mode: state.mode,
    profile: scope ?? undefined,
    remoteAuthMode: authMode,
    remoteToken: authMode === 'token' ? remoteToken.trim() || undefined : undefined,
    remoteUrl: trimmedUrl
  })

  const save = async (apply: boolean) => {
    if (state.mode === 'remote' && !canUseRemote) {
      notify({
        kind: 'warning',
        title: g.incompleteTitle,
        message: authMode === 'oauth' ? g.incompleteSignIn : g.incompleteToken
      })

      return
    }

    if (scope === null && state.mode === 'remote' && !state.selectedConnectionName.trim()) {
      setConnectionNameInvalid(true)
      notify({ kind: 'warning', title: g.connectionNameRequiredTitle, message: g.connectionNameRequired })

      return
    }

    setSaving(true)

    try {
      const next = apply
        ? await window.hermesDesktop.applyConnectionConfig(payload())
        : await window.hermesDesktop.saveConnectionConfig(payload())

      if (apply) {
        acceptSavedConfig(next)
      } else {
        acceptPersistedConfig(next)
      }

      setRemoteToken('')

      if (scope === null) {
        setGlobalPanel(null)
      }

      const applyingGlobalConnection = apply && scope === null

      notify({
        kind: 'success',
        title: applyingGlobalConnection ? g.switchingTitle : apply ? g.restartingTitle : g.savedTitle,
        message: applyingGlobalConnection ? g.switchingMessage : apply ? g.restartingMessage : g.savedMessage
      })
    } catch (err) {
      notifyError(err, apply ? g.applyFailed : g.saveFailed)
    } finally {
      setSaving(false)
    }
  }

  // OAuth sign-in: persist the URL + oauth mode first (so the saved config has
  // the URL the login window needs), then open the gateway login window and
  // refresh the connection status from the saved config once it completes.
  const signIn = async () => {
    if (!trimmedUrl) {
      notify({ kind: 'warning', title: g.incompleteTitle, message: g.enterUrlFirst })

      return
    }

    if (scope === null && !state.selectedConnectionName.trim()) {
      setConnectionNameInvalid(true)
      notify({ kind: 'warning', title: g.connectionNameRequiredTitle, message: g.connectionNameRequired })

      return
    }

    const draftConnectionId = state.selectedConnectionId
    const draftConnectionName = state.selectedConnectionName

    setSigningIn(true)

    try {
      // Save (don't apply/restart) so the login window has a URL to use and the
      // oauth mode is persisted, without yet flipping the live connection.
      const saved = await window.hermesDesktop.saveConnectionConfig({
        connectionId: scope === null ? state.selectedConnectionId : undefined,
        connectionName: scope === null ? state.selectedConnectionName.trim() || undefined : undefined,
        mode: state.mode,
        preserveSelection: scope === null,
        profile: scope ?? undefined,
        remoteAuthMode: 'oauth',
        remoteUrl: trimmedUrl
      })

      const savedConnection = acceptPersistedRemoteConfig(saved, draftConnectionId, draftConnectionName)

      const result = await window.hermesDesktop.oauthLoginConnectionConfig(trimmedUrl)

      if (result.connected) {
        const refreshed = await window.hermesDesktop.getConnectionConfig(scope)
        acceptPersistedRemoteConfig(refreshed, savedConnection?.id ?? draftConnectionId, draftConnectionName)
        notify({ kind: 'success', title: g.signedIn, message: g.connectedTo(providerLabel) })
      } else {
        notify({
          kind: 'warning',
          title: t.boot.failure.signInIncompleteTitle,
          message: t.boot.failure.signInIncompleteMessage
        })
      }
    } catch (err) {
      notifyError(err, g.signInFailed)
    } finally {
      setSigningIn(false)
    }
  }

  const signOut = async () => {
    setSigningIn(true)

    try {
      await window.hermesDesktop.oauthLogoutConnectionConfig(trimmedUrl || undefined)
      const refreshed = await window.hermesDesktop.getConnectionConfig(scope)
      acceptPersistedRemoteConfig(refreshed, state.selectedConnectionId, state.selectedConnectionName)
      notify({ kind: 'success', title: g.signedOutTitle, message: g.signedOutMessage })
    } catch (err) {
      notifyError(err, g.signOutFailed)
    } finally {
      setSigningIn(false)
    }
  }

  // --- Hermes Cloud handlers ---

  // Pull the discovered agent list over the shared portal session. Tolerant of
  // a lapsed session: a needsCloudLogin error flips us back to signed-out.
  // `org` scopes discovery for multi-org users; when discovery comes back with
  // needsOrgSelection we surface the org list and show a picker instead.
  const discoverCloud = async (org?: string) => {
    const desktop = window.hermesDesktop

    if (!desktop?.cloud) {
      return
    }

    setCloudDiscover('loading')

    try {
      const result = await desktop.cloud.discover(org)

      if ('needsOrgSelection' in result && result.needsOrgSelection) {
        // Multi-org user with no org chosen yet: show the picker. Don't clear a
        // previously-chosen org list on a refresh.
        setCloudOrgs(result.orgs)
        setCloudAgents([])
        setCloudDiscover('done')

        return
      }

      // Single org (or org now chosen): we have agents.
      setCloudAgents('agents' in result ? result.agents : [])

      // Record the org AUTHORITATIVELY from the response (NAS echoes the org the
      // list was scoped to), falling back to the org we requested. This is what
      // gets persisted on connect, so it must be set even on single-membership
      // auto-resolve where no picker ran and no `org` arg was passed.
      const resolvedOrgRef = 'org' in result && result.org ? (result.org.slug ?? result.org.id) : null

      if (resolvedOrgRef) {
        setCloudOrg(resolvedOrgRef)
      } else if (org) {
        setCloudOrg(org)
      }

      setCloudDiscover('done')
    } catch (err) {
      setCloudAgents([])
      setCloudDiscover('error')

      // A lapsed/absent portal session means we're effectively signed out.
      if (err && typeof err === 'object' && 'needsCloudLogin' in err) {
        setCloudSignedIn(false)
      }

      notifyError(err, g.cloudDiscoverFailed)
    }
  }

  // User picked an org from the multi-org picker: remember it and re-run
  // discovery scoped to it.
  const selectCloudOrg = (org: DesktopCloudOrg) => {
    const ref = org.slug ?? org.id
    setCloudOrg(ref)
    void discoverCloud(ref)
  }

  // "Change org": clear the selected org and re-discover with no org arg. A
  // multi-org user gets NAS's 409 → the picker; a single-org user auto-resolves
  // back to their one org. Also clear the agent list so the current org's
  // agents don't linger under the picker while discovery re-runs.
  const changeCloudOrg = () => {
    setCloudOrg(null)
    setCloudAgents([])
    void discoverCloud()
  }

  // On entering cloud mode (or scope change), read the portal session status and
  // auto-discover when already signed in, so the picker is populated on open.
  useEffect(() => {
    if (!cloudPanelOpen) {
      return
    }

    const desktop = window.hermesDesktop

    if (!desktop?.cloud) {
      return
    }

    let cancelled = false
    desktop.cloud
      .status()
      .then(status => {
        if (cancelled) {
          return
        }

        setCloudSignedIn(status.signedIn)

        if (status.signedIn) {
          // Restore the persisted org (if any) so we reopen straight into that
          // org's agent list instead of the picker; discoverCloud(org) also
          // records it as the selected org. Empty → normal discovery (single-org
          // resolves automatically; multi-org shows the picker).
          const savedOrg = state.cloudOrg || ''

          if (savedOrg) {
            setCloudOrg(savedOrg)
          }

          void discoverCloud(savedOrg || undefined)
        } else {
          setCloudAgents([])
          setCloudOrgs([])
          setCloudOrg(null)
          setCloudDiscover('idle')
        }
      })
      .catch(() => {
        if (!cancelled) {
          setCloudSignedIn(false)
        }
      })

    return () => void (cancelled = true)
    // eslint-disable-next-line react-hooks/exhaustive-deps -- reload on mode/scope change only
  }, [cloudPanelOpen, scope])

  const cloudSignIn = async () => {
    const desktop = window.hermesDesktop

    if (!desktop?.cloud) {
      return
    }

    setCloudSigningIn(true)

    try {
      const result = await desktop.cloud.login()
      setCloudSignedIn(result.signedIn)

      if (result.signedIn) {
        await discoverCloud()
      }
    } catch (err) {
      notifyError(err, g.cloudSignInFailed)
    } finally {
      setCloudSigningIn(false)
    }
  }

  const cloudSignOut = async () => {
    const desktop = window.hermesDesktop

    if (!desktop?.cloud) {
      return
    }

    setCloudSigningIn(true)

    try {
      await desktop.cloud.logout()
      setCloudSignedIn(false)
      setCloudAgents([])
      setCloudOrgs([])
      setCloudOrg(null)
      setCloudDiscover('idle')
      notify({ kind: 'success', title: g.cloudSignedOutTitle, message: g.cloudSignedOutMessage })
    } catch (err) {
      notifyError(err, g.signOutFailed)
    } finally {
      setCloudSigningIn(false)
    }
  }

  // Select a discovered agent: drive the silent per-agent cascade (no second
  // prompt — the shared portal session auto-approves), then persist a cloud-mode
  // connection pointed at its dashboardUrl and apply it (soft-reconnects in place).
  const connectCloudAgent = async (agent: DesktopCloudAgent) => {
    if (!agent.dashboardUrl) {
      return
    }

    const desktop = window.hermesDesktop

    if (!desktop?.cloud) {
      return
    }

    setCloudConnectingId(agent.id)

    try {
      const result = await desktop.cloud.agentSignIn(agent.dashboardUrl)

      if (!result.connected) {
        notify({
          kind: 'warning',
          title: t.boot.failure.signInIncompleteTitle,
          message: t.boot.failure.signInIncompleteMessage
        })

        return
      }

      // Persist a cloud-mode connection (remote-shaped, oauth) and soft-reconnect.
      // Include the selected org so Settings reopens into the same org + instance.
      // Read the REF (not the cloudOrg state) so a just-resolved org from
      // discovery in this same render tick is captured, not a stale null.
      const next = await desktop.applyConnectionConfig({
        mode: 'cloud',
        profile: scope ?? undefined,
        remoteAuthMode: 'oauth',
        remoteUrl: agent.dashboardUrl,
        cloudOrg: cloudOrgRef.current ?? undefined
      })

      acceptSavedConfig(next)

      if (scope === null) {
        setGlobalPanel(null)
      }

      notify({ kind: 'success', title: g.cloudConnectedTitle, message: g.cloudConnectedTo(agent.name) })
    } catch (err) {
      if (err && typeof err === 'object' && 'needsCloudLogin' in err) {
        setCloudSignedIn(false)
      }

      notifyError(err, g.cloudConnectFailed)
    } finally {
      setCloudConnectingId(null)
    }
  }

  const testRemote = async () => {
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

    try {
      const result = await window.hermesDesktop.testConnectionConfig({
        mode: 'remote',
        profile: scope ?? undefined,
        remoteAuthMode: authMode,
        remoteToken: authMode === 'token' ? remoteToken.trim() || undefined : undefined,
        remoteUrl: trimmedUrl
      })

      const message = g.connectedTo(result.baseUrl, result.version ?? undefined)
      setLastTest(message)
      notify({ kind: 'success', title: g.reachableTitle, message })
    } catch (err) {
      notifyError(err, g.testFailed)
    } finally {
      setTesting(false)
    }
  }

  if (loading) {
    return <LoadingState label={g.loading} />
  }

  if (!window.hermesDesktop?.getConnectionConfig) {
    return <EmptyState description={g.unavailableDesc} title={g.unavailableTitle} />
  }

  return (
    <SettingsContent bare={embedded}>
      {embedded ? null : (
        <div className="mb-5">
          <div className="flex items-center gap-2 text-[length:var(--conversation-text-font-size)] font-medium">
            <Globe className="size-4 text-muted-foreground" />
            {g.title}
            {state.envOverride ? <Pill tone="primary">{g.envOverride}</Pill> : null}
          </div>
          <p className="mt-2 max-w-2xl text-[length:var(--conversation-caption-font-size)] leading-(--conversation-caption-line-height) text-(--ui-text-tertiary)">
            {scope === null ? g.connectionsIntro : g.intro}
          </p>
        </div>
      )}

      {namedProfiles.length > 0 ? (
        <div className="mb-5 grid gap-2">
          <div className="text-[length:var(--conversation-caption-font-size)] font-medium text-(--ui-text-secondary)">
            {g.appliesTo}
          </div>
          <div className="flex flex-wrap gap-1.5">
            <ScopeChip active={scope === null} label={g.allProfiles} onSelect={() => setScope(null)} />
            {namedProfiles.map(profile => (
              <ScopeChip
                active={scope === profile.name}
                key={profile.name}
                label={profile.name}
                onSelect={() => setScope(profile.name)}
              />
            ))}
          </div>
          <p className="text-[length:var(--conversation-caption-font-size)] leading-(--conversation-caption-line-height) text-(--ui-text-tertiary)">
            {scope === null ? g.defaultConnection : g.profileConnection(scope)}
          </p>
        </div>
      ) : null}

      {scope === null ? (
        <div className="mb-5 grid gap-2">
          <div className="flex items-center justify-between gap-3">
            <div>
              <div className="text-[length:var(--conversation-caption-font-size)] font-medium text-(--ui-text-secondary)">
                {g.savedConnectionsTitle}
              </div>
              <p className="mt-1 text-[length:var(--conversation-caption-font-size)] leading-(--conversation-caption-line-height) text-(--ui-text-tertiary)">
                {g.savedConnectionsDesc}
              </p>
            </div>
            <Button
              disabled={activeConfig.envOverride || saving}
              onClick={startNewConnection}
              size="sm"
              variant="outline"
            >
              <Plus />
              {g.addConnection}
            </Button>
          </div>

          <div className="grid gap-1">
            <ListRow
              action={
                !activeConfig.envOverride && activeConfig.mode === 'local' ? (
                  <Pill tone="primary">
                    <Check className="size-3" /> {g.connectedConnection}
                  </Pill>
                ) : (
                  <Button disabled={saving} onClick={() => void connectLocal()} size="sm">
                    {g.connectConnection}
                  </Button>
                )
              }
              description={g.localDesc}
              title={g.localTitle}
            />

            {activeConfig.connections.map(connection => {
              const active =
                !activeConfig.envOverride &&
                activeConfig.mode === 'remote' &&
                activeConfig.selectedConnectionId === connection.id

              return (
                <div
                  aria-current={active ? 'true' : undefined}
                  aria-label={connection.name}
                  className={cn('px-3', selectableCardClass({ active, prominent: true }))}
                  key={connection.id}
                  role="group"
                >
                  <ListRow
                    action={
                      <div className="flex items-center gap-1.5">
                        {active ? (
                          <Pill tone="primary">
                            <Check className="size-3" /> {g.connectedConnection}
                          </Pill>
                        ) : (
                          <Button
                            disabled={activeConfig.envOverride || saving}
                            onClick={() => void connectSavedConnection(connection.id)}
                            size="sm"
                          >
                            {g.connectConnection}
                          </Button>
                        )}
                        <Button
                          aria-label={`${t.common.change} ${connection.name}`}
                          disabled={activeConfig.envOverride || saving}
                          onClick={() => editSavedConnection(connection)}
                          size="icon-sm"
                          variant="ghost"
                        >
                          <Pencil />
                        </Button>
                        <Button
                          aria-label={g.deleteConnection(connection.name)}
                          disabled={activeConfig.envOverride || saving}
                          onClick={() => void deleteSavedConnection(connection)}
                          size="icon-sm"
                          variant="ghost"
                        >
                          <Trash2 />
                        </Button>
                      </div>
                    }
                    description={g.connectionShortcut(connection.id, connection.remoteUrl)}
                    title={connection.name}
                  />
                </div>
              )
            })}

            <ListRow
              action={
                <div className="flex items-center gap-1.5">
                  {!activeConfig.envOverride && activeConfig.mode === 'cloud' ? (
                    <Pill tone="primary">
                      <Check className="size-3" /> {g.connectedConnection}
                    </Pill>
                  ) : null}
                  <Button
                    disabled={activeConfig.envOverride || saving}
                    onClick={configureCloud}
                    size="sm"
                    variant={activeConfig.mode === 'cloud' ? 'outline' : 'default'}
                  >
                    {activeConfig.mode === 'cloud' ? t.common.change : t.common.choose}
                  </Button>
                </div>
              }
              description={g.cloudDesc}
              title={g.cloudTitle}
            />
          </div>
        </div>
      ) : null}

      {state.envOverride ? (
        <div className="mb-5 flex items-start gap-2 rounded-xl border border-destructive/30 bg-destructive/10 px-3 py-2.5 text-[length:var(--conversation-caption-font-size)] text-destructive">
          <AlertCircle className="mt-0.5 size-4 shrink-0" />
          <div>
            <div className="font-medium">{g.envOverrideTitle}</div>
            <div className="mt-1 leading-5">{g.envOverrideDesc}</div>
          </div>
        </div>
      ) : null}

      {scope !== null ? (
        <div className="mb-5 grid gap-2">
          <div className="text-[length:var(--conversation-caption-font-size)] font-medium text-(--ui-text-secondary)">
            {g.modeTitle}
          </div>
          <div className="grid auto-rows-fr grid-cols-1 gap-2 min-[42rem]:grid-cols-3">
            <ModeCard
              active={state.mode === 'local'}
              description={g.localDesc}
              disabled={state.envOverride}
              icon={Monitor}
              onSelect={() => setState(current => ({ ...current, mode: 'local' }))}
              title={g.localTitle}
            />
            <ModeCard
              active={state.mode === 'cloud'}
              description={g.cloudDesc}
              disabled={state.envOverride}
              icon={Cloud}
              onSelect={() => setState(current => ({ ...current, mode: 'cloud' }))}
              title={g.cloudTitle}
            />
            <ModeCard
              active={state.mode === 'remote'}
              description={g.remoteDesc}
              disabled={state.envOverride}
              hint={g.remoteAuthHint}
              icon={Globe}
              onSelect={selectRemoteMode}
              title={g.remoteTitle}
            />
          </div>
        </div>
      ) : null}

      {/* Hermes Cloud panel: one portal sign-in, then a discovered-agent picker
          whose selection drives the silent per-agent cascade + a cloud
          connection. Replaces the URL/token form while in cloud mode. */}
      {cloudPanelOpen && !state.envOverride ? (
        <div
          className={cn('mt-5 grid gap-1', scope === null && 'rounded-xl border border-(--ui-stroke-tertiary) p-3')}
          ref={scope === null ? editorRef : undefined}
        >
          {scope === null ? (
            <div className="mb-2 flex items-center justify-between gap-3">
              <div className="text-[length:var(--conversation-text-font-size)] font-medium">{g.cloudTitle}</div>
              <Button onClick={cancelGlobalEditor} size="sm" variant="text">
                {t.common.cancel}
              </Button>
            </div>
          ) : null}
          <ListRow
            action={
              cloudSignedIn ? (
                <div className="flex items-center gap-2">
                  <Pill tone="primary">
                    <Check className="size-3" /> {g.cloudSignedIn}
                  </Pill>
                  <Button disabled={cloudSigningIn} onClick={() => void cloudSignOut()} variant="outline">
                    {cloudSigningIn ? <Loader2 className="animate-spin" /> : null}
                    {g.signOut}
                  </Button>
                </div>
              ) : (
                <Button disabled={cloudSigningIn} onClick={() => void cloudSignIn()}>
                  {cloudSigningIn ? <Loader2 className="animate-spin" /> : <LogIn />}
                  {g.cloudSignIn}
                </Button>
              )
            }
            description={cloudSignedIn ? g.cloudSignedInDesc : g.cloudNeedsSignIn}
            title={g.cloudSignInTitle}
          />

          {cloudSignedIn ? (
            cloudOrgs.length > 0 && !cloudOrg ? (
              // Multi-org user who hasn't picked an org yet: show the org picker
              // instead of the agent list. Selecting one re-runs discovery
              // scoped to it.
              <div className="mt-3">
                <div className="mb-2 text-[length:var(--conversation-caption-font-size)] font-medium text-(--ui-text-secondary)">
                  {g.cloudOrgPickerTitle}
                </div>
                <div className="grid gap-1">
                  {cloudOrgs.map(orgEntry => (
                    <ListRow
                      action={
                        <Button onClick={() => selectCloudOrg(orgEntry)} size="sm">
                          {g.cloudOrgSelect}
                        </Button>
                      }
                      description={g.cloudOrgRole(orgEntry.role)}
                      key={orgEntry.id}
                      title={orgEntry.name}
                    />
                  ))}
                </div>
              </div>
            ) : (
              <div className="mt-3">
                <div className="mb-2 flex items-center justify-between">
                  <div className="text-[length:var(--conversation-caption-font-size)] font-medium text-(--ui-text-secondary)">
                    {g.cloudAgentsTitle}
                  </div>
                  <div className="flex items-center gap-2">
                    {cloudOrg ? (
                      // Let the user switch orgs. Gating on cloudOrgs.length would
                      // hide this after a restore-open (which discovers straight
                      // into the saved org and never populates the org list). So
                      // show it whenever an org is selected: clicking clears the
                      // org and re-runs discovery with no org arg — a multi-org
                      // user gets the picker (NAS 409), a single-org user simply
                      // auto-resolves back to their one org (harmless).
                      <Button onClick={() => changeCloudOrg()} size="sm" variant="text">
                        {g.cloudOrgChange}
                      </Button>
                    ) : null}
                    <Button
                      disabled={cloudDiscover === 'loading'}
                      onClick={() => void discoverCloud(cloudOrg ?? undefined)}
                      size="sm"
                      variant="text"
                    >
                      {cloudDiscover === 'loading' ? <Loader2 className="animate-spin" /> : <RefreshCw />}
                      {g.cloudRefresh}
                    </Button>
                  </div>
                </div>

                {cloudDiscover === 'loading' ? (
                  <div className="flex items-center gap-2 py-3 text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
                    <Loader2 className="size-4 animate-spin" />
                    {g.cloudLoadingAgents}
                  </div>
                ) : cloudAgents.length === 0 ? (
                  <div className="flex items-start gap-2 py-3 text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
                    <AlertCircle className="mt-0.5 size-4 shrink-0" />
                    <span>
                      {g.cloudNoAgents.before}
                      <ExternalLink href="https://portal.nousresearch.com/agents" showExternalIcon={false}>
                        {g.cloudNoAgents.linkText}
                      </ExternalLink>
                      {g.cloudNoAgents.after}
                    </span>
                  </div>
                ) : (
                  <div className="grid gap-1">
                    {cloudAgents.map(agent => {
                      const connected = isConnectedAgent(agent)

                      return (
                        <div
                          className={cn('rounded-md px-2', connected && 'bg-primary/5 ring-1 ring-primary/25')}
                          key={agent.id}
                        >
                          <ListRow
                            action={
                              connected ? (
                                <Pill tone="primary">
                                  <Check className="mr-1 inline size-3" />
                                  {g.cloudConnectedPill}
                                </Pill>
                              ) : (
                                <Button
                                  disabled={!agent.dashboardUrl || cloudConnectingId !== null}
                                  onClick={() => void connectCloudAgent(agent)}
                                  size="sm"
                                >
                                  {cloudConnectingId === agent.id ? <Loader2 className="animate-spin" /> : null}
                                  {agent.dashboardUrl
                                    ? cloudConnectingId === agent.id
                                      ? g.cloudConnecting
                                      : g.cloudConnect
                                    : g.cloudAgentProvisioning}
                                </Button>
                              )
                            }
                            description={g.cloudStatusLabel(agent.dashboardGatewayState)}
                            title={agent.name}
                          />
                        </div>
                      )
                    })}
                  </div>
                )}
              </div>
            )
          ) : null}
        </div>
      ) : null}

      {remoteEditorOpen && !state.envOverride ? (
        <div
          className={cn('mt-5 grid gap-1', scope === null && 'rounded-xl border border-(--ui-stroke-tertiary) p-3')}
          ref={scope === null ? editorRef : undefined}
        >
          {scope === null ? (
            <div className="mb-2 flex items-center justify-between gap-3">
              <div className="text-[length:var(--conversation-text-font-size)] font-medium">
                {state.selectedConnectionId ? state.selectedConnectionName : g.addConnection}
              </div>
              <Button onClick={cancelGlobalEditor} size="sm" variant="text">
                {t.common.cancel}
              </Button>
            </div>
          ) : null}
          {scope === null ? (
            <ListRow
              action={
                <Input
                  aria-invalid={connectionNameInvalid}
                  aria-label={g.connectionNameTitle}
                  className={cn('h-8', CONTROL_TEXT)}
                  onChange={event => {
                    const value = event.target.value

                    if (value.trim()) {
                      setConnectionNameInvalid(false)
                    }

                    setState(current => ({ ...current, selectedConnectionName: value }))
                  }}
                  placeholder={g.connectionNamePlaceholder}
                  ref={connectionNameInputRef}
                  value={state.selectedConnectionName}
                />
              }
              description={
                connectionNameInvalid
                  ? g.connectionNameRequired
                  : state.selectedConnectionId
                    ? g.connectionIdDesc(state.selectedConnectionId)
                    : g.connectionNameDesc
              }
              title={g.connectionNameTitle}
            />
          ) : null}
          <ListRow
            action={
              <Input
                aria-label={g.remoteUrlTitle}
                className={cn('h-8', CONTROL_TEXT)}
                disabled={state.envOverride}
                onChange={event => setState(current => ({ ...current, remoteUrl: event.target.value }))}
                placeholder="https://gateway.example.com/hermes"
                value={state.remoteUrl}
              />
            }
            description={g.remoteUrlDesc}
            title={g.remoteUrlTitle}
          />

          {state.mode === 'remote' && probeStatus === 'probing' ? (
            <div className="flex items-center gap-2 py-3 text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
              <Loader2 className="size-4 animate-spin" />
              {g.probing}
            </div>
          ) : null}

          {state.mode === 'remote' && probeStatus === 'error' ? (
            <div className="flex items-start gap-2 py-3 text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
              <AlertCircle className="mt-0.5 size-4 shrink-0" />
              {g.probeError}
            </div>
          ) : null}

          {/* OAuth / password gateways: present a sign-in button + connection status. */}
          {state.mode === 'remote' && authResolved && authMode === 'oauth' ? (
            <ListRow
              action={
                oauthConnected ? (
                  <div className="flex items-center gap-2">
                    <Pill tone="primary">
                      <Check className="size-3" /> {g.signedIn}
                    </Pill>
                    <Button disabled={signingIn || state.envOverride} onClick={() => void signOut()} variant="outline">
                      {signingIn ? <Loader2 className="animate-spin" /> : null}
                      {g.signOut}
                    </Button>
                  </div>
                ) : (
                  <Button disabled={signingIn || state.envOverride || !trimmedUrl} onClick={() => void signIn()}>
                    {signingIn ? <Loader2 className="animate-spin" /> : <LogIn />}
                    {isPasswordProvider ? g.signIn : g.signInWith(providerLabel)}
                  </Button>
                )
              }
              description={
                oauthConnected
                  ? isPasswordProvider
                    ? g.authSignedInPassword
                    : g.authSignedInOauth
                  : isPasswordProvider
                    ? g.authNeedsPassword
                    : g.authNeedsOauth(providerLabel)
              }
              title={g.authTitle}
            />
          ) : null}

          {/* Session-token gateways: keep the existing token entry box. */}
          {state.mode === 'remote' && authResolved && authMode === 'token' ? (
            <ListRow
              action={
                <Input
                  aria-label={g.tokenTitle}
                  autoComplete="off"
                  className={cn('h-8 font-mono', CONTROL_TEXT)}
                  disabled={state.envOverride}
                  onChange={event => setRemoteToken(event.target.value)}
                  placeholder={
                    state.remoteTokenSet
                      ? g.existingToken(state.remoteTokenPreview ?? g.savedToken)
                      : g.pasteSessionToken
                  }
                  type="password"
                  value={remoteToken}
                />
              }
              description={g.tokenDesc}
              title={g.tokenTitle}
            />
          ) : null}
        </div>
      ) : null}

      {lastTest ? <div className="mt-4 text-xs text-primary">{lastTest}</div> : null}

      {/* Test/Save apply to local + remote. Cloud connects via the agent picker
          above (which applies a cloud connection on select), so its only
          bottom-row action would be redundant — hidden in cloud mode. */}
      {bottomActionsVisible ? (
        <div className="mt-6 flex flex-wrap items-center justify-end gap-4">
          {remoteEditorOpen ? (
            <Button
              className="mr-auto"
              disabled={state.envOverride || testing || !canUseRemote}
              onClick={() => void testRemote()}
              size="sm"
              variant="text"
            >
              {testing ? <Loader2 className="animate-spin" /> : null}
              {g.testRemote}
            </Button>
          ) : null}
          {embedded || scope === null ? null : (
            <Button
              disabled={state.envOverride || saving}
              onClick={() => void save(false)}
              size="sm"
              variant="textStrong"
            >
              {g.saveForRestart}
            </Button>
          )}
          <Button disabled={state.envOverride || saving} onClick={() => void save(true)} size="sm">
            {saving ? <Loader2 className="animate-spin" /> : null}
            {scope === null ? g.saveAndConnect : g.saveAndReconnect}
          </Button>
        </div>
      ) : null}

      {embedded ? null : (
        <div className="mt-6 grid gap-1">
          <ListRow
            action={
              <Button onClick={() => void window.hermesDesktop?.revealLogs()} size="sm" variant="textStrong">
                <FileText />
                {g.openLogs}
              </Button>
            }
            description={g.diagnosticsDesc}
            title={g.diagnostics}
          />
        </div>
      )}
    </SettingsContent>
  )
}
