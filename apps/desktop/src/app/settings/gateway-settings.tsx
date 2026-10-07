import { isGatewayReauthRequired } from '@rabbit/shared'
import { useStore } from '@nanostores/react'
import { useEffect, useRef, useState } from 'react'

import { RemoteSetupFields } from '@/components/remote-setup/fields'
import { useRemoteSetup } from '@/components/remote-setup/use-remote-setup'
import { Button } from '@/components/ui/button'
import { ConfirmDialog } from '@/components/ui/confirm-dialog'
import { Input } from '@/components/ui/input'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Tip } from '@/components/ui/tooltip'
import type { DesktopConnectionConfigInput, DesktopRegistryConnection } from '@/global'
import { useI18n } from '@/i18n'
import { ExternalLink } from '@/lib/external-link'
import {
  AlertCircle,
  Check,
  FileText,
  Globe,
  HelpCircle,
  Loader2,
  Monitor,
  Terminal
} from '@/lib/icons'
import { coerceRemoteUrlScheme } from '@/lib/remote-url'
import { selectableCardClass } from '@/lib/selectable-card'
import { cn } from '@/lib/utils'
import {
  $activeConnectionId,
  $connectionsRegistry,
  refreshConnectionsRegistry,
  selectConnection
} from '@/store/connections'
import { managedUpdatesSupported } from '@/store/managed-updates'
import { notify, notifyError, readableError } from '@/store/notifications'

import { ConnectionsRegistrySection } from './connections-registry'
import { CONTROL_TEXT } from './constants'
import { ManagedUpdatesSection } from './managed-updates-section'
import { EmptyState, ListRow, Pill, SettingsContent, SettingsSkeleton, ToggleRow } from './primitives'
import { SETTING_IDS, settingElementId } from './settings-manifest'
import { enrichSelectedSshHost, selectSshHost } from './ssh-host-selection'
import { useSettingDeepLink } from './use-setting-deep-link'

type Mode = 'local' | 'remote' | 'ssh'
type AuthMode = 'oauth' | 'token'

export interface GatewaySettingsState {
  envOverride: boolean
  mode: Mode
  remoteAuthMode: AuthMode
  remoteOauthConnected: boolean
  remoteTokenPreview: string | null
  remoteTokenSet: boolean
  // Whether OS-keychain-backed encryption (Electron safeStorage) is available.
  // Default true so we never gate on a value we haven't hydrated yet.
  secureTokenStorage: boolean
  // Whether the currently-persisted remote token is stored as plain text on
  // disk (opted-in on a machine without secure storage). Drives the warning banner.
  remoteTokenPlainText: boolean
  remoteUrl: string
  sshHost: string
  sshUser: string
  sshPort: number | null
  sshKeyPath: string
  sshRemoteRabbitPath: string
  sshRemoteProfile: string
}

const SSH_HOST_CUSTOM = '__custom__'

const EMPTY_STATE: GatewaySettingsState = {
  envOverride: false,
  mode: 'local',
  remoteAuthMode: 'token',
  remoteOauthConnected: false,
  remoteTokenPreview: null,
  remoteTokenSet: false,
  secureTokenStorage: true,
  remoteTokenPlainText: false,
  remoteUrl: '',
  sshHost: '',
  sshUser: '',
  sshPort: null,
  sshKeyPath: '',
  sshRemoteRabbitPath: '',
  sshRemoteProfile: ''
}

export function normalizeGatewaySettingsState(
  config: Partial<GatewaySettingsState> | null | undefined
): GatewaySettingsState {
  if (!config || typeof config !== 'object') {
    return { ...EMPTY_STATE }
  }

  const defined = Object.fromEntries(Object.entries(config).filter(([, value]) => value != null))

  return { ...EMPTY_STATE, ...defined }
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

interface GatewaySettingsProps {
  embedded?: boolean
  subpage?: string
}

export function GatewaySettings({ embedded = false, subpage }: GatewaySettingsProps = {}) {
  useSettingDeepLink('gateway', page => subpage === undefined || page === subpage)

  // Recovery always keeps the complete connection form, regardless of a
  // settings destination. Other tasks never mount that form or its probes.
  if (!embedded && subpage === 'devices') {
    return (
      <SettingsContent>
        <ConnectionsRegistrySection />
      </SettingsContent>
    )
  }

  if (!embedded && subpage === 'managed-updates') {
    return <GatewayManagedUpdates />
  }

  return <GatewayConnectionSettings embedded={embedded} standalone={subpage !== undefined} />
}

function GatewayManagedUpdates() {
  const { t } = useI18n()
  const registry = useStore($connectionsRegistry)
  const [loading, setLoading] = useState(true)
  const [failed, setFailed] = useState(false)
  const supported = managedUpdatesSupported()

  useEffect(() => {
    if (!supported) {
      return
    }

    let active = true
    void refreshConnectionsRegistry()
      .catch(() => {
        if (active) {
          setFailed(true)
        }
      })
      .finally(() => {
        if (active) {
          setLoading(false)
        }
      })

    return () => {
      active = false
    }
  }, [supported])

  if (supported && loading) {
    return <SettingsSkeleton sections={[{ heading: true, rows: 3 }]} />
  }

  const hasSsh = registry?.connections.some(connection => connection.kind === 'ssh')

  return (
    <SettingsContent>
      {supported && !failed && hasSsh ? (
        <ManagedUpdatesSection />
      ) : (
        <EmptyState
          description={
            !supported
              ? t.settings.subpages.gatewayManagedUpdatesUnavailable
              : failed
                ? t.settings.gateway.failedLoad
                : t.settings.subpages.gatewayManagedUpdatesEmpty
          }
          title={t.settings.managedUpdates.title}
        />
      )}
    </SettingsContent>
  )
}

// `embedded` trims the page chrome for reuse inside the boot-failure recovery
// card: the outer title/intro, the "Save for next restart" action, and the
// Diagnostics row are redundant there (the card owns its header + a single
// reconnect action), so only the connection controls render.
function GatewayConnectionSettings({ embedded, standalone }: { embedded: boolean; standalone: boolean }) {
  const { t } = useI18n()
  const g = t.settings.gateway
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [testing, setTesting] = useState(false)
  const [state, setState] = useState<GatewaySettingsState>(EMPTY_STATE)

  const remote = useRemoteSetup({
    host: 'settings',
    enabled: !loading && state.mode === 'remote',
    beforeOAuthLogin: async (payload: DesktopConnectionConfigInput): Promise<void> => {
      await window.rabbitDesktop.saveConnectionConfig(payload)
    },
    onNotice: notify
  })

  const [lastTest, setLastTest] = useState<null | string>(null)
  const [sshHostSuggestions, setSshHostSuggestions] = useState<string[]>([])
  const [sshCustomHost, setSshCustomHost] = useState(false)
  const sshResolveSeq = useRef(0)
  const sshTestSeq = useRef(0)
  const saveSeq = useRef(0)
  const saveOwner = useRef<number | null>(null)
  const signingSeq = useRef(0)
  const contextSeq = useRef(0)
  const registry = useStore($connectionsRegistry)
  const activeConnectionId = useStore($activeConnectionId)

  useEffect(() => {
    void refreshConnectionsRegistry().catch(err => notifyError(err, g.failedLoad))
  }, [g.failedLoad])

  // Opt-in OS-keychain encryption for stored gateway secrets. Read lazily via
  // IPC (never touches the keychain); flipping it re-encodes stored secrets
  // in the main process and can legitimately prompt for keychain access.
  const [keychainEncryption, setKeychainEncryptionState] = useState(false)
  const [keychainEncryptionBusy, setKeychainEncryptionBusy] = useState(false)

  useEffect(() => {
    let cancelled = false

    void window.rabbitDesktop
      ?.getSecretStorageEncryption?.()
      .then(res => {
        if (!cancelled && res) {
          setKeychainEncryptionState(res.on === true)
        }
      })
      .catch(() => {})

    return () => {
      cancelled = true
    }
  }, [])

  const setKeychainEncryption = async (on: boolean) => {
    setKeychainEncryptionBusy(true)
    // Optimistic paint; the IPC result (or a failure rollback) gets the last word.
    setKeychainEncryptionState(on)

    try {
      const res = await window.rabbitDesktop.setSecretStorageEncryption(on)

      setKeychainEncryptionState(res?.on === true)
    } catch (err) {
      setKeychainEncryptionState(!on)
      notifyError(err, g.keychainEncryptionFailed)
    } finally {
      setKeychainEncryptionBusy(false)
    }
  }

  const acceptSavedConfig = (config: GatewaySettingsState): void => {
    const normalized = normalizeGatewaySettingsState(config)

    setState(normalized)
    remote.reset({
      url: normalized.remoteUrl,
      authMode: normalized.remoteAuthMode,
      oauthConnected: normalized.remoteOauthConnected,
      tokenSet: normalized.remoteTokenSet,
      tokenPreview: normalized.remoteTokenPreview
    })
  }

  // When set, the plain-text opt-in dialog is open; `apply` remembers whether
  // the gated action was Save-for-restart (false) or Save-and-reconnect (true)
  // so confirm resumes the right one.
  const [plainTextConfirm, setPlainTextConfirm] = useState<null | { apply: boolean }>(null)

  useEffect(() => {
    let cancelled = false
    const desktop = window.rabbitDesktop

    if (!desktop?.getConnectionConfig) {
      setLoading(false)

      return () => void (cancelled = true)
    }

    setLoading(true)

    desktop
      .getConnectionConfig(null)
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
    // eslint-disable-next-line react-hooks/exhaustive-deps -- load once on mount; copy is stable
  }, [])

  // Debounced probe of the entered remote URL. Only runs in remote mode with a
  // syntactically plausible URL. The probe result drives whether we render the
  // OAuth login button or the session-token entry box. The effective auth mode
  // prefers a fresh probe result over the saved value.
  const trimmedUrl = coerceRemoteUrlScheme(state.remoteUrl)


  useEffect(() => {
    // One-directional: a saved host that isn't in the suggestions must render
    // the free-text input (rehydration). Never force custom OFF here — that
    // instantly snapped the just-clicked-Custom (empty-host) input back to the
    // dropdown, making a raw-IP host impossible to type. The way back to the
    // dropdown is the input's onBlur (empty host + suggestions).
    if (state.sshHost && !sshHostSuggestions.includes(state.sshHost)) {
      setSshCustomHost(true)
    }
  }, [state.sshHost, sshHostSuggestions])

  useEffect(() => {
    if (state.mode !== 'ssh' || !window.rabbitDesktop?.sshConfigHosts) {
      return
    }

    let cancelled = false
    void window.rabbitDesktop
      .sshConfigHosts()
      .then(result => {
        if (!cancelled) {
          setSshHostSuggestions(result.hosts)
        }
      })
      .catch(() => {
        if (!cancelled) {
          setSshHostSuggestions([])
        }
      })

    return () => void (cancelled = true)
  }, [state.mode])

  // eslint-disable-next-line no-restricted-syntax -- monotonic request-sequence counters, not an atom mirror
  useEffect(() => {
    contextSeq.current += 1
    sshTestSeq.current += 1
    saveSeq.current += 1
    signingSeq.current += 1
    setLastTest(null)
  }, [
    state.mode,
    remote.credentials.url,
    remote.credentials.token,
    remote.credentials.authMode,
    state.sshHost,
    state.sshUser,
    state.sshPort,
    state.sshKeyPath,
    state.sshRemoteRabbitPath,
    state.sshRemoteProfile
  ])

  const payload = (allowPlainTextToken?: boolean): DesktopConnectionConfigInput => ({
    ...(state.mode === 'remote'
      ? remote.payload
      : {
          mode: state.mode,
          remoteAuthMode: state.remoteAuthMode,
          remoteUrl: coerceRemoteUrlScheme(state.remoteUrl),
          sshHost: state.sshHost.trim(),
          // Send an explicit '' for cleared fields. Main merges with `??`, so
          // `undefined` means "inherit the saved value" and a cleared Identity
          // file (or User) could never be removed once saved: the form kept
          // refilling the stale path, and the stale key path made the v1 SSH
          // route's identity differ from the registered gateway's, leaving
          // the window unscoped.
          sshUser: state.sshUser.trim(),
          sshPort: state.sshPort,
          sshKeyPath: state.sshKeyPath.trim(),
          sshRemoteRabbitPath: state.sshRemoteRabbitPath.trim(),
          // A blank clears an existing remote-profile mapping.
          sshRemoteProfile: state.sshRemoteProfile.trim()
        }),
    ...(allowPlainTextToken ? { allowPlainTextToken: true } : {})
  })

  // A pending Save/Apply would write a NEW token to disk in plain text when
  // we're on a remote-like connection using token auth, the user typed a token,
  // and this machine has no OS keyring (safeStorage unavailable). In that case
  // we must get an explicit opt-in before persisting.
  const wouldPersistPlainTextToken =
    state.mode === 'remote' &&
    remote.credentials.authMode === 'token' &&
    Boolean(remote.credentials.token.trim()) &&
    state.secureTokenStorage === false

  const performSave = async (apply: boolean, allowPlainTextToken: boolean): Promise<void> => {
    const seq = ++saveSeq.current
    saveOwner.current = seq
    setSaving(true)

    try {
      const next = apply
        ? await window.rabbitDesktop.applyConnectionConfig(payload(allowPlainTextToken))
        : await window.rabbitDesktop.saveConnectionConfig(payload(allowPlainTextToken))

      if (seq !== saveSeq.current) {
        return
      }

      acceptSavedConfig(next)
      notify({
        kind: 'success',
        title: apply ? g.restartingTitle : g.savedTitle,
        message: apply ? g.restartingMessage : g.savedMessage
      })
    } catch (err) {
      if (seq !== saveSeq.current) {
        return
      }

      // The plain-text opt-in path runs inside ConfirmDialog's onConfirm, which
      // keeps the dialog open with an inline error when it throws — rethrow a
      // readable message there so a failed save can't play the success beat.
      if (allowPlainTextToken) {
        throw new Error(readableError(err, apply ? g.applyFailed : g.saveFailed).message)
      }

      const sshError = err && typeof err === 'object' && 'sshError' in err ? String(err.sshError) : ''

      const errors = {
        'auth-failed': g.sshErrAuth,
        'rabbit-not-found': g.sshErrNotInstalled,
        'host-key-changed': g.sshErrHostKey,
        timeout: g.sshErrTimeout,
        unreachable: g.sshErrUnreachable,
        'unsupported-platform': g.sshErrPlatform,
        'update-required': g.sshErrUpdateRequired
      }

      if (state.mode === 'ssh' && sshError) {
        notify({
          kind: 'error',
          title: apply ? g.applyFailed : g.saveFailed,
          message: (errors as Record<string, string>)[sshError] || g.sshErrUnknown
        })
      } else {
        notifyError(err, apply ? g.applyFailed : g.saveFailed)
      }
    } finally {
      // A stale response cannot replace the draft, but its request must release busy state.
      if (seq === saveOwner.current) {
        saveOwner.current = null
        setSaving(false)
      }
    }
  }

  const save = async (apply: boolean): Promise<void> => {
    if (state.mode === 'remote' && !remote.canCommit) {
      notify({
        kind: 'warning',
        title: g.incompleteTitle,
        message: remote.credentials.authMode === 'oauth' ? g.incompleteSignIn : g.incompleteToken
      })

      return
    }

    // Defer to the opt-in dialog; confirm resumes with allowPlainTextToken.
    if (wouldPersistPlainTextToken) {
      setPlainTextConfirm({ apply })

      return
    }

    await performSave(apply, false)
  }

  const resolveSshHost = async (host: string) => {
    if (!host || !window.rabbitDesktop?.sshResolveHost) {
      return
    }

    const seq = ++sshResolveSeq.current

    try {
      const resolved = await window.rabbitDesktop.sshResolveHost(host)

      if (seq !== sshResolveSeq.current) {
        return
      }

      setState(current => enrichSelectedSshHost(current, host, resolved))
    } catch {
      return
    }
  }

  const selectHost = (value: string) => {
    if (value === SSH_HOST_CUSTOM) {
      setSshCustomHost(true)
      setState(current => selectSshHost(current, ''))

      return
    }

    setSshCustomHost(false)
    setState(current => selectSshHost(current, value))
    void resolveSshHost(value)
  }

  const testSsh = async () => {
    const seq = ++sshTestSeq.current

    if (!state.sshHost.trim()) {
      notify({ kind: 'warning', title: g.incompleteTitle, message: g.sshIncompleteHost })

      return
    }

    setTesting(true)
    setLastTest(null)

    try {
      const result = await window.rabbitDesktop.testConnectionConfig(payload())

      if (seq !== sshTestSeq.current) {
        return
      }

      if (!result.reachable) {
        const errors = {
          'auth-failed': g.sshErrAuth,
          'rabbit-not-found': g.sshErrNotInstalled,
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
      if (seq === sshTestSeq.current) {
        notifyError(err, g.testFailed)
      }
    } finally {
      if (seq === sshTestSeq.current) {
        setTesting(false)
      }
    }
  }

  if (loading) {
    return (
      <SettingsSkeleton
        sections={[
          { heading: true, rows: 3 },
          { heading: true, rows: 3 }
        ]}
      />
    )
  }

  if (!window.rabbitDesktop?.getConnectionConfig) {
    return <EmptyState description={g.unavailableDesc} title={g.unavailableTitle} />
  }

  return (
    <SettingsContent bare={embedded}>
      {embedded || standalone ? null : (
        <div className="mb-5">
          <div className="flex items-center gap-2 text-[length:var(--conversation-text-font-size)] font-medium">
            <Globe className="size-4 text-muted-foreground" />
            {g.title}
            {state.envOverride ? <Pill tone="primary">{g.envOverride}</Pill> : null}
          </div>
          <p className="mt-2 max-w-2xl text-[length:var(--conversation-caption-font-size)] leading-(--conversation-caption-line-height) text-(--ui-text-tertiary)">
            {g.intro}
          </p>
        </div>
      )}

      {state.envOverride ? (
        <div className="mb-5 flex items-start gap-2 rounded-xl border border-destructive/30 bg-destructive/10 px-3 py-2.5 text-[length:var(--conversation-caption-font-size)] text-destructive">
          <AlertCircle className="mt-0.5 size-4 shrink-0" />
          <div>
            <div className="font-medium">{g.envOverrideTitle}</div>
            <div className="mt-1 leading-5">{g.envOverrideDesc}</div>
          </div>
        </div>
      ) : null}

      <div className="mb-5 grid gap-2" id={settingElementId(SETTING_IDS.gateway.connectionMode)}>
        <div className="text-[length:var(--conversation-caption-font-size)] font-medium text-(--ui-text-secondary)">
          {g.modeTitle}
        </div>
        <div className="grid auto-rows-fr grid-cols-1 gap-2 sm:grid-cols-2 min-[72rem]:grid-cols-4">
          <ModeCard
            active={state.mode === 'local'}
            description={g.localDesc}
            disabled={state.envOverride}
            icon={Monitor}
            onSelect={() => setState(current => ({ ...current, mode: 'local' }))}
            title={g.localTitle}
          />
          <ModeCard
            active={state.mode === 'remote'}
            description={g.remoteDesc}
            disabled={state.envOverride}
            hint={g.remoteAuthHint}
            icon={Globe}
            onSelect={() => setState(current => ({ ...current, mode: 'remote' }))}
            title={g.remoteTitle}
          />
          <ModeCard
            active={state.mode === 'ssh'}
            description={g.sshDesc}
            disabled={state.envOverride}
            hint={g.sshTrustHint}
            icon={Terminal}
            onSelect={() => setState(current => ({ ...current, mode: 'ssh' }))}
            title={g.sshTitle}
          />
        </div>
      </div>

      {/* An env-pinned remote (RABBIT_DESKTOP_REMOTE_URL) still renders this
          block: the override pins the URL/mode, but the browser SESSION is not
          env-owned — docs promise "you still sign in from the Gateway settings
          panel" (user-guide/desktop.md). Hiding it left a lapsed session with
          no sign-in anywhere in Settings, and the boot-recovery card routes
          every remote failure here, so "Use local gateway" became the only way
          back in (#114856). The URL input and Save/Test stay env-gated. */}
      {state.mode === 'remote' ? (
        <div className="mt-5">
          <RemoteSetupFields disabled={saving} setup={remote} urlDisabled={state.envOverride} />
          {remote.credentials.authMode === 'token' && state.remoteTokenPlainText ? (
            <div className="mt-2 text-sm text-destructive">
              <div className="font-medium">{g.plainTextStoredTitle}</div>
              <div>{g.plainTextStoredDesc}</div>
            </div>
          ) : null}
        </div>
      ) : null}

      {state.mode === 'ssh' && !state.envOverride ? (
        <div className="mt-5 grid gap-1">
          {sshHostSuggestions.length > 0 && !sshCustomHost ? (
            <ListRow
              action={
                <Select
                  onValueChange={selectHost}
                  // Only report an actual host here. Leaving the value at the
                  // empty string shows the placeholder; using SSH_HOST_CUSTOM
                  // while the dropdown is still rendered would make the FIRST
                  // "Custom" click a no-op (Radix suppresses onValueChange when
                  // a controlled value doesn't change), so the custom-host
                  // input would never mount without a round-trip through
                  // another option.
                  value={sshHostSuggestions.includes(state.sshHost) ? state.sshHost : ''}
                >
                  <SelectTrigger className={cn('h-8', CONTROL_TEXT)}>
                    <SelectValue placeholder={g.sshHostPick} />
                  </SelectTrigger>
                  <SelectContent>
                    {sshHostSuggestions.map(host => (
                      <SelectItem key={host} value={host}>
                        {host}
                      </SelectItem>
                    ))}
                    <SelectItem value={SSH_HOST_CUSTOM}>{g.sshHostCustom}</SelectItem>
                  </SelectContent>
                </Select>
              }
              description={g.sshHostPickDesc}
              title={g.sshHostPickTitle}
            />
          ) : (
            <ListRow
              action={
                <Input
                  autoFocus={sshCustomHost}
                  className={cn('h-8', CONTROL_TEXT)}
                  onBlur={() => {
                    // Empty host on blur with suggestions available = the user backed
                    // out of Custom; return to the dropdown.
                    if (!state.sshHost.trim() && sshHostSuggestions.length > 0) {
                      setSshCustomHost(false)

                      return
                    }

                    void resolveSshHost(state.sshHost)
                  }}
                  onChange={event => setState(current => selectSshHost(current, event.target.value))}
                  value={state.sshHost}
                />
              }
              description={g.sshHostDesc}
              title={g.sshHostTitle}
            />
          )}
          <ListRow
            action={
              <Input
                className={cn('h-8', CONTROL_TEXT)}
                onChange={event => setState(current => ({ ...current, sshUser: event.target.value }))}
                placeholder={g.sshUserPlaceholder}
                value={state.sshUser}
              />
            }
            description={g.sshUserDesc}
            title={g.sshUserTitle}
          />
          <ListRow
            action={
              <Input
                className={cn('h-8', CONTROL_TEXT)}
                inputMode="numeric"
                onChange={event =>
                  setState(current => ({ ...current, sshPort: event.target.value ? Number(event.target.value) : null }))
                }
                placeholder="22"
                value={state.sshPort ?? ''}
              />
            }
            description={g.sshPortDesc}
            title={g.sshPortTitle}
          />
          <ListRow
            action={
              <Input
                className={cn('h-8 font-mono', CONTROL_TEXT)}
                onChange={event => setState(current => ({ ...current, sshKeyPath: event.target.value }))}
                value={state.sshKeyPath}
              />
            }
            description={g.sshKeyDesc}
            title={g.sshKeyTitle}
          />
          <ListRow
            action={
              <Input
                className={cn('h-8 font-mono', CONTROL_TEXT)}
                onChange={event => setState(current => ({ ...current, sshRemoteRabbitPath: event.target.value }))}
                placeholder={g.sshRabbitPathPlaceholder}
                value={state.sshRemoteRabbitPath}
              />
            }
            description={g.sshRabbitPathDesc}
            title={g.sshRabbitPathTitle}
          />
        </div>
      ) : null}

      {lastTest ? <div className="mt-4 text-xs text-primary">{lastTest}</div> : null}

      {(
        <div className="mt-6 flex flex-wrap items-center justify-end gap-4">
          {state.mode === 'remote' ? (
            <Button
              className="mr-auto"
              disabled={state.envOverride || saving || remote.testing || !remote.canTest}
              onClick={() => void remote.test()}
              size="sm"
              variant="text"
            >
              {remote.testing ? <Loader2 className="animate-spin" /> : null}
              {g.testRemote}
            </Button>
          ) : state.mode === 'ssh' ? (
            <Button
              className="mr-auto"
              disabled={testing || !state.sshHost.trim()}
              onClick={() => void testSsh()}
              size="sm"
              variant="text"
            >
              {testing ? <Loader2 className="animate-spin" /> : null}
              {g.sshTestConnection}
            </Button>
          ) : null}
          {embedded ? null : (
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
            {g.saveAndReconnect}
          </Button>
        </div>
      )}

      {embedded ? null : (
        <div className="mt-6 grid gap-1">
          <ToggleRow
            checked={keychainEncryption}
            description={g.keychainEncryptionDesc}
            disabled={keychainEncryptionBusy}
            id={settingElementId(SETTING_IDS.gateway.keychainEncryption)}
            label={g.keychainEncryptionTitle}
            onChange={on => void setKeychainEncryption(on)}
          />
          <ListRow
            action={
              <Button onClick={() => void window.rabbitDesktop?.revealLogs()} size="sm" variant="textStrong">
                <FileText />
                {g.openLogs}
              </Button>
            }
            description={g.diagnosticsDesc}
            id={settingElementId(SETTING_IDS.gateway.diagnostics)}
            title={g.diagnostics}
          />
        </div>
      )}

      {/* Preserve the full legacy page outside subpage navigation, without
          mounting registry editors in connection-only or recovery views. */}
      {embedded || standalone ? null : (
        <>
          <ConnectionsRegistrySection />
          {/* Per-connection driver for the transactional managed SSH update
              engine (#95942). Renders only when SSH sources are registered and
              the Electron main exposes connections.updateManaged. */}
          <ManagedUpdatesSection />
        </>
      )}

      {/* Plain-text token opt-in: gated when secure storage is unavailable and a
          new token would be persisted. Confirm resumes the remembered save/apply. */}
      <ConfirmDialog
        confirmLabel={g.plainTextConfirmAction}
        description={g.plainTextConfirmDesc}
        destructive
        onClose={() => setPlainTextConfirm(null)}
        onConfirm={async () => {
          if (!plainTextConfirm) {
            return
          }

          await performSave(plainTextConfirm.apply, true)
        }}
        open={plainTextConfirm !== null}
        title={g.plainTextConfirmTitle}
      />
    </SettingsContent>
  )
}
