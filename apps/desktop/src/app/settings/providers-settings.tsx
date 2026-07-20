import { useStore } from '@nanostores/react'
import type { ReactNode } from 'react'
import { useCallback, useEffect, useMemo, useState } from 'react'

import { runInTerminal } from '@/app/right-sidebar/store'
import {
  FEATURED_ID,
  FeaturedProviderRow,
  FireworksProviderRow,
  OpenRouterProviderRow,
  ProviderRow,
  providerTitle,
  sortProviders
} from '@/components/onboarding'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { RowButton } from '@/components/ui/row-button'
import { SearchField } from '@/components/ui/search-field'
import { disconnectOAuthProvider, getCustomEndpoint, listOAuthProviders, saveCustomEndpoint } from '@/hermes'
import { useI18n } from '@/i18n'
import { Check, ChevronDown, ChevronRight, KeyRound, Loader2, Terminal, Trash2 } from '@/lib/icons'
import { normalize } from '@/lib/text'
import { cn } from '@/lib/utils'
import { notify, notifyError } from '@/store/notifications'
import { $desktopOnboarding, startManualProviderOAuth } from '@/store/onboarding'
import type { CustomEndpointConfig, EnvVarInfo, OAuthProvider } from '@/types/hermes'

import { CREDENTIAL_CONTROL_CLASS, isKeyVar, ProviderKeyRows } from './credential-key-ui'
import { SettingsCategoryHeading, useEnvCredentials } from './env-credentials'
import { providerGroup, providerMeta, providerPriority } from './helpers'
import { ListRow, LoadingState, SettingsContent } from './primitives'

// The embedded terminal (and thus the "run disconnect command" path) only
// exists in the Electron desktop shell, not the web dashboard.
const canRunInTerminal = () => typeof window !== 'undefined' && Boolean(window.hermesDesktop?.terminal)

// Parallel group headers ("Connected", "Other providers") so the expanded list
// reads as its own section instead of bleeding into the connected group.
function GroupLabel({ children }: { children: ReactNode }) {
  return (
    <p className="mt-3 px-0.5 text-[length:var(--conversation-caption-font-size)] font-medium text-(--ui-text-tertiary)">
      {children}
    </p>
  )
}

// Sub-views surfaced as a sidebar subnav: account sign-in vs raw API keys.
export const PROVIDER_VIEWS = ['accounts', 'keys'] as const

export type ProviderView = (typeof PROVIDER_VIEWS)[number]

// Group the env catalog by provider — one ListRow per vendor plus optional
// advanced overrides (base URL, region, etc.). Groups without a key field are
// skipped.
//
// Grouping key precedence:
//   1. Backend `provider_label` / `provider` (from the unified provider catalog
//      in hermes_cli/provider_catalog.py) — the SAME provider identity
//      `hermes model` uses. This is authoritative: a provider tagged by the
//      backend always renders a card, even with no PROVIDER_GROUPS row.
//   2. Desktop prefix match (`providerGroup`) — legacy fallback for provider
//      env vars that predate the backend tagging.
// Only entries that resolve to neither (the "Other" bucket) are skipped.
function buildProviderKeyGroups(vars: Record<string, EnvVarInfo>): ProviderKeyGroup[] {
  const buckets = new Map<string, [string, EnvVarInfo][]>()

  for (const [key, info] of Object.entries(vars)) {
    if (info.category !== 'provider') {
      continue
    }

    // Prefer the backend-supplied provider label/id so the Keys tab groups by
    // the same identity the CLI picker uses; fall back to the prefix guess.
    const name = info.provider_label?.trim() || info.provider?.trim() || providerGroup(key)

    if (name === 'Other') {
      continue
    }

    buckets.set(name, [...(buckets.get(name) ?? []), [key, info]])
  }

  const groups: ProviderKeyGroup[] = []

  for (const [name, entries] of buckets) {
    const primary = entries.find(([k, i]) => !i.advanced && isKeyVar(k, i)) ?? entries.find(([k, i]) => isKeyVar(k, i))

    if (!primary) {
      continue
    }

    // Presentation overlay (priority, blurb, docs) is keyed by the prefix-based
    // group name; when the backend introduced this provider it may have no
    // overlay entry, so fall back to the backend/env metadata for display.
    const meta = providerMeta(name)

    groups.push({
      // Advanced = the provider's non-key knobs (base URL, region, deployment).
      // Skip redundant alias key vars (e.g. ANTHROPIC_TOKEN vs ANTHROPIC_API_KEY)
      // so we never render a second "Paste key" input — unless one is already
      // set, in which case keep it visible so it stays clearable.
      advanced: entries
        .filter(([k, i]) => k !== primary[0] && (!isKeyVar(k, i) || i.is_set))
        .sort(([a], [b]) => a.localeCompare(b)),
      description: meta?.description ?? primary[1].description,
      docsUrl: meta?.docsUrl ?? primary[1].url ?? undefined,
      hasAnySet: entries.some(([, i]) => i.is_set),
      name,
      primary,
      priority: providerPriority(name)
    })
  }

  return groups.sort((a, b) => a.priority - b.priority || a.name.localeCompare(b.name))
}

// Deliberately a near-1:1 replica of the first-run onboarding picker
// (`Picker` in desktop-onboarding-overlay): same recommended card, same
// Fireworks #2 quick-key row, same provider rows, same "Other providers"
// disclosure, same OpenRouter quick-key row, and the same bottom-right
// "I have an API key" affordance. The leaf cards are the exact shared
// components, so the two surfaces stay visually identical. Selecting a
// provider hands off to the shared onboarding overlay, which runs that
// provider's real sign-in flow; the key affordances open the API-key
// catalog below.
function OAuthPicker({
  disconnecting,
  onDisconnect,
  onTerminalDisconnect,
  onWantApiKey,
  providers
}: {
  disconnecting: null | string
  onDisconnect: (provider: OAuthProvider) => void
  onTerminalDisconnect: (provider: OAuthProvider) => void
  onWantApiKey: () => void
  providers: OAuthProvider[]
}) {
  const { t } = useI18n()
  const p = t.settings.providers
  const [showAll, setShowAll] = useState(false)
  const ordered = useMemo(() => sortProviders(providers), [providers])

  if (ordered.length === 0) {
    return null
  }

  const select = (p: OAuthProvider) => startManualProviderOAuth(p.id)

  const featured = ordered.find(p => p.id === FEATURED_ID && !p.status?.logged_in) ?? null
  const rest = featured ? ordered.filter(p => p.id !== FEATURED_ID) : ordered
  // Keep connected accounts grouped and always visible; only the unconnected
  // providers hide behind the disclosure, so the page leads with what's set up.
  // Both lists preserve `sortProviders` order (curated priority, then name).
  const connected = rest.filter(p => p.status?.logged_in)
  const others = rest.filter(p => !p.status?.logged_in)
  const collapsible = others.length > 0
  const showOthers = !collapsible || showAll

  return (
    <section className="mb-5 grid gap-2">
      <div className="flex flex-wrap items-baseline justify-between gap-x-3">
        <SettingsCategoryHeading icon={KeyRound} title={p.connectAccount} />
        <Button
          className="text-[length:var(--conversation-caption-font-size)]"
          onClick={onWantApiKey}
          size="inline"
          type="button"
          variant="textStrong"
        >
          {p.haveApiKey}
        </Button>
      </div>
      <p className="-mt-2 mb-1 text-[length:var(--conversation-caption-font-size)] leading-(--conversation-caption-line-height) text-(--ui-text-tertiary)">
        {p.intro}
      </p>
      {featured && <FeaturedProviderRow onSelect={select} provider={featured} />}
      {/* Slot #2 — always visible, matching onboarding / CANONICAL_PROVIDERS. */}
      <FireworksProviderRow onClick={onWantApiKey} />
      {connected.length > 0 && (
        <>
          <GroupLabel>{p.connected}</GroupLabel>
          {connected.map(p => (
            <ConnectedProviderRow
              disconnecting={disconnecting === p.id}
              key={p.id}
              onDisconnect={onDisconnect}
              onSelect={select}
              onTerminalDisconnect={onTerminalDisconnect}
              provider={p}
            />
          ))}
        </>
      )}
      {showOthers && (
        <>
          {connected.length > 0 && <GroupLabel>{p.otherProviders}</GroupLabel>}
          {others.map(p => (
            <ProviderRow key={p.id} onSelect={select} provider={p} />
          ))}
          <OpenRouterProviderRow onClick={onWantApiKey} />
        </>
      )}
      {collapsible && (
        <Button
          className="py-1 text-[length:var(--conversation-caption-font-size)]"
          onClick={() => setShowAll(v => !v)}
          size="inline"
          type="button"
          variant="text"
        >
          {showAll ? p.collapse : connected.length > 0 ? p.connectAnother : p.otherProviders}
          <ChevronDown className={cn('size-3.5 transition', showAll && 'rotate-180')} />
        </Button>
      )}
    </section>
  )
}

function ConnectedProviderRow({
  disconnecting,
  onDisconnect,
  onSelect,
  onTerminalDisconnect,
  provider
}: {
  disconnecting: boolean
  onDisconnect: (provider: OAuthProvider) => void
  onSelect: (provider: OAuthProvider) => void
  onTerminalDisconnect: (provider: OAuthProvider) => void
  provider: OAuthProvider
}) {
  const { t } = useI18n()
  const copy = t.settings.providers
  const title = providerTitle(provider)
  const Trail = provider.flow === 'external' ? Terminal : ChevronRight
  // Hermes can clear this provider's creds via the API.
  const canDisconnect = provider.disconnectable ?? provider.flow !== 'external'
  // External (CLI-managed) provider Hermes can't clear via the API, but ships a
  // command we can run in the embedded terminal (Electron shell only).
  const terminalDisconnect = !canDisconnect && Boolean(provider.disconnect_command) && canRunInTerminal()
  // Only fall back to a static "remove it elsewhere" hint when we offer no button.
  const showHint = !canDisconnect && !terminalDisconnect

  return (
    <div className="group grid grid-cols-[minmax(0,1fr)_auto] items-center gap-1 rounded-[6px] transition-colors hover:bg-(--ui-control-hover-background)">
      <RowButton className="min-w-0 px-3 py-2.5 text-left" onClick={() => onSelect(provider)}>
        <div className="flex min-w-0 items-center gap-2">
          <span className="truncate text-[length:var(--conversation-text-font-size)] font-semibold">{title}</span>
          <span className="inline-flex shrink-0 items-center gap-1 bg-primary/10 px-2 py-0.5 text-xs font-medium text-primary">
            <Check className="size-3" />
            {copy.connected}
          </span>
        </div>
        <p className="mt-1 text-xs leading-5 text-muted-foreground">{t.onboarding.flowSubtitles[provider.flow]}</p>
        {showHint && (
          <p className="mt-0.5 truncate text-[0.68rem] leading-5 text-muted-foreground/70">
            {provider.flow === 'external' ? copy.removeExternalGeneric(title) : copy.removeKeyManaged(title)}
          </p>
        )}
      </RowButton>
      <div className="flex items-center gap-1 pr-2">
        <Trail className="size-4 text-muted-foreground transition group-hover:text-foreground" />
        {canDisconnect && (
          <Button
            aria-label={`${t.common.remove} ${title}`}
            disabled={disconnecting}
            onClick={() => onDisconnect(provider)}
            size="icon-xs"
            title={`${t.common.remove} ${title}`}
            type="button"
            variant="ghost"
          >
            {disconnecting ? <Loader2 className="size-3 animate-spin" /> : <Trash2 className="size-3" />}
          </Button>
        )}
        {terminalDisconnect && (
          <Button
            aria-label={`${copy.disconnect} ${title}`}
            onClick={() => onTerminalDisconnect(provider)}
            size="icon-xs"
            title={copy.disconnectInTerminal}
            type="button"
            variant="ghost"
          >
            <Trash2 className="size-3" />
          </Button>
        )}
      </div>
    </div>
  )
}

function NoProviderKeys() {
  const { t } = useI18n()

  return (
    <div className="grid min-h-32 place-items-center px-4 py-8 text-center text-[length:var(--conversation-caption-font-size)] text-muted-foreground">
      {t.settings.providers.noProviderKeys}
    </div>
  )
}

function CustomEndpointKeyCard() {
  const { t } = useI18n()
  const ce = t.settings.providers.customEndpoint
  const [expanded, setExpanded] = useState(false)
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [savedBaseUrl, setSavedBaseUrl] = useState('')
  const [baseUrlDraft, setBaseUrlDraft] = useState('')
  const [apiKeyDraft, setApiKeyDraft] = useState<string | null>(null)
  const [apiKeySet, setApiKeySet] = useState(false)

  useEffect(() => {
    let cancelled = false
    void getCustomEndpoint()
      .then(config => {
        if (cancelled) return
        setSavedBaseUrl(config.base_url)
        setBaseUrlDraft(config.base_url)
        setApiKeySet(config.api_key_set)
        setApiKeyDraft(null)
      })
      .catch(err => notifyError(err, ce.failedLoad))
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [])

  const keyEditing = apiKeyDraft !== null
  const baseUrlDirty = baseUrlDraft.trim() !== savedBaseUrl
  const apiKeyDirty = Boolean(apiKeyDraft?.trim())
  const dirty = baseUrlDirty || apiKeyDirty
  const busy = saving

  function startKeyEdit() {
    if (!keyEditing) setApiKeyDraft('')
  }

  function cancelKeyEdit() {
    setApiKeyDraft(null)
  }

  async function save() {
    if (!baseUrlDraft.trim() || !dirty) return
    setSaving(true)
    try {
      const saved = await saveCustomEndpoint({
        ...(apiKeyDirty ? { api_key: apiKeyDraft!.trim() } : {}),
        base_url: baseUrlDraft
      })
      setSavedBaseUrl(saved.base_url)
      setBaseUrlDraft(saved.base_url)
      setApiKeyDraft(null)
      setApiKeySet(apiKeyDirty || saved.api_key_set)
      notify({ kind: 'success', message: ce.saved })
    } catch (err) {
      notifyError(err, ce.failedSave)
    } finally {
      setSaving(false)
    }
  }

  async function removeApiKey() {
    if (!window.confirm(ce.removeConfirm)) return
    if (!savedBaseUrl.trim()) return
    setSaving(true)
    try {
      await saveCustomEndpoint({ api_key: '', base_url: savedBaseUrl })
      setApiKeyDraft(null)
      setApiKeySet(false)
      notify({ kind: 'success', message: ce.apiKeyRemoved })
    } catch (err) {
      notifyError(err, ce.failedRemove)
    } finally {
      setSaving(false)
    }
  }

  return (
    <div
      className={cn(
        '@container group/card rounded-[6px] p-3 transition-colors',
        expanded ? 'bg-(--ui-bg-quaternary) ring-1 ring-(--ui-stroke-secondary)' : 'cursor-pointer row-hover'
      )}
      onClick={() => setExpanded(value => !value)}
      onKeyDown={e => {
        if (e.target !== e.currentTarget) return
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault()
          setExpanded(value => !value)
        }
      }}
      role="button"
      tabIndex={0}
    >
      <div className="grid grid-cols-1 items-start gap-x-3 gap-y-1.5 @2xl:grid-cols-[minmax(0,1fr)_minmax(15rem,22rem)] @2xl:gap-y-3">
        <div className="flex h-8 min-w-0 items-center gap-2">
          <span
            className={cn('size-2 shrink-0 rounded-full', savedBaseUrl ? 'bg-primary' : 'bg-(--ui-stroke-secondary)')}
          />
          <span className="min-w-0 truncate text-[length:var(--conversation-text-font-size)] font-medium text-foreground">
            {ce.title}
          </span>
          <ChevronDown
            className={cn(
              'size-3.5 shrink-0 text-muted-foreground transition',
              expanded ? 'rotate-180 opacity-100' : 'opacity-0 group-hover/card:opacity-100'
            )}
          />
        </div>

        <div className="min-w-0" onClick={e => e.stopPropagation()} onFocus={() => setExpanded(true)}>
          {apiKeySet && !keyEditing ? (
            <Input
              className={cn(
                CREDENTIAL_CONTROL_CLASS,
                !expanded && 'border-0! bg-transparent! shadow-none! h-auto! p-0! @2xl:h-8! @2xl:px-2.5! @2xl:py-1.5!',
                'cursor-pointer text-muted-foreground'
              )}
              onFocus={startKeyEdit}
              readOnly
              value="••••••••"
            />
          ) : (
            <div className="grid grid-cols-[minmax(0,1fr)_auto] items-center gap-2">
              <Input
                autoFocus={keyEditing}
                className={cn(
                  CREDENTIAL_CONTROL_CLASS,
                  !expanded && 'border-0! bg-transparent! shadow-none! h-auto! p-0! @2xl:h-8! @2xl:px-2.5! @2xl:py-1.5!'
                )}
                onChange={e => setApiKeyDraft(e.target.value)}
                onFocus={startKeyEdit}
                onKeyDown={e => {
                  if (e.key === 'Enter' && dirty) void save()
                  if (e.key === 'Escape' && keyEditing) {
                    e.preventDefault()
                    e.stopPropagation()
                    cancelKeyEdit()
                  }
                }}
                placeholder={ce.pasteApiKey}
                type="password"
                value={apiKeyDraft ?? ''}
              />
              {keyEditing && (apiKeySet || apiKeyDirty) && (
                <div className="flex items-center gap-1">
                  {apiKeySet && (
                    <Button
                      aria-label={ce.removeApiKey}
                      className="text-muted-foreground hover:text-destructive"
                      disabled={busy}
                      onClick={() => void removeApiKey()}
                      size="icon-xs"
                      title={ce.removeApiKey}
                      type="button"
                      variant="ghost"
                    >
                      <Trash2 />
                    </Button>
                  )}
                  {dirty && (
                    <Button
                      className="h-8"
                      disabled={busy || !baseUrlDraft.trim()}
                      onClick={() => void save()}
                      size="sm"
                    >
                      {busy ? <Loader2 className="animate-spin" /> : <Check />}
                      {t.common.save}
                    </Button>
                  )}
                </div>
              )}
            </div>
          )}
        </div>

        {expanded && (
          <div className="grid gap-3 @2xl:col-span-2" onClick={e => e.stopPropagation()}>
            <p className="text-[length:var(--conversation-caption-font-size)] leading-(--conversation-caption-line-height) text-(--ui-text-tertiary)">
              {ce.description}
            </p>
            <ListRow
              action={
                <Input
                  className={CREDENTIAL_CONTROL_CLASS}
                  onChange={e => setBaseUrlDraft(e.target.value)}
                  placeholder="https://…/v1"
                  value={loading ? '' : baseUrlDraft}
                />
              }
              title={ce.baseUrlLabel}
            />
            {baseUrlDirty && !keyEditing && (
              <div>
                <Button className="h-8" disabled={busy || !baseUrlDraft.trim()} onClick={() => void save()} size="sm">
                  {busy ? <Loader2 className="animate-spin" /> : <Check />}
                  {t.common.save}
                </Button>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

export function ProvidersSettings({
  onClose,
  onConfigSaved,
  onMainModelChanged,
  onViewChange,
  view
}: ProvidersSettingsProps) {
  const { t } = useI18n()
  const { rowProps, vars } = useEnvCredentials()
  const [oauthProviders, setOauthProviders] = useState<OAuthProvider[]>([])
  const [openProvider, setOpenProvider] = useState<null | string>(null)
  const [disconnecting, setDisconnecting] = useState<null | string>(null)
  // Free-text filter for the API-keys view (provider name / env-var key / desc).
  const [keyQuery, setKeyQuery] = useState('')
  // The onboarding overlay owns the OAuth flow. Watch its `manual` flag so we
  // re-read connection state when the user finishes (or dismisses) a sign-in
  // they launched from this page — otherwise the cards keep their stale status.
  const onboardingActive = useStore($desktopOnboarding).manual

  const refreshOAuthProviders = useCallback(async () => {
    // OAuth providers are best-effort — a failure here just hides the panel.
    const { providers } = await listOAuthProviders()
    setOauthProviders(providers)
  }, [])

  useEffect(() => {
    let cancelled = false

    void (async () => {
      if (onboardingActive) {
        return
      }

      try {
        const { providers } = await listOAuthProviders()

        if (!cancelled) {
          setOauthProviders(providers)
        }
      } catch {
        // Ignore — the OAuth panel just won't render.
      }
    })()

    return () => void (cancelled = true)
  }, [onboardingActive])

  // External (CLI-managed) providers can't be cleared via the API by design —
  // Hermes never deletes creds another tool owns behind a silent API call.
  // Instead we run the documented removal command in the embedded terminal so
  // the user sees exactly what executes, then return them to chat to watch it.
  function handleTerminalDisconnect(provider: OAuthProvider) {
    const command = provider.disconnect_command

    if (!command) {
      return
    }

    const name = providerTitle(provider)

    if (!window.confirm(t.settings.providers.removeTerminalConfirm(name, command))) {
      return
    }

    // Leave the settings overlay so the terminal pane (chat-only) is visible.
    onClose()
    runInTerminal(command)
    notify({
      kind: 'info',
      title: t.settings.providers.removedTitle,
      message: t.settings.providers.removeTerminalRunning(name)
    })
  }

  async function handleDisconnect(provider: OAuthProvider) {
    const name = providerTitle(provider)

    if (!window.confirm(t.settings.providers.removeConfirm(name))) {
      return
    }

    setDisconnecting(provider.id)

    try {
      await disconnectOAuthProvider(provider.id)
      notify({
        durationMs: 3_000,
        kind: 'success',
        title: t.settings.providers.removedTitle,
        message: t.settings.providers.removedMessage(name)
      })
      await refreshOAuthProviders().catch(() => undefined)
    } catch (err) {
      notifyError(err, t.settings.providers.failedRemove(name))
    } finally {
      setDisconnecting(null)
    }
  }

  if (!vars) {
    return <LoadingState label={t.settings.providers.loading} />
  }

  const hasOauth = oauthProviders.length > 0
  // The sidebar subnav owns the Accounts/API-keys split now; with no OAuth
  // providers there's nothing for the "Accounts" view to show, so fall to keys.
  const showApiKeys = view === 'keys' || !hasOauth

  const keyGroups = buildProviderKeyGroups(vars)

  if (showApiKeys) {
    const q = normalize(keyQuery)

    const visibleGroups = q
      ? keyGroups.filter(group => {
          const haystack = [group.name, group.description ?? '', group.primary[0], ...group.advanced.map(([k]) => k)]

          return haystack.some(s => s.toLowerCase().includes(q))
        })
      : keyGroups

    return (
      <SettingsContent>
        {keyGroups.length > 0 ? (
          <div className="grid gap-3">
            <SearchField
              aria-label={t.settings.providers.searchKeys}
              containerClassName="w-full"
              onChange={setKeyQuery}
              placeholder={t.settings.providers.searchKeys}
              value={keyQuery}
            />
            {visibleGroups.length > 0 ? (
              <div className="grid gap-2">
                {visibleGroups.map(group => (
                  <ProviderKeyRows
                    expanded={openProvider === group.name}
                    group={group}
                    key={group.name}
                    onExpand={() => setOpenProvider(group.name)}
                    onToggle={() => setOpenProvider(prev => (prev === group.name ? null : group.name))}
                    rowProps={rowProps}
                  />
                ))}
                <CustomEndpointKeyCard />
              </div>
            ) : (
              <div className="grid min-h-24 place-items-center px-4 py-6 text-center text-[length:var(--conversation-caption-font-size)] text-muted-foreground">
                {t.settings.providers.noKeysMatch}
              </div>
            )}
          </div>
        ) : (
          <div className="grid gap-2">
            <NoProviderKeys />
            <CustomEndpointKeyCard />
          </div>
        )}
      </SettingsContent>
    )
  }

  return (
    <SettingsContent>
      <OAuthPicker
        disconnecting={disconnecting}
        onDisconnect={provider => void handleDisconnect(provider)}
        onTerminalDisconnect={handleTerminalDisconnect}
        onWantApiKey={() => onViewChange('keys')}
        providers={oauthProviders}
      />
    </SettingsContent>
  )
}

interface ProviderKeyGroup {
  advanced: [string, EnvVarInfo][]
  description?: string
  docsUrl?: string
  hasAnySet: boolean
  name: string
  primary: [string, EnvVarInfo]
  priority: number
}

interface ProvidersSettingsProps {
  onClose: () => void
  onConfigSaved?: () => void
  onMainModelChanged?: (provider: string, model: string) => void
  onViewChange: (view: ProviderView) => void
  view: ProviderView
}
