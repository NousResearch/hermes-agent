import { useStore } from '@nanostores/react'
import type { ReactNode } from 'react'
import { useCallback, useEffect, useMemo, useState } from 'react'

import { runInTerminal } from '@/app/right-sidebar/store'
import {
  FEATURED_ID,
  FeaturedProviderRow,
  KeyProviderRow,
  ProviderRow,
  providerTitle,
  sortProviders
} from '@/components/desktop-onboarding-overlay'
import { Button } from '@/components/ui/button'
import { disconnectOAuthProvider, listOAuthProviders } from '@/hermes'
import { useI18n } from '@/i18n'
import { Check, ChevronDown, ChevronRight, KeyRound, Loader2, Terminal, Trash2 } from '@/lib/icons'
import { cn } from '@/lib/utils'
import { notify, notifyError } from '@/store/notifications'
import { $desktopOnboarding, startManualProviderOAuth } from '@/store/onboarding'
import type { EnvVarInfo, OAuthProvider } from '@/types/hermes'

import { isKeyVar, ProviderKeyRows } from './credential-key-ui'
import { SettingsCategoryHeading, useEnvCredentials } from './env-credentials'
import { providerGroup, providerMeta, providerPriority } from './helpers'
import { LoadingState, SettingsContent } from './primitives'

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

const PROVIDER_ALIAS_KEY_VARS = new Set([
  'ANTHROPIC_TOKEN',
  'CLAUDE_CODE_OAUTH_TOKEN',
  'GH_TOKEN',
  'GITHUB_TOKEN'
])

const PROVIDER_ALIAS_GROUP_KEYS = new Set([
  'GITHUB_TOKEN'
])

const shouldIncludeProviderKey = (key: string, info: EnvVarInfo) =>
  info.category === 'provider' || PROVIDER_ALIAS_GROUP_KEYS.has(key)

const shouldShowAdvancedProviderKey = (key: string, info: EnvVarInfo) =>
  !isKeyVar(key, info) || info.is_set || PROVIDER_ALIAS_KEY_VARS.has(key)

const providerKeyGroupName = (key: string, info: EnvVarInfo) =>
  info.provider_group_name?.trim() || info.provider_name?.trim() || providerGroup(key)

// Group the env catalog by provider — one ListRow per vendor plus optional
// advanced overrides (base URL, region, etc.). Groups without a key field and
// the "Other" bucket are skipped.
function buildProviderKeyGroups(vars: Record<string, EnvVarInfo>): ProviderKeyGroup[] {
  const buckets = new Map<string, [string, EnvVarInfo][]>()

  for (const [key, info] of Object.entries(vars)) {
    if (!shouldIncludeProviderKey(key, info)) {
      continue
    }

    const name = providerKeyGroupName(key, info)

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

    const meta = providerMeta(name)
    const providerDescription =
      primary[1].provider_group_description?.trim() || primary[1].provider_description?.trim()
    const providerDocsUrl = primary[1].provider_url?.trim()

    groups.push({
      // Advanced = the provider's non-key knobs (base URL, region, deployment).
      // Skip redundant alias key vars (e.g. ANTHROPIC_TOKEN vs ANTHROPIC_API_KEY)
      // so we never render a second "Paste key" input — unless one is already
      // set or is an explicit provider alias users may need to configure.
      advanced: entries
        .filter(([k, i]) => k !== primary[0] && shouldShowAdvancedProviderKey(k, i))
        .sort(([a], [b]) => a.localeCompare(b)),
      description: providerDescription || meta?.description || primary[1].description,
      docsUrl: providerDocsUrl || meta?.docsUrl || primary[1].url || undefined,
      hasAnySet: entries.some(([, i]) => i.is_set),
      name,
      primary,
      priority: providerPriority(name)
    })
  }

  return groups.sort((a, b) => a.name.localeCompare(b.name))
}

// Deliberately a near-1:1 replica of the first-run onboarding picker
// (`Picker` in desktop-onboarding-overlay): same recommended card, same
// provider rows, same "Other providers" disclosure, same OpenRouter quick-key
// row, and the same bottom-right "I have an API key" affordance. The leaf cards
// are the exact shared components, so the two surfaces stay visually identical.
// Selecting a provider hands off to the shared onboarding overlay, which runs
// that provider's real sign-in flow; the key affordances open the API-key
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
          <KeyProviderRow onClick={onWantApiKey} />
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
      <button className="min-w-0 px-3 py-2.5 text-left" onClick={() => onSelect(provider)} type="button">
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
      </button>
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

export function ProvidersSettings({ onClose, onViewChange, view }: ProvidersSettingsProps) {
  const { t } = useI18n()
  const { rowProps, vars } = useEnvCredentials()
  const [oauthProviders, setOauthProviders] = useState<OAuthProvider[]>([])
  const [openProvider, setOpenProvider] = useState<null | string>(null)
  const [disconnecting, setDisconnecting] = useState<null | string>(null)
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
    notify({ kind: 'info', title: t.settings.providers.removedTitle, message: t.settings.providers.removeTerminalRunning(name) })
  }

  async function handleDisconnect(provider: OAuthProvider) {
    const name = providerTitle(provider)

    if (!window.confirm(t.settings.providers.removeConfirm(name))) {
      return
    }

    setDisconnecting(provider.id)

    try {
      await disconnectOAuthProvider(provider.id)
      notify({ durationMs: 3_000, kind: 'success', title: t.settings.providers.removedTitle, message: t.settings.providers.removedMessage(name) })
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
    return (
      <SettingsContent>
        {keyGroups.length > 0 ? (
          <div className="grid gap-2">
            {keyGroups.map(group => (
              <ProviderKeyRows
                expanded={openProvider === group.name}
                group={group}
                key={group.name}
                onExpand={() => setOpenProvider(group.name)}
                onToggle={() => setOpenProvider(prev => (prev === group.name ? null : group.name))}
                rowProps={rowProps}
              />
            ))}
          </div>
        ) : (
          <NoProviderKeys />
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
  onViewChange: (view: ProviderView) => void
  view: ProviderView
}
