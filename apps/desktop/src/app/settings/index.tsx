import { useStore } from '@nanostores/react'
import { useCallback, useEffect, useMemo, useRef } from 'react'
import { Navigate, useLocation, useNavigate } from 'react-router'

import { codiconIcon } from '@/components/ui/codicon'
import { KbdCombo } from '@/components/ui/kbd'
import { Tip } from '@/components/ui/tooltip'
import { getHermesConfigDefaults, getHermesConfigRecord, profileScopeKey, saveHermesConfig } from '@/hermes'
import { useI18n } from '@/i18n'
import { triggerHaptic } from '@/lib/haptics'
import {
  Archive,
  BarChart3,
  Bell,
  Cpu,
  Download,
  Globe,
  Info,
  Keyboard,
  KeyRound,
  RefreshCw,
  Search,
  Settings2,
  ShieldLock,
  Upload,
  Wrench,
  Zap
} from '@/lib/icons'
import { isEditableTarget } from '@/lib/keybinds/combo'
import { typeToFocusChar } from '@/lib/keybinds/composer-focus-keys'
import { cn } from '@/lib/utils'
import { $commandPaletteOpen, openCommandPalettePage } from '@/store/command-palette'
import { confirm } from '@/store/confirm'
import { bindingsFor } from '@/store/keybinds'
import { $localModelsEnabled } from '@/store/local-models-flag'
import { notifyError } from '@/store/notifications'
import { $settingsOwner } from '@/store/settings-scope'

import { invalidateHermesConfig } from '../hooks/use-config-record'
import { useRouteEnumParam } from '../hooks/use-route-enum-param'
import { OverlayIconButton } from '../overlays/overlay-chrome'
import { OverlayMain, OverlayNav, type OverlayNavGroup, OverlaySplitLayout } from '../overlays/overlay-split-layout'
import { OverlayView } from '../overlays/overlay-view'

import { AboutSettings } from './about-settings'
import { AppearanceSettings } from './appearance-settings'
import { BILLING_VIEWS, BillingSettings, type BillingSubView } from './billing'
import { deriveBillingView, useBillingState, useSubscriptionState } from './billing/use-billing-state'
import { ConfigSettings } from './config-settings'
import { SECTIONS } from './constants'
import { GatewaySettings } from './gateway-settings'
import { KeybindSettings } from './keybind-settings'
import { KEYS_VIEWS, KeysSettings, type KeysView } from './keys-settings'
import { movedSettingsTabRedirect } from './moved-tabs'
import { NotificationsSettings } from './notifications-settings'
import { SettingsBreadcrumbContext } from './primitives'
import { PROVIDER_VIEWS, ProvidersSettings, type ProviderView } from './providers-settings'
import { SessionsSettings } from './sessions-settings'
import { SettingsSubpageHeader } from './subpage-navigation'
import { resolveSettingsSubpage, settingsSubpageIcon, settingsSubpages } from './subpages'
import type { SettingsPageProps, SettingsView as SettingsViewId } from './types'
import { VaultSettings } from './vault-settings'

const SETTINGS_VIEWS: readonly SettingsViewId[] = [
  ...SECTIONS.map(s => `config:${s.id}` as SettingsViewId),
  'providers',
  'gateway',
  // Legacy alias: the Connections page merged into Gateways. Kept in the enum
  // so saved `?tab=connections` deep links still resolve (redirected below).
  'connections',
  'keybinds',
  'keys',
  'vault',
  'notifications',
  'billing',
  'sessions',
  'about'
]

export function notifySettingsOwnerError(
  owner: NonNullable<ReturnType<typeof $settingsOwner.get>>,
  error: unknown,
  fallback: string
) {
  if ($settingsOwner.get() === owner) {
    notifyError(error, fallback)
  }
}

export function SettingsView({ onClose, onConfigSaved, onMainModelChanged }: SettingsPageProps) {
  const settingsOwner = useStore($settingsOwner)

  const settingsOwnerKey = settingsOwner ? profileScopeKey(settingsOwner) : 'unavailable'

  const { t } = useI18n()
  const navigate = useNavigate()
  const { hash, pathname, search } = useLocation()

  // MCP and Plugins moved out of Settings into Capabilities. Keep old
  // `/settings?tab=mcp|plugins` deep links working — `useRouteEnumParam` would
  // silently coerce the unknown tab to the default view otherwise.
  useEffect(() => {
    const redirect = movedSettingsTabRedirect(search)

    if (redirect) {
      navigate(redirect, { replace: true })
    }
  }, [navigate, search])

  const [activeView] = useRouteEnumParam('tab', SETTINGS_VIEWS, 'config:model' as SettingsViewId)
  const params = new URLSearchParams(search)
  const requestedSubpage = params.get('page')
  const subpage = resolveSettingsSubpage(activeView, params)
  const needsSubpageRedirect = Boolean(subpage && subpage !== requestedSubpage)
  const subpageSearch = new URLSearchParams(search)

  if (subpage) {
    subpageSearch.set('page', subpage)
  }

  const openSettingsPage = useCallback(
    (view: SettingsViewId, page?: string) => {
      const next = new URLSearchParams(search)

      for (const key of [
        'page',
        'field',
        'setting',
        'key',
        'aux',
        'session',
        'kind',
        'label',
        'origin',
        'pview',
        'kview',
        'bview'
      ]) {
        next.delete(key)
      }

      next.set('tab', view)
      const destination = page ?? settingsSubpages(view)[0]?.id

      if (destination) {
        next.set('page', destination)
      }

      navigate({ hash, pathname, search: `?${next}` }, { replace: true })
    },
    [hash, navigate, pathname, search]
  )

  const setActiveView = useCallback((view: SettingsViewId) => openSettingsPage(view), [openSettingsPage])

  // Connections merged into the unified Gateways page: land old
  // `?tab=connections` routes/bookmarks there instead of a dead entry.
  useEffect(() => {
    if (activeView === 'connections') {
      setActiveView('gateway')
    }
  }, [activeView, setActiveView])
  // Providers subnav (Accounts vs API keys) lives in its own param so each
  // sub-view is deep-linkable and survives a refresh.
  const [providerView, setProviderView] = useRouteEnumParam<ProviderView>('pview', PROVIDER_VIEWS, 'accounts')
  const [keysView] = useRouteEnumParam<KeysView>('kview', KEYS_VIEWS, 'tools')
  const [billingView] = useRouteEnumParam<BillingSubView>('bview', BILLING_VIEWS, 'overview')
  const billingState = useBillingState()
  const subscriptionState = useSubscriptionState()
  const billingPresentation = deriveBillingView(billingState.data, subscriptionState.data)
  const canViewPlans = billingPresentation.status === 'normal' && Boolean(billingPresentation.plan?.action)

  // Jump to a section + its sub-view in one navigate. Two sequential setters
  // would each read the same stale `search` and the second would clobber the
  // first's `tab` — so the sub-view never opened on narrow screens.
  const openSubView = useCallback(
    (tab: SettingsViewId, param: string, value: string, fallback: string) => {
      const params = new URLSearchParams(search)

      for (const key of ['page', 'field', 'setting', 'key', 'aux', 'session', 'kind', 'label', 'origin']) {
        params.delete(key)
      }

      params.set('tab', tab)

      if (value === fallback) {
        params.delete(param)
      } else {
        params.set(param, value)
      }

      const qs = params.toString()
      navigate({ hash, pathname, search: qs ? `?${qs}` : '' }, { replace: true })
    },
    [hash, navigate, pathname, search]
  )

  const openProviderView = useCallback(
    (view: ProviderView) => openSubView('providers', 'pview', view, 'accounts'),
    [openSubView]
  )

  const openKeysView = useCallback((view: KeysView) => openSubView('keys', 'kview', view, 'tools'), [openSubView])

  const importInputRef = useRef<HTMLInputElement | null>(null)

  const exportConfig = async () => {
    const owner = $settingsOwner.get()

    if (!owner) {
      notifyError(new Error(t.settings.config.failedLoad), t.settings.config.failedLoad)

      return
    }

    try {
      const cfg = await getHermesConfigRecord(owner)

      if ($settingsOwner.get() !== owner) {
        return
      }

      const blob = new Blob([JSON.stringify(cfg, null, 2)], { type: 'application/json' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = 'hermes-config.json'
      a.click()
      URL.revokeObjectURL(url)
      triggerHaptic('success')
    } catch (err) {
      notifySettingsOwnerError(owner, err, t.settings.exportFailed)
    }
  }

  const resetConfig = async () => {
    const owner = $settingsOwner.get()

    if (!owner) {
      notifyError(new Error(t.settings.config.failedLoad), t.settings.config.failedLoad)

      return
    }

    const ok = await confirm({
      confirmLabel: t.settings.resetToDefaults,
      destructive: true,
      title: t.settings.resetConfirm
    })

    if (!ok || $settingsOwner.get() !== owner) {
      return
    }

    try {
      const defaults = await getHermesConfigDefaults(owner)

      if ($settingsOwner.get() !== owner) {
        return
      }

      await saveHermesConfig(defaults, owner)
      void invalidateHermesConfig(owner)

      if ($settingsOwner.get() === owner) {
        triggerHaptic('success')
        onConfigSaved?.()
      }
    } catch (err) {
      notifySettingsOwnerError(owner, err, t.settings.resetFailed)
    }
  }

  const navGroups: OverlayNavGroup[] = useMemo(
    () =>
      (
        [
          ...SECTIONS.flatMap(s => {
            const view = `config:${s.id}` as SettingsViewId

            const entry = {
              active: activeView === view,
              icon: s.icon,
              id: view,
              label: t.settings.sections[s.id] ?? s.label,
              onSelect: () => setActiveView(view)
            }

            // Credential Vault lives beside the Browser section: it feeds the
            // browser's model-blind vault fill, so the two are one mental unit.
            if (s.id === 'browser') {
              return [
                entry,
                {
                  active: activeView === 'vault',
                  icon: ShieldLock,
                  id: 'vault',
                  label: t.settings.nav.vault,
                  onSelect: () => setActiveView('vault')
                }
              ]
            }

            return [entry]
          }),
          {
            active: activeView === 'notifications',
            icon: Bell,
            id: 'notifications',
            label: t.settings.nav.notifications,
            onSelect: () => setActiveView('notifications')
          },
          {
            active: activeView === 'billing',
            children: [
              {
                active: activeView === 'billing' && (billingView === 'overview' || !canViewPlans),
                icon: BarChart3,
                id: 'bview:overview',
                label: t.settings.subpages.billingOverview,
                onSelect: () => openSubView('billing', 'bview', 'overview', 'overview')
              },
              ...(canViewPlans
                ? [
                    {
                      active: activeView === 'billing' && billingView === 'plans',
                      icon: BarChart3,
                      id: 'bview:plans',
                      label: t.settings.subpages.billingPlans,
                      onSelect: () => openSubView('billing', 'bview', 'plans', 'overview')
                    }
                  ]
                : [])
            ],
            icon: BarChart3,
            id: 'billing',
            label: t.settings.nav.billing,
            onSelect: () => setActiveView('billing')
          },
          {
            active: activeView === 'providers',
            children: [
              {
                active: activeView === 'providers' && providerView === 'accounts',
                icon: codiconIcon('account'),
                id: 'pview:accounts',
                label: t.settings.nav.providerAccounts,
                onSelect: () => openProviderView('accounts')
              },
              {
                active: activeView === 'providers' && providerView === 'keys',
                icon: KeyRound,
                id: 'pview:keys',
                label: t.settings.nav.providerApiKeys,
                onSelect: () => openProviderView('keys')
              },
              {
                active: activeView === 'providers' && providerView === 'custom-endpoints',
                icon: Globe,
                id: 'pview:custom-endpoints',
                label: t.settings.nav.providerCustomEndpoints,
                onSelect: () => openProviderView('custom-endpoints')
              },
              // Local models ships behind the --local launch flag: no flag, no
              // nav entry (the pane itself also refuses to render, so a stale
              // ?pview=local deep link falls back to accounts-shaped emptiness
              // rather than a hidden feature).
              ...($localModelsEnabled.get()
                ? [
                    {
                      active: activeView === 'providers' && providerView === 'local',
                      icon: Cpu,
                      id: 'pview:local',
                      label: t.settings.nav.providerLocalModels,
                      onSelect: () => openProviderView('local')
                    }
                  ]
                : [])
            ],
            gapBefore: true,
            icon: Zap,
            id: 'providers',
            label: t.settings.nav.providers,
            onSelect: () => setActiveView('providers')
          },
          {
            active: activeView === 'gateway',
            icon: Globe,
            id: 'gateway',
            label: t.settings.nav.gateway,
            onSelect: () => setActiveView('gateway')
          },
          {
            active: activeView === 'keybinds',
            icon: Keyboard,
            id: 'keybinds',
            label: t.settings.nav.keybinds,
            onSelect: () => setActiveView('keybinds')
          },
          {
            active: activeView === 'keys',
            children: [
              {
                active: activeView === 'keys' && keysView === 'tools',
                icon: Wrench,
                id: 'kview:tools',
                label: t.settings.nav.keysTools,
                onSelect: () => openKeysView('tools')
              },
              {
                active: activeView === 'keys' && keysView === 'settings',
                icon: Settings2,
                id: 'kview:settings',
                label: t.settings.nav.keysSettings,
                onSelect: () => openKeysView('settings')
              }
            ],
            icon: KeyRound,
            id: 'keys',
            label: t.settings.nav.apiKeys,
            onSelect: () => setActiveView('keys')
          },
          {
            active: activeView === 'sessions',
            icon: Archive,
            id: 'sessions',
            label: t.settings.nav.sessions,
            onSelect: () => setActiveView('sessions')
          },
          {
            active: activeView === 'about',
            gapBefore: true,
            icon: Info,
            id: 'about',
            label: t.settings.nav.about,
            onSelect: () => setActiveView('about')
          }
        ] as OverlayNavGroup[]
      ).map(group => {
        const view = group.id as SettingsViewId
        const children = settingsSubpages(view)

        return children.length
          ? {
              ...group,
              children: children.map(page => ({
                active: group.active && subpage === page.id,
                icon: settingsSubpageIcon(page, group.icon),
                id: `${view}:${page.id}`,
                label: t.settings.subpages[page.labelKey],
                onSelect: () => openSettingsPage(view, page.id)
              }))
            }
          : group
      }),
    [
      activeView,
      billingView,
      canViewPlans,
      keysView,
      providerView,
      subpage,
      t,
      setActiveView,
      openProviderView,
      openKeysView,
      openSettingsPage,
      openSubView
    ]
  )

  const activeGroup = navGroups.find(group => group.active)
  const activeChild = activeGroup?.children?.find(child => child.active)

  // Type-to-search: printable keystrokes on the Settings surface (outside any
  // field) open the settings-scoped palette, seeded with the character — same
  // reflex as the chat surface's type-to-focus, pointed at search instead.
  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if ($commandPaletteOpen.get() || isEditableTarget(event.target)) {
        return
      }

      const char = typeToFocusChar(event)

      if (char === null || char === ' ') {
        return
      }

      event.preventDefault()
      openCommandPalettePage('settings', char)
    }

    window.addEventListener('keydown', onKeyDown)

    return () => window.removeEventListener('keydown', onKeyDown)
  }, [])

  // Fake search pill riding the card's top edge, dead-center and half off it.
  // Clicking (or just typing) opens the ⌘K palette scoped to settings; while
  // the palette is up the pill hands over to it — grows slightly and fades,
  // then fades back when the palette closes. It sits outside the raised card,
  // so it needs its own opaque glass surface to mask the content underneath.
  const searchCombo = bindingsFor('nav.commandPalette')[0]
  const paletteOpen = useStore($commandPaletteOpen)

  const searchPill = (
    <button
      className={cn(
        'flex h-(--titlebar-control-height) items-center gap-1.5 rounded-full border border-(--ui-stroke-secondary) bg-(--ui-chat-surface-background) px-2.5 text-(--ui-text-tertiary) shadow-sm transition-all duration-200 ease-out hover:text-foreground motion-reduce:transition-none',
        paletteOpen && 'pointer-events-none scale-110 opacity-0'
      )}
      data-glass-opaque=""
      onClick={() => {
        triggerHaptic('open')
        openCommandPalettePage('settings')
      }}
      tabIndex={paletteOpen ? -1 : undefined}
      type="button"
    >
      <Search className="size-3" />
      <span className="text-xs">{t.settings.search.pill}</span>
      {searchCombo && <KbdCombo combo={searchCombo} size="sm" variant="ghost" />}
    </button>
  )

  const navFooter = (
    <>
      <Tip label={t.settings.exportConfig}>
        <OverlayIconButton aria-label={t.settings.exportConfig} onClick={() => void exportConfig()}>
          <Download />
        </OverlayIconButton>
      </Tip>
      <Tip label={t.settings.importConfig}>
        <OverlayIconButton
          aria-label={t.settings.importConfig}
          onClick={() => {
            triggerHaptic('open')
            importInputRef.current?.click()
          }}
        >
          <Upload />
        </OverlayIconButton>
      </Tip>
      <Tip label={t.settings.resetToDefaults}>
        <OverlayIconButton
          aria-label={t.settings.resetToDefaults}
          className="hover:text-destructive"
          onClick={() => {
            triggerHaptic('warning')
            void resetConfig()
          }}
        >
          <RefreshCw />
        </OverlayIconButton>
      </Tip>
    </>
  )

  const activeSettingsContent =
    activeView === 'config:appearance' ? (
      <AppearanceSettings subpage={subpage} />
    ) : activeView === 'about' ? (
      <AboutSettings subpage={subpage} />
    ) : activeView === 'gateway' || activeView === 'connections' ? (
      // 'connections' renders the unified page too so the frame before
      // the alias redirect lands doesn't flash the fallback view.
      <GatewaySettings subpage={subpage} />
    ) : activeView === 'keybinds' ? (
      <KeybindSettings subpage={subpage} />
    ) : activeView.startsWith('config:') ? (
      <ConfigSettings
        activeSectionId={activeView.slice('config:'.length)}
        importInputRef={importInputRef}
        onConfigSaved={onConfigSaved}
        onMainModelChanged={onMainModelChanged}
        subpage={subpage}
      />
    ) : activeView === 'providers' ? (
      <ProvidersSettings
        key={settingsOwnerKey}
        onClose={onClose}
        onConfigSaved={onConfigSaved}
        onMainModelChanged={onMainModelChanged}
        onViewChange={setProviderView}
        view={providerView}
      />
    ) : activeView === 'keys' ? (
      <KeysSettings key={settingsOwnerKey} view={keysView} />
    ) : activeView === 'notifications' ? (
      <NotificationsSettings subpage={subpage} />
    ) : activeView === 'billing' ? (
      <BillingSettings />
    ) : activeView === 'vault' ? (
      <VaultSettings key={settingsOwnerKey} settingsOwner={settingsOwner ?? undefined} subpage={subpage} />
    ) : (
      <SessionsSettings key={settingsOwnerKey} settingsOwner={settingsOwner} subpage={subpage} />
    )

  return (
    <OverlayView closeLabel={t.settings.closeSettings} edgeBadge={searchPill} onClose={onClose}>
      <OverlaySplitLayout>
        <OverlayNav footer={navFooter} groups={navGroups} />

        <OverlayMain className="px-0 pb-0">
          <SettingsBreadcrumbContext.Provider value>
            {activeGroup && <SettingsSubpageHeader child={activeChild} group={activeGroup} />}
            {needsSubpageRedirect ? (
              <Navigate replace to={{ hash, pathname, search: `?${subpageSearch}` }} />
            ) : (
              activeSettingsContent
            )}
          </SettingsBreadcrumbContext.Provider>
        </OverlayMain>
      </OverlaySplitLayout>
    </OverlayView>
  )
}

export { SettingsView as SettingsPage }
