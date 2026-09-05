import { useStore } from '@nanostores/react'
import { useCallback, useEffect, useMemo, useRef } from 'react'
import { useLocation, useNavigate } from 'react-router'

import { codiconIcon } from '@/components/ui/codicon'
import { KbdCombo } from '@/components/ui/kbd'
import { Tip } from '@/components/ui/tooltip'
import { getHermesConfigDefaults, getHermesConfigRecord, saveHermesConfig } from '@/hermes'
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
  Package,
  RefreshCw,
  Search,
  Settings2,
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

import { useRouteEnumParam } from '../hooks/use-route-enum-param'
import { OverlayIconButton } from '../overlays/overlay-chrome'
import { OverlayMain, OverlayNav, type OverlayNavGroup, OverlaySplitLayout } from '../overlays/overlay-split-layout'
import { OverlayView } from '../overlays/overlay-view'
import { SKILLS_ROUTE } from '../routes'

import { AboutSettings } from './about-settings'
import { AppearanceSettings } from './appearance-settings'
import { BillingSettings } from './billing'
import { ConfigSettings } from './config-settings'
import { SECTIONS } from './constants'
import { GatewaySettings } from './gateway-settings'
import { KeybindSettings } from './keybind-settings'
import { KEYS_VIEWS, KeysSettings, type KeysView } from './keys-settings'
import { NotificationsSettings } from './notifications-settings'
import { PluginsSettings } from './plugins-settings'
import { PROVIDER_VIEWS, ProvidersSettings, type ProviderView } from './providers-settings'
import { SessionsSettings } from './sessions-settings'
import type { SettingsPageProps, SettingsView as SettingsViewId } from './types'

const SETTINGS_VIEWS: readonly SettingsViewId[] = [
  ...SECTIONS.map(s => `config:${s.id}` as SettingsViewId),
  'providers',
  'gateway',
  // Legacy alias: the Connections page merged into Gateways. Kept in the enum
  // so saved `?tab=connections` deep links still resolve (redirected below).
  'connections',
  'keybinds',
  'keys',
  'notifications',
  'billing',
  'plugins',
  'sessions',
  'about'
]

export function SettingsView({ onClose, onConfigSaved, onMainModelChanged }: SettingsPageProps) {
  const { t } = useI18n()
  const navigate = useNavigate()
  const { hash, pathname, search } = useLocation()

  // MCP moved out of Settings into Capabilities (/skills?tab=mcp). Keep old
  // `/settings?tab=mcp` deep links working — `useRouteEnumParam` would silently
  // coerce the unknown tab to the default view otherwise. Preserve `server=` so
  // an old bookmark still lands on (and highlights) the selected server.
  useEffect(() => {
    const params = new URLSearchParams(search)

    if (params.get('tab') === 'mcp') {
      const server = params.get('server')
      const suffix = server ? `&server=${encodeURIComponent(server)}` : ''
      navigate(`${SKILLS_ROUTE}?tab=mcp${suffix}`, { replace: true })
    }
  }, [navigate, search])

  const [activeView, setActiveView] = useRouteEnumParam('tab', SETTINGS_VIEWS, 'config:model' as SettingsViewId)

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

  // Jump to a section + its sub-view in one navigate. Two sequential setters
  // would each read the same stale `search` and the second would clobber the
  // first's `tab` — so the sub-view never opened on narrow screens.
  const openSubView = useCallback(
    (tab: SettingsViewId, param: string, value: string, fallback: string) => {
      const params = new URLSearchParams(search)
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
    try {
      const cfg = await getHermesConfigRecord()
      const blob = new Blob([JSON.stringify(cfg, null, 2)], { type: 'application/json' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = 'hermes-config.json'
      a.click()
      URL.revokeObjectURL(url)
      triggerHaptic('success')
    } catch (err) {
      notifyError(err, t.settings.exportFailed)
    }
  }

  const resetConfig = async () => {
    const ok = await confirm({
      confirmLabel: t.settings.resetToDefaults,
      destructive: true,
      title: t.settings.resetConfirm
    })

    if (!ok) {
      return
    }

    try {
      await saveHermesConfig(await getHermesConfigDefaults())
      triggerHaptic('success')
      onConfigSaved?.()
    } catch (err) {
      notifyError(err, t.settings.resetFailed)
    }
  }

  const navGroups: OverlayNavGroup[] = useMemo(
    () => [
      ...SECTIONS.map(s => {
        const view = `config:${s.id}` as SettingsViewId

        return {
          active: activeView === view,
          icon: s.icon,
          id: view,
          label: t.settings.sections[s.id] ?? s.label,
          onSelect: () => setActiveView(view)
        }
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
        active: activeView === 'plugins',
        icon: Package,
        id: 'plugins',
        label: t.settings.nav.plugins,
        onSelect: () => setActiveView('plugins')
      },
      {
        active: activeView === 'sessions',
        icon: Archive,
        id: 'sessions',
        label: t.settings.nav.archivedChats,
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
    ],
    [activeView, keysView, providerView, t, setActiveView, openProviderView, openKeysView]
  )

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
  // then fades back when the palette closes. It renders as chrome, not an
  // input — no border, recessed fill, live ⌘K hint.
  const searchCombo = bindingsFor('nav.commandPalette')[0]
  const paletteOpen = useStore($commandPaletteOpen)

  const searchPill = (
    <button
      className={cn(
        'flex h-(--titlebar-control-height) items-center gap-1.5 rounded-full border border-(--ui-stroke-secondary) bg-(--ui-chat-surface-background) px-2.5 text-(--ui-text-tertiary) shadow-sm transition-all duration-200 ease-out hover:text-foreground motion-reduce:transition-none',
        paletteOpen && 'pointer-events-none scale-110 opacity-0'
      )}
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
        <OverlayIconButton onClick={() => void exportConfig()}>
          <Download />
        </OverlayIconButton>
      </Tip>
      <Tip label={t.settings.importConfig}>
        <OverlayIconButton
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
      <AppearanceSettings />
    ) : activeView === 'about' ? (
      <AboutSettings />
    ) : activeView === 'gateway' || activeView === 'connections' ? (
      // 'connections' renders the unified page too so the frame before
      // the alias redirect lands doesn't flash the fallback view.
      <GatewaySettings />
    ) : activeView === 'keybinds' ? (
      <KeybindSettings />
    ) : activeView.startsWith('config:') ? (
      <ConfigSettings
        activeSectionId={activeView.slice('config:'.length)}
        importInputRef={importInputRef}
        onConfigSaved={onConfigSaved}
        onMainModelChanged={onMainModelChanged}
      />
    ) : activeView === 'providers' ? (
      <ProvidersSettings
        onClose={onClose}
        onConfigSaved={onConfigSaved}
        onMainModelChanged={onMainModelChanged}
        onViewChange={setProviderView}
        view={providerView}
      />
    ) : activeView === 'keys' ? (
      <KeysSettings view={keysView} />
    ) : activeView === 'notifications' ? (
      <NotificationsSettings />
    ) : activeView === 'billing' ? (
      <BillingSettings />
    ) : activeView === 'plugins' ? (
      <PluginsSettings />
    ) : (
      <SessionsSettings />
    )

  return (
    <OverlayView closeLabel={t.settings.closeSettings} edgeBadge={searchPill} onClose={onClose}>
      <OverlaySplitLayout>
        <OverlayNav footer={navFooter} groups={navGroups} />

        <OverlayMain className="px-0 pb-0">{activeSettingsContent}</OverlayMain>
      </OverlaySplitLayout>
    </OverlayView>
  )
}

export { SettingsView as SettingsPage }
