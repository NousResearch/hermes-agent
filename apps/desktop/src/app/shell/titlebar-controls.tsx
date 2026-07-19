import { useStore } from '@nanostores/react'
import {
  type ComponentProps,
  type MouseEvent,
  type ReactNode,
  useCallback,
  useEffect,
  useState
} from 'react'
import { useLocation, useNavigate } from 'react-router-dom'

import { toggleLayoutEditMode } from '@/components/pane-shell/edit-mode'
import { resetLayoutTree } from '@/components/pane-shell/tree/store'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger
} from '@/components/ui/dropdown-menu'
import { Tip, TipKeybindLabel } from '@/components/ui/tooltip'
import type { DesktopConnectionConfig } from '@/global'
import { useI18n } from '@/i18n'
import { triggerHaptic } from '@/lib/haptics'
import { cn } from '@/lib/utils'
import { $gatewaySwitching } from '@/store/gateway-switch'
import { $hapticsMuted, toggleHapticsMuted } from '@/store/haptics'
import {
  $fileBrowserOpen,
  $sidebarOpen,
  toggleFileBrowserOpen,
  togglePanesFlipped,
  toggleSidebarOpen
} from '@/store/layout'
import { notify, notifyError } from '@/store/notifications'
import { $connection } from '@/store/session'

import { appViewForPath, isOverlayView, SETTINGS_ROUTE } from '../routes'

import { titlebarButtonClass } from './titlebar'

export interface TitlebarTool {
  id: string
  label: string
  active?: boolean
  className?: string
  disabled?: boolean
  hidden?: boolean
  href?: string
  icon: ReactNode
  onSelect?: (event?: MouseEvent) => void
  /** Keybind action id — when set, the tooltip shows the label + keybind hint. */
  actionId?: string
  title?: string
  to?: string
}

export type TitlebarToolSide = 'left' | 'right'
export type SetTitlebarToolGroup = (id: string, tools: readonly TitlebarTool[], side?: TitlebarToolSide) => void

type GatewayMode = 'local' | 'remote' | 'cloud'

interface GatewayTarget {
  mode: 'remote' | 'cloud'
  remoteUrl: string
  remoteAuthMode: 'oauth' | 'token'
  cloudOrg?: string
}

interface TitlebarControlsProps extends ComponentProps<'div'> {
  leftTools?: readonly TitlebarTool[]
  tools?: readonly TitlebarTool[]
  onOpenSettings: () => void
}

const LAST_REMOTE_KEY = 'hermes.desktop.gateway-last-remote'
const LAST_CLOUD_KEY = 'hermes.desktop.gateway-last-cloud'

/**
 * The layout button's glyph. Morphs into its composite reset form — the
 * layout icon wearing a small counter-clockwise arrow badge ("layout, back
 * to how it was") — ONLY while the pointer is on the button AND ⌘/Ctrl is
 * held: hover gates via CSS (`group/tool` on the button), the modifier via
 * the window listener. Pressing the modifier elsewhere changes nothing.
 */
function LayoutGlyph({ modHeld }: { modHeld: boolean }) {
  return (
    <>
      <span className={cn('inline-flex', modHeld && 'group-hover/tool:hidden')}>
        <Codicon name="layout" />
      </span>
      <span className={cn('relative hidden', modHeld && 'group-hover/tool:inline-flex')}>
        <Codicon name="layout" />
        <span className="absolute -bottom-1 -right-1.5 grid place-items-center rounded-full bg-(--ui-bg-chrome) p-px">
          <Codicon className="-scale-x-100" name="refresh" size="0.5625rem" />
        </span>
      </span>
    </>
  )
}

/** Live ⌘/Ctrl tracking — mod-click affordances telegraph themselves (the
 *  layout button morphs into its reset form while the modifier is down). */
function useModifierHeld(): boolean {
  const [held, setHeld] = useState(false)

  useEffect(() => {
    const sync = (event: KeyboardEvent) => setHeld(event.metaKey || event.ctrlKey)
    const clear = () => setHeld(false)

    window.addEventListener('keydown', sync)
    window.addEventListener('keyup', sync)
    window.addEventListener('blur', clear)

    return () => {
      window.removeEventListener('keydown', sync)
      window.removeEventListener('keyup', sync)
      window.removeEventListener('blur', clear)
    }
  }, [])

  return held
}

function modeIcon(mode: GatewayMode, busy = false): string {
  if (busy) {
    return 'loading'
  }

  if (mode === 'cloud') {
    return 'cloud'
  }

  if (mode === 'remote') {
    return 'globe'
  }

  return 'home'
}

function canUseRemoteLike(config: DesktopConnectionConfig | null | undefined): boolean {
  if (!config?.remoteUrl) {
    return false
  }

  return config.remoteAuthMode === 'oauth' ? config.remoteOauthConnected : config.remoteTokenSet
}

function readLastTarget(key: string): GatewayTarget | null {
  if (typeof window === 'undefined') {
    return null
  }

  try {
    const raw = window.localStorage.getItem(key)

    if (!raw) {
      return null
    }

    const parsed = JSON.parse(raw) as Partial<GatewayTarget>
    const remoteUrl = String(parsed.remoteUrl || '').trim()
    const mode = parsed.mode === 'cloud' ? 'cloud' : parsed.mode === 'remote' ? 'remote' : null
    const remoteAuthMode = parsed.remoteAuthMode === 'oauth' ? 'oauth' : 'token'

    if (!mode || !remoteUrl) {
      return null
    }

    return {
      mode,
      remoteUrl,
      remoteAuthMode,
      cloudOrg: mode === 'cloud' ? String(parsed.cloudOrg || '').trim() || undefined : undefined
    }
  } catch {
    return null
  }
}

function writeLastTarget(key: string, target: GatewayTarget): void {
  if (typeof window === 'undefined') {
    return
  }

  try {
    window.localStorage.setItem(key, JSON.stringify(target))
  } catch {
    // Quota / private mode — menu still works from the live saved config.
  }
}

function targetFromConfig(config: DesktopConnectionConfig): GatewayTarget | null {
  const remoteUrl = String(config.remoteUrl || '').trim()

  if (!remoteUrl) {
    return null
  }

  if (config.mode === 'cloud') {
    return {
      mode: 'cloud',
      remoteUrl,
      remoteAuthMode: config.remoteAuthMode === 'oauth' ? 'oauth' : 'token',
      cloudOrg: String(config.cloudOrg || '').trim() || undefined
    }
  }

  // A remote-shaped block (or residual url after local hop) is a remote target.
  return {
    mode: 'remote',
    remoteUrl,
    remoteAuthMode: config.remoteAuthMode === 'oauth' ? 'oauth' : 'token'
  }
}

function rememberTargetFromConfig(config: DesktopConnectionConfig | null | undefined): void {
  if (!config) {
    return
  }

  const target = targetFromConfig(config)

  if (!target) {
    return
  }

  // Only cache targets that still authenticate — a dead OAuth session should
  // not keep advertising a one-click hop that will just bounce to Settings.
  if (!canUseRemoteLike(config) && config.mode !== 'local') {
    return
  }

  if (target.mode === 'cloud' || config.mode === 'cloud') {
    writeLastTarget(LAST_CLOUD_KEY, { ...target, mode: 'cloud' })
  }

  if (target.mode === 'remote' || config.mode === 'remote') {
    writeLastTarget(LAST_REMOTE_KEY, { ...target, mode: 'remote' })
  }
}

export function TitlebarControls({ leftTools = [], tools = [], onOpenSettings }: TitlebarControlsProps) {
  const { t } = useI18n()
  const navigate = useNavigate()
  const location = useLocation()
  const modHeld = useModifierHeld()
  const hapticsMuted = useStore($hapticsMuted)
  const fileBrowserOpen = useStore($fileBrowserOpen)
  const sidebarOpen = useStore($sidebarOpen)
  const connection = useStore($connection)
  const gatewaySwitching = useStore($gatewaySwitching)
  const [gatewayBusy, setGatewayBusy] = useState(false)
  const [menuOpen, setMenuOpen] = useState(false)
  const [savedConfig, setSavedConfig] = useState<DesktopConnectionConfig | null>(null)
  const [lastRemote, setLastRemote] = useState<GatewayTarget | null>(() => readLastTarget(LAST_REMOTE_KEY))
  const [lastCloud, setLastCloud] = useState<GatewayTarget | null>(() => readLastTarget(LAST_CLOUD_KEY))

  const busy = gatewayBusy || gatewaySwitching
  // Saved provenance (cloud/remote/local) for the checkmarks; live connection is
  // only ever local|remote so cloud needs the saved config to show correctly.
  const selectedMode: GatewayMode =
    savedConfig?.mode === 'cloud' || savedConfig?.mode === 'remote' || savedConfig?.mode === 'local'
      ? savedConfig.mode
      : connection?.mode === 'remote'
        ? 'remote'
        : 'local'

  const refreshSavedConfig = useCallback(async () => {
    const desktop = window.hermesDesktop

    if (!desktop?.getConnectionConfig) {
      return
    }

    try {
      const next = await desktop.getConnectionConfig()
      setSavedConfig(next)
      rememberTargetFromConfig(next)
      setLastRemote(readLastTarget(LAST_REMOTE_KEY))
      setLastCloud(readLastTarget(LAST_CLOUD_KEY))
    } catch {
      // Keep last known config; a transient IPC miss shouldn't blank the menu.
    }
  }, [])

  useEffect(() => {
    void refreshSavedConfig()
  }, [connection?.mode, connection?.baseUrl, refreshSavedConfig])

  useEffect(() => {
    if (menuOpen) {
      void refreshSavedConfig()
    }
  }, [menuOpen, refreshSavedConfig])

  const toggleHaptics = () => {
    if (!hapticsMuted) {
      triggerHaptic('tap')
    }

    toggleHapticsMuted()

    if (hapticsMuted) {
      window.requestAnimationFrame(() => triggerHaptic('success'))
    }
  }

  // Soft-switch the saved connection mode. Remote and cloud targets are cached
  // separately so Cloud → Local → Cloud still works even though the persisted
  // connection config intentionally drops cloud provenance on a local hop.
  const selectGatewayMode = useCallback(
    async (mode: GatewayMode) => {
      const desktop = window.hermesDesktop
      const copy = t.titlebar

      if (!desktop?.applyConnectionConfig || !desktop.getConnectionConfig || busy) {
        return
      }

      if (selectedMode === mode && !busy) {
        // Already on this mode — still open settings for remote/cloud so the
        // user can re-auth / re-pick without hunting the nav.
        if (mode !== 'local') {
          navigate(`${SETTINGS_ROUTE}?tab=gateway`)
        }

        return
      }

      setGatewayBusy(true)
      triggerHaptic('tap')

      try {
        if (mode === 'local') {
          // Snapshot the live target BEFORE local apply. Cloud→local clears
          // cloud provenance in coerceDesktopConnectionConfig; without this
          // cache the menu loses Cloud entirely.
          const live = await desktop.getConnectionConfig().catch(() => null)
          rememberTargetFromConfig(live)

          const next = await desktop.applyConnectionConfig({ mode: 'local' })
          setSavedConfig(next)
          setLastRemote(readLastTarget(LAST_REMOTE_KEY))
          setLastCloud(readLastTarget(LAST_CLOUD_KEY))
          notify({ kind: 'success', message: copy.gatewaySwitchLocalMessage, title: copy.gatewayModeLocal })

          return
        }

        const config = await desktop.getConnectionConfig()
        const cached = mode === 'cloud' ? readLastTarget(LAST_CLOUD_KEY) : readLastTarget(LAST_REMOTE_KEY)
        const liveMatches =
          config.mode === mode && canUseRemoteLike(config)
            ? targetFromConfig(config)
            : config.mode !== 'local' && canUseRemoteLike(config) && mode === 'remote' && config.mode === 'remote'
              ? targetFromConfig(config)
              : null

        // Prefer a still-authenticated live match; else the last successful
        // target of that mode. Cached OAuth targets still need a live cookie.
        let target = liveMatches || cached

        if (target && target.mode !== mode) {
          target = { ...target, mode }
        }

        if (!target?.remoteUrl) {
          navigate(`${SETTINGS_ROUTE}?tab=gateway`)
          notify({
            kind: 'info',
            message: mode === 'cloud' ? copy.gatewaySwitchNeedsCloudMessage : copy.gatewaySwitchNeedsSetupMessage,
            title: mode === 'cloud' ? copy.gatewaySwitchNeedsCloudTitle : copy.gatewaySwitchNeedsSetupTitle
          })

          return
        }

        // For OAuth, the live cookie jar must still be signed in for this URL.
        // If the saved config points at the same URL and reports disconnected,
        // bounce to Settings instead of a doomed apply.
        if (
          target.remoteAuthMode === 'oauth' &&
          config.remoteUrl === target.remoteUrl &&
          !config.remoteOauthConnected
        ) {
          navigate(`${SETTINGS_ROUTE}?tab=gateway`)
          notify({
            kind: 'info',
            message: mode === 'cloud' ? copy.gatewaySwitchNeedsCloudMessage : copy.gatewaySwitchNeedsSetupMessage,
            title: mode === 'cloud' ? copy.gatewaySwitchNeedsCloudTitle : copy.gatewaySwitchNeedsSetupTitle
          })

          return
        }

        const next = await desktop.applyConnectionConfig({
          mode,
          remoteAuthMode: target.remoteAuthMode,
          remoteUrl: target.remoteUrl,
          cloudOrg: mode === 'cloud' ? target.cloudOrg : undefined
        })
        setSavedConfig(next)
        rememberTargetFromConfig(next)
        setLastRemote(readLastTarget(LAST_REMOTE_KEY))
        setLastCloud(readLastTarget(LAST_CLOUD_KEY))
        notify({
          kind: 'success',
          message: mode === 'cloud' ? copy.gatewaySwitchCloudMessage : copy.gatewaySwitchRemoteMessage,
          title: mode === 'cloud' ? copy.gatewayModeCloud : copy.gatewayModeRemote
        })
      } catch (err) {
        notifyError(err, t.titlebar.gatewaySwitchFailed)
      } finally {
        setGatewayBusy(false)
      }
    },
    [busy, navigate, selectedMode, t.titlebar]
  )

  // POSITIONAL toggles: each button shows/hides everything on its physical
  // side of the main zone (the layout tree collapses the whole side), so they
  // stay correct through flips and rearranges. $sidebarOpen ≙ left side,
  // $fileBrowserOpen ≙ right side. Never an active highlight — plain
  // show/hide affordances.
  const leftEdge = { open: sidebarOpen, toggle: toggleSidebarOpen }
  const rightEdge = { open: fileBrowserOpen, toggle: toggleFileBrowserOpen }

  const leftToolbarTools: TitlebarTool[] = [
    {
      actionId: 'view.toggleSidebar',
      icon: <Codicon name="layout-sidebar-left" />,
      id: 'sidebar',
      label: leftEdge.open ? t.titlebar.hideSidebar : t.titlebar.showSidebar,
      onSelect: () => {
        triggerHaptic('tap')
        leftEdge.toggle()
      }
    },
    {
      actionId: 'view.flipPanes',
      icon: <Codicon name="arrow-swap" />,
      id: 'flip-panes',
      label: t.titlebar.swapSidebarSides,
      onSelect: () => {
        triggerHaptic('tap')
        togglePanesFlipped()
      },
      title: t.titlebar.swapSidebarSidesTitle
    },
    ...leftTools
  ]

  const rightSidebarTool: TitlebarTool = {
    actionId: 'view.toggleRightSidebar',
    icon: <Codicon name="layout-sidebar-right" />,
    id: 'right-sidebar',
    label: rightEdge.open ? t.titlebar.hideRightSidebar : t.titlebar.showRightSidebar,
    onSelect: () => {
      triggerHaptic('tap')
      rightEdge.toggle()
    }
  }

  // Static system tools — always pinned to the screen's right edge. Gateway
  // mode is rendered separately as a dropdown so the click opens Local /
  // Remote / Cloud instead of toggling.
  const systemTools: TitlebarTool[] = [
    {
      className: 'group/tool',
      // Hover + held ⌘/Ctrl morphs the glyph into its reset form (see
      // LayoutGlyph) — the mod-click telegraphs itself before it happens.
      icon: <LayoutGlyph modHeld={modHeld} />,
      id: 'layout',
      label: t.titlebar.layoutEditor,
      onSelect: event => {
        if (event?.metaKey || event?.ctrlKey) {
          triggerHaptic('warning')
          resetLayoutTree()

          return
        }

        triggerHaptic('open')
        toggleLayoutEditMode()
      },
      title: t.titlebar.layoutEditorTitle
    },
    {
      active: hapticsMuted,
      icon: <Codicon name={hapticsMuted ? 'mute' : 'unmute'} />,
      id: 'haptics',
      label: hapticsMuted ? t.titlebar.unmuteHaptics : t.titlebar.muteHaptics,
      onSelect: toggleHaptics
    },
    {
      actionId: 'keybinds.openPanel',
      icon: <Codicon name="keyboard" />,
      id: 'keybinds',
      label: t.titlebar.openKeybinds,
      onSelect: () => {
        triggerHaptic('open')
        navigate(`${SETTINGS_ROUTE}?tab=keybinds`)
      }
    },
    {
      actionId: 'nav.settings',
      icon: <Codicon name="settings-gear" />,
      id: 'settings',
      label: t.titlebar.openSettings,
      onSelect: () => {
        triggerHaptic('open')
        onOpenSettings()
      }
    }
  ]

  // While a full-screen overlay (settings, command center, …) is open it should
  // visually own the window. These control clusters are `fixed` at a higher
  // z-index than the overlay card, so they'd otherwise bleed over it — hide them
  // and let the overlay's own chrome (close button, drag region) take over.
  if (isOverlayView(appViewForPath(location.pathname))) {
    return null
  }

  const visibleSystemTools = systemTools.filter(tool => !tool.hidden)
  const settingsTool = visibleSystemTools.find(tool => tool.id === 'settings')
  const visibleSystemToolsBeforeSettings = visibleSystemTools.filter(tool => tool.id !== 'settings')
  const visiblePaneTools = tools.filter(tool => !tool.hidden)

  const gatewayTriggerClass = cn(
    titlebarButtonClass,
    'bg-transparent select-none',
    selectedMode !== 'local' && 'text-foreground'
  )

  // Only list gateways the user can actually switch to. Unconfigured remote /
  // cloud stay out of the menu — use "Gateway settings" to add them. Last
  // successful targets survive Cloud/Remote → Local so the round-trip stays
  // one click. The currently-selected mode always stays visible.
  const remoteConfigured =
    Boolean(lastRemote?.remoteUrl) ||
    (canUseRemoteLike(savedConfig) && savedConfig?.mode === 'remote') ||
    (canUseRemoteLike(savedConfig) && savedConfig?.mode !== 'cloud' && Boolean(savedConfig?.remoteUrl))
  const cloudConfigured =
    Boolean(lastCloud?.remoteUrl) || (canUseRemoteLike(savedConfig) && savedConfig?.mode === 'cloud')
  const showRemote = selectedMode === 'remote' || remoteConfigured
  const showCloud = selectedMode === 'cloud' || cloudConfigured

  const modeItems: { mode: GatewayMode; icon: string; label: string; hint: string }[] = [
    {
      mode: 'local',
      icon: 'home',
      label: t.titlebar.gatewayModeLocal,
      hint: t.titlebar.gatewayModeLocalHint
    },
    ...(showRemote
      ? [
          {
            mode: 'remote' as const,
            icon: 'globe',
            label: t.titlebar.gatewayModeRemote,
            hint: t.titlebar.gatewayModeRemoteHint
          }
        ]
      : []),
    ...(showCloud
      ? [
          {
            mode: 'cloud' as const,
            icon: 'cloud',
            label: t.titlebar.gatewayModeCloud,
            hint: t.titlebar.gatewayModeCloudHint
          }
        ]
      : [])
  ]

  return (
    <>
      <div
        aria-label={t.shell.windowControls}
        className="fixed left-(--titlebar-controls-left) top-(--titlebar-controls-top) z-70 flex translate-y-0.5 flex-row items-center gap-x-1 pointer-events-auto select-none [-webkit-app-region:no-drag]"
      >
        {leftToolbarTools
          .filter(tool => !tool.hidden)
          .map(tool => (
            <TitlebarToolButton key={tool.id} navigate={navigate} tool={tool} />
          ))}
      </div>

      {/*
        Pane-scoped tools (preview's monitor / devtools / refresh / X) render
        as their own fixed cluster. AppShell sets --shell-preview-toolbar-gap
        to either the static cluster's width (file-browser closed → cluster
        sits flush against system tools) or the file-browser pane's width
        (file-browser open → cluster sits flush against the file-browser pane,
        i.e. at the preview pane's right edge). No margin hacks needed.
      */}
      {visiblePaneTools.length > 0 && (
        <div
          aria-label={t.shell.paneControls}
          className="fixed top-[calc(var(--titlebar-controls-top)+var(--right-rail-top-inset,0px))] right-[calc(var(--titlebar-tools-right)+var(--shell-preview-toolbar-gap,0))] z-70 flex flex-row items-center gap-x-1 pointer-events-auto select-none [-webkit-app-region:no-drag]"
        >
          {visiblePaneTools.map(tool => (
            <TitlebarToolButton key={tool.id} navigate={navigate} tool={tool} />
          ))}
        </div>
      )}

      <div
        aria-label={t.shell.appControls}
        className="fixed right-(--titlebar-tools-right) top-(--titlebar-controls-top) z-70 flex flex-row items-center justify-end gap-x-1 pointer-events-auto select-none [-webkit-app-region:no-drag]"
      >
        {visibleSystemToolsBeforeSettings.map(tool => (
          <TitlebarToolButton key={tool.id} navigate={navigate} tool={tool} />
        ))}

        <DropdownMenu onOpenChange={setMenuOpen} open={menuOpen}>
          <Tip label={t.titlebar.gatewayModeMenuTitle}>
            <DropdownMenuTrigger asChild>
              <Button
                aria-label={t.titlebar.gatewayModeMenuTitle}
                aria-pressed={selectedMode !== 'local' || undefined}
                className={gatewayTriggerClass}
                disabled={busy}
                onPointerDown={event => event.stopPropagation()}
                size="icon-titlebar"
                type="button"
                variant="ghost"
              >
                <Codicon name={modeIcon(selectedMode, busy)} spinning={busy} />
              </Button>
            </DropdownMenuTrigger>
          </Tip>
          <DropdownMenuContent align="end" className="min-w-52" side="bottom" sideOffset={6}>
            <DropdownMenuLabel>{t.titlebar.gatewayModeMenuTitle}</DropdownMenuLabel>
            <DropdownMenuSeparator />
            {modeItems.map(item => {
              const active = selectedMode === item.mode

              return (
                <DropdownMenuItem
                  className="gap-2"
                  disabled={busy}
                  key={item.mode}
                  onSelect={() => {
                    void selectGatewayMode(item.mode)
                  }}
                >
                  <Codicon name={item.icon} size="0.875rem" />
                  <span className="flex min-w-0 flex-1 flex-col">
                    <span className="truncate font-medium">{item.label}</span>
                    <span className="truncate text-[0.65rem] text-muted-foreground">{item.hint}</span>
                  </span>
                  {active ? <Codicon className="text-foreground" name="check" size="0.75rem" /> : null}
                </DropdownMenuItem>
              )
            })}
            <DropdownMenuSeparator />
            <DropdownMenuItem
              className="gap-2"
              onSelect={() => {
                navigate(`${SETTINGS_ROUTE}?tab=gateway`)
              }}
            >
              <Codicon name="settings-gear" size="0.875rem" />
              <span className="truncate">{t.titlebar.gatewayModeOpenSettings}</span>
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>

        {settingsTool && <TitlebarToolButton navigate={navigate} tool={settingsTool} />}
        <TitlebarToolButton navigate={navigate} tool={rightSidebarTool} />
      </div>
    </>
  )
}

function TitlebarToolButton({ navigate, tool }: { navigate: ReturnType<typeof useNavigate>; tool: TitlebarTool }) {
  // Titlebar actions never show an active background — state reads from the
  // icon itself (e.g. the mute/unmute glyph). aria-pressed still carries it
  // for a11y.
  const className = cn(titlebarButtonClass, 'bg-transparent select-none', tool.className)

  const tooltipLabel = tool.actionId ? (
    <TipKeybindLabel actionId={tool.actionId} text={tool.title ?? tool.label} />
  ) : (
    (tool.title ?? tool.label)
  )

  if (tool.href) {
    return (
      <Tip label={tooltipLabel}>
        <Button asChild className={className} size="icon-titlebar" variant="ghost">
          <a
            aria-label={tool.label}
            href={tool.href}
            onPointerDown={event => event.stopPropagation()}
            rel="noreferrer"
            target="_blank"
          >
            {tool.icon}
          </a>
        </Button>
      </Tip>
    )
  }

  return (
    <Tip label={tooltipLabel}>
      <Button
        aria-label={tool.label}
        aria-pressed={tool.active ?? undefined}
        className={className}
        disabled={tool.disabled}
        onClick={event => {
          if (tool.to) {
            navigate(tool.to)
          }

          tool.onSelect?.(event)
        }}
        onPointerDown={event => event.stopPropagation()}
        size="icon-titlebar"
        type="button"
        variant="ghost"
      >
        {tool.icon}
      </Button>
    </Tip>
  )
}
