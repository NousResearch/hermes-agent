import { useStore } from '@nanostores/react'
import { type ComponentProps, type ReactNode, useEffect, useMemo, useState } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from '@/components/ui/dropdown-menu'
import { Tip } from '@/components/ui/tooltip'
import type { DesktopConnectionConfig } from '@/global'
import { useI18n } from '@/i18n'
import { triggerHaptic } from '@/lib/haptics'
import { Check, Cloud, Globe, Loader2, Monitor } from '@/lib/icons'
import { cn } from '@/lib/utils'
import { $hapticsMuted, toggleHapticsMuted } from '@/store/haptics'
import { toggleKeybindPanel } from '@/store/keybinds'
import {
  $fileBrowserOpen,
  $panesFlipped,
  $sidebarOpen,
  toggleFileBrowserOpen,
  togglePanesFlipped,
  toggleSidebarOpen
} from '@/store/layout'
import { notify, notifyError } from '@/store/notifications'
import { $activeGatewayProfile } from '@/store/profile'

import { appViewForPath, isOverlayView } from '../routes'

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
  onSelect?: () => void
  title?: string
  to?: string
}

export type TitlebarToolSide = 'left' | 'right'
export type SetTitlebarToolGroup = (id: string, tools: readonly TitlebarTool[], side?: TitlebarToolSide) => void

interface TitlebarControlsProps extends ComponentProps<'div'> {
  leftTools?: readonly TitlebarTool[]
  tools?: readonly TitlebarTool[]
  onOpenSettings: () => void
}

interface GatewayToggleProps {
  activeGatewayProfile: string
}

export function GatewayToggle({ activeGatewayProfile }: GatewayToggleProps) {
  const { t } = useI18n()
  const navigate = useNavigate()
  const [config, setConfig] = useState<DesktopConnectionConfig | null>(null)
  const [switching, setSwitching] = useState(false)
  const [open, setOpen] = useState(false)

  useEffect(() => {
    const refreshConfig = async () => {
      const desktop = window.hermesDesktop

      if (!desktop?.getConnectionConfig) {
        return
      }

      try {
        const cfg = await desktop.getConnectionConfig(activeGatewayProfile)

        setConfig(cfg)
      } catch (err) {
        console.error('Failed to get connection config:', err)
      }
    }

    void refreshConfig()

    const offConnectionApplied = window.hermesDesktop?.onConnectionApplied?.(() => {
      void refreshConfig()
    })

    return () => {
      offConnectionApplied?.()
    }
  }, [activeGatewayProfile])

  const isConfigured = (mode: 'local' | 'remote' | 'cloud') => {
    if (mode === 'local') {
      return true
    }

    if (!config) {
      return false
    }

    const hasUrl = Boolean(config.remoteUrl?.trim())

    if (!hasUrl) {
      return false
    }

    if (mode === 'remote') {
      if (config.remoteAuthMode === 'oauth') {
        return config.remoteOauthConnected
      }

      return config.remoteTokenSet
    }

    if (mode === 'cloud') {
      return config.remoteOauthConnected
    }

    return false
  }

  const handleSelectMode = async (mode: 'local' | 'remote' | 'cloud') => {
    if (config?.mode === mode) {
      return
    }

    if (!isConfigured(mode)) {
      triggerHaptic('warning')

      notify({
        kind: 'warning',
        title: t.settings.gateway.incompleteTitle,
        message: mode === 'cloud'
          ? t.settings.gateway.cloudNeedsSignIn
          : (config?.remoteAuthMode === 'oauth'
            ? t.settings.gateway.incompleteSignIn
            : t.settings.gateway.incompleteToken)
      })

      navigate('/settings?tab=gateway')

      return
    }

    const desktop = window.hermesDesktop

    if (!desktop?.applyConnectionConfig) {
      return
    }

    triggerHaptic('tap')
    setSwitching(true)

    try {
      const payload = {
        mode,
        profile: activeGatewayProfile === 'default' ? undefined : activeGatewayProfile,
        remoteAuthMode: config?.remoteAuthMode ?? 'token',
        remoteUrl: config?.remoteUrl ?? '',
        cloudOrg: config?.cloudOrg ?? ''
      }

      const next = await desktop.applyConnectionConfig(payload)

      setConfig(next)

      notify({
        kind: 'success',
        title: t.settings.gateway.restartingTitle,
        message: t.settings.gateway.restartingMessage
      })
    } catch (err: any) {
      if (err?.needsOauthLogin && config?.remoteUrl) {
        try {
          const result = await desktop.oauthLoginConnectionConfig(config.remoteUrl)

          if (result.connected) {
            const nextPayload = {
              mode,
              profile: activeGatewayProfile === 'default' ? undefined : activeGatewayProfile,
              remoteAuthMode: 'oauth',
              remoteUrl: config.remoteUrl,
              cloudOrg: config.cloudOrg
            }

            const next = await desktop.applyConnectionConfig(nextPayload)

            setConfig(next)

            notify({
              kind: 'success',
              title: t.settings.gateway.restartingTitle,
              message: t.settings.gateway.restartingMessage
            })
          } else {
            notify({
              kind: 'warning',
              title: t.boot.failure.signInIncompleteTitle,
              message: t.boot.failure.signInIncompleteMessage
            })
          }
        } catch (loginErr) {
          notifyError(loginErr, t.settings.gateway.signInFailed)
        }
      } else {
        notifyError(err, t.settings.gateway.applyFailed)
      }
    } finally {
      setSwitching(false)
    }
  }

  const activeMode = config?.mode ?? 'local'

  const icon = useMemo(() => {
    if (switching) {
      return <Loader2 className="size-4 animate-spin" />
    }

    switch (activeMode) {
      case 'local':
        return <Monitor className="size-4" />

      case 'cloud':
        return <Cloud className="size-4" />

      case 'remote':
        return <Globe className="size-4" />

      default:
        return <Monitor className="size-4" />
    }
  }, [activeMode, switching])

  return (
    <DropdownMenu onOpenChange={setOpen} open={switching ? false : open}>
      <Tip label={t.settings.gateway.title}>
        <DropdownMenuTrigger asChild>
          <Button
            aria-label={t.settings.gateway.title}
            className={cn(titlebarButtonClass, 'bg-transparent select-none')}
            disabled={switching}
            size="icon-titlebar"
            type="button"
            variant="ghost"
          >
            {icon}
          </Button>
        </DropdownMenuTrigger>
      </Tip>

      <DropdownMenuContent align="end" className="w-48" side="bottom" sideOffset={8}>
        <DropdownMenuItem
          className="gap-2 text-foreground focus:bg-accent [&_svg]:size-4"
          onClick={() => void handleSelectMode('local')}
        >
          <Monitor className="size-4 shrink-0 text-muted-foreground" />
          <span className="truncate">{t.settings.gateway.localTitle}</span>
          {activeMode === 'local' && <Check className="ml-auto size-4 shrink-0 text-primary" />}
        </DropdownMenuItem>

        <DropdownMenuItem
          className="gap-2 text-foreground focus:bg-accent [&_svg]:size-4"
          onClick={() => void handleSelectMode('remote')}
        >
          <Globe className="size-4 shrink-0 text-muted-foreground" />
          <span className="truncate">{t.settings.gateway.remoteTitle}</span>
          {activeMode === 'remote' && <Check className="ml-auto size-4 shrink-0 text-primary" />}
        </DropdownMenuItem>

        <DropdownMenuItem
          className="gap-2 text-foreground focus:bg-accent [&_svg]:size-4"
          onClick={() => void handleSelectMode('cloud')}
        >
          <Cloud className="size-4 shrink-0 text-muted-foreground" />
          <span className="truncate">{t.settings.gateway.cloudTitle}</span>
          {activeMode === 'cloud' && <Check className="ml-auto size-4 shrink-0 text-primary" />}
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}

export function TitlebarControls({ leftTools = [], tools = [], onOpenSettings }: TitlebarControlsProps) {
  const { t } = useI18n()
  const navigate = useNavigate()
  const location = useLocation()
  const hapticsMuted = useStore($hapticsMuted)
  const fileBrowserOpen = useStore($fileBrowserOpen)
  const sidebarOpen = useStore($sidebarOpen)
  const panesFlipped = useStore($panesFlipped)
  const activeGatewayProfile = useStore($activeGatewayProfile)

  const toggleHaptics = () => {
    if (!hapticsMuted) {
      triggerHaptic('tap')
    }

    toggleHapticsMuted()

    if (hapticsMuted) {
      window.requestAnimationFrame(() => triggerHaptic('success'))
    }
  }

  // Each titlebar button controls the pane physically on its side, so a flip
  // swaps which pane each one toggles. Default: sessions left, file browser
  // right. Flipped: file browser left, sessions right. Sidebar toggles never
  // carry an active highlight — they're plain show/hide affordances.
  const fileBrowserEdge = { open: fileBrowserOpen, toggle: toggleFileBrowserOpen }
  const sessionsEdge = { open: sidebarOpen, toggle: toggleSidebarOpen }
  const leftEdge = panesFlipped ? fileBrowserEdge : sessionsEdge
  const rightEdge = panesFlipped ? sessionsEdge : fileBrowserEdge

  const leftToolbarTools: TitlebarTool[] = [
    {
      icon: <Codicon name="layout-sidebar-left" />,
      id: 'sidebar',
      label: leftEdge.open ? t.titlebar.hideSidebar : t.titlebar.showSidebar,
      onSelect: () => {
        triggerHaptic('tap')
        leftEdge.toggle()
      }
    },
    {
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
    icon: <Codicon name="layout-sidebar-right" />,
    id: 'right-sidebar',
    label: rightEdge.open ? t.titlebar.hideRightSidebar : t.titlebar.showRightSidebar,
    onSelect: () => {
      triggerHaptic('tap')
      rightEdge.toggle()
    }
  }

  // Static system tools — always pinned to the screen's right edge.
  const systemTools: TitlebarTool[] = [
    {
      active: hapticsMuted,
      icon: <Codicon name={hapticsMuted ? 'mute' : 'unmute'} />,
      id: 'haptics',
      label: hapticsMuted ? t.titlebar.unmuteHaptics : t.titlebar.muteHaptics,
      onSelect: toggleHaptics
    },
    {
      icon: <Codicon name="keyboard" />,
      id: 'keybinds',
      label: t.titlebar.openKeybinds,
      onSelect: () => {
        triggerHaptic('open')
        toggleKeybindPanel()
      }
    },
    {
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
        <GatewayToggle activeGatewayProfile={activeGatewayProfile} />
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

  if (tool.href) {
    return (
      <Tip label={tool.title ?? tool.label}>
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
    <Tip label={tool.title ?? tool.label}>
      <Button
        aria-label={tool.label}
        aria-pressed={tool.active ?? undefined}
        className={className}
        disabled={tool.disabled}
        onClick={() => {
          if (tool.to) {
            navigate(tool.to)
          }

          tool.onSelect?.()
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
