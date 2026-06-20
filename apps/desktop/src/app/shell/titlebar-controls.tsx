import { useStore } from '@nanostores/react'
import { type ComponentProps, type ReactNode, useEffect, useState } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { useI18n } from '@/i18n'
import { triggerHaptic } from '@/lib/haptics'
import { cn } from '@/lib/utils'
import { openAccountDialog } from '@/store/account'
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
import { $workflowCopilotOpen, toggleWorkflowCopilotOpen } from '@/store/workflow'

import { appViewForPath, isOverlayView } from '../routes'

import { titlebarButtonClass } from './titlebar'

const SIGNED_OUT_LABEL = '登录 / 注册'

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
  text?: string
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

export function TitlebarControls({ leftTools = [], tools = [], onOpenSettings }: TitlebarControlsProps) {
  const { t } = useI18n()
  const navigate = useNavigate()
  const location = useLocation()
  const hapticsMuted = useStore($hapticsMuted)
  const fileBrowserOpen = useStore($fileBrowserOpen)
  const sidebarOpen = useStore($sidebarOpen)
  const panesFlipped = useStore($panesFlipped)
  const workflowCopilotOpen = useStore($workflowCopilotOpen)
  const currentView = appViewForPath(location.pathname)
  const [accountLabel, setAccountLabel] = useState(SIGNED_OUT_LABEL)
  const signedOut = accountLabel === SIGNED_OUT_LABEL

  useEffect(() => {
    let cancelled = false

    const refreshAccount = () => {
      const account = window.hermesDesktop?.account

      if (!account?.status) {
        setAccountLabel(SIGNED_OUT_LABEL)

        return
      }

      void account
        .status()
        .then(status => {
          if (cancelled) {
            return
          }

          setAccountLabel(status.loggedIn ? status.username || status.email || '账号' : SIGNED_OUT_LABEL)
        })
        .catch(() => {
          // Keep the last known label on a transient status error — don't flash
          // back to "登录" while the user is actually signed in.
        })
    }

    refreshAccount()
    window.addEventListener('hermes-account-changed', refreshAccount)

    return () => {
      cancelled = true
      window.removeEventListener('hermes-account-changed', refreshAccount)
    }
  }, [])

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

  const rightSidebarTool: TitlebarTool =
    currentView === 'workflow'
      ? {
          active: workflowCopilotOpen,
          icon: <Codicon name="comment-discussion" />,
          id: 'workflow-copilot',
          label: workflowCopilotOpen ? '收回爱马仕 Copilot' : '弹出爱马仕 Copilot',
          onSelect: () => {
            triggerHaptic('tap')
            toggleWorkflowCopilotOpen()
          },
          title: workflowCopilotOpen ? '收回爱马仕 Copilot' : '弹出爱马仕 Copilot'
        }
      : {
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
      icon: null,
      id: 'account',
      label: accountLabel,
      text: accountLabel,
      title: signedOut ? '登录 / 注册 EasyHermes' : '账号 · 余额 · 消费明细',
      onSelect: () => {
        triggerHaptic('open')
        openAccountDialog()
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
  if (isOverlayView(currentView)) {
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
          className="fixed top-(--titlebar-controls-top) right-[calc(var(--titlebar-tools-right)+var(--shell-preview-toolbar-gap,0))] z-70 flex flex-row items-center gap-x-1 pointer-events-auto select-none [-webkit-app-region:no-drag]"
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
  const size = tool.text ? ('sm' as const) : ('icon-titlebar' as const)

  const textClassName = tool.text
    ? 'h-(--titlebar-control-height) rounded-[4px] px-2 text-[0.75rem] leading-none'
    : ''

  if (tool.href) {
    return (
      <Button asChild className={cn(className, textClassName)} size={size} variant="ghost">
        <a
          aria-label={tool.label}
          href={tool.href}
          onPointerDown={event => event.stopPropagation()}
          rel="noreferrer"
          target="_blank"
          title={tool.title ?? tool.label}
        >
          {tool.icon}
        </a>
      </Button>
    )
  }

  return (
    <Button
      aria-label={tool.label}
      aria-pressed={tool.active ?? undefined}
      className={cn(className, textClassName)}
      disabled={tool.disabled}
      onClick={() => {
        if (tool.to) {
          navigate(tool.to)
        }

        tool.onSelect?.()
      }}
      onPointerDown={event => event.stopPropagation()}
      size={size}
      title={tool.title ?? tool.label}
      type="button"
      variant="ghost"
    >
      {tool.icon}
      {tool.text ? <span className="max-w-24 truncate">{tool.text}</span> : null}
    </Button>
  )
}
