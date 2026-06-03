import { useStore } from '@nanostores/react'
import type { ComponentProps, ReactNode } from 'react'
import { useNavigate } from 'react-router-dom'

import { Codicon } from '@/components/ui/codicon'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger
} from '@/components/ui/dropdown-menu'
import { useTranslation } from '@/i18n'
import { triggerHaptic } from '@/lib/haptics'
import { Volume2, VolumeX } from '@/lib/icons'
import { cn } from '@/lib/utils'
import { $hapticsMuted, toggleHapticsMuted } from '@/store/haptics'
import { $fileBrowserOpen, $sidebarOpen, toggleFileBrowserOpen, toggleSidebarOpen } from '@/store/layout'

import { PROFILES_ROUTE } from '../routes'

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
  commandCenterOpen?: boolean
  onOpenSettings: () => void
  onOpenSearch: () => void
}

export function TitlebarControls({
  leftTools = [],
  tools = [],
  commandCenterOpen = false,
  onOpenSettings,
  onOpenSearch
}: TitlebarControlsProps) {
  const navigate = useNavigate()
  const t = useTranslation()
  const hapticsMuted = useStore($hapticsMuted)
  const fileBrowserOpen = useStore($fileBrowserOpen)
  const sidebarOpen = useStore($sidebarOpen)

  const toggleHaptics = () => {
    if (!hapticsMuted) {
      triggerHaptic('tap')
    }

    toggleHapticsMuted()

    if (hapticsMuted) {
      window.requestAnimationFrame(() => triggerHaptic('success'))
    }
  }

  const leftToolbarTools: TitlebarTool[] = [
    {
      icon: <Codicon name="layout-sidebar-left" />,
      id: 'sidebar',
      label: sidebarOpen ? t('shell.titlebar.hideSidebar') : t('shell.titlebar.showSidebar'),
      onSelect: () => {
        triggerHaptic('tap')
        toggleSidebarOpen()
      }
    },
    {
      active: commandCenterOpen,
      icon: <Codicon name="search" />,
      id: 'search',
      label: t('shell.titlebar.search'),
      onSelect: () => {
        triggerHaptic('open')
        onOpenSearch()
      },
      title: t('shell.titlebar.searchTitle')
    },
    ...leftTools
  ]

  const rightSidebarTool: TitlebarTool = {
    active: fileBrowserOpen,
    icon: <Codicon name="layout-sidebar-right" />,
    id: 'right-sidebar',
    label: fileBrowserOpen ? t('shell.titlebar.hideRightSidebar') : t('shell.titlebar.showRightSidebar'),
    onSelect: () => {
      triggerHaptic('tap')
      toggleFileBrowserOpen()
    }
  }

  // Static system tools — always pinned to the screen's right edge.
  const systemTools: TitlebarTool[] = [
    {
      active: hapticsMuted,
      icon: hapticsMuted ? <VolumeX /> : <Volume2 />,
      id: 'haptics',
      label: hapticsMuted ? t('shell.titlebar.unmuteHaptics') : t('shell.titlebar.muteHaptics'),
      onSelect: toggleHaptics
    },
    {
      icon: <Codicon name="settings-gear" />,
      id: 'settings',
      label: t('shell.titlebar.openSettings'),
      onSelect: () => {
        triggerHaptic('open')
        onOpenSettings()
      }
    }
  ]

  const visibleSystemTools = systemTools.filter(tool => !tool.hidden)
  const settingsTool = visibleSystemTools.find(tool => tool.id === 'settings')
  const visibleSystemToolsBeforeSettings = visibleSystemTools.filter(tool => tool.id !== 'settings')
  const visiblePaneTools = tools.filter(tool => !tool.hidden)

  return (
    <>
      <div
        aria-label={t('shell.titlebar.windowControls')}
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
          aria-label={t('shell.titlebar.paneControls')}
          className="fixed top-(--titlebar-controls-top) right-[calc(var(--titlebar-tools-right)+var(--shell-preview-toolbar-gap,0))] z-70 flex flex-row items-center gap-x-1 pointer-events-auto select-none [-webkit-app-region:no-drag]"
        >
          {visiblePaneTools.map(tool => (
            <TitlebarToolButton key={tool.id} navigate={navigate} tool={tool} />
          ))}
        </div>
      )}

      <div
        aria-label={t('shell.titlebar.appControls')}
        className="fixed right-(--titlebar-tools-right) top-(--titlebar-controls-top) z-70 flex flex-row items-center justify-end gap-x-1 pointer-events-auto select-none [-webkit-app-region:no-drag]"
      >
        {visibleSystemToolsBeforeSettings.map(tool => (
          <TitlebarToolButton key={tool.id} navigate={navigate} tool={tool} />
        ))}
        <ProfilesMenuButton navigate={navigate} />
        {settingsTool && <TitlebarToolButton navigate={navigate} tool={settingsTool} />}
        <TitlebarToolButton navigate={navigate} tool={rightSidebarTool} />
      </div>
    </>
  )
}

function ProfilesMenuButton({ navigate }: { navigate: ReturnType<typeof useNavigate> }) {
  const t = useTranslation()

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <button
          aria-label={t('shell.titlebar.profiles')}
          className={cn(titlebarButtonClass, 'grid place-items-center bg-transparent select-none [&_svg]:size-4')}
          onPointerDown={event => event.stopPropagation()}
          title={t('shell.titlebar.profiles')}
          type="button"
        >
          <Codicon name="account" />
        </button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-64" sideOffset={8}>
        <DropdownMenuLabel>
          <div className="text-sm font-medium text-foreground">{t('shell.titlebar.profiles')}</div>
          <div className="mt-1 text-xs font-normal leading-4 text-muted-foreground">
            {t('shell.titlebar.profilesDescription')}
          </div>
        </DropdownMenuLabel>
        <DropdownMenuSeparator />
        <DropdownMenuItem
          onSelect={() => {
            triggerHaptic('open')
            navigate(PROFILES_ROUTE)
          }}
        >
          <Codicon name="account" size="1rem" />
          <span>{t('shell.titlebar.manageProfiles')}</span>
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}

function TitlebarToolButton({ navigate, tool }: { navigate: ReturnType<typeof useNavigate>; tool: TitlebarTool }) {
  const className = cn(
    titlebarButtonClass,
    'grid place-items-center bg-transparent select-none [&_svg]:size-4',
    tool.active && 'bg-(--ui-control-active-background)! text-foreground!',
    tool.className
  )

  if (tool.href) {
    return (
      <a
        aria-label={tool.label}
        className={className}
        href={tool.href}
        onPointerDown={event => event.stopPropagation()}
        rel="noreferrer"
        target="_blank"
        title={tool.title ?? tool.label}
      >
        {tool.icon}
      </a>
    )
  }

  return (
    <button
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
      title={tool.title ?? tool.label}
      type="button"
    >
      {tool.icon}
    </button>
  )
}
