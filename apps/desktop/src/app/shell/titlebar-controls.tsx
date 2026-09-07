import { useStore } from '@nanostores/react'
import { type ComponentProps, type MouseEvent, type ReactNode, useEffect, useState } from 'react'
import { useLocation, useNavigate } from 'react-router'

import { hudTargetSessionId } from '@/app/hud/handoff'
import { toggleLayoutEditMode } from '@/components/pane-shell/edit-mode'
import { resetLayoutTree } from '@/components/pane-shell/tree/store'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Tip, TipKeybindLabel } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'
import { compactNumber } from '@/lib/format'
import { triggerHaptic } from '@/lib/haptics'
import { formatModifierToken } from '@/lib/keybinds/combo'
import { cn } from '@/lib/utils'
import { $hapticsMuted, toggleHapticsMuted } from '@/store/haptics'
import { toggleHud } from '@/store/hud'
import {
  $fileBrowserOpen,
  $panesFlipped,
  $sidebarOpen,
  toggleFileBrowserOpen,
  togglePanesFlipped,
  toggleSidebarOpen
} from '@/store/layout'
import { $unreadSessionCount } from '@/store/session-dot-state'

import { appViewForPath, isOverlayView } from '../routes'

import {
  TITLEBAR_ICON_BADGE_SCALE,
  titlebarButtonClass,
  titlebarIconSizeCss,
  titlebarToolClusterClass
} from './titlebar'
import { TitlebarIcon } from './titlebar-icon'

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
  /** Overlay count on the glyph (unread sessions). Hidden when 0/undefined. */
  badge?: number
  title?: string
  to?: string
  /** Durable `data-tour` handle. Tools are addressed by icon and translated
   *  label otherwise, and neither survives a theme or a locale change. */
  tour?: string
}

export type TitlebarToolSide = 'left' | 'right'
export type SetTitlebarToolGroup = (id: string, tools: readonly TitlebarTool[], side?: TitlebarToolSide) => void

interface TitlebarControlsProps extends ComponentProps<'div'> {
  leftTools?: readonly TitlebarTool[]
  tools?: readonly TitlebarTool[]
  onOpenSettings: () => void
}

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
        <TitlebarIcon name="layout" />
      </span>
      <span className={cn('relative hidden', modHeld && 'group-hover/tool:inline-flex')}>
        <TitlebarIcon name="layout" />
        <span className="absolute -bottom-1 -right-1.5 grid place-items-center rounded-full bg-(--ui-bg-chrome) p-px">
          <TitlebarIcon className="-scale-x-100" name="refresh" size={titlebarIconSizeCss(TITLEBAR_ICON_BADGE_SCALE)} />
        </span>
      </span>
    </>
  )
}

/** Overlay count on a titlebar glyph. Hidden when count is 0/undefined. */
function withCountBadge(icon: ReactNode, count: number | undefined): ReactNode {
  if (!count) {
    return icon
  }

  return (
    <span className="relative inline-flex">
      {icon}
      <span className="pointer-events-none absolute -top-2.5 -right-1.5 z-1">
        <Badge aria-hidden size="overlay" variant="solid">
          {compactNumber(count)}
        </Badge>
      </span>
    </span>
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

export function TitlebarControls({ leftTools = [], tools = [], onOpenSettings }: TitlebarControlsProps) {
  const { t } = useI18n()
  const navigate = useNavigate()
  const location = useLocation()
  const modHeld = useModifierHeld()
  const hapticsMuted = useStore($hapticsMuted)
  const fileBrowserOpen = useStore($fileBrowserOpen)
  const panesFlipped = useStore($panesFlipped)
  const sidebarOpen = useStore($sidebarOpen)
  const unreadCount = useStore($unreadSessionCount)
  const unreadBadge = unreadCount > 0 ? unreadCount : undefined
  const unreadHint = unreadBadge ? ` · ${t.titlebar.unreadSessions(unreadBadge)}` : ''

  const toggleHaptics = () => {
    if (!hapticsMuted) {
      triggerHaptic('tap')
    }

    toggleHapticsMuted()

    if (hapticsMuted) {
      window.requestAnimationFrame(() => triggerHaptic('success'))
    }
  }

  // POSITIONAL toggles: each button shows/hides everything on its physical
  // side of the main zone (the layout tree collapses the whole side), so they
  // stay correct through flips and rearranges. $sidebarOpen ≙ left side,
  // $fileBrowserOpen ≙ right side. Never an active highlight — plain
  // show/hide affordances.
  const leftEdge = { open: sidebarOpen, toggle: toggleSidebarOpen }
  const rightEdge = { open: fileBrowserOpen, toggle: toggleFileBrowserOpen }
  const leftLabel = leftEdge.open ? t.titlebar.hideSidebar : t.titlebar.showSidebar
  const rightLabel = rightEdge.open ? t.titlebar.hideRightSidebar : t.titlebar.showRightSidebar

  const leftToolbarTools: TitlebarTool[] = [
    {
      actionId: 'view.toggleSidebar',
      badge: panesFlipped ? undefined : unreadBadge,
      icon: <TitlebarIcon name="layout-sidebar-left" />,
      id: 'sidebar',
      label: `${leftLabel}${panesFlipped ? '' : unreadHint}`,
      onSelect: () => {
        triggerHaptic('tap')
        leftEdge.toggle()
      }
    },
    {
      actionId: 'view.flipPanes',
      icon: <TitlebarIcon name="arrow-swap" />,
      id: 'flip-panes',
      label: t.titlebar.swapSidebarSides,
      onSelect: () => {
        triggerHaptic('tap')
        togglePanesFlipped()
      }
    },
    ...leftTools
  ]

  const rightSidebarTool: TitlebarTool = {
    actionId: 'view.toggleRightSidebar',
    badge: panesFlipped ? unreadBadge : undefined,
    icon: <TitlebarIcon name="layout-sidebar-right" />,
    id: 'right-sidebar',
    label: `${rightLabel}${panesFlipped ? unreadHint : ''}`,
    onSelect: () => {
      triggerHaptic('tap')
      rightEdge.toggle()
    },
    tour: 'right-pane-toggle'
  }

  // Static system tools — always pinned to the screen's right edge.
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
      title: t.titlebar.layoutEditorTitle(formatModifierToken('mod'))
    },
    {
      // No `title`: TitlebarToolButton passes `title` to TipKeybindLabel as a
      // text OVERRIDE, so a long sentence there replaces the short label and
      // crowds the ⌘⇧H hint off the tooltip. Label only — the hint is appended
      // from the action registry, same as every other tool here.
      actionId: 'view.toggleHud',
      icon: <TitlebarIcon name="comment-discussion" />,
      id: 'hud',
      label: t.titlebar.enterHud,
      onSelect: () => {
        triggerHaptic('open')
        toggleHud(hudTargetSessionId())
      }
    },
    {
      active: hapticsMuted,
      icon: <TitlebarIcon name={hapticsMuted ? 'mute' : 'unmute'} />,
      id: 'haptics',
      label: hapticsMuted ? t.titlebar.unmuteHaptics : t.titlebar.muteHaptics,
      onSelect: toggleHaptics
    },
    {
      actionId: 'nav.settings',
      icon: <TitlebarIcon name="settings-gear" />,
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
  const visiblePaneTools = tools.filter(tool => !tool.hidden)

  return (
    <>
      <div
        aria-label={t.shell.windowControls}
        className={cn(
          titlebarToolClusterClass,
          'left-(--titlebar-controls-left) top-(--titlebar-controls-top) translate-y-(--titlebar-controls-y-nudge)'
        )}
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
          className={cn(
            titlebarToolClusterClass,
            'top-[calc(var(--titlebar-controls-top)+var(--right-rail-top-inset,0px))] right-[calc(var(--titlebar-tools-right)+var(--shell-preview-toolbar-gap,0))]'
          )}
        >
          {visiblePaneTools.map(tool => (
            <TitlebarToolButton key={tool.id} navigate={navigate} tool={tool} />
          ))}
        </div>
      )}

      <div
        aria-label={t.shell.appControls}
        className={cn(titlebarToolClusterClass, 'right-(--titlebar-tools-right) top-(--titlebar-controls-top)')}
      >
        {visibleSystemTools.map(tool => (
          <TitlebarToolButton key={tool.id} navigate={navigate} tool={tool} />
        ))}
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
            data-tour={tool.tour}
            href={tool.href}
            onPointerDown={event => event.stopPropagation()}
            rel="noreferrer"
            target="_blank"
          >
            {withCountBadge(tool.icon, tool.badge)}
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
        data-tour={tool.tour}
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
        {withCountBadge(tool.icon, tool.badge)}
      </Button>
    </Tip>
  )
}
