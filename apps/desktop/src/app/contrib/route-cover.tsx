/**
 * Pages cover the chat; they do not replace it. In the redesign a route overlay (Settings, Agents,
 * Schedules, …) is painted over the whole workspace and a workspace page (Capabilities, Library, …)
 * over the chat inside its pane, while the chat underneath stays mounted: hidden from paint, out of
 * the tab order and the accessibility tree, its transcript scroll, composer draft and focus intact.
 * Returning to the chat is then a visibility flip, not a remount of the transcript and the sidebar.
 * Beside a workspace page the chat sidebar is collapsed for the route, not unmounted
 * (pane-shell/tree/store-route-collapse.ts). Background sessions keep streaming into their stores.
 */
import { memo, type ReactNode, useLayoutEffect, useRef, useState } from 'react'
import { type Location, Routes, useLocation } from 'react-router'

import { usePortalShell } from '@/store/shell-variant'

import { appViewForPath, isOverlayView } from '../routes'

import { AtLocation, type RouterLocation, useRouterLocation } from './frozen-location'

/** Remembers the last focused element inside a covered region and gives focus back to it when the
 *  region is uncovered, unless the user has since put focus somewhere else. `inert` blurs it. */
function useFocusReturn(covered: boolean) {
  const ref = useRef<HTMLDivElement>(null)
  const lastFocused = useRef<HTMLElement | null>(null)

  useLayoutEffect(() => {
    const region = ref.current

    if (!region) {
      return
    }

    const remember = (event: FocusEvent) => {
      if (event.target instanceof HTMLElement) {
        lastFocused.current = event.target
      }
    }

    region.addEventListener('focusin', remember)

    return () => region.removeEventListener('focusin', remember)
  }, [])

  useLayoutEffect(() => {
    const target = lastFocused.current
    const focus = document.activeElement

    if (!covered && target?.isConnected && (!focus || focus === document.body)) {
      target.focus({ preventScroll: true })
    }
  }, [covered])

  return ref
}

interface CoveredRegionProps {
  children: ReactNode
  /** The box it takes: exactly the one its content had without it. */
  className: string
  covered: boolean
}

/** A region that something else paints over: skipped by style, layout and paint while covered
 *  (`content-visibility: hidden` keeps its rendering state, so uncovering is a flip, not a rebuild),
 *  inert and hidden from assistive tech. */
function CoveredRegion({ children, className, covered }: CoveredRegionProps) {
  const ref = useFocusReturn(covered)

  return (
    <div
      aria-hidden={covered || undefined}
      className={className}
      data-route-covered={covered || undefined}
      inert={covered || undefined}
      ref={ref}
    >
      {children}
    </div>
  )
}

/** The layout tree, covered while a route overlay is open (the redesign; classic overlays float
 *  as cards over the shell they dim, so its tree stays as it is). */
export function LayoutTreeCover({ children }: { children: ReactNode }) {
  const { pathname } = useLocation()
  const covered = usePortalShell() && isOverlayView(appViewForPath(pathname))

  return (
    <CoveredRegion className="flex min-h-0 min-w-0 flex-1" covered={covered}>
      {children}
    </CoveredRegion>
  )
}

/** The same place: a fresh navigation key alone must not re-render the kept chat. */
function samePlace(a: Location, b: Location) {
  return a.pathname === b.pathname && a.search === b.search && a.hash === b.hash && a.state === b.state
}

interface ChatAtProps {
  chat: ReactNode
  location: RouterLocation
}

/** The chat's routes at a location of their own. Memoized on it, so while a page covers the chat no
 *  navigation (its own or between pages) re-renders the transcript or anything that reads the router. */
const ChatAt = memo(function ChatAt({ chat, location }: ChatAtProps) {
  return (
    <AtLocation location={location}>
      <Routes>{chat}</Routes>
    </AtLocation>
  )
})

interface KeptChatRoutesProps {
  /** Every route the router reads as a chat location (index, `:sessionId`, redirects). */
  chat: ReactNode
  /** Every other route (workspace pages, plugin pages, overlay paths), painted over the chat. */
  pages: ReactNode
}

/** The workspace pane's routes with the chat kept alive under pages. Until the chat has been shown
 *  once (a deep link straight to a page) there is nothing to keep, so nothing renders under it. */
export function KeptChatRoutes({ chat, pages }: KeptChatRoutesProps) {
  const current = useRouterLocation()
  const onChat = appViewForPath(current.location.pathname) === 'chat'
  const [kept, setKept] = useState<RouterLocation | null>(onChat ? current : null)

  if (onChat && (!kept || !samePlace(kept.location, current.location))) {
    setKept(current)
  }

  return (
    <>
      {kept && (
        <CoveredRegion className="absolute inset-0" covered={!onChat}>
          <ChatAt chat={chat} location={kept} />
        </CoveredRegion>
      )}
      {!onChat && (
        <div className="absolute inset-0" data-slot="workspace-page-layer">
          <Routes>{pages}</Routes>
        </div>
      )}
    </>
  )
}
