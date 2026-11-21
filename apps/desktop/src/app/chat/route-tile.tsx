/**
 * ROUTE (PAGE) TILES — a full-page view rendered as a layout-tree pane BESIDE
 * the main thread, the page analog of session tiles. Built-in pages
 * (Capabilities / Messaging / Artifacts) render their view; plugin pages render
 * their `ROUTES_AREA` contribution. Lifecycle mirrors session tiles:
 * `openRouteTile(path)` -> `watchRouteTiles` registers a pane docked beside
 * main -> tree adoption lands it on the chosen edge; closing removes it.
 */

import { type ReactNode, Suspense } from 'react'

import { ContribBoundary, ContribRender } from '@/contrib/react/boundary'
import { useContributions } from '@/contrib/react/use-contributions'
import { runtimeTranslations, type Translations, useI18n } from '@/i18n'
import { $routeTiles, closeRouteTile, type RouteTile } from '@/store/route-tiles'
import { $portalShell, usePortalShell } from '@/store/shell-variant'

import { ArtifactsView, CapabilitiesView, MessagingView } from '../page-views'
import {
  $routesVersion,
  ARTIFACTS_ROUTE,
  CAPABILITIES_ROUTE,
  contributedRoutes,
  MESSAGING_ROUTE,
  ROUTES_AREA
} from '../routes'

import { paneMirror } from './pane-mirror'

interface BuiltinPage {
  render: () => ReactNode
  /** Classic's fixed tab title. */
  title: string
  /** The redesign's: the page's nav label, in the app's language and the shell's wording. */
  label: (t: Translations) => string
}

// Built-in page views + their pane titles, keyed by route.
const BUILTIN_PAGES: Record<string, BuiltinPage> = {
  [ARTIFACTS_ROUTE]: { render: () => <ArtifactsView />, title: 'Artifacts', label: t => t.sidebar.nav.artifacts },
  [MESSAGING_ROUTE]: { render: () => <MessagingView />, title: 'Messaging', label: t => t.sidebar.nav.messaging },
  [CAPABILITIES_ROUTE]: {
    render: () => <CapabilitiesView />,
    title: 'Capabilities',
    label: t => t.sidebar.nav.capabilities
  }
}

/** Humanize a route path into a tab title: `/my-atlas` → `My Atlas`. */
const humanizePath = (path: string): string =>
  path
    .replace(/^\/+/, '')
    .split(/[/-]/)
    .filter(Boolean)
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ') || path

/** Title for a route tile: the built-in name, the contribution's own `title`,
 *  else a humanized path — never the internal `${source}:${id}` key. */
function routeTitle(path: string): string {
  if (BUILTIN_PAGES[path]) {
    return BUILTIN_PAGES[path].title
  }

  return contributedRoutes().find(r => r.path === path)?.title ?? humanizePath(path)
}

/** A built-in page's tab label, live: it follows a language or shell switch without re-registering. */
function BuiltinPageTitle({ page }: { page: BuiltinPage }) {
  const { t } = useI18n()
  const portal = usePortalShell()

  return <>{portal ? page.label(t) : page.title}</>
}

export function RouteTilePane({ path }: { path: string }) {
  const builtin = BUILTIN_PAGES[path]

  // Subscribe so a plugin page tile appears the moment its route registers.
  // The snapshot feeds the lookup: under React Compiler an independently
  // called contributedRoutes() can stay memoized across that registration.
  const contributions = useContributions(ROUTES_AREA)
  const contrib = builtin ? null : contributedRoutes(contributions).find(r => r.path === path)

  if (builtin) {
    return (
      <ContribBoundary id={path}>
        <Suspense fallback={null}>
          <ContribRender render={builtin.render} />
        </Suspense>
      </ContribBoundary>
    )
  }

  if (contrib) {
    return (
      <ContribBoundary id={path}>
        <ContribRender render={contrib.render} />
      </ContribBoundary>
    )
  }

  return (
    <div className="grid h-full place-items-center font-mono text-[11px] text-(--ui-text-quaternary)">
      no page at {path}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Route tile -> pane contribution sync (call once from the app root).
// ---------------------------------------------------------------------------

/** Keep pane contributions mirroring `$routeTiles`. Call once from the root. */
export const watchRouteTiles = paneMirror<RouteTile>({
  source: $routeTiles,
  // A tile restored before its plugin route registers must pick up the
  // contribution's title once it lands, not keep the humanized-path fallback.
  also: [$routesVersion],
  key: t => t.path,
  prefix: 'route-tile',
  dir: t => t.dir,
  minWidth: '22rem',
  title: routeTitle,
  tabTitle: path => BUILTIN_PAGES[path] && <BuiltinPageTitle page={BUILTIN_PAGES[path]} />,
  tabTitleText: path =>
    BUILTIN_PAGES[path] && $portalShell.get() ? BUILTIN_PAGES[path].label(runtimeTranslations()) : routeTitle(path),
  render: path => <RouteTilePane path={path} />,
  close: closeRouteTile
})
