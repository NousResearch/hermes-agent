import { lazy, Suspense } from 'react'

import { Button } from '@/components/ui/button'
import { ContribBoundary, ContribRender } from '@/contrib/react/boundary'
import { useContributions } from '@/contrib/react/use-contributions'
import { translateNow } from '@/i18n'
import { openRouteTile } from '@/store/route-tiles'

import { contributedRoutes, ROUTES_AREA, routePathname, type AppView } from '../routes'
import { Panel } from '../overlays/panel'

const CapabilitiesView = lazy(async () => ({ default: (await import('../capabilities')).CapabilitiesView }))
const MessagingView = lazy(async () => ({ default: (await import('../messaging')).MessagingView }))
const ArtifactsView = lazy(async () => ({ default: (await import('../artifacts')).ArtifactsView }))

interface PageOverlayHostProps {
  onClose: () => void
  path: string | null
  view: AppView | null
}

export function PageOverlayHost({ onClose, path, view }: PageOverlayHostProps) {
  const routeSnapshot = useContributions(ROUTES_AREA)
  const route =
    view === 'extension' && path
      ? contributedRoutes(routeSnapshot).find(candidate => candidate.path === routePathname(path))
      : undefined

  if (!path || !view) {
    return null
  }

  return (
    <Suspense fallback={null}>
      <Panel onClose={onClose}>
        <div className="mb-2 flex shrink-0 justify-end pr-8">
          <Button
            onClick={() => {
              openRouteTile(routePathname(path))
              onClose()
            }}
            size="sm"
            variant="ghost"
          >
            {translateNow('sidebar.row.openInNewTab')}
          </Button>
        </div>
        {view === 'capabilities' && <CapabilitiesView onClose={onClose} overlay />}
        {view === 'messaging' && <MessagingView overlay />}
        {view === 'artifacts' && <ArtifactsView overlay />}
        {view === 'extension' && route && (
          <ContribBoundary id={route.key}>
            <ContribRender render={route.render} />
          </ContribBoundary>
        )}
      </Panel>
    </Suspense>
  )
}
