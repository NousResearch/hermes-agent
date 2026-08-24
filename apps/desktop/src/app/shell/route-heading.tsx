import { useLocation } from 'react-router'

import { useI18n } from '@/i18n'
import { appViewForPath } from '@/app/routes'

/**
 * Screen-reader-only top-level heading for the desktop shell.
 *
 * Electron windows don't map to document-style heading rules the way a web
 * page does, but screen-reader users still benefit from a stable h1 naming
 * the current route (audit #38072, finding 5). The heading is visually hidden
 * so the visual shell is untouched; it tracks the routed `AppView` so the
 * announced name stays in sync with the screen.
 */
export function RouteHeading() {
  const { t } = useI18n()
  const location = useLocation()
  const view = appViewForPath(location.pathname)

  return <h1 className="sr-only">{t.shell.routeTitles[view]}</h1>
}
