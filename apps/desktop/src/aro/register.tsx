/**
 * ARO DESIGN SYSTEM — REGISTRATION (desktop port wiring)
 *
 * Side-effect module: mounts the `/aro-design` reference page and its
 * sidebar row through the host's own contribution registry, exactly the
 * way first-party surfaces and the Kanban plugin register theirs —
 *
 *   · ROUTES_AREA        → a full page in the workspace pane at `data.path`
 *   · SIDEBAR_NAV_AREA   → a data row in the sidebar's top nav
 *
 * Imported once from src/main.tsx (`import './aro/register'`). Because the
 * page is a contributed route, the rest of the app needs zero changes: the
 * router reserves the path, the sidebar renders + highlights the row, and
 * nothing about existing screens is touched.
 */

import { registry } from '@/contrib/registry'
import { ROUTES_AREA, SIDEBAR_NAV_AREA, type RouteContribution, type SidebarNavContribution } from '@/app/routes'

import { AroDesignSystemPage } from './DesignSystemAro'

/** Route of the Aro design-system reference page (contributed, one segment). */
export const ARO_DESIGN_ROUTE = '/aro-design'

registry.registerMany([
  {
    id: 'aro-design',
    area: ROUTES_AREA,
    title: 'Aro Design',
    data: { path: ARO_DESIGN_ROUTE } satisfies RouteContribution,
    render: () => <AroDesignSystemPage />,
  },
  {
    id: 'aro-design',
    area: SIDEBAR_NAV_AREA,
    order: 50,
    data: { codicon: 'symbol-color', label: 'Aro Design', path: ARO_DESIGN_ROUTE } satisfies SidebarNavContribution,
  },
])
