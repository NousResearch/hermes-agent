import { useStore } from '@nanostores/react'
import { atom, computed, onMount } from 'nanostores'
import type { ReactNode } from 'react'
import { useNavigate } from 'react-router'

import { bandPillReplacesStrip } from '@/app/redesign/portal-conversation-header'
import { findGroupOfPane } from '@/components/pane-shell/tree/model'
import {
  $hiddenTreePanes,
  $layoutTree,
  $narrowViewport,
  shownPanesInGroup,
  tabStripVisibleForGroup
} from '@/components/pane-shell/tree/store'
import { $routeCollapsedPanes } from '@/components/pane-shell/tree/store-route-collapse'
import { Button } from '@/components/ui/button'
import { Tip, TipKeybindLabel } from '@/components/ui/tooltip'
import { useContributions } from '@/contrib/react/use-contributions'
import { registry } from '@/contrib/registry'
import { useI18n } from '@/i18n'
import { recordAction } from '@/store/desktop-metrics'
import { requestSimplifiedNewSession } from '@/store/simplified-new-session'
import { $tabStripDefault } from '@/store/tabstrip-prefs'

import { $workspaceIsPage, NEW_CHAT_ROUTE } from '../routes'

import { TitlebarIcon } from './titlebar-icon'

interface WorkspaceTabChrome {
  tabTitle?: () => ReactNode
}

/** The pane registrations, as a store: which tabs a zone shows depends on them. */
const $paneRegistrations = atom(registry.getArea('panes'))

onMount($paneRegistrations, () => {
  $paneRegistrations.set(registry.getArea('panes'))

  return registry.subscribeArea('panes', () => $paneRegistrations.set(registry.getArea('panes')))
})

/** The main chat zone's front tab when the band pill stands for it, else null: once the zone shows
 *  its own strip (two or more tabs, unless the Tab Strip setting hides it), that strip rides the band
 *  as the tab pills and the lone pill would only repeat one of them. A store of one id, so a sash
 *  drag (which rewrites the tree's sizes every frame) never re-renders the band. */
export const $bandPillPane = computed(
  [$layoutTree, $paneRegistrations, $tabStripDefault, $hiddenTreePanes, $narrowViewport, $routeCollapsedPanes],
  tree => {
    const zone = tree ? findGroupOfPane(tree, 'workspace') : null

    if (!zone) {
      return 'workspace'
    }

    const shown = shownPanesInGroup(zone)

    if (tabStripVisibleForGroup(zone) && !bandPillReplacesStrip(shown)) {
      return null
    }

    return shown.includes(zone.active) ? zone.active : (shown[0] ?? 'workspace')
  }
)

/** The redesigned band's session tab (Figma 906:19481 + 906:19483): the front chat's title as a
 *  pill at the stage's leading edge, then "+" for a new session. Each pane already publishes its
 *  live title (stored title, or the draft's) — the same one its tab shows in a strip. */
export function PortalSessionTab() {
  const { t } = useI18n()
  const navigate = useNavigate()
  const panes = useContributions('panes')
  const page = useStore($workspaceIsPage)
  const paneId = useStore($bandPillPane)
  const pane = panes.find(candidate => candidate.id === paneId)

  if (!pane || page) {
    return null
  }

  const title = (pane.data as undefined | WorkspaceTabChrome)?.tabTitle?.() ?? pane.title
  const label = t.sidebar.nav['new-session']

  return (
    <div className="portal-session-tab" data-slot="portal-session-tab">
      {/* The widest thing mid-band and not interactive, so it drags the window like the band
          around it (its cluster is no-drag for the "+"). No tooltip: a drag region gets no hover. */}
      <span className="portal-session-tab-title [-webkit-app-region:drag]">{title}</span>
      <Tip label={<TipKeybindLabel actionId="session.new" text={label} />} placement="toolbar">
        <Button
          aria-label={label}
          className="portal-chrome-button"
          onClick={() => {
            recordAction('session.new', 'click')

            if (!requestSimplifiedNewSession()) {
              navigate(NEW_CHAT_ROUTE)
            }
          }}
          onPointerDown={event => event.stopPropagation()}
          size="icon"
          variant="ghost"
        >
          <TitlebarIcon name="add" />
        </Button>
      </Tip>
    </div>
  )
}
