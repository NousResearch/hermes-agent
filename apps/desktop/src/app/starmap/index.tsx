import { useStore } from '@nanostores/react'
import { useEffect, useMemo, useState } from 'react'
import { useLocation, useNavigate } from 'react-router'

import { PageLoader } from '@/components/page-loader'
import { SegmentedControl } from '@/components/ui/segmented-control'
import { useI18n } from '@/i18n'
import { $starmapError, $starmapGeneration, $starmapGraph, $starmapLoading, $starmapOwner, loadStarmapGraph } from '@/store/starmap'
import type { StarmapGraph } from '@/types/hermes'

import { Panel, PanelEmpty } from '../overlays/panel'

import { MemoryList } from './memory-list'
import { StarMap } from './star-map'
import type { MemoryGraphSource } from './types'
import { VaultView } from './vault-view'

// Star map overlay: a top-down map of what Hermes has learned for a profile,
// over a radial time axis. Data is fetched on demand into the $starmap* atoms;
// the map itself lives in ./star-map. The chrome is owned by the map itself
// (timeline scrubber + legend float over the canvas), so there's no panel
// header here.
export function StarmapView({ onClose }: { onClose: () => void }) {
  const { locale, t } = useI18n()
  const graph = useStore($starmapGraph)
  const loading = useStore($starmapLoading)
  const error = useStore($starmapError)
  const owner = useStore($starmapOwner)
  const generation = useStore($starmapGeneration)
  const location = useLocation()
  const navigate = useNavigate()

  // A pasted share code populates the map with someone else's (or an exported)
  // graph, overriding the live profile scan. Cleared by "back to my map" and
  // whenever a fresh profile graph loads in.
  const [imported, setImported] = useState<StarmapGraph | null>(null)
  // The vault (the user's Obsidian notes) is the primary memory view; the
  // learned-skills star map and the flat list stay one click away.
  const viewParam = new URLSearchParams(location.search).get('view')
  const view = viewParam === 'list' || viewParam === 'graph' ? viewParam : 'vault'

  useEffect(() => {
    void loadStarmapGraph()
  }, [])

  // Drop a stale import when the underlying profile graph changes out from under it.
  useEffect(() => {
    setImported(null)
  }, [graph])

  const shown = imported ?? graph

  const source = useMemo<MemoryGraphSource | null>(() => {
    if (imported) {return { kind: 'imported', import_id: 'shared-map', graph: imported }}

    if (graph) {return { kind: 'owned', owner: owner ?? { connectionId: 'local', profile: 'default' }, generation, graph }}

    return null
  }, [generation, graph, imported, owner])

  const chooseView = (next: 'graph' | 'list' | 'vault') => {
    const params = new URLSearchParams(location.search)
    params.set('view', next)
    navigate(`${location.pathname}?${params.toString()}`, { replace: true })
  }

  const pl = locale === 'pl'

  const tabs = [
    { id: 'vault', label: pl ? 'Vault' : 'Vault' },
    { id: 'graph', label: pl ? 'Umiejętności' : 'Skills' },
    { id: 'list', label: t.starmap.memory }
  ] as const

  const learned =
    error ? (
      <PanelEmpty description={error} icon="warning" title={t.starmap.loadFailed} />
    ) : !shown && loading ? (
      <PageLoader aria-label={t.starmap.loading} className="min-h-0 flex-1" />
    ) : shown && shown.nodes.length === 0 && !imported ? (
      <PanelEmpty description={t.starmap.emptyDesc} icon="lightbulb" title={t.starmap.emptyTitle} />
    ) : shown ? (
      view === 'list' && source ? (
        <MemoryList source={source} />
      ) : (
        <StarMap graph={shown} imported={imported !== null} onImport={setImported} onResetMap={() => setImported(null)} source={source ?? undefined} />
      )
    ) : null

  return (
    <Panel closeLabel={t.starmap.close} onClose={onClose}>
      <div className="pointer-events-auto absolute right-14 top-2 z-30 [-webkit-app-region:no-drag]">
        <SegmentedControl onChange={chooseView} options={tabs} value={view} />
      </div>
      {view === 'vault' ? <VaultView /> : learned}
    </Panel>
  )
}
