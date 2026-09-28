import { useEffect, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { SegmentedControl } from '@/components/ui/segmented-control'
import { Tip } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'
import { openRouteTile } from '@/store/route-tiles'

import { Panel, PanelHeader } from '../overlays/panel'
import { AGENTS_ROUTE } from '../routes'

import { SessionOverview } from './sessions'
import { SpawnTreeView } from './spawn-tree'

export { SubagentRow } from './spawn-tree'

export function AgentsView({ onClose, embedded = false }: { onClose: () => void; embedded?: boolean }) {
  const { t } = useI18n()
  const [tab, setTab] = useState<'sessions' | 'tree'>('sessions')
  const [summary, setSummary] = useState('')
  useEffect(() => {
    if (embedded) {
      return
    }

    const previous = document.activeElement

    return () => {
      // Closing the short task restores its caller, never a background refresh.
      if (previous instanceof HTMLElement && previous.isConnected) {
        previous.focus({ preventScroll: true })
      }
    }
  }, [embedded])

  const content = (
    <>
      <PanelHeader
        actions={
          <>
            <SegmentedControl
              onChange={setTab}
              options={[
                { id: 'sessions', label: t.agents.sessionsTab },
                { id: 'tree', label: t.agents.treeTab }
              ]}
              value={tab}
            />
            {!embedded ? (
              <Tip label={t.agents.openAsTab}>
                <Button
                  aria-label={t.agents.openAsTab}
                  className="text-muted-foreground"
                  onClick={() => {
                    openRouteTile(AGENTS_ROUTE)
                    onClose()
                  }}
                  size="icon-sm"
                  variant="ghost"
                >
                  <Codicon name="layout-sidebar-right" size="0.875rem" />
                </Button>
              </Tip>
            ) : null}
          </>
        }
        subtitle={tab === 'sessions' ? summary || t.agents.overviewSubtitle : t.agents.subtitle}
        title={t.agents.title}
      />
      {tab === 'sessions' ? <SessionOverview onSummary={setSummary} /> : <SpawnTreeView />}
    </>
  )

  return embedded ? (
    <div className="flex h-full min-h-0 flex-col p-4">{content}</div>
  ) : (
    <Panel closeLabel={t.agents.close} onClose={onClose}>
      {content}
    </Panel>
  )
}
