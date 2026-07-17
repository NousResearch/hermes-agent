import { useCallback, useEffect, useState } from 'react'
import { useSearchParams } from 'react-router-dom'

import { Codicon } from '@/components/ui/codicon'
import { getMemoryContent } from '@/hermes'
import { useI18n } from '@/i18n'
import type { MemoryContentResponse } from '@/types/hermes'

import { Panel, PanelBlock, PanelBody, PanelDetail, PanelEmpty, PanelHeader, PanelList, PanelListRow } from '../overlays/panel'

type MemoryTab = 'memory' | 'user'

export function MemoryViewer({ onClose }: { onClose: () => void }) {
  const { t } = useI18n()
  const mt = t.memoryViewer
  const [searchParams] = useSearchParams()

  const [content, setContent] = useState<MemoryContentResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const [activeTab, setActiveTab] = useState<MemoryTab>(() => {
    const tab = searchParams.get('tab')

    return tab === 'user' ? 'user' : 'memory'
  })

  useEffect(() => {
    let cancelled = false
    setLoading(true)
    setError(null)
    getMemoryContent()
      .then(data => {
        if (!cancelled) {
          setContent(data)
          setLoading(false)
        }
      })
      .catch(err => {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : String(err))
          setLoading(false)
        }
      })

    return () => {
      cancelled = true
    }
  }, [])

  const onSelectMemory = useCallback(() => setActiveTab('memory'), [])
  const onSelectUser = useCallback(() => setActiveTab('user'), [])

  const activeContent = content ? (activeTab === 'memory' ? content.memory : content.user) : ''
  const hasContent = activeContent.trim().length > 0

  // Parse entries delimited by §
  const entries = hasContent
    ? activeContent.split('\n§\n').map((entry, i) => ({
        key: i,
        text: entry.trim()
      }))
    : []

  return (
    <Panel closeLabel={mt.close} onClose={onClose}>
      <PanelHeader
        subtitle={mt.subtitle}
        title={mt.title}
      />
      {error ? (
        <PanelEmpty description={error} icon="warning" title={mt.loadFailed} />
      ) : loading ? (
        <PanelEmpty icon="loading~spin" title={mt.loading} />
      ) : (
        <PanelBody>
          <PanelList>
            <PanelListRow
              active={activeTab === 'memory'}
              icon="notebook"
              meta={content?.memory ? `${content.memory.length}` : undefined}
              onSelect={onSelectMemory}
              rowKey="memory"
              title={mt.memoryTab}
            />
            <PanelListRow
              active={activeTab === 'user'}
              icon="account"
              meta={content?.user ? `${content.user.length}` : undefined}
              onSelect={onSelectUser}
              rowKey="user"
              title={mt.userTab}
            />
          </PanelList>
          <PanelDetail>
            {!hasContent ? (
              <PanelEmpty
                description={activeTab === 'memory' ? mt.memoryEmpty : mt.userEmpty}
                icon="lightbulb"
                title={mt.emptyTitle}
              />
            ) : (
              <div className="space-y-3">
                <div className="flex items-center gap-2 text-xs text-muted-foreground">
                  <Codicon name={activeTab === 'memory' ? 'notebook' : 'account'} size="0.875rem" />
                  <span>{activeTab === 'memory' ? mt.memoryTab : mt.userTab}</span>
                  <span className="text-muted-foreground/50">·</span>
                  <span>{mt.entryCount(entries.length)}</span>
                </div>
                {entries.map(entry => (
                  <PanelBlock key={entry.key}>{entry.text}</PanelBlock>
                ))}
              </div>
            )}
          </PanelDetail>
        </PanelBody>
      )}
    </Panel>
  )
}
