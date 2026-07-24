import { useCallback, useEffect, useMemo, useState } from 'react'
import { useSearchParams } from 'react-router-dom'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { getMemoryContent, saveMemoryContent } from '@/hermes'
import { useI18n } from '@/i18n'
import type { MemoryContentResponse, MemoryContentUpdate } from '@/types/hermes'

import {
  Panel,
  PanelBlock,
  PanelBody,
  PanelDetail,
  PanelEmpty,
  PanelHeader,
  PanelList,
  PanelListRow
} from '../overlays/panel'

type MemoryTab = 'memory' | 'user'

export function MemoryViewer({ onClose }: { onClose: () => void }) {
  const { t } = useI18n()
  const mt = t.memoryViewer
  const [searchParams] = useSearchParams()

  const [content, setContent] = useState<MemoryContentResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [saving, setSaving] = useState(false)
  const [editing, setEditing] = useState(false)

  const [activeTab, setActiveTab] = useState<MemoryTab>(() => {
    const tab = searchParams.get('tab')

    return tab === 'user' ? 'user' : 'memory'
  })

  const activeContent = useMemo(
    () => (content ? (activeTab === 'memory' ? content.memory : content.user) : ''),
    [activeTab, content]
  )

  const [draft, setDraft] = useState(activeContent)

  useEffect(() => {
    setDraft(activeContent)
  }, [activeContent, editing])

  const load = useCallback(() => {
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

  useEffect(() => {
    return load()
  }, [load])

  const onEdit = useCallback(() => setEditing(true), [])

  const onCancel = useCallback(() => {
    setEditing(false)
    setDraft(activeContent)
  }, [activeContent])

  const onSave = useCallback(async () => {
    if (!content) {
      return
    }

    setSaving(true)
    setError(null)

    const body: MemoryContentUpdate = {
      memory: activeTab === 'memory' ? draft : content.memory,
      user: activeTab === 'user' ? draft : content.user
    }

    try {
      await saveMemoryContent(body)
      setEditing(false)
      load()
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setSaving(false)
    }
  }, [activeTab, content, draft, load])

  const onChangeTab = useCallback(
    (tab: MemoryTab) => {
      if (editing && draft !== activeContent) {
        if (!window.confirm(mt.discardConfirm)) {
          return
        }
      }

      setActiveTab(tab)
      setEditing(false)
    },
    [activeContent, draft, editing, mt.discardConfirm]
  )

  const hasContent = activeContent.trim().length > 0

  // Parse entries delimited by §
  const entries = useMemo(
    () =>
      hasContent
        ? activeContent.split('\n§\n').map((entry, i) => ({
            key: i,
            text: entry.trim()
          }))
        : [],
    [activeContent, hasContent]
  )

  const dirty = draft !== activeContent

  const headerActions = editing ? (
    <>
      <Button disabled={saving} onClick={onCancel} size="xs" variant="text">
        {mt.cancel}
      </Button>
      <Button disabled={saving || !dirty} onClick={onSave} size="xs" variant="outline">
        {saving ? mt.saving : mt.save}
      </Button>
    </>
  ) : (
    <Button disabled={loading || saving} onClick={onEdit} size="xs" variant="outline">
      <Codicon name="edit" size="0.75rem" />
      {mt.edit}
    </Button>
  )

  return (
    <Panel closeLabel={mt.close} onClose={onClose}>
      <PanelHeader actions={headerActions} subtitle={mt.subtitle} title={mt.title} />
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
              onSelect={() => onChangeTab('memory')}
              rowKey="memory"
              title={mt.memoryTab}
            />
            <PanelListRow
              active={activeTab === 'user'}
              icon="account"
              meta={content?.user ? `${content.user.length}` : undefined}
              onSelect={() => onChangeTab('user')}
              rowKey="user"
              title={mt.userTab}
            />
          </PanelList>
          <PanelDetail>
            {editing ? (
              <div className="flex h-full flex-col gap-2">
                <div className="flex items-center gap-2 text-xs text-muted-foreground">
                  <Codicon name={activeTab === 'memory' ? 'notebook' : 'account'} size="0.875rem" />
                  <span>{activeTab === 'memory' ? mt.memoryTab : mt.userTab}</span>
                </div>
                <textarea
                  className="min-h-0 flex-1 resize-none rounded-md border border-(--ui-stroke-secondary) bg-(--ui-bg-quinary) p-3 font-mono text-[0.75rem] leading-relaxed text-(--ui-text-primary) outline-none focus-visible:border-ring focus-visible:ring-[0.1875rem] focus-visible:ring-ring/50"
                  onChange={event => setDraft(event.target.value)}
                  value={draft}
                />
              </div>
            ) : !hasContent ? (
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
