import { useRef, useState } from 'react'
import { useNavigate } from 'react-router'

import { Button } from '@/components/ui/button'
import { Switch } from '@/components/ui/switch'
import { type ProfileScope, setToolsetEnabled } from '@/hermes'
import { useI18n } from '@/i18n'
import { queryClient } from '@/lib/query-client'
import { invalidateSlashCompletions } from '@/lib/slash-completion-cache'
import { notifyError } from '@/store/notifications'
import type { ToolsetInfo } from '@/types/hermes'

import { asText, toolNames, toolsetDisplayLabel } from '../../settings/helpers'
import { CapabilitySearch } from '../ui/capability-search'

import { ToolsetDetail } from './toolset-detail'
import { filteredToolsets, toolsetsQueryKey } from './toolsets-data'

interface AgentToolsetsProps {
  profile: ProfileScope
  toolsets: ToolsetInfo[]
  query: string
  onQueryChange: (query: string) => void
}

export function AgentToolsets({ profile, toolsets, query, onQueryChange }: AgentToolsetsProps) {
  const { t } = useI18n()
  const navigate = useNavigate()
  const writing = useRef(false)
  const [busy, setBusy] = useState(false)
  const [showAll, setShowAll] = useState(false)
  const visible = filteredToolsets(toolsets, query, {}, true)
  const on = visible.filter(toolset => toolset.enabled)
  const off = visible.filter(toolset => !toolset.enabled)

  const refresh = () => {
    void queryClient.invalidateQueries({ queryKey: toolsetsQueryKey(profile), exact: true })
  }

  async function toggle(toolset: ToolsetInfo, enabled: boolean) {
    if (writing.current) {
      return
    }

    writing.current = true
    setBusy(true)
    const key = toolsetsQueryKey(profile)

    try {
      await queryClient.cancelQueries({ queryKey: key, exact: true })
      queryClient.setQueryData<ToolsetInfo[]>(key, current =>
        current?.map(row => (row.name === toolset.name ? { ...row, enabled } : row))
      )
      await setToolsetEnabled(toolset.name, enabled, profile)
    } catch (error) {
      queryClient.setQueryData<ToolsetInfo[]>(key, current =>
        current?.map(row => (row.name === toolset.name ? { ...row, enabled: toolset.enabled } : row))
      )
      notifyError(error, t.skills.failedToUpdate(toolsetDisplayLabel(toolset)))
    } finally {
      writing.current = false
      setBusy(false)
      invalidateSlashCompletions()
      refresh()
    }
  }

  const rows = (items: ToolsetInfo[]) =>
    items.map(toolset => (
      <div className="flex items-start gap-3 py-3" key={toolset.name}>
        <ToolsetConfiguration onConfiguredChange={refresh} profile={profile} toolset={toolset} />
        {!toolset.configured && <span className="text-xs text-(--ui-text-tertiary)">{t.skills.needsKeys}</span>}
        <Switch
          aria-label={t.skills.toggleToolset(toolsetDisplayLabel(toolset), !toolset.enabled)}
          checked={toolset.enabled}
          disabled={busy}
          onCheckedChange={enabled => void toggle(toolset, enabled)}
        />
      </div>
    ))

  return (
    <div className="min-h-0 flex-1 overflow-y-auto">
      <div className="agent-capability-list">
        <div className="agent-capability-banner">
          <p>{t.skills.agentScopeHint}</p>
          <Button onClick={() => navigate('/capabilities?tab=toolsets')} variant="outline">
            {t.skills.openToolLibrary}
          </Button>
        </div>
        <CapabilitySearch
          onChange={onQueryChange}
          placeholder={t.skills.searchToolsets}
          scope={t.skills.tabToolsets}
          value={query}
        />
        <section aria-label={t.common.on}>
          <h3 className="text-sm text-(--ui-text-secondary)">
            {t.common.on} · {on.length}
          </h3>
          <div className="agent-capability-rows">{rows(on)}</div>
        </section>
        <section aria-label={t.common.off}>
          <h3 className="text-sm text-(--ui-text-secondary)">
            {t.common.off} · {off.length}
          </h3>
          <div className="agent-capability-rows">{rows(showAll || query ? off : off.slice(0, 9))}</div>
          {!showAll && !query && off.length > 9 && (
            <Button onClick={() => setShowAll(true)} variant="textStrong">
              {t.skills.showAllToolsets(off.length)}
            </Button>
          )}
        </section>
      </div>
    </div>
  )
}

function ToolsetConfiguration({
  profile,
  toolset,
  onConfiguredChange
}: {
  profile: ProfileScope
  toolset: ToolsetInfo
  onConfiguredChange: () => void
}) {
  const { t } = useI18n()
  const [open, setOpen] = useState(false)

  return (
    <details className="group/toolset min-w-0 flex-1" onToggle={event => setOpen(event.currentTarget.open)}>
      <summary className="cursor-pointer text-sm">
        {toolsetDisplayLabel(toolset)}
        <span className="ml-2 text-xs text-(--ui-text-tertiary)">
          {t.catalog.toolsets.toolCount(toolNames(toolset).length)}
        </span>
        <span className="mt-1 line-clamp-2 text-xs text-(--ui-text-tertiary) group-open/toolset:line-clamp-none">
          {asText(toolset.description)}
        </span>
      </summary>
      {open && (
        <div className="grid gap-4 pt-4">
          <ul className="flex flex-wrap gap-2 text-xs text-(--ui-text-secondary)">
            {toolNames(toolset).map(name => (
              <li className="break-all font-mono" key={name}>
                {name}
              </li>
            ))}
          </ul>
          <p className="text-xs text-(--ui-text-tertiary)">{t.skills.toolSecretsHint}</p>
          <ToolsetDetail onConfiguredChange={onConfiguredChange} profile={profile} toolCalls={{}} toolset={toolset} />
        </div>
      )}
    </details>
  )
}
