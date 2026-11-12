import { useId, useRef, useState } from 'react'
import { useNavigate } from 'react-router'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { Switch } from '@/components/ui/switch'
import { type ProfileScope, setToolsetEnabled } from '@/hermes'
import { useI18n } from '@/i18n'
import { queryClient } from '@/lib/query-client'
import { invalidateSlashCompletions } from '@/lib/slash-completion-cache'
import { notifyError } from '@/store/notifications'
import type { ToolsetInfo } from '@/types/hermes'

import { asText, toolNames, toolsetDisplayLabel } from '../../settings/helpers'
import {
  AgentCapabilityBanner,
  AgentCapabilityMore,
  AgentCapabilityRow,
  AgentCapabilitySection
} from '../ui/agent-capability'
import { useAgentCapabilityCopy } from '../ui/agent-capability-copy'
import { CapabilitySearch } from '../ui/capability-search'

import { ToolsetDetail } from './toolset-detail'
import { filteredToolsets, toolsetsQueryKey } from './toolsets-data'

/** Off toolsets the Off card lists before "Show all" (895:14956). */
const OFF_PREVIEW = 9

interface AgentToolsetsProps {
  profile: ProfileScope
  toolsets: ToolsetInfo[]
  query: string
  onQueryChange: (query: string) => void
}

/** One agent's toolsets (Figma 895:13886): On, then Off; each row's switch writes this profile. */
export function AgentToolsets({ profile, toolsets, query, onQueryChange }: AgentToolsetsProps) {
  const { t } = useI18n()
  const copy = useAgentCapabilityCopy()
  const navigate = useNavigate()
  const writing = useRef(false)
  const [busy, setBusy] = useState(false)
  const [showAll, setShowAll] = useState(false)
  const visible = filteredToolsets(toolsets, query, {}, true)
  const on = visible.filter(toolset => toolset.enabled)
  const off = visible.filter(toolset => !toolset.enabled)
  const offShown = showAll || query ? off : off.slice(0, OFF_PREVIEW)

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
      <AgentToolsetRow
        busy={busy}
        key={toolset.name}
        onConfiguredChange={refresh}
        onToggle={enabled => void toggle(toolset, enabled)}
        profile={profile}
        toolset={toolset}
      />
    ))

  return (
    <div className="agent-capability-list">
      <AgentCapabilityBanner
        action={
          <Button onClick={() => navigate('/capabilities?tab=toolsets')} variant="outline">
            {t.skills.openToolLibrary}
          </Button>
        }
        description={copy.tools.bannerBody}
        title={copy.tools.bannerTitle}
      />
      <CapabilitySearch
        onChange={onQueryChange}
        placeholder={t.skills.searchToolsets}
        scope={t.skills.tabToolsets}
        value={query}
      />
      {on.length > 0 && (
        <AgentCapabilitySection description={copy.tools.onDesc} title={t.common.on}>
          {rows(on)}
        </AgentCapabilitySection>
      )}
      {off.length > 0 && (
        <AgentCapabilitySection
          description={copy.tools.offDesc(offShown.length, off.length)}
          footer={
            offShown.length < off.length && (
              <AgentCapabilityMore label={copy.showAll(off.length)} onClick={() => setShowAll(true)} />
            )
          }
          title={t.common.off}
        >
          {rows(offShown)}
        </AgentCapabilitySection>
      )}
    </div>
  )
}

interface AgentToolsetRowProps {
  profile: ProfileScope
  toolset: ToolsetInfo
  busy: boolean
  onToggle: (enabled: boolean) => void
  onConfiguredChange: () => void
}

/** "2 tools ⌄" opens the row to its tool keys and the toolset's own keys/provider settings. */
function AgentToolsetRow({ profile, toolset, busy, onToggle, onConfiguredChange }: AgentToolsetRowProps) {
  const { t } = useI18n()
  const copy = useAgentCapabilityCopy()
  const [open, setOpen] = useState(false)
  const detailId = useId()
  const label = toolsetDisplayLabel(toolset)
  const tools = toolNames(toolset)
  const expander = open ? copy.less : t.catalog.toolsets.toolCount(tools.length)

  return (
    <AgentCapabilityRow
      below={
        open && (
          <div className="agent-capability-detail" id={detailId}>
            <ul aria-label={label} className="agent-capability-tags">
              {tools.map(name => (
                <li key={name}>{name}</li>
              ))}
            </ul>
            <p className="agent-capability-meta">{t.skills.toolSecretsHint}</p>
            <ToolsetDetail onConfiguredChange={onConfiguredChange} profile={profile} toolCalls={{}} toolset={toolset} />
          </div>
        )
      }
      controls={
        <>
          {!toolset.configured && <span className="agent-capability-meta">{t.skills.needsKeys}</span>}
          <button
            aria-controls={open ? detailId : undefined}
            aria-expanded={open}
            aria-label={`${label}: ${expander}`}
            className="agent-capability-expander"
            onClick={() => setOpen(value => !value)}
            type="button"
          >
            {expander}
            <Codicon aria-hidden name={open ? 'chevron-up' : 'chevron-down'} size="0.75rem" />
          </button>
          <Switch
            aria-label={t.skills.toggleToolset(label, !toolset.enabled)}
            checked={toolset.enabled}
            disabled={busy}
            onCheckedChange={onToggle}
          />
        </>
      }
      description={asText(toolset.description)}
      expanded={open}
      name={label}
    />
  )
}
