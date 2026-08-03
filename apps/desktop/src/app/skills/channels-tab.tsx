import { useStore } from '@nanostores/react'
import { useQuery } from '@tanstack/react-query'
import type { Dispatch, SetStateAction } from 'react'
import { useEffect, useMemo, useRef, useState } from 'react'

import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { RowButton } from '@/components/ui/row-button'
import { SegmentedControl } from '@/components/ui/segmented-control'
import { Switch } from '@/components/ui/switch'
import { getChannelCapabilities, updateChannelCapabilities } from '@/hermes'
import { useI18n } from '@/i18n'
import { AlertTriangle, Lock } from '@/lib/icons'
import { queryClient } from '@/lib/query-client'
import { cn } from '@/lib/utils'
import { notify, notifyError } from '@/store/notifications'
import { $activeGatewayProfile, normalizeProfileKey } from '@/store/profile'
import type { ChannelMcpMode } from '@/types/hermes'

import { useOnProfileSwitch } from '../hooks/use-on-profile-switch'
import { DetailColumn, ListColumn, MasterDetail, ToolChip } from '../master-detail'

const CHANNEL_CAPABILITIES_QUERY_KEY = ['channel-capabilities'] as const
const HIGH_IMPACT = new Set(['terminal', 'file', 'code_execution', 'computer_use', 'delegation', 'cronjob'])

interface ChannelsTabProps {
  query: string
}

export function ChannelsTab({ query }: ChannelsTabProps) {
  const { t } = useI18n()
  const activeProfile = useStore($activeGatewayProfile)
  const profileKey = normalizeProfileKey(activeProfile)

  const { data: channels, isError, isLoading } = useQuery({
    queryKey: [...CHANNEL_CAPABILITIES_QUERY_KEY, profileKey],
    queryFn: getChannelCapabilities,
    staleTime: 0
  })

  const [selected, setSelected] = useState<string | null>(null)
  const [enabledToolsets, setEnabledToolsets] = useState<Set<string>>(new Set())
  const [mcpMode, setMcpMode] = useState<ChannelMcpMode>('all')
  const [mcpServers, setMcpServers] = useState<Set<string>>(new Set())
  const [savingProfile, setSavingProfile] = useState<string | null>(null)
  const saveGeneration = useRef(0)
  const activeProfileRef = useRef(profileKey)
  activeProfileRef.current = profileKey
  const saving = savingProfile === profileKey

  // A profile swap changes the backend authority. Drop every local draft
  // before another click can route profile A's selections to profile B.
  useOnProfileSwitch(() => {
    saveGeneration.current += 1
    setSelected(null)
    setEnabledToolsets(new Set())
    setMcpMode('all')
    setMcpServers(new Set())
    setSavingProfile(null)
  })

  const visibleChannels = useMemo(() => {
    const needle = query.trim().toLowerCase()
    const rows = channels ?? []

    if (!needle) {
      return rows
    }

    return rows.filter(
      channel =>
        channel.label.toLowerCase().includes(needle) ||
        channel.platform.toLowerCase().includes(needle) ||
        channel.toolsets.some(
          toolset =>
            toolset.label.toLowerCase().includes(needle) ||
            toolset.name.toLowerCase().includes(needle) ||
            toolset.tools.some(tool => tool.toLowerCase().includes(needle))
        )
    )
  }, [channels, query])

  // Filtering only changes the list. It must not switch the detail pane and
  // overwrite an unsaved draft for the currently selected channel.
  const active = useMemo(
    () =>
      channels?.find(channel => channel.platform === selected) ??
      channels?.[0] ??
      null,
    [channels, selected]
  )

  useEffect(() => {
    if (!active) {
      return
    }

    setEnabledToolsets(new Set(active.toolsets.filter(toolset => toolset.enabled).map(toolset => toolset.name)))
    setMcpMode(active.mcp.mode)
    setMcpServers(new Set(active.mcp.selected))
  }, [active])

  const updateSet = (
    setter: Dispatch<SetStateAction<Set<string>>>,
    name: string,
    enabled: boolean
  ) =>
    setter(current => {
      const next = new Set(current)

      if (enabled) {
        next.add(name)
      } else {
        next.delete(name)
      }

      return next
    })

  const save = async () => {
    if (!active || saving) {
      return
    }

    const requestProfile = profileKey
    const requestGeneration = ++saveGeneration.current
    setSavingProfile(requestProfile)

    const isCurrentRequest = () =>
      saveGeneration.current === requestGeneration &&
      activeProfileRef.current === requestProfile

    try {
      await updateChannelCapabilities(active.platform, {
        toolsets: [...enabledToolsets].sort(),
        mcp_mode: mcpMode,
        mcp_servers: mcpMode === 'allowlist' ? [...mcpServers].sort() : []
      })

      if (!isCurrentRequest()) {
        return
      }

      await queryClient.invalidateQueries({
        queryKey: [...CHANNEL_CAPABILITIES_QUERY_KEY, profileKey]
      })
      notify({
        kind: 'success',
        title: t.skills.channels.savedTitle,
        message: t.skills.channels.savedMessage(active.label)
      })
    } catch (error) {
      if (isCurrentRequest()) {
        notifyError(error, t.skills.channels.saveFailed(active.label))
      }
    } finally {

      if (isCurrentRequest()) {
        setSavingProfile(null)
      }
    }
  }

  if (isLoading) {
    return <div className="flex h-full items-center justify-center text-xs text-muted-foreground">{t.skills.loading}</div>
  }

  if (isError || !channels) {
    return (
      <div className="flex h-full items-center justify-center text-xs text-muted-foreground">
        {t.skills.channels.loadFailed}
      </div>
    )
  }

  return (
    <MasterDetail>
      <ListColumn>
        {visibleChannels.map(channel => (
          <RowButton
            className={cn(
              'flex w-full items-center gap-2 rounded-[4px] px-2.5 py-2 text-left transition-colors',
              active?.platform === channel.platform
                ? 'bg-(--ui-control-active-background) text-foreground'
                : 'text-muted-foreground hover:bg-(--chrome-action-hover) hover:text-foreground'
            )}
            disabled={saving}
            key={channel.platform}
            onClick={() => setSelected(channel.platform)}
          >
            <span className="min-w-0 flex-1 truncate text-xs font-medium">{channel.label}</span>
          </RowButton>
        ))}
      </ListColumn>

      <DetailColumn
        actionBar={
          active && (
            <>
              <p className="mr-auto text-[0.65rem] text-muted-foreground">{t.skills.changesApplyNewSessions}</p>
              <Button disabled={saving} onClick={() => void save()} size="xs">
                {saving ? t.common.saving : t.skills.channels.save}
              </Button>
            </>
          )
        }
      >
        {active && (
          <>
            <header>
              <div className="flex flex-wrap items-center gap-2">
                <h3 className="text-[0.9375rem] font-semibold tracking-tight">{active.label}</h3>
                <Badge variant="muted">
                  {active.explicit ? t.skills.channels.customBoundary : t.skills.channels.inheritedDefaults}
                </Badge>
              </div>
              <p className="mt-1 text-[length:var(--conversation-caption-font-size)] leading-(--conversation-caption-line-height) text-(--ui-text-tertiary)">
                {t.skills.channels.description}
              </p>
            </header>

            <section>
              <div className="mb-2 flex items-center justify-between gap-3">
                <h4 className="text-xs font-semibold">{t.skills.channels.toolsets}</h4>
                <span className="text-[0.65rem] tabular-nums text-(--ui-text-quaternary)">
                  {enabledToolsets.size}/{active.toolsets.length}
                </span>
              </div>
              <div className="space-y-3">
                {active.toolsets.map(toolset => (
                  <div className="flex items-start gap-3" key={toolset.name}>
                    <Switch
                      aria-label={t.skills.channels.toggleToolset(toolset.label)}
                      checked={enabledToolsets.has(toolset.name)}
                      disabled={saving || (enabledToolsets.size === 1 && enabledToolsets.has(toolset.name))}
                      onCheckedChange={checked => updateSet(setEnabledToolsets, toolset.name, checked)}
                      size="xs"
                    />
                    <div className="min-w-0 flex-1">
                      <div className="flex flex-wrap items-center gap-1.5">
                        <span className="text-xs font-medium">{toolset.label}</span>
                        {HIGH_IMPACT.has(toolset.name) && (
                          <Badge variant="warn">
                            <AlertTriangle />
                            {t.skills.channels.highImpact}
                          </Badge>
                        )}
                      </div>
                      <p className="mt-0.5 text-[0.68rem] text-(--ui-text-tertiary)">{toolset.description}</p>
                      {toolset.tools.length > 0 && (
                        <div className="mt-1.5 flex flex-wrap gap-1">
                          {toolset.tools.map(tool => (
                            <ToolChip key={tool}>{tool}</ToolChip>
                          ))}
                        </div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </section>

            {active.implicit_toolsets.length > 0 && (
              <section>
                <h4 className="mb-2 text-xs font-semibold">{t.skills.channels.required}</h4>
                <div className="flex flex-wrap gap-1">
                  {active.implicit_toolsets.map(toolset => (
                    <ToolChip key={toolset.name}>{toolset.label}</ToolChip>
                  ))}
                </div>
              </section>
            )}

            <section>
              <div className="mb-2 flex items-center gap-1.5">
                <Lock className="size-3.5 text-(--ui-text-tertiary)" />
                <h4 className="text-xs font-semibold">{t.skills.channels.mcpAccess}</h4>
              </div>
              <SegmentedControl
                disabled={saving}
                onChange={setMcpMode}
                options={[
                  { id: 'all', label: t.skills.channels.mcpAll },
                  { id: 'none', label: t.skills.channels.mcpNone },
                  { id: 'allowlist', label: t.skills.channels.mcpSelected }
                ]}
                value={mcpMode}
              />
              {mcpMode === 'allowlist' && (
                <div className="mt-3 space-y-2">
                  {active.mcp.available.length === 0 ? (
                    <p className="text-[0.68rem] text-(--ui-text-tertiary)">{t.skills.channels.noMcp}</p>
                  ) : (
                    active.mcp.available.map(server => (
                      <label className="flex items-center gap-2 text-xs" key={server}>
                        <Switch
                          checked={mcpServers.has(server)}
                          disabled={saving}
                          onCheckedChange={checked => updateSet(setMcpServers, server, checked)}
                          size="xs"
                        />
                        <span className="font-mono">{server}</span>
                      </label>
                    ))
                  )}
                </div>
              )}
            </section>
          </>
        )}
      </DetailColumn>
    </MasterDetail>
  )
}
