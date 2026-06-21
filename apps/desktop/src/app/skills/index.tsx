import type * as React from 'react'
import { useCallback, useEffect, useMemo, useState } from 'react'

import { PageLoader } from '@/components/page-loader'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { Switch } from '@/components/ui/switch'
import { TextTab, TextTabMeta } from '@/components/ui/text-tab'
import { getSkills, getToolsets, toggleSkill, toggleToolset } from '@/hermes'
import { useI18n } from '@/i18n'
import { cn } from '@/lib/utils'
import { notify, notifyError } from '@/store/notifications'
import type { SkillInfo, ToolsetInfo } from '@/types/hermes'

import { useRefreshHotkey } from '../hooks/use-refresh-hotkey'
import { useRouteEnumParam } from '../hooks/use-route-enum-param'
import { PAGE_INSET_X } from '../layout-constants'
import { PageSearchShell } from '../page-search-shell'
import { asText, includesQuery, prettyName, toolNames, toolsetDisplayLabel } from '../settings/helpers'
import { ToolsetConfigPanel } from '../settings/toolset-config-panel'
import type { SetStatusbarItemGroup } from '../shell/statusbar-controls'

const SKILLS_MODES = ['skills', 'toolsets'] as const
type SkillsMode = (typeof SKILLS_MODES)[number]

const SKILL_SOURCE_FILTERS = ['all', 'custom', 'bundled'] as const
type SkillSourceFilter = (typeof SKILL_SOURCE_FILTERS)[number]

function skillSourceFor(skill: SkillInfo): Exclude<SkillInfo['source'], 'hub' | undefined> | 'hub' {
  if (skill.source === 'bundled' || skill.source === 'hub') {
    return skill.source
  }
  return 'custom'
}

function matchesSkillSource(skill: SkillInfo, sourceFilter: SkillSourceFilter): boolean {
  if (sourceFilter === 'all') {
    return true
  }
  return skillSourceFor(skill) === sourceFilter
}

function skillSourceLabel(skill: SkillInfo): string {
  const source = skillSourceFor(skill)
  if (source === 'bundled') {
    return 'Bundled'
  }
  if (source === 'hub') {
    return 'Hub'
  }
  return 'Custom'
}

function categoryFor(skill: SkillInfo): string {
  return asText(skill.category) || 'general'
}

function filteredSkills(
  skills: SkillInfo[],
  query: string,
  category: string | null,
  sourceFilter: SkillSourceFilter
): SkillInfo[] {
  const q = query.trim().toLowerCase()

  return skills
    .filter(skill => {
      if (!matchesSkillSource(skill, sourceFilter)) {
        return false
      }

      if (category && categoryFor(skill) !== category) {
        return false
      }

      if (!q) {
        return true
      }

      return includesQuery(skill.name, q) || includesQuery(skill.description, q) || includesQuery(skill.category, q)
    })
    .sort((a, b) => asText(a.name).localeCompare(asText(b.name)))
}

function filteredToolsets(toolsets: ToolsetInfo[], query: string): ToolsetInfo[] {
  const q = query.trim().toLowerCase()

  return toolsets
    .filter(toolset => {
      if (!q) {
        return true
      }

      const label = toolsetDisplayLabel(toolset)

      return (
        includesQuery(toolset.name, q) ||
        includesQuery(label, q) ||
        includesQuery(toolset.label, q) ||
        includesQuery(toolset.description, q) ||
        toolNames(toolset).some(name => includesQuery(name, q))
      )
    })
    .sort((a, b) => toolsetDisplayLabel(a).localeCompare(toolsetDisplayLabel(b)))
}

interface SkillsViewProps extends React.ComponentProps<'section'> {
  setStatusbarItemGroup?: SetStatusbarItemGroup
}

export function SkillsView({ setStatusbarItemGroup: _setStatusbarItemGroup, ...props }: SkillsViewProps) {
  const { t } = useI18n()
  const [mode, setMode] = useRouteEnumParam('tab', SKILLS_MODES, 'skills')

  const [query, setQuery] = useState('')
  const [skills, setSkills] = useState<SkillInfo[] | null>(null)
  const [toolsets, setToolsets] = useState<ToolsetInfo[] | null>(null)
  const [activeCategory, setActiveCategory] = useState<string | null>(null)
  const [sourceFilter, setSourceFilter] = useState<SkillSourceFilter>('all')
  const [refreshing, setRefreshing] = useState(false)
  const [savingSkill, setSavingSkill] = useState<string | null>(null)
  const [savingToolset, setSavingToolset] = useState<string | null>(null)
  const [expandedToolset, setExpandedToolset] = useState<string | null>(null)

  const skillsForSource = useMemo(
    () => (skills ? skills.filter(skill => matchesSkillSource(skill, sourceFilter)) : []),
    [skills, sourceFilter]
  )

  const refreshCapabilities = useCallback(async () => {
    setRefreshing(true)

    try {
      const [nextSkills, nextToolsets] = await Promise.all([getSkills(), getToolsets()])
      setSkills(nextSkills)
      setToolsets(nextToolsets)
    } catch (err) {
      notifyError(err, t.skills.skillsLoadFailed)
    } finally {
      setRefreshing(false)
    }
  }, [t])

  const refreshToolsets = useCallback(() => {
    getToolsets()
      .then(setToolsets)
      .catch(err => notifyError(err, t.skills.toolsetsRefreshFailed))
  }, [t])

  useRefreshHotkey(refreshCapabilities)

  useEffect(() => {
    void refreshCapabilities()
  }, [refreshCapabilities])

  const categories = useMemo(() => {
    if (!skillsForSource) {
      return []
    }

    const counts = new Map<string, number>()

    for (const skill of skillsForSource) {
      const key = categoryFor(skill)
      counts.set(key, (counts.get(key) || 0) + 1)
    }

    return Array.from(counts.entries())
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([key, count]) => ({ key, count }))
  }, [skillsForSource])

  const visibleSkills = useMemo(
    () => (skills ? filteredSkills(skills, query, mode === 'skills' ? activeCategory : null, sourceFilter) : []),
    [activeCategory, mode, query, skills, sourceFilter]
  )

  const visibleToolsets = useMemo(() => (toolsets ? filteredToolsets(toolsets, query) : []), [query, toolsets])

  const skillGroups = useMemo(() => {
    const groups = new Map<string, SkillInfo[]>()

    for (const skill of visibleSkills) {
      const key = categoryFor(skill)
      groups.set(key, [...(groups.get(key) || []), skill])
    }

    return Array.from(groups.entries()).sort(([a], [b]) => a.localeCompare(b))
  }, [visibleSkills])

  const totalSkills = skills?.length || 0
  const customSkills = skills?.filter(skill => skillSourceFor(skill) === 'custom').length || 0
  const bundledSkills = skills?.filter(skill => skillSourceFor(skill) === 'bundled').length || 0
  const enabledToolsets = toolsets?.filter(toolset => toolset.enabled).length || 0

  async function handleToggleSkill(skill: SkillInfo, enabled: boolean) {
    setSavingSkill(skill.name)

    try {
      await toggleSkill(skill.name, enabled)
      setSkills(current => current?.map(row => (row.name === skill.name ? { ...row, enabled } : row)) ?? current)
      notify({
        kind: 'success',
        title: enabled ? t.skills.skillEnabled : t.skills.skillDisabled,
        message: t.skills.appliesToNewSessions(skill.name)
      })
    } catch (err) {
      notifyError(err, t.skills.failedToUpdate(skill.name))
    } finally {
      setSavingSkill(null)
    }
  }

  async function handleToggleToolset(toolset: ToolsetInfo, enabled: boolean) {
    setSavingToolset(toolset.name)

    try {
      await toggleToolset(toolset.name, enabled)
      setToolsets(
        current =>
          current?.map(row => (row.name === toolset.name ? { ...row, enabled, available: enabled } : row)) ?? current
      )
      notify({
        kind: 'success',
        title: enabled ? t.skills.toolsetEnabled : t.skills.toolsetDisabled,
        message: t.skills.appliesToNewSessions(toolsetDisplayLabel(toolset))
      })
    } catch (err) {
      notifyError(err, t.skills.failedToUpdate(toolsetDisplayLabel(toolset)))
    } finally {
      setSavingToolset(null)
    }
  }

  return (
    <PageSearchShell
      {...props}
      filters={
        mode === 'skills' && categories.length > 0 ? (
          <>
            <TextTab
              active={sourceFilter === 'all' && activeCategory === null}
              onClick={() => {
                setSourceFilter('all')
                setActiveCategory(null)
              }}
            >
              {t.skills.all} <TextTabMeta>{totalSkills}</TextTabMeta>
            </TextTab>
            <TextTab
              active={sourceFilter === 'custom'}
              onClick={() => {
                setSourceFilter('custom')
                setActiveCategory(null)
              }}
            >
              Custom <TextTabMeta>{customSkills}</TextTabMeta>
            </TextTab>
            <TextTab
              active={sourceFilter === 'bundled'}
              onClick={() => {
                setSourceFilter('bundled')
                setActiveCategory(null)
              }}
            >
              Bundled <TextTabMeta>{bundledSkills}</TextTabMeta>
            </TextTab>
            {categories.map(category => (
              <TextTab
                active={activeCategory === category.key}
                key={category.key}
                onClick={() => setActiveCategory(activeCategory === category.key ? null : category.key)}
              >
                {prettyName(category.key)} <TextTabMeta>{category.count}</TextTabMeta>
              </TextTab>
            ))}
          </>
        ) : undefined
      }
      onSearchChange={setQuery}
      searchHidden={mode === 'skills' ? (skills?.length ?? 0) === 0 : (toolsets?.length ?? 0) === 0}
      searchPlaceholder={mode === 'skills' ? t.skills.searchSkills : t.skills.searchToolsets}
      searchTrailingAction={
        <Button
          aria-label={refreshing ? t.skills.refreshing : t.skills.refresh}
          className="text-(--ui-text-tertiary) hover:bg-transparent hover:text-foreground"
          disabled={refreshing}
          onClick={() => void refreshCapabilities()}
          size="icon-xs"
          title={refreshing ? t.skills.refreshing : t.skills.refresh}
          type="button"
          variant="ghost"
        >
          <Codicon name="refresh" size="0.875rem" spinning={refreshing} />
        </Button>
      }
      searchValue={query}
      tabs={
        <>
          <TextTab active={mode === 'skills'} onClick={() => setMode('skills')}>
            {t.skills.tabSkills}
          </TextTab>
          <TextTab active={mode === 'toolsets'} onClick={() => setMode('toolsets')}>
            {t.skills.tabToolsets}
          </TextTab>
        </>
      }
    >
      {!skills || !toolsets ? (
        <PageLoader label={t.skills.loading} />
      ) : mode === 'skills' ? (
        <div className={cn('h-full overflow-y-auto py-3', PAGE_INSET_X)}>
          {visibleSkills.length === 0 ? (
            <EmptyState description={t.skills.noSkillsDesc} title={t.skills.noSkillsTitle} />
          ) : (
            <div className="space-y-4">
              {skillGroups.map(([category, list]) => (
                <div className="space-y-1.5" key={category}>
                  {activeCategory === null && (
                    <div className="text-[0.68rem] font-semibold uppercase tracking-[0.12em] text-muted-foreground">
                      {prettyName(category)}
                    </div>
                  )}
                  <div>
                    {list.map(skill => (
                      <div
                        className="grid gap-3 px-0 py-2.5 sm:grid-cols-[minmax(0,1fr)_auto] sm:items-center"
                        key={skill.name}
                      >
                        <div className="min-w-0">
                          <div className="flex items-center gap-2">
                            <div className="truncate text-sm font-medium">{skill.name}</div>
                            <Badge
                              className={cn(
                                'shrink-0',
                                skillSourceFor(skill) === 'custom'
                                  ? 'bg-(--ui-accent-subtle) text-(--ui-accent-strong)'
                                  : 'bg-(--ui-bg-quinary) text-(--ui-text-tertiary)'
                              )}
                            >
                              {skillSourceLabel(skill)}
                            </Badge>
                          </div>
                          <p className="mt-0.5 text-xs text-muted-foreground">
                            {asText(skill.description) || t.skills.noDescription}
                          </p>
                        </div>
                        <Switch
                          checked={skill.enabled}
                          disabled={savingSkill === skill.name}
                          onCheckedChange={checked => void handleToggleSkill(skill, checked)}
                        />
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      ) : (
        <div className={cn('h-full overflow-y-auto py-3', PAGE_INSET_X)}>
          {visibleToolsets.length === 0 ? (
            <EmptyState description={t.skills.noToolsetsDesc} title={t.skills.noToolsetsTitle} />
          ) : (
            <div className="space-y-2">
              <div className="text-xs text-muted-foreground">
                {t.skills.toolsetsEnabled(enabledToolsets, toolsets.length)}
              </div>
              <div>
                {visibleToolsets.map(toolset => {
                  const tools = toolNames(toolset)
                  const label = toolsetDisplayLabel(toolset)
                  const expanded = expandedToolset === toolset.name

                  return (
                    <div className="px-0 py-2.5" key={toolset.name}>
                      <div className="flex items-center justify-between gap-2">
                        <div className="truncate text-sm font-medium">{label}</div>
                        <div className="flex shrink-0 items-center gap-1.5">
                          <button
                            aria-expanded={expanded}
                            aria-label={t.skills.configureToolset(label)}
                            className="cursor-pointer rounded-full outline-none focus-visible:ring-2 focus-visible:ring-ring/50"
                            onClick={() =>
                              setExpandedToolset(current => (current === toolset.name ? null : toolset.name))
                            }
                            type="button"
                          >
                            <StatusPill active={toolset.configured}>
                              {toolset.configured ? t.skills.configured : t.skills.needsKeys}
                            </StatusPill>
                          </button>
                          <Switch
                            aria-label={t.skills.toggleToolset(label)}
                            checked={toolset.enabled}
                            disabled={savingToolset === toolset.name}
                            onCheckedChange={checked => void handleToggleToolset(toolset, checked)}
                          />
                        </div>
                      </div>
                      <p className="mt-1 text-xs text-muted-foreground">
                        {asText(toolset.description) || t.skills.noDescription}
                      </p>
                      {tools.length > 0 && (
                        <div className="mt-2 flex flex-wrap gap-1">
                          {tools.map(name => (
                            <span
                              className="rounded-md bg-(--ui-bg-quinary) px-1.5 py-0.5 font-mono text-[0.65rem] text-(--ui-text-tertiary)"
                              key={name}
                            >
                              {name}
                            </span>
                          ))}
                        </div>
                      )}
                      {expanded && <ToolsetConfigPanel onConfiguredChange={refreshToolsets} toolset={toolset.name} />}
                    </div>
                  )
                })}
              </div>
            </div>
          )}
        </div>
      )}
    </PageSearchShell>
  )
}

function StatusPill({ active, children }: { active: boolean; children: string }) {
  return (
    <Badge
      className={
        active ? 'bg-(--ui-bg-tertiary) text-(--ui-text-secondary)' : 'bg-(--ui-bg-quinary) text-(--ui-text-tertiary)'
      }
    >
      {children}
    </Badge>
  )
}

function EmptyState({ title, description }: { title: string; description: string }) {
  return (
    <div className="grid min-h-52 place-items-center text-center">
      <div>
        <div className="text-sm font-medium">{title}</div>
        <div className="mt-1 text-xs text-muted-foreground">{description}</div>
      </div>
    </div>
  )
}
