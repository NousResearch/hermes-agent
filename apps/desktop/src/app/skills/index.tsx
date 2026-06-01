import type * as React from 'react'
import { useCallback, useEffect, useMemo, useState } from 'react'

import { PageLoader } from '@/components/page-loader'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { Switch } from '@/components/ui/switch'
import { TextTab, TextTabMeta } from '@/components/ui/text-tab'
import { getSkills, getToolsets, toggleSkill } from '@/hermes'
import { cn } from '@/lib/utils'
import { notify, notifyError } from '@/store/notifications'
import type { SkillInfo, ToolsetInfo } from '@/types/hermes'

import { useRouteEnumParam } from '../hooks/use-route-enum-param'
import { PageSearchShell } from '../page-search-shell'
import { asText, includesQuery, prettyName, toolNames } from '../settings/helpers'
import type { SetStatusbarItemGroup } from '../shell/statusbar-controls'

const SKILLS_MODES = ['skills', 'toolsets'] as const
type SkillsMode = (typeof SKILLS_MODES)[number]

function categoryFor(skill: SkillInfo): string {
  return asText(skill.category) || 'general'
}

function filteredSkills(skills: SkillInfo[], query: string, category: string | null): SkillInfo[] {
  const q = query.trim().toLowerCase()

  return skills
    .filter(skill => {
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

      return (
        includesQuery(toolset.name, q) ||
        includesQuery(toolset.label, q) ||
        includesQuery(toolset.description, q) ||
        toolNames(toolset).some(name => includesQuery(name, q))
      )
    })
    .sort((a, b) => asText(a.label || a.name).localeCompare(asText(b.label || b.name)))
}

interface SkillsViewProps extends React.ComponentProps<'section'> {
  setStatusbarItemGroup?: SetStatusbarItemGroup
}

export function SkillsView({ setStatusbarItemGroup: _setStatusbarItemGroup, ...props }: SkillsViewProps) {
  const [mode, setMode] = useRouteEnumParam('tab', SKILLS_MODES, 'skills')

  const [query, setQuery] = useState('')
  const [skills, setSkills] = useState<SkillInfo[] | null>(null)
  const [toolsets, setToolsets] = useState<ToolsetInfo[] | null>(null)
  const [activeCategory, setActiveCategory] = useState<string | null>(null)
  const [refreshing, setRefreshing] = useState(false)
  const [savingSkill, setSavingSkill] = useState<string | null>(null)

  const refreshCapabilities = useCallback(async () => {
    setRefreshing(true)

    try {
      const [nextSkills, nextToolsets] = await Promise.all([getSkills(), getToolsets()])
      setSkills(nextSkills)
      setToolsets(nextToolsets)
    } catch (err) {
      notifyError(err, 'Skills failed to load')
    } finally {
      setRefreshing(false)
    }
  }, [])

  useEffect(() => {
    void refreshCapabilities()
  }, [refreshCapabilities])

  const categories = useMemo(() => {
    if (!skills) {
      return []
    }

    const counts = new Map<string, number>()

    for (const skill of skills) {
      const key = categoryFor(skill)
      counts.set(key, (counts.get(key) || 0) + 1)
    }

    return Array.from(counts.entries())
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([key, count]) => ({ key, count }))
  }, [skills])

  const visibleSkills = useMemo(
    () => (skills ? filteredSkills(skills, query, mode === 'skills' ? activeCategory : null) : []),
    [activeCategory, mode, query, skills]
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
  const enabledToolsets = toolsets?.filter(toolset => toolset.enabled).length || 0

  async function handleToggleSkill(skill: SkillInfo, enabled: boolean) {
    setSavingSkill(skill.name)

    try {
      await toggleSkill(skill.name, enabled)
      setSkills(current => current?.map(row => (row.name === skill.name ? { ...row, enabled } : row)) ?? current)
      notify({
        kind: 'success',
        title: enabled ? 'Skill enabled' : 'Skill disabled',
        message: `${skill.name} applies to new sessions.`
      })
    } catch (err) {
      notifyError(err, `Failed to update ${skill.name}`)
    } finally {
      setSavingSkill(null)
    }
  }

  return (
    <PageSearchShell
      {...props}
      filters={
        <>
          <div className="flex flex-wrap items-center justify-center gap-x-2 gap-y-1">
            <TextTab active={mode === 'skills'} onClick={() => setMode('skills')}>
              Skills
            </TextTab>
            <TextTab active={mode === 'toolsets'} onClick={() => setMode('toolsets')}>
              Toolsets
            </TextTab>
          </div>
          {mode === 'skills' && categories.length > 0 && (
            <div className="flex flex-wrap justify-center gap-x-2 gap-y-1">
              <TextTab active={activeCategory === null} onClick={() => setActiveCategory(null)}>
                All <TextTabMeta>{totalSkills}</TextTabMeta>
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
            </div>
          )}
        </>
      }
      onSearchChange={setQuery}
      searchPlaceholder={mode === 'skills' ? 'Search skills...' : 'Search toolsets...'}
      searchTrailingAction={
        <Button
          aria-label={refreshing ? 'Refreshing skills' : 'Refresh skills'}
          className="text-(--ui-text-tertiary) hover:bg-transparent hover:text-foreground"
          disabled={refreshing}
          onClick={() => void refreshCapabilities()}
          size="icon-xs"
          title={refreshing ? 'Refreshing skills' : 'Refresh skills'}
          type="button"
          variant="ghost"
        >
          <Codicon name="refresh" size="0.875rem" spinning={refreshing} />
        </Button>
      }
      searchValue={query}
    >
      {!skills || !toolsets ? (
        <PageLoader label="Loading capabilities..." />
      ) : mode === 'skills' ? (
        <div className="h-full overflow-y-auto px-4 py-3">
          {visibleSkills.length === 0 ? (
            <EmptyState description="Try a broader search or different category." title="No skills found" />
          ) : (
            <div className="space-y-4">
              {skillGroups.map(([category, list]) => (
                <div className="space-y-1.5" key={category}>
                  <div className="text-[0.68rem] font-semibold uppercase tracking-[0.12em] text-muted-foreground">
                    {prettyName(category)}
                  </div>
                  <div className="divide-y divide-(--ui-stroke-quaternary)">
                    {list.map(skill => (
                      <div
                        className="grid gap-3 px-0 py-2.5 sm:grid-cols-[minmax(0,1fr)_auto] sm:items-center"
                        key={skill.name}
                      >
                        <div className="min-w-0">
                          <div className="truncate text-sm font-medium">{skill.name}</div>
                          <p className="mt-0.5 text-xs text-muted-foreground">
                            {asText(skill.description) || 'No description.'}
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
        <div className="h-full overflow-y-auto px-4 py-3">
          {visibleToolsets.length === 0 ? (
            <EmptyState description="Try a broader search query." title="No toolsets found" />
          ) : (
            <div className="space-y-2">
              <div className="text-xs text-muted-foreground">
                {enabledToolsets}/{toolsets.length} toolsets enabled
              </div>
              <div className="divide-y divide-(--ui-stroke-quaternary)">
                {visibleToolsets.map(toolset => {
                  const tools = toolNames(toolset)
                  const label = asText(toolset.label || toolset.name)

                  return (
                    <div className="px-0 py-2.5" key={toolset.name}>
                      <div className="flex items-center justify-between gap-2">
                        <div className="truncate text-sm font-medium">{label}</div>
                        <div className="flex items-center gap-1.5">
                          <StatusPill active={toolset.enabled}>{toolset.enabled ? 'Enabled' : 'Disabled'}</StatusPill>
                          <StatusPill active={toolset.configured}>
                            {toolset.configured ? 'Configured' : 'Needs keys'}
                          </StatusPill>
                        </div>
                      </div>
                      <p className="mt-1 text-xs text-muted-foreground">
                        {asText(toolset.description) || 'No description.'}
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
    <span
      className={cn(
        'inline-flex items-center rounded-full px-1.5 py-0.5 text-[0.64rem]',
        active ? 'bg-(--ui-bg-tertiary) text-(--ui-text-secondary)' : 'bg-(--ui-bg-quinary) text-(--ui-text-tertiary)'
      )}
    >
      {children}
    </span>
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
