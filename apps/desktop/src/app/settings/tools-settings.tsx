import { useCallback, useEffect, useMemo, useState } from 'react'

import { Switch } from '@/components/ui/switch'
import { getSkills, getToolsets, toggleSkill, toggleToolset } from '@/hermes'
import { Brain, Wrench } from '@/lib/icons'
import { notify, notifyError } from '@/store/notifications'
import type { SkillInfo, ToolsetInfo } from '@/types/hermes'

import { asText, includesQuery, prettyName, toolNames } from './helpers'
import { ListRow, LoadingState, Pill, SectionHeading, SettingsContent } from './primitives'
import { ToolsetConfigPanel } from './toolset-config-panel'
import type { SearchProps } from './types'

export function ToolsSettings({ query }: SearchProps) {
  const [skills, setSkills] = useState<SkillInfo[] | null>(null)
  const [toolsets, setToolsets] = useState<ToolsetInfo[] | null>(null)
  const [savingSkill, setSavingSkill] = useState<string | null>(null)
  const [savingToolset, setSavingToolset] = useState<string | null>(null)
  const [expandedToolset, setExpandedToolset] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    Promise.all([getSkills(), getToolsets()])
      .then(([s, t]) => {
        if (cancelled) {
          return
        }

        setSkills(s)
        setToolsets(t)
      })
      .catch(err => notifyError(err, 'Capabilities failed to load'))

    return () => void (cancelled = true)
  }, [])

  const refreshToolsets = useCallback(() => {
    getToolsets()
      .then(setToolsets)
      .catch(err => notifyError(err, 'Toolsets failed to refresh'))
  }, [])

  const filteredSkills = useMemo(() => {
    if (!skills) {
      return []
    }

    const q = query.trim().toLowerCase()

    return skills
      .filter(s => !q || includesQuery(s.name, q) || includesQuery(s.description, q) || includesQuery(s.category, q))
      .sort(
        (a, b) => asText(a.category).localeCompare(asText(b.category)) || asText(a.name).localeCompare(asText(b.name))
      )
  }, [query, skills])

  const filteredToolsets = useMemo(() => {
    if (!toolsets) {
      return []
    }

    const q = query.trim().toLowerCase()

    return toolsets
      .filter(t => {
        if (!q) {
          return true
        }

        return (
          includesQuery(t.name, q) ||
          includesQuery(t.label, q) ||
          includesQuery(t.description, q) ||
          toolNames(t).some(n => includesQuery(n, q))
        )
      })
      .sort((a, b) => asText(a.label || a.name).localeCompare(asText(b.label || b.name)))
  }, [query, toolsets])

  const skillGroups = useMemo(() => {
    const groups = new Map<string, SkillInfo[]>()

    for (const skill of filteredSkills) {
      const cat = asText(skill.category) || 'other'
      groups.set(cat, [...(groups.get(cat) ?? []), skill])
    }

    return Array.from(groups).sort(([a], [b]) => a.localeCompare(b))
  }, [filteredSkills])

  async function handleToggleSkill(skill: SkillInfo, enabled: boolean) {
    setSavingSkill(skill.name)

    try {
      await toggleSkill(skill.name, enabled)
      setSkills(c => c?.map(s => (s.name === skill.name ? { ...s, enabled } : s)) ?? c)
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

  async function handleToggleToolset(toolset: ToolsetInfo, enabled: boolean) {
    setSavingToolset(toolset.name)

    try {
      await toggleToolset(toolset.name, enabled)
      setToolsets(c => c?.map(t => (t.name === toolset.name ? { ...t, enabled, available: enabled } : t)) ?? c)
      notify({
        kind: 'success',
        title: enabled ? 'Toolset enabled' : 'Toolset disabled',
        message: `${asText(toolset.label || toolset.name)} applies to new sessions.`
      })
    } catch (err) {
      notifyError(err, `Failed to update ${asText(toolset.label || toolset.name)}`)
    } finally {
      setSavingToolset(null)
    }
  }

  if (!skills || !toolsets) {
    return <LoadingState label="Loading skills and toolsets..." />
  }

  return (
    <SettingsContent>
      <div className="mb-6">
        <SectionHeading icon={Brain} meta={`${filteredSkills.filter(s => s.enabled).length} enabled`} title="Skills" />
        {skillGroups.map(([category, list]) => (
          <div className="mt-4 first:mt-0" key={category}>
            <div className="mb-1 text-[0.68rem] font-semibold uppercase tracking-[0.12em] text-muted-foreground">
              {prettyName(category)}
            </div>
            <div className="divide-y divide-border/40">
              {list.map(skill => (
                <ListRow
                  action={
                    <Switch
                      checked={skill.enabled}
                      disabled={savingSkill === skill.name}
                      onCheckedChange={c => void handleToggleSkill(skill, c)}
                    />
                  }
                  description={asText(skill.description)}
                  key={asText(skill.name)}
                  title={asText(skill.name)}
                />
              ))}
            </div>
          </div>
        ))}
      </div>

      <div className="mb-6">
        <SectionHeading
          icon={Wrench}
          meta={`${filteredToolsets.filter(t => t.enabled).length} enabled`}
          title="Toolsets"
        />
        <div className="divide-y divide-border/40">
          {filteredToolsets.map(toolset => {
            const tools = toolNames(toolset)
            const label = asText(toolset.label || toolset.name)
            const expanded = expandedToolset === toolset.name

            return (
              <ListRow
                action={
                  <div className="flex shrink-0 items-center gap-1.5">
                    <button
                      aria-expanded={expanded}
                      aria-label={`Configure ${label}`}
                      className="cursor-pointer rounded-full outline-none focus-visible:ring-2 focus-visible:ring-ring/50"
                      onClick={() => setExpandedToolset(c => (c === toolset.name ? null : toolset.name))}
                      type="button"
                    >
                      <Pill tone={toolset.configured ? 'primary' : 'muted'}>
                        {toolset.configured ? 'Configured' : 'Needs keys'}
                      </Pill>
                    </button>
                    <Switch
                      aria-label={`Toggle ${label} toolset`}
                      checked={toolset.enabled}
                      disabled={savingToolset === toolset.name}
                      onCheckedChange={c => void handleToggleToolset(toolset, c)}
                    />
                  </div>
                }
                below={
                  <>
                    {tools.length > 0 && (
                      <div className="mt-3 flex flex-wrap gap-1">
                        {tools.slice(0, 10).map(t => (
                          <span
                            className="rounded-md bg-muted px-1.5 py-0.5 font-mono text-[0.64rem] text-muted-foreground"
                            key={t}
                          >
                            {t}
                          </span>
                        ))}
                        {tools.length > 10 && (
                          <span className="rounded-md bg-muted px-1.5 py-0.5 text-[0.64rem] text-muted-foreground">
                            +{tools.length - 10} more
                          </span>
                        )}
                      </div>
                    )}
                    {expanded && (
                      <ToolsetConfigPanel onConfiguredChange={refreshToolsets} toolset={toolset.name} />
                    )}
                  </>
                }
                description={asText(toolset.description)}
                key={asText(toolset.name) || label}
                title={label}
              />
            )
          })}
        </div>
      </div>
    </SettingsContent>
  )
}
