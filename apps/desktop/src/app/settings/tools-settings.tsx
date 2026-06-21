import { useEffect, useMemo, useState } from 'react'

import { Switch } from '@/components/ui/switch'
import { getSkills, getToolsets, toggleSkill } from '@/hermes'
import { dt } from '@/lib/i18n'
import { Brain, Wrench } from '@/lib/icons'
import { $desktopLanguage } from '@/store/language'
import { notify, notifyError } from '@/store/notifications'
import { useStore } from '@nanostores/react'
import type { SkillInfo, ToolsetInfo } from '@/types/hermes'

import { asText, includesQuery, prettyName, toolNames } from './helpers'
import { ListRow, LoadingState, Pill, SectionHeading, SettingsContent } from './primitives'
import type { SearchProps } from './types'

export function ToolsSettings({ query }: SearchProps) {
  const language = useStore($desktopLanguage)
  const [skills, setSkills] = useState<SkillInfo[] | null>(null)
  const [toolsets, setToolsets] = useState<ToolsetInfo[] | null>(null)
  const [savingSkill, setSavingSkill] = useState<string | null>(null)

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
      .catch(err => notifyError(err, language === 'zh' ? '能力列表加载失败' : 'Capabilities failed to load'))

    return () => void (cancelled = true)
  }, [language])

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
        title: enabled
          ? language === 'zh'
            ? '技能已启用'
            : 'Skill enabled'
          : language === 'zh'
            ? '技能已禁用'
            : 'Skill disabled',
        message:
          language === 'zh' ? `${skill.name} 将应用于新的会话。` : `${skill.name} applies to new sessions.`
      })
    } catch (err) {
      notifyError(err, `Failed to update ${skill.name}`)
    } finally {
      setSavingSkill(null)
    }
  }

  if (!skills || !toolsets) {
    return <LoadingState label={language === 'zh' ? '正在加载技能和工具集...' : 'Loading skills and toolsets...'} />
  }

  return (
    <SettingsContent>
      <div className="mb-6">
        <SectionHeading
          icon={Brain}
          meta={
            language === 'zh'
              ? `已启用 ${filteredSkills.filter(s => s.enabled).length} 个`
              : `${filteredSkills.filter(s => s.enabled).length} enabled`
          }
          title={dt(language, 'skills', 'Skills')}
        />
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
          meta={
            language === 'zh'
              ? `已启用 ${filteredToolsets.filter(t => t.enabled).length} 个`
              : `${filteredToolsets.filter(t => t.enabled).length} enabled`
          }
          title={dt(language, 'tools', 'Toolsets')}
        />
        <div className="divide-y divide-border/40">
          {filteredToolsets.map(toolset => {
            const tools = toolNames(toolset)
            const label = asText(toolset.label || toolset.name)

            return (
              <ListRow
                action={
                  <div className="flex shrink-0 items-center gap-1.5">
                    <Pill tone={toolset.enabled ? 'primary' : 'muted'}>
                      {toolset.enabled
                        ? language === 'zh'
                          ? '已启用'
                          : 'Enabled'
                        : language === 'zh'
                          ? '已禁用'
                          : 'Disabled'}
                    </Pill>
                    <Pill tone={toolset.configured ? 'primary' : 'muted'}>
                      {toolset.configured
                        ? language === 'zh'
                          ? '已配置'
                          : 'Configured'
                        : language === 'zh'
                          ? '需要密钥'
                          : 'Needs keys'}
                    </Pill>
                  </div>
                }
                below={
                  tools.length > 0 && (
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
                          {language === 'zh' ? `另有 ${tools.length - 10} 个` : `+${tools.length - 10} more`}
                        </span>
                      )}
                    </div>
                  )
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
