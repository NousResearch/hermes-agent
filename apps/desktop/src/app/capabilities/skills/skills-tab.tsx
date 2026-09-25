import { compactNumber } from '@hermes/shared'
import { useStore } from '@nanostores/react'
import { useQuery } from '@tanstack/react-query'
import { type ReactNode, useCallback, useMemo, useState } from 'react'

import { ArchiveSkillConfirmDialog } from '@/app/learning/archive-skill-confirm-dialog'
import { CodeEditor } from '@/components/chat/code-editor'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import {
  editLearningNode,
  getLearningNode,
  getOfficialSkills,
  profileScopeKey,
  setSkillCategoryEnabled,
  setSkillEnabled,
  type ProfileScope
} from '@/hermes'
import { useI18n } from '@/i18n'
import { Loader2 } from '@/lib/icons'
import { Codecs, persistentAtom } from '@/lib/persisted'
import { queryClient } from '@/lib/query-client'
import { invalidateSlashCompletions } from '@/lib/slash-completion-cache'
import { useStoreSelector } from '@/lib/use-session-slice'
import { $hubActions, installHubSkill, notifyHubActionFailed, OFFICIAL_SKILLS_KEY } from '@/store/hub-actions'
import { notify, notifyError } from '@/store/notifications'
import type { OfficialSkillInfo, SkillInfo } from '@/types/hermes'

import {
  CapRow,
  DetailColumn,
  DetailPane,
  ListColumn,
  ListStrip,
  ListStripMenu,
  type ListStripMenuToggle,
  MasterDetail
} from '../../master-detail'
import { prettyName } from '../../settings/helpers'
import { CapabilityEmpty, GroupHeaderRow, SortButton } from '../primitives'

import { OfficialSkillDetail } from './official-skill-detail'
import { SkillDetail } from './skill-detail'
import { categoryFor, filteredOfficial, filteredSkills, skillsQueryKey, usageOf } from './skills-data'

// Sort direction for the Skills list — persisted so the tab remembers
// most/least-used across navigations and restarts.
const $skillsSortDesc = persistentAtom('hermes.desktop.capabilities.skillsSortDesc', true, Codecs.bool)

// Row subtitle: category, with non-default origins badged.
function skillSubtitle(skill: SkillInfo): ReactNode {
  const category = prettyName(categoryFor(skill))
  const provenance = skill.provenance

  return (
    <>
      <span className="truncate">{category}</span>
      {provenance === 'agent' && (
        <Badge className="shrink-0 normal-case" variant="default">
          learned
        </Badge>
      )}
      {provenance === 'hub' && (
        <Badge className="shrink-0 normal-case" variant="muted">
          hub
        </Badge>
      )}
    </>
  )
}

interface SkillsTabProps {
  /** The scope's skill list, straight from the shell's query. */
  skills: SkillInfo[]
  /** The (connection, profile) scope every read and write routes to. */
  profile: ProfileScope
  query: string
  /** Page-level refresh: a saved skill edit reloads the same way the refresh
   *  hotkey does, counts and slash completions included. */
  onRefresh: () => void
}

/** The Skills tab: installed skills, official optional skills, and learned-skill editing. */
export function SkillsTab({ onRefresh, profile, query, skills }: SkillsTabProps) {
  const { t } = useI18n()
  const skillsSortDesc = useStore($skillsSortDesc)
  const [bulkBusy, setBulkBusy] = useState(false)
  const [selectedSkill, setSelectedSkill] = useState<string | null>(null)
  const [selectedOfficial, setSelectedOfficial] = useState<string | null>(null)

  const { data: officialData } = useQuery({
    queryKey: [...OFFICIAL_SKILLS_KEY, profileScopeKey(profile)],
    queryFn: () => getOfficialSkills(profile),
    staleTime: 60_000,
    retry: false
  })

  // Learned/local skills are editable + archivable, mirroring the memory
  // graph (same /api/learning/node endpoints — delete archives, restorable
  // via `hermes curator restore`).
  const [skillEditor, setSkillEditor] = useState<null | { content: string; name: string }>(null)
  const [skillDraft, setSkillDraft] = useState('')
  const [skillSaving, setSkillSaving] = useState(false)
  const [archiveTarget, setArchiveTarget] = useState<null | string>(null)

  // Optimistic write-through against the scoped Skills key: toggles/bulk/
  // archive repaint instantly; the next background refetch reconciles.
  const setSkills = useCallback(
    (fn: (cur: SkillInfo[] | undefined) => SkillInfo[] | undefined) =>
      queryClient.setQueryData<SkillInfo[]>(skillsQueryKey(profile), prev => fn(prev) ?? prev),
    [profile]
  )

  const visibleSkills = useMemo(() => filteredSkills(skills, query, skillsSortDesc), [query, skills, skillsSortDesc])

  // The installed list renders as category sections (header + its rows, list
  // sort preserved inside a section). The header switch's state and write
  // target the WHOLE category, never the search-filtered view — same contract
  // as the strip-menu bulk items.
  const groupedSkills = useMemo(() => {
    const byCategory = new Map<string, SkillInfo[]>()
    for (const skill of visibleSkills) {
      const key = categoryFor(skill)
      byCategory.set(key, [...(byCategory.get(key) ?? []), skill])
    }

    return Array.from(byCategory.entries()).sort((a, b) => a[0].localeCompare(b[0]))
  }, [visibleSkills])

  const categoryState = useMemo(() => {
    const state = new Map<string, { allEnabled: boolean; size: number }>()
    const rowsByCategory = new Map<string, SkillInfo[]>()
    for (const skill of skills) {
      const key = categoryFor(skill)
      rowsByCategory.set(key, [...(rowsByCategory.get(key) ?? []), skill])
    }

    for (const [category, rows] of rowsByCategory) {
      state.set(category, { allEnabled: rows.every(skill => skill.enabled), size: rows.length })
    }

    return state
  }, [skills])

  // Installed-name set stays unfiltered so search cannot make a skill look absent.
  const installedSkillNames = useMemo(() => new Set(skills.map(s => s.name)), [skills])

  const visibleOfficial = useMemo(() => {
    const catalog = (officialData?.skills ?? []).filter(
      skill => !skill.installed && !installedSkillNames.has(skill.name)
    )

    return filteredOfficial(catalog, query)
  }, [installedSkillNames, officialData, query])

  const runningInstallKey = useStoreSelector($hubActions, actions =>
    Object.keys(actions)
      .filter(key => actions[key]?.running)
      .sort()
      .join('|')
  )

  const runningInstalls = useMemo(() => new Set(runningInstallKey.split('|').filter(Boolean)), [runningInstallKey])

  // Keep a valid selection: fall back to the first visible row when the
  // current selection is filtered out (or nothing is selected yet).
  const activeSkill = useMemo(
    () => visibleSkills.find(s => s.name === selectedSkill) ?? visibleSkills[0] ?? null,
    [selectedSkill, visibleSkills]
  )

  const activeOfficial = useMemo(
    () => visibleOfficial.find(skill => skill.identifier === selectedOfficial) ?? null,
    [selectedOfficial, visibleOfficial]
  )

  function handleInstallOfficial(skill: OfficialSkillInfo) {
    notify({ kind: 'success', title: t.skills.hub.installStarted(skill.name), message: t.skills.hub.actionLog })
    void installHubSkill(skill.identifier, profile).catch(err =>
      notifyHubActionFailed(err, t.skills.hub.actionFailed, skill.name, profile)
    )
  }

  async function handleToggleSkill(skill: SkillInfo, enabled: boolean) {
    setSkills(current => current?.map(row => (row.name === skill.name ? { ...row, enabled } : row)) ?? current)

    try {
      await setSkillEnabled(skill.name, enabled, profile)
      // A disabled skill loses its `/name` command, so the composer's cached
      // `/` list has to be dropped along with the row repaint.
      invalidateSlashCompletions()
    } catch (err) {
      setSkills(
        current => current?.map(row => (row.name === skill.name ? { ...row, enabled: !enabled } : row)) ?? current
      )
      notifyError(err, t.skills.failedToUpdate(skill.name))
    }
  }

  // Sequential on purpose: each toggle is a config read-modify-write on the
  // backend; parallel calls would race the disabled-list save.
  async function bulkApply(targets: SkillInfo[], enabled: boolean) {
    if (bulkBusy || targets.length === 0) {
      return
    }

    setBulkBusy(true)

    let done = 0

    try {
      for (const row of targets) {
        await setSkillEnabled(row.name, enabled, profile)
        setSkills(cur => cur?.map(r => (r.name === row.name ? { ...r, enabled } : r)) ?? cur)
        done += 1
      }

      notify({ kind: 'success', title: t.skills.bulkUpdated(done), message: '' })
    } catch (err) {
      notifyError(err, t.skills.failedToUpdate(t.skills.tabSkills))
    } finally {
      invalidateSlashCompletions()
      setBulkBusy(false)
    }
  }

  // Bulk actions ("All" master switch, "Disable unused") and the master-switch
  // state target the WHOLE tab, never the search-filtered view — a tab-wide
  // control that silently scoped to the current query would be a lie.
  const allEnabled = skills.length > 0 && skills.every(s => s.enabled)

  // One switch line covering enable-all/disable-all.
  const bulkSwitch: ListStripMenuToggle = {
    checked: allEnabled,
    disabled: bulkBusy,
    label: t.skills.all,
    onToggle: checked =>
      void bulkApply(
        skills.filter(row => row.enabled !== checked),
        checked
      )
  }

  // Per-category bulk: one backend request per category (a single
  // skills.disabled write server-side); rows repaint from response names.
  async function bulkCategoryApply(category: string, enabled: boolean) {
    if (bulkBusy) {
      return
    }

    setBulkBusy(true)

    try {
      const result = await setSkillCategoryEnabled(
        category === 'general' ? null : category,
        enabled,
        profile
      )
      const names = new Set(result.names)
      setSkills(cur => cur?.map(row => (names.has(row.name) ? { ...row, enabled } : row)) ?? cur)
      notify({ kind: 'success', title: t.skills.bulkUpdated(result.names.length), message: '' })
    } catch (err) {
      notifyError(err, t.skills.failedToUpdate(category))
    } finally {
      invalidateSlashCompletions()
      setBulkBusy(false)
    }
  }

  const categoryBulkItems = useMemo(() => {
    const byCategory = new Map<string, SkillInfo[]>()
    for (const row of skills) {
      const key = categoryFor(row)
      byCategory.set(key, [...(byCategory.get(key) ?? []), row])
    }

    return Array.from(byCategory.entries()).map(([category, rows]) => {
      const anyEnabled = rows.some(row => row.enabled)
      return {
        disabled: bulkBusy,
        label: anyEnabled
          ? t.skills.categoryDisableAll(prettyName(category), rows.length)
          : t.skills.categoryEnableAll(prettyName(category), rows.length),
        onSelect: () => void bulkCategoryApply(category, !anyEnabled)
      }
    })
  }, [bulkBusy, skills, t])

  // "Never used" = zero recorded activity. The pruning move for a 100+ skill
  // install: keep the workhorses, shed the noise.
  const disableUnused = () =>
    bulkApply(
      skills.filter(skill => skill.enabled && usageOf(skill) === 0),
      false
    )

  const openSkillEditor = async (name: string) => {
    try {
      const node = await getLearningNode(name, profile)

      setSkillEditor({ content: node.content, name })
      setSkillDraft(node.content)
    } catch (err) {
      notifyError(err, name)
    }
  }

  const saveSkillEdit = async () => {
    if (!skillEditor) {
      return
    }

    setSkillSaving(true)

    try {
      await editLearningNode(skillEditor.name, skillDraft, profile)
      notify({
        kind: 'success',
        title: t.skills.skillUpdated,
        message: t.skills.appliesToNewSessions(skillEditor.name)
      })
      setSkillEditor(null)
      onRefresh()
    } catch (err) {
      notifyError(err, skillEditor.name)
    } finally {
      setSkillSaving(false)
    }
  }

  const skillEditorPane = skillEditor && (
    <DetailPane
      actions={
        <Button disabled={skillSaving} onClick={() => void saveSkillEdit()} size="xs">
          {skillSaving ? t.common.saving : t.common.save}
        </Button>
      }
      id="skill-editor"
      onClose={() => setSkillEditor(null)}
      title={<span className="text-[0.68rem] font-normal text-muted-foreground/60">{skillEditor.name}/SKILL.md</span>}
    >
      <CodeEditor
        filePath="SKILL.md"
        initialValue={skillEditor.content}
        key={skillEditor.name}
        onCancel={() => setSkillEditor(null)}
        onChange={setSkillDraft}
        onSave={() => void saveSkillEdit()}
      />
    </DetailPane>
  )

  return (
    <>
      {visibleSkills.length === 0 && visibleOfficial.length === 0 ? (
        <CapabilityEmpty noun="skills" query={query} />
      ) : (
        <MasterDetail pane={skillEditorPane} resizeId="capabilities-split" split="wide">
          <ListColumn
            header={
              <ListStrip
                left={<SortButton desc={skillsSortDesc} onFlip={() => $skillsSortDesc.set(!$skillsSortDesc.get())} />}
                right={
                  <ListStripMenu
                    items={[
                      {
                        disabled: bulkBusy,
                        label: t.skills.disableUnused,
                        onSelect: () => void disableUnused()
                      },
                      ...categoryBulkItems
                    ]}
                    label={t.skills.tabSkills}
                    toggle={bulkSwitch}
                  />
                }
              />
            }
          >
            {groupedSkills.map(([category, rows]) => {
              const state = categoryState.get(category)
              const size = state?.size ?? rows.length

              return (
                <div className="contents" key={category}>
                  <GroupHeaderRow
                    busy={bulkBusy}
                    count={size}
                    enabled={state?.allEnabled ?? false}
                    label={prettyName(category)}
                    onToggle={checked => void bulkCategoryApply(category, checked)}
                  />
                  {rows.map(skill => (
                    <CapRow
                      active={activeOfficial === null && activeSkill?.name === skill.name}
                      busy={bulkBusy}
                      enabled={skill.enabled}
                      key={skill.name}
                      meta={usageOf(skill) > 0 ? `×${compactNumber(usageOf(skill))}` : undefined}
                      onSelect={() => {
                        setSelectedSkill(skill.name)
                        setSelectedOfficial(null)
                      }}
                      onToggle={enabled => void handleToggleSkill(skill, enabled)}
                      subtitle={skillSubtitle(skill)}
                      title={skill.name}
                      toggleLabel={skill.name}
                    />
                  ))}
                </div>
              )
            })}
            {visibleOfficial.length > 0 && (
              <div className="flex h-7 shrink-0 items-end px-2 pb-1 text-[0.62rem] font-medium uppercase tracking-wide text-(--ui-text-quaternary)">
                {t.skills.officialCatalog}
              </div>
            )}
            {visibleOfficial.map(skill => {
              const installing = runningInstalls.has(skill.identifier)

              return (
                <CapRow
                  action={
                    <Button disabled={installing} onClick={() => handleInstallOfficial(skill)} size="xs" variant="text">
                      {installing && <Loader2 className="size-3 animate-spin" />}
                      {installing ? t.skills.hub.installing : t.skills.hub.install}
                    </Button>
                  }
                  active={activeOfficial?.identifier === skill.identifier}
                  enabled={false}
                  key={skill.identifier}
                  onSelect={() => setSelectedOfficial(skill.identifier)}
                  subtitle={prettyName(skill.category)}
                  title={skill.name}
                />
              )
            })}
          </ListColumn>
          <DetailColumn footer={t.skills.changesApplyNewSessions}>
            {activeOfficial ? (
              <OfficialSkillDetail
                installing={runningInstalls.has(activeOfficial.identifier)}
                onInstall={() => handleInstallOfficial(activeOfficial)}
                profile={profile}
                skill={activeOfficial}
              />
            ) : (
              activeSkill && (
                <SkillDetail
                  onArchive={() => setArchiveTarget(activeSkill.name)}
                  onEdit={() => void openSkillEditor(activeSkill.name)}
                  profile={profile}
                  skill={activeSkill}
                />
              )
            )}
          </DetailColumn>
        </MasterDetail>
      )}
      {archiveTarget && (
        <ArchiveSkillConfirmDialog
          onApply={() => {
            const name = archiveTarget
            const snapshot = skills

            setSkills(current => current?.filter(skill => skill.name !== name) ?? current)
            invalidateSlashCompletions()

            if (skillEditor?.name === name) {
              setSkillEditor(null)
            }

            return () => setSkills(() => snapshot)
          }}
          onClose={() => setArchiveTarget(null)}
          onFailure={(err, name) => notifyError(err, name)}
          open
          profile={profile}
          skillId={archiveTarget}
          skillName={archiveTarget}
        />
      )}
    </>
  )
}
