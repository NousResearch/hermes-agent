import { compactNumber } from '@hermes/shared'
import { useStore } from '@nanostores/react'
import { type ReactNode, useCallback, useMemo, useState } from 'react'

import { ArchiveSkillConfirmDialog } from '@/app/learning/archive-skill-confirm-dialog'
import { CodeEditor } from '@/components/chat/code-editor'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { editLearningNode, getLearningNode, type ProfileScope, setSkillEnabled } from '@/hermes'
import { useI18n } from '@/i18n'
import { Codecs, persistentAtom } from '@/lib/persisted'
import { queryClient } from '@/lib/query-client'
import { invalidateSlashCompletions } from '@/lib/slash-completion-cache'
import { notify, notifyError } from '@/store/notifications'
import type { SkillInfo } from '@/types/hermes'

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
import type { CapabilityView } from '../capability-tabs'
import { CapabilityEmpty, SortButton } from '../primitives'

import { SkillCatalog } from './skill-catalog'
import { SkillDetail } from './skill-detail'
import { categoryFor, filteredSkills, skillsQueryKey, usageOf } from './skills-data'

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
  onQueryChange: (value: string) => void
  /** Installed list or the hub catalog (the shell owns the toggle). */
  view: CapabilityView
  /** Page-level refresh: a saved skill edit reloads the same way the refresh
   *  hotkey does, counts and slash completions included. */
  onRefresh: () => void
}

/** THE Skills tab: the installed list with its inspector, the learned-skill
 *  editor/archive flow, and the hub catalog behind Browse. */
export function SkillsTab({ onQueryChange, onRefresh, profile, query, skills, view }: SkillsTabProps) {
  const { t } = useI18n()
  const skillsSortDesc = useStore($skillsSortDesc)
  const [bulkBusy, setBulkBusy] = useState(false)
  const [selectedSkill, setSelectedSkill] = useState<string | null>(null)
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

  // Installed-name set for the catalog's already-installed guard — the
  // UNFILTERED list on purpose (search must not make a skill look absent).
  const installedSkillNames = useMemo(() => new Set(skills.map(s => s.name)), [skills])

  // Keep a valid selection: fall back to the first visible row when the
  // current selection is filtered out (or nothing is selected yet).
  const activeSkill = useMemo(
    () => visibleSkills.find(s => s.name === selectedSkill) ?? visibleSkills[0] ?? null,
    [selectedSkill, visibleSkills]
  )

  // Single toggles are optimistic and silent on success (the row repaints
  // immediately — a toast per flip would spam rapid customization). Errors
  // revert and notify.
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
    onToggle: checked => void bulkApply(skills.filter(row => row.enabled !== checked), checked)
  }

  // "Never used" = zero recorded activity. The pruning move for a 100+ skill
  // install: keep the workhorses, shed the noise.
  const disableUnused = () => bulkApply(skills.filter(skill => skill.enabled && usageOf(skill) === 0), false)

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
      {view === 'browse' ? (
        <SkillCatalog
          installedNames={installedSkillNames}
          onQueryChange={onQueryChange}
          profile={profile}
          query={query}
        />
      ) : visibleSkills.length === 0 ? (
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
                      }
                    ]}
                    label={t.skills.tabSkills}
                    toggle={bulkSwitch}
                  />
                }
              />
            }
          >
            {visibleSkills.map(skill => (
              <CapRow
                active={activeSkill?.name === skill.name}
                busy={bulkBusy}
                enabled={skill.enabled}
                key={skill.name}
                meta={usageOf(skill) > 0 ? `×${compactNumber(usageOf(skill))}` : undefined}
                onSelect={() => setSelectedSkill(skill.name)}
                onToggle={enabled => void handleToggleSkill(skill, enabled)}
                subtitle={skillSubtitle(skill)}
                title={skill.name}
                toggleLabel={skill.name}
              />
            ))}
          </ListColumn>
          <DetailColumn footer={t.skills.changesApplyNewSessions}>
            {activeSkill && (
              <SkillDetail
                onArchive={() => setArchiveTarget(activeSkill.name)}
                onEdit={() => void openSkillEditor(activeSkill.name)}
                profile={profile}
                skill={activeSkill}
              />
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
