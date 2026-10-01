import { useCallback, useEffect, useRef, useState } from 'react'

import { ArchiveSkillConfirmDialog } from '@/app/learning/archive-skill-confirm-dialog'
import { CodeEditor } from '@/components/chat/code-editor'
import { Button } from '@/components/ui/button'
import { Loader } from '@/components/ui/loader'
import { Switch } from '@/components/ui/switch'
import { editLearningNode, getLearningNode, type ProfileScope, profileScopeKey, setSkillEnabled } from '@/hermes'
import { useI18n } from '@/i18n'
import { queryClient } from '@/lib/query-client'
import { invalidateSlashCompletions } from '@/lib/slash-completion-cache'
import { notify, notifyError } from '@/store/notifications'
import type { SkillInfo } from '@/types/hermes'

import { DetailPane, ListStripMenu, type ListStripMenuToggle } from '../../master-detail'
import { CatalogAlert } from '../catalog/catalog-alert'
import { SkillCatalog } from '../catalog/skill-catalog'
import { UpdateSkillsButton } from '../catalog/update-skills-button'

  return (
    <>
      <span className="truncate">{category}</span>
      {provenance === 'agent' && (
        <Badge className="shrink-0 normal-case" variant="default">
          learned
        </Badge>
      )}
      {provenance === 'bundled' && (
        <Badge className="shrink-0 normal-case" variant="muted">
          built-in
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
import { SkillDetail } from './skill-detail'
import { skillsQueryKey, usageOf } from './skills-data'

interface SkillsTabProps {
  /** The scope's skill list, straight from the shell's query. */
  skills: SkillInfo[]
  /** Every read and write targets this connection/profile pair. */
  profile: ProfileScope
  query: string
  onQueryChange?: (value: string) => void
  installedPending?: boolean
  installedError?: unknown
  onRefresh: () => void
}

/** One management controller for the unified catalog, remounted on scope changes
 * so an old profile's editor, confirmation or pending write cannot enter another. */
export function SkillsTab(props: SkillsTabProps) {
  return <ScopedSkillsTab key={profileScopeKey(props.profile)} {...props} />
}

function ScopedSkillsTab({
  onRefresh,
  profile,
  query,
  onQueryChange,
  skills,
  installedPending = false,
  installedError
}: SkillsTabProps) {
  const { t } = useI18n()
  const mounted = useRef(true)
  const mutationBusy = useRef(false)
  const editorRequest = useRef(0)
  const saving = useRef(false)
  const [busy, setBusy] = useState(false)
  const [skillEditor, setSkillEditor] = useState<null | { name: string }>(null)
  const [skillDraft, setSkillDraft] = useState('')
  const [skillSaving, setSkillSaving] = useState(false)
  const [archiveTarget, setArchiveTarget] = useState<null | string>(null)

  // eslint-disable-next-line no-restricted-syntax -- lifecycle guard drops stale async completions; it does not mirror an atom
  useEffect(() => {
    mounted.current = true

    return () => {
      mounted.current = false
      editorRequest.current += 1
    }
  }, [])

  const setSkills = useCallback(
    (fn: (cur: SkillInfo[] | undefined) => SkillInfo[] | undefined) =>
      queryClient.setQueryData<SkillInfo[]>(skillsQueryKey(profile), prev => fn(prev) ?? prev),
    [profile]
  )

  // Provenance counts make it clear why a skill appears in this list (agent,
  // bundled, or Skills Hub) instead of implying that only one source exists.
  // Older backends do not send provenance at all; do not fabricate an
  // `agent` count because that would turn missing metadata into a false claim.
  const provenanceSummary = useMemo(() => {
    if (!skills || skills.some(skill => !skill.provenance)) {
      return null
    }

    const counts = { agent: 0, bundled: 0, hub: 0 }

    for (const skill of skills) {
      const provenance = skill.provenance

      if (!provenance) {
        return null
      }

      counts[provenance] += 1
    }

    return t.skills.provenanceSummary(counts.agent, counts.bundled, counts.hub)
  }, [skills, t.skills])

  const visibleSkills = useMemo(() => filteredSkills(skills, query, skillsSortDesc), [query, skills, skillsSortDesc])

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
  // The backend saves one disabled-list config value: serialize individual and
  // bulk changes together, not merely the members of a bulk action.
  async function applyEnabled(targets: SkillInfo[], enabled: boolean, bulk = false) {
    if (mutationBusy.current || installedPending || installedError || targets.length === 0) {
      return
    }

    mutationBusy.current = true
    setBusy(true)
    let done = 0

    try {
      await queryClient.cancelQueries({ queryKey: skillsQueryKey(profile), exact: true })

      for (const row of targets) {
        if (!mounted.current) {
          break
        }

        const previous =
          queryClient.getQueryData<SkillInfo[]>(skillsQueryKey(profile))?.find(skill => skill.name === row.name) ?? row

        setSkills(current => current?.map(skill => (skill.name === row.name ? { ...skill, enabled } : skill)))

        try {
          await setSkillEnabled(row.name, enabled, profile)
          done += 1
        } catch (err) {
          if (mounted.current) {
            setSkills(current =>
              current?.map(skill =>
                skill.name === row.name && skill.enabled === enabled ? { ...skill, enabled: previous.enabled } : skill
              )
            )
          }

          throw err
        }
      }

      if (bulk && mounted.current) {
        notify({ kind: 'success', title: t.skills.bulkUpdated(done), message: '' })
      }
    } catch (err) {
      if (mounted.current) {
        notifyError(err, t.skills.failedToUpdate(bulk ? t.skills.tabSkills : targets[0].name))
      }
    } finally {
      invalidateSlashCompletions()
      void queryClient.invalidateQueries({ queryKey: skillsQueryKey(profile), exact: true })
      mutationBusy.current = false

      if (mounted.current) {
        setBusy(false)
      }
    }
  }

  const controlsDisabled = busy || installedPending || Boolean(installedError)

  // Bulk always means the whole profile, never just a search/filter result.
  const bulkSwitch: ListStripMenuToggle = {
    checked: skills.length > 0 && skills.every(skill => skill.enabled),
    disabled: controlsDisabled || skills.length === 0,
    label: t.skills.all,
    onToggle: checked =>
      void applyEnabled(
        skills.filter(skill => skill.enabled !== checked),
        checked,
        true
      )
  }

  const openSkillEditor = async (name: string) => {
    if (saving.current || skillEditor?.name === name) {
      return
    }

    const request = ++editorRequest.current

    try {
      const node = await getLearningNode(name, profile)

      if (!mounted.current || request !== editorRequest.current) {
        return
      }

      setSkillEditor({ name })
      setSkillDraft(node.content)
    } catch (err) {
      if (mounted.current && request === editorRequest.current) {
        notifyError(err, name)
      }
    }
  }

  const closeSkillEditor = () => {
    editorRequest.current += 1
    setSkillEditor(null)
  }

  const saveSkillEdit = async () => {
    if (!skillEditor || saving.current) {
      return
    }

    const editor = skillEditor
    const request = editorRequest.current
    saving.current = true
    setSkillSaving(true)

    try {
      const result = await editLearningNode(editor.name, skillDraft, profile)

      if (!result.ok) {
        throw new Error(result.message)
      }

      void queryClient.invalidateQueries({ queryKey: ['skill-content', editor.name, profileScopeKey(profile)] })
      void queryClient.invalidateQueries({ queryKey: skillsQueryKey(profile), exact: true })
      invalidateSlashCompletions()

      if (!mounted.current) {
        return
      }

      notify({ kind: 'success', title: t.skills.skillUpdated, message: t.skills.appliesToNewSessions(editor.name) })

      if (request === editorRequest.current) {
        setSkillEditor(null)
      }

      onRefresh()
    } catch (err) {
      if (mounted.current) {
        notifyError(err, editor.name)
      }
    } finally {
      saving.current = false

      if (mounted.current) {
        setSkillSaving(false)
      }
    }
  }

  const notice = installedError ? (
    <CatalogAlert onRetry={onRefresh} retryLabel={t.skills.refresh} title={t.skills.skillsLoadFailed}>
      {installedError instanceof Error ? installedError.message : null}
    </CatalogAlert>
  ) : installedPending ? (
    <Loader className="mx-auto my-2 size-6 text-(--ui-text-tertiary)" label={t.skills.loading} type="rose-curve" />
  ) : null

  return (
    <>
      {visibleSkills.length === 0 && visibleOfficial.length === 0 ? (
        <CapabilityEmpty noun="skills" query={query} />
      ) : (
        <MasterDetail pane={skillEditorPane} resizeId="capabilities-split" split="wide">
          <ListColumn
            header={
              <>
                {provenanceSummary && (
                  <div className="border-b border-(--ui-stroke-secondary) px-3 py-1 text-[0.65rem] text-(--ui-text-tertiary)">
                    {provenanceSummary}
                  </div>
                )}
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
              </>
            }
          >
            {visibleSkills.map(skill => (
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
            {visibleOfficial.length > 0 && (
              <div className="flex h-7 shrink-0 items-end px-2 pb-1 text-[0.62rem] font-medium uppercase tracking-wide text-(--ui-text-quaternary)">
                {t.skills.officialCatalog}
              </div>
      <SkillCatalog
        actions={
          <>
            <UpdateSkillsButton profile={profile} />
            <ListStripMenu
              items={[
                {
                  disabled: controlsDisabled || !skills.some(skill => skill.enabled && usageOf(skill) === 0),
                  label: t.skills.disableUnused,
                  onSelect: () =>
                    void applyEnabled(
                      skills.filter(skill => skill.enabled && usageOf(skill) === 0),
                      false,
                      true
                    )
                }
              ]}
              label={t.skills.tabSkills}
              toggle={bulkSwitch}
            />
          </>
        }
        installedPending={installedPending || Boolean(installedError)}
        notice={notice}
        onQueryChange={onQueryChange}
        profile={profile}
        query={query}
        renderInstalledAction={skill => (
          <Switch
            aria-label={skill.name}
            checked={skill.enabled}
            disabled={controlsDisabled}
            onCheckedChange={enabled => void applyEnabled([skill], enabled)}
            size="xs"
          />
        )}
        renderInstalledDetail={skill => (
          <>
            {usageOf(skill) > 0 && (
              <p className="text-xs text-(--ui-text-tertiary)">{t.skills.usageCount(usageOf(skill))}</p>
            )}
            <SkillDetail
              onArchive={() => {
                if (!saving.current) {
                  setArchiveTarget(skill.name)
                }
              }}
              onEdit={() => void openSkillEditor(skill.name)}
              profile={profile}
              skill={skill}
            />
            <p className="text-xs text-(--ui-text-tertiary)">{t.skills.changesApplyNewSessions}</p>
            {skillEditor?.name === skill.name && (
              <DetailPane
                actions={
                  <Button disabled={skillSaving} onClick={() => void saveSkillEdit()} size="xs">
                    {skillSaving ? t.common.saving : t.common.save}
                  </Button>
                }
                id="skill-editor"
                onClose={closeSkillEditor}
                title={
                  <span className="text-[0.68rem] font-normal text-muted-foreground/60">
                    {skillEditor.name}/SKILL.md
                  </span>
                }
              >
                <CodeEditor
                  disabled={skillSaving}
                  filePath="SKILL.md"
                  initialValue={skillDraft}
                  key={skillEditor.name}
                  onCancel={closeSkillEditor}
                  onChange={setSkillDraft}
                  onSave={() => void saveSkillEdit()}
                />
              </DetailPane>
            )}
          </>
        )}
        skills={skills}
      />
      {archiveTarget && (
        <ArchiveSkillConfirmDialog
          onApply={() => {
            const name = archiveTarget

            const snapshot =
              queryClient.getQueryData<SkillInfo[]>(skillsQueryKey(profile))?.find(skill => skill.name === name) ??
              skills.find(skill => skill.name === name)

            void queryClient.cancelQueries({ queryKey: skillsQueryKey(profile), exact: true })
            setSkills(current => current?.filter(skill => skill.name !== name))
            invalidateSlashCompletions()

            if (skillEditor?.name === name) {
              closeSkillEditor()
            }

            // Restore only this row; never clobber intervening toggles or installs.
            return () => {
              if (mounted.current) {
                setSkills(current =>
                  snapshot && current && !current.some(skill => skill.name === name) ? [...current, snapshot] : current
                )
              }

              void queryClient.invalidateQueries({ queryKey: skillsQueryKey(profile), exact: true })
            }
          }}
          onClose={() => setArchiveTarget(null)}
          onFailure={(err, name) => {
            if (mounted.current) {
              notifyError(err, name)
            }
          }}
          onSuccess={() => {
            void queryClient.invalidateQueries({ queryKey: skillsQueryKey(profile), exact: true })

            if (mounted.current) {
              onRefresh()
            }
          }}
          open
          profile={profile}
          skillId={archiveTarget}
          skillName={archiveTarget}
        />
      )}
    </>
  )
}
