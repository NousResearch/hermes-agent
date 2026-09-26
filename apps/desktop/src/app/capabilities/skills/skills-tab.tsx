import { useCallback, useEffect, useRef, useState } from 'react'

import { ArchiveSkillConfirmDialog } from '@/app/learning/archive-skill-confirm-dialog'
import { CodeEditor } from '@/components/chat/code-editor'
import { Button } from '@/components/ui/button'
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
    <p className="px-3 py-2 text-xs text-(--ui-text-tertiary)" role="status">
      {t.skills.loading}
    </p>
  ) : null

  return (
    <>
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
