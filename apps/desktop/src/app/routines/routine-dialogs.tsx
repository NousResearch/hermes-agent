import { useStore } from '@nanostores/react'
import { useQuery } from '@tanstack/react-query'
import { useState } from 'react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { ConfirmDialog } from '@/components/ui/confirm-dialog'
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import { Field } from '@/components/ui/field'
import { Input } from '@/components/ui/input'
import { type AutomationBlueprint, type CronJob, getAutomationBlueprints, updateCronJob } from '@/hermes'
import { useI18n } from '@/i18n'
import { $composerDraft, setComposerDraft } from '@/store/composer'
import { notify } from '@/store/notifications'

import { startSimplifiedAgentSession } from '../chat/sidebar/simplified-new-session-model'
import { truncateText } from '../cron/job-state'

import { RoutineBlueprintForm } from './routine-blueprint-form'
import { blankDraft, draftFromJob } from './routine-draft'
import { RoutineEditorForm } from './routine-editor-form'
import { routineTitle } from './routine-model'
import { RoutinePicker } from './routine-picker'
import type { RoutineSource } from './routine-source'
import { useRoutinesCopy } from './routines-copy'
import {
  $routineEditor,
  $routinePendingDelete,
  $routineRename,
  closeRoutineEditor,
  openRoutineEditor,
  type RoutineEditor
} from './routines-store'
import type { RoutineActions } from './use-routine-actions'

interface RoutineDialogsProps {
  source: RoutineSource
  actions: RoutineActions
  /** A routine was just created or saved (the rail page opens it). */
  onSaved?: (job: CronJob) => void
}

/** The editor (R3 → R4) and the Delete confirm, mounted once by whichever surface hosts routines. */
export function RoutineDialogs({ actions, onSaved, source }: RoutineDialogsProps) {
  const { t } = useI18n()
  const editor = useStore($routineEditor)
  const pendingDelete = useStore($routinePendingDelete)

  return (
    <>
      <RoutineEditorDialog editor={editor} onSaved={onSaved} source={source} />
      <RenameDialog source={source} />
      <ConfirmDialog
        busyLabel={t.cron.deleting}
        confirmLabel={t.common.delete}
        description={
          pendingDelete ? (
            <>
              {t.cron.deleteDescPrefix}
              <span className="font-medium text-foreground">{truncateText(routineTitle(pendingDelete), 60)}</span>
              {t.cron.deleteDescSuffix}
            </>
          ) : null
        }
        destructive
        onClose={() => $routinePendingDelete.set(null)}
        onConfirm={async () => {
          if (pendingDelete) {
            await actions.remove(pendingDelete)
          }
        }}
        open={pendingDelete !== null}
        title={t.cron.deleteTitle}
      />
    </>
  )
}

interface RoutineEditorDialogProps {
  editor: RoutineEditor
  source: RoutineSource
  onSaved?: (job: CronJob) => void
}

type OpenEditor = Exclude<RoutineEditor, { mode: 'closed' }>

/** The ready-made catalog for the editor's agent (built-ins plus that agent's plugin blueprints),
 *  read while the picker or a ready-made form is open. */
function useEditorBlueprints(editor: RoutineEditor, source: RoutineSource) {
  const profile = editor.mode === 'pick' || editor.mode === 'create' ? editor.profile : ''
  const wanted = editor.mode === 'pick' || (editor.mode === 'create' && Boolean(editor.blueprintKey))

  return useQuery({
    queryKey: ['cron-blueprints', source.connectionId ?? '', profile],
    queryFn: async () => (await getAutomationBlueprints(profile, source.connectionId)).blueprints,
    enabled: wanted && Boolean(profile)
  })
}

function RoutineEditorDialog({ editor, onSaved, source }: RoutineEditorDialogProps) {
  const blueprints = useEditorBlueprints(editor, source)

  const done = (job: CronJob) => {
    closeRoutineEditor()
    onSaved?.(job)
  }

  return (
    <Dialog onOpenChange={value => !value && closeRoutineEditor()} open={editor.mode !== 'closed'}>
      {editor.mode !== 'closed' && (
        <DialogContent className="routines-dialog">
          <EditorBody
            blueprints={blueprints.data ?? []}
            done={done}
            editor={editor}
            loading={blueprints.isPending}
            source={source}
          />
        </DialogContent>
      )}
    </Dialog>
  )
}

interface EditorBodyProps {
  editor: OpenEditor
  blueprints: readonly AutomationBlueprint[]
  loading: boolean
  source: RoutineSource
  done: (job: CronJob) => void
}

function EditorHeader({ back, description, title }: { back?: () => void; description?: string; title: string }) {
  const { t } = useI18n()

  return (
    <DialogHeader>
      {back && (
        <button className="routines-dialog-back" onClick={back} type="button">
          <Codicon aria-hidden name="chevron-left" size="0.75rem" />
          {t.cron.createTitle}
        </button>
      )}
      <DialogTitle>{title}</DialogTitle>
      <DialogDescription className={description ? undefined : 'sr-only'}>{description ?? title}</DialogDescription>
    </DialogHeader>
  )
}

/** One step of the editor per mode: "What kind of routine?", a ready-made form, the blank form, an edit. */
function EditorBody({ blueprints, done, editor, loading, source }: EditorBodyProps) {
  const { t } = useI18n()
  const copy = useRoutinesCopy()

  if (editor.mode === 'edit') {
    return (
      <>
        <EditorHeader title={routineTitle(editor.job)} />
        <RoutineEditorForm
          initial={draftFromJob(editor.job)}
          job={editor.job}
          key={editor.job.id}
          lockProfile
          onCancel={closeRoutineEditor}
          onDone={done}
          source={source}
        />
      </>
    )
  }

  if (editor.mode === 'pick') {
    return (
      <>
        <EditorHeader description={copy.pickDescription} title={t.cron.createTitle} />
        <RoutinePicker
          blueprints={blueprints}
          loading={loading}
          lockProfile={editor.lockProfile}
          onAskInChat={() => {
            closeRoutineEditor()
            startSimplifiedAgentSession({ connectionId: source.connectionId, profile: editor.profile })
            seedChatDraft(copy.askInChatPrompt)
          }}
          onBlank={() => openRoutineEditor({ ...editor, fromPick: true, mode: 'create' })}
          onProfileChange={next => openRoutineEditor({ ...editor, profile: next })}
          onUse={item => openRoutineEditor({ ...editor, blueprintKey: item.key, fromPick: true, mode: 'create' })}
          profile={editor.profile}
        />
      </>
    )
  }

  const blueprint = editor.blueprintKey ? blueprints.find(item => item.key === editor.blueprintKey) : undefined

  const back = editor.fromPick
    ? () => openRoutineEditor({ lockProfile: editor.lockProfile, mode: 'pick', profile: editor.profile })
    : undefined

  return (
    <>
      <EditorHeader back={back} title={blueprint?.title ?? t.cron.createTitle} />
      {blueprint ? (
        <RoutineBlueprintForm
          blueprint={blueprint}
          key={`${editor.profile}:${blueprint.key}`}
          lockProfile={editor.lockProfile}
          onCancel={closeRoutineEditor}
          onDone={done}
          onProfileChange={next => openRoutineEditor({ ...editor, profile: next })}
          profile={editor.profile}
          source={source}
        />
      ) : (
        !editor.blueprintKey && (
          <RoutineEditorForm
            initial={blankDraft(editor.profile)}
            lockProfile={editor.lockProfile}
            onCancel={closeRoutineEditor}
            onDone={done}
            source={source}
          />
        )
      )}
    </>
  )
}

/** "Ask in chat" starts the new chat's composer on a starter sentence, once the fresh draft has
 *  replaced the previous one, and never over words the user already typed there. */
function seedChatDraft(text: string) {
  window.setTimeout(() => {
    if (!$composerDraft.get().trim()) {
      setComposerDraft(text)
    }
  }, 300)
}

/** Rename, a rail quick action (Figma 1098:8624): the routine's name only. */
function RenameDialog({ source }: { source: RoutineSource }) {
  const job = useStore($routineRename)

  return (
    <Dialog onOpenChange={open => !open && $routineRename.set(null)} open={job !== null}>
      {job && (
        <DialogContent className="routines-dialog">
          <RenameForm job={job} key={job.id} source={source} />
        </DialogContent>
      )}
    </Dialog>
  )
}

function RenameForm({ job, source }: { job: CronJob; source: RoutineSource }) {
  const { t } = useI18n()
  const copy = useRoutinesCopy()
  const [name, setName] = useState(routineTitle(job))
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<null | string>(null)

  async function submit(event: React.FormEvent) {
    event.preventDefault()
    setSaving(true)
    setError(null)

    try {
      const saved = await source.mutate(() => updateCronJob(job.id, { name: name.trim() }, source.scopeFor(job)))

      if (saved) {
        notify({ kind: 'success', title: t.cron.updated, message: routineTitle(saved) })
        $routineRename.set(null)
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : t.cron.failedSave)
    } finally {
      setSaving(false)
    }
  }

  return (
    <>
      <EditorHeader title={copy.renameTitle} />
      <form className="routines-form" onSubmit={submit}>
        <Field htmlFor="routine-rename" label={copy.name}>
          <Input id="routine-rename" onChange={event => setName(event.target.value)} value={name} />
        </Field>
        {error && (
          <p className="routines-form-error" role="alert">
            {error}
          </p>
        )}
        <DialogFooter>
          <Button disabled={saving} onClick={() => $routineRename.set(null)} type="button" variant="outline">
            {t.common.cancel}
          </Button>
          <Button disabled={!name.trim()} loading={saving} type="submit">
            {copy.rename}
          </Button>
        </DialogFooter>
      </form>
    </>
  )
}
