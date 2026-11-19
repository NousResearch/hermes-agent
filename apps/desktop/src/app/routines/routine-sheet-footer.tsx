import { useEffect, useRef } from 'react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import type { CronJob } from '@/hermes'
import { useI18n } from '@/i18n'
import { firstRunText, nextFirstRun } from '@/lib/routine-first-run'
import { $composerDraft, setComposerDraft } from '@/store/composer'

import { startSimplifiedAgentSession } from '../chat/sidebar/simplified-new-session-model'

import type { RoutineDraft } from './routine-draft'
import type { RoutineSource } from './routine-source'
import { useRoutinesCopy } from './routines-copy'
import { closeRoutineSheet } from './routines-store'

export const ROUTINE_FORM_ID = 'routine-sheet-form'

interface RoutineSheetFooterProps {
  job: CronJob | null
  draft: RoutineDraft
  source: RoutineSource
  saving: boolean
  confirmingDelete: boolean
  onConfirmDelete: (confirming: boolean) => void
  onDelete: () => void
}

/** The sheet's footer, on the sheet's own opaque fill so nothing scrolled under it shows through.
 *  New: when it first runs, then Cancel, Create in chat, Create routine. Edit: Delete routine, Cancel,
 *  Save & update routine; Delete swaps the row for its confirmation (rev 1407:9605). */
export function RoutineSheetFooter(props: RoutineSheetFooterProps) {
  if (props.confirmingDelete) {
    return <DeleteConfirmation onCancel={() => props.onConfirmDelete(false)} onDelete={props.onDelete} />
  }

  return props.job ? <EditFooter {...props} /> : <CreateFooter {...props} />
}

function CreateFooter({ draft, saving, source }: RoutineSheetFooterProps) {
  const { locale, t } = useI18n()
  const copy = useRoutinesCopy()
  const now = Date.now()
  const first = nextFirstRun(draft.pick, now)
  const when = first ? firstRunText(first, locale, now) : ''

  return (
    <footer className="routines-sheet-footer">
      {/* Hidden until a frequency is set, so the blank sheet says nothing. */}
      {when && (
        <p aria-live="polite" className="routines-first-run">
          <Codicon aria-hidden name="clock" size="0.875rem" />
          {first?.style === 'once' ? copy.runsOnce(when) : copy.firstRun(when)}
        </p>
      )}
      <div className="routines-sheet-actions">
        <Button disabled={saving} onClick={closeRoutineSheet} type="button" variant="ghost">
          {t.common.cancel}
        </Button>
        <span className="routines-sheet-actions-end">
          <Button
            disabled={saving}
            onClick={() => {
              closeRoutineSheet()
              startSimplifiedAgentSession({ connectionId: source.connectionId, profile: draft.profile })
              seedChatDraft(copy.askInChatPrompt)
            }}
            type="button"
            variant="secondary"
          >
            <Codicon name="comment-discussion" />
            {copy.createInChat}
          </Button>
          <Button form={ROUTINE_FORM_ID} loading={saving} type="submit">
            {t.cron.createAction}
          </Button>
        </span>
      </div>
    </footer>
  )
}

function EditFooter({ onConfirmDelete, saving }: RoutineSheetFooterProps) {
  const { t } = useI18n()
  const copy = useRoutinesCopy()

  return (
    <footer className="routines-sheet-footer">
      <div className="routines-sheet-actions">
        <Button disabled={saving} onClick={() => onConfirmDelete(true)} type="button" variant="destructiveOutline">
          {copy.deleteRoutine}
        </Button>
        <span className="routines-sheet-actions-end">
          <Button disabled={saving} onClick={closeRoutineSheet} type="button" variant="ghost">
            {t.common.cancel}
          </Button>
          <Button form={ROUTINE_FORM_ID} loading={saving} type="submit">
            {copy.saveRoutine}
          </Button>
        </span>
      </div>
    </footer>
  )
}

/** "Delete this routine?" in place of the footer's buttons; Keep it puts them back. Focus starts on
 *  Keep it, the safe answer. */
function DeleteConfirmation({ onCancel, onDelete }: { onCancel: () => void; onDelete: () => void }) {
  const copy = useRoutinesCopy()
  const keep = useRef<HTMLButtonElement>(null)

  useEffect(() => keep.current?.focus(), [])

  return (
    <footer className="routines-sheet-footer" data-confirming="">
      <div aria-labelledby="routine-delete-confirm" className="routines-sheet-actions" role="group">
        <div className="routines-delete-text">
          <p className="routines-delete-title" id="routine-delete-confirm">
            {copy.deleteConfirm}
          </p>
          <p className="routines-field-note">{copy.deleteConfirmHint}</p>
        </div>
        <span className="routines-sheet-actions-end">
          <Button onClick={onCancel} ref={keep} type="button" variant="secondary">
            {copy.keepIt}
          </Button>
          <Button onClick={onDelete} type="button" variant="destructive">
            {copy.actions.delete}
          </Button>
        </span>
      </div>
    </footer>
  )
}

/** "Create in chat" starts the new chat's composer on a starter sentence, once the fresh draft has
 *  replaced the previous one, and never over words the user already typed there. */
function seedChatDraft(text: string) {
  window.setTimeout(() => {
    if (!$composerDraft.get().trim()) {
      setComposerDraft(text)
    }
  }, 300)
}
