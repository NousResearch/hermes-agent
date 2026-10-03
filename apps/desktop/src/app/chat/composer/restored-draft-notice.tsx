import { useStore } from '@nanostores/react'

import { Button } from '@/components/ui/button'
import { useI18n } from '@/i18n'
import {
  $restoredDraftNotice,
  $salvagedEditNoticesBySession,
  type ComposerAttachment,
  dismissRestoredDraftNotice,
  dismissSalvagedEdit,
  takeSalvagedEditAttachments,
  undoRestoredDraft,
  undoSalvagedEdit
} from '@/store/composer'

interface RestoredDraftNoticeProps {
  /** The composer is showing the fresh draft (no session scope). */
  freshDraft: boolean
  /** The composer's queue session key, when it has one. */
  sessionKey: string | null
  /** Clear the editor after Undo emptied the fresh draft. */
  onUndone: () => void
  /** Repaint the editor after Undo put text back into it. */
  onRestored: (text: string, attachments?: ComposerAttachment[]) => void
  /** Latest live editor text — Undo only applies while it is still what was restored. */
  readLiveText: () => string
  /** The composer surface id, when this composer has one (#88621 review R7):
   * a record published by another pane of the same session is that pane's to
   * consume, never this one's. */
  surfaceId?: string | null
}

/**
 * "Restored your unsent message" strip above the fresh draft's input
 * (#111868), and its session-scoped sibling for a salvaged queued edit
 * (#88621). Offers, never hijacks: no focus move, no navigation, no toast —
 * the text is simply in the composer with a way to put it back. The restored
 * notice renders nothing outside the fresh draft, so opening another session
 * hides it without consuming the Undo; the salvage notice renders only on the
 * composer whose session it names.
 */
export function RestoredDraftNotice({
  freshDraft,
  onUndone,
  onRestored,
  readLiveText,
  sessionKey,
  surfaceId
}: RestoredDraftNoticeProps) {
  const notice = useStore($restoredDraftNotice)
  const salvagedBySession = useStore($salvagedEditNoticesBySession)
  const { t } = useI18n()

  const salvaged = sessionKey ? (salvagedBySession[sessionKey] ?? null) : null

  if (salvaged) {
    // On the vanished-entry teardown path the dirty buffer stayed in the
    // editor: Undo would repaint what is already there. Offer no button that
    // reads as a no-op — the notice is informational there (#88621).
    const undoApplies = salvaged.currentText !== salvaged.undoText

    // A record that names another pane of this same session belongs to that
    // pane (#88621 review R7): this one must not consume it — and its
    // dismiss control stays, so this pane can still clear the strip.
    const ownsRecord = !salvaged.surfaceKey || !surfaceId || salvaged.surfaceKey === surfaceId

    return (
      <div
        className="flex items-center justify-between gap-2 rounded-lg border border-[color:mix(in_srgb,var(--dt-composer-ring)_32%,transparent)] bg-accent/18 px-2 py-1"
        data-slot="composer-salvaged-edit"
        role="status"
      >
        <div className="min-w-0 text-[0.7rem] text-muted-foreground/88">{t.composer.salvagedEditNotice}</div>
        <div className="flex shrink-0 items-center gap-1">
          {undoApplies && (
            <Button
              className="h-6 rounded-md px-2 text-[0.68rem]"
              onClick={() => {
                if (!sessionKey) {
                  return
                }

                // Consume only what this surface owns; the payload Undo paints
                // is the salvaged text AND its attachment membership (#88621
                // review N1) — never the typed words on top of whatever chips
                // happen to be live now.
                const attachments = takeSalvagedEditAttachments(sessionKey)
                const text = undoSalvagedEdit(sessionKey, readLiveText(), surfaceId ?? undefined)

                if (text !== null) {
                  onRestored(text, attachments)
                }
              }}
              type="button"
              variant="ghost"
            >
              {t.composer.salvagedEditUndo}
            </Button>
          )}
          <Button
            aria-label={t.common.close}
            className="h-6 rounded-md px-2 text-[0.68rem]"
            onClick={() => {
              if (sessionKey && ownsRecord) {
                dismissSalvagedEdit(sessionKey)
              }
            }}
            type="button"
            variant="ghost"
          >
            ×
          </Button>
        </div>
      </div>
    )
  }

  if (!notice || !freshDraft) {
    return null
  }

  return (
    <div
      className="flex items-center justify-between gap-2 rounded-lg border border-[color-mix(in_srgb,var(--dt-composer-ring)_32%,transparent)] bg-accent/18 px-2 py-1"
      data-slot="composer-restored-draft"
      role="status"
    >
      <div className="min-w-0 text-[0.7rem] text-muted-foreground/88">{t.composer.restoredDraftNotice}</div>
      <div className="flex shrink-0 items-center gap-1">
        <Button
          className="h-6 rounded-md px-2 text-[0.68rem]"
          onClick={() => {
            if (undoRestoredDraft(readLiveText())) {
              onUndone()
            }
          }}
          type="button"
          variant="ghost"
        >
          {t.composer.restoredDraftUndo}
        </Button>
        <Button
          aria-label={t.common.close}
          className="h-6 rounded-md px-2 text-[0.68rem]"
          onClick={dismissRestoredDraftNotice}
          type="button"
          variant="ghost"
        >
          ×
        </Button>
      </div>
    </div>
  )
}
