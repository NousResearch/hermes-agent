import { useStore } from '@nanostores/react'

import { Button } from '@/components/ui/button'
import { useI18n } from '@/i18n'
import { $restoredDraftNotice, dismissRestoredDraftNotice, undoRestoredDraft } from '@/store/composer'

interface RestoredDraftNoticeProps {
  /** The composer is showing the fresh draft (no session scope). */
  freshDraft: boolean
  /** Clear the editor after Undo emptied the fresh draft. */
  onUndone: () => void
  /** Latest live editor text — Undo only applies while it is still what was restored. */
  readLiveText: () => string
}

/**
 * "Restored your unsent message" strip above the fresh draft's input
 * (#111868). Offers, never hijacks: no focus move, no navigation, no toast —
 * the text is simply in the composer with a way to put it back. Renders
 * nothing outside the fresh draft, so opening another session hides it
 * without consuming the Undo.
 */
export function RestoredDraftNotice({ freshDraft, onUndone, readLiveText }: RestoredDraftNoticeProps) {
  const notice = useStore($restoredDraftNotice)
  const { t } = useI18n()

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
