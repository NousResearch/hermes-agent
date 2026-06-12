import { StatusRow } from '@/components/chat/status-row'
import { StatusSection } from '@/components/chat/status-section'
import { Button } from '@/components/ui/button'
import { type Translations, useI18n } from '@/i18n'
import { cn } from '@/lib/utils'
import type { QueuedPromptEntry } from '@/store/composer-queue'

interface QueuePanelProps {
  busy: boolean
  editingId: null | string
  entries: QueuedPromptEntry[]
  onDelete: (id: string) => void
  onEdit: (entry: QueuedPromptEntry) => void
  onSendNow: (id: string) => void
}

const entryPreview = (entry: QueuedPromptEntry, c: Translations['composer']) =>
  entry.text.trim() || (entry.attachments.length > 0 ? c.attachmentOnly : c.emptyTurn)

export function QueuePanel({ busy, editingId, entries, onDelete, onEdit, onSendNow }: QueuePanelProps) {
  const { t } = useI18n()
  const c = t.composer

  if (entries.length === 0) {
    return null
  }

  return (
    <StatusSection label={c.queued(entries.length)}>
      {entries.map(entry => {
        const isEditing = editingId === entry.id
        const attachmentsCount = entry.attachments.length

        return (
          <StatusRow
            className={cn(
              'border border-transparent',
              isEditing && 'border-[color-mix(in_srgb,var(--dt-composer-ring)_40%,transparent)] bg-accent/25'
            )}
            key={entry.id}
            leading={
              <span aria-hidden className="size-3.5 shrink-0 rounded-full border border-foreground/35 bg-transparent" />
            }
            trailing={
              <>
                <Button
                  disabled={Boolean(editingId) && !isEditing}
                  onClick={() => onEdit(entry)}
                  size="micro"
                  type="button"
                  variant="text"
                >
                  {c.queueEdit}
                </Button>
                <Button
                  disabled={isEditing}
                  onClick={() => onSendNow(entry.id)}
                  size="micro"
                  type="button"
                  variant="secondary"
                >
                  {busy ? c.queueSendNext : c.queueSend}
                </Button>
                <Button onClick={() => onDelete(entry.id)} size="micro" type="button" variant="text">
                  {c.queueDelete}
                </Button>
              </>
            }
            trailingVisible={isEditing}
          >
            <div className="min-w-0 flex-1">
              <p className="truncate text-[0.73rem] leading-4 text-foreground/92">{entryPreview(entry, c)}</p>
              {(attachmentsCount > 0 || isEditing) && (
                <div className="mt-0.5 flex items-center gap-1.5 text-[0.64rem] text-muted-foreground/75">
                  {attachmentsCount > 0 && <span>{c.attachments(attachmentsCount)}</span>}
                  {isEditing && (
                    <span className="text-[color-mix(in_srgb,var(--dt-composer-ring)_78%,var(--muted-foreground))]">
                      {c.editingInComposer}
                    </span>
                  )}
                </div>
              )}
            </div>
          </StatusRow>
        )
      })}
    </StatusSection>
  )
}
