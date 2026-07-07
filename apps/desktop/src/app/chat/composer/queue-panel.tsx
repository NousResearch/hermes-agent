import { StatusRow } from '@/components/chat/status-row'
import { StatusSection } from '@/components/chat/status-section'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { Tip } from '@/components/ui/tooltip'
import { type Translations, useI18n } from '@/i18n'
import { ArrowUp, iconSize, Pencil, Trash2 } from '@/lib/icons'
import { cn } from '@/lib/utils'
import type { QueuedPromptEntry } from '@/store/composer-queue'

interface QueuePanelProps {
  busy: boolean
  editingId: null | string
  entries: QueuedPromptEntry[]
  onDelete: (id: string) => void
  onEdit: (entry: QueuedPromptEntry) => void
  onSendAll: () => void
  onSendNow: (id: string) => void
}

const entryPreview = (entry: QueuedPromptEntry, c: Translations['composer']) =>
  entry.text.trim() || (entry.attachments.length > 0 ? c.attachmentOnly : c.emptyTurn)

export function QueuePanel({ busy, editingId, entries, onDelete, onEdit, onSendAll, onSendNow }: QueuePanelProps) {
  const { t } = useI18n()
  const c = t.composer
  const sendAllLabel = busy ? c.queueSendAllNext : c.queueSendAll

  if (entries.length === 0) {
    return null
  }

  return (
    <StatusSection
      accessory={
        <Button
          aria-label={sendAllLabel}
          disabled={Boolean(editingId)}
          onClick={onSendAll}
          size="micro"
          type="button"
          variant="text"
        >
          {sendAllLabel}
        </Button>
      }
      icon={<Codicon className="text-muted-foreground/70" name="layers" size="0.8rem" />}
      label={c.queued(entries.length)}
    >
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
            trailing={
              <>
                <Tip label={c.queueEdit}>
                  <Button
                    aria-label={c.queueEdit}
                    className="size-5 rounded-md"
                    disabled={Boolean(editingId) && !isEditing}
                    onClick={() => onEdit(entry)}
                    size="icon-xs"
                    type="button"
                    variant="ghost"
                  >
                    <Pencil className={iconSize.xs} />
                  </Button>
                </Tip>
                <Tip label={busy ? c.queueSendNext : c.queueSend}>
                  <Button
                    aria-label={busy ? c.queueSendNext : c.queueSend}
                    className="size-5 rounded-md"
                    disabled={isEditing}
                    onClick={() => onSendNow(entry.id)}
                    size="icon-xs"
                    type="button"
                    variant="ghost"
                  >
                    <ArrowUp className={iconSize.xs} />
                  </Button>
                </Tip>
                <Tip label={c.queueDelete}>
                  <Button
                    aria-label={c.queueDelete}
                    className="size-5 rounded-md"
                    onClick={() => onDelete(entry.id)}
                    size="icon-xs"
                    type="button"
                    variant="ghost"
                  >
                    <Trash2 className={iconSize.xs} />
                  </Button>
                </Tip>
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
