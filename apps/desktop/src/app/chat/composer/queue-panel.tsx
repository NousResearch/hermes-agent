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
  /** Lift a park (explicit Stop/Esc halt) and let the queue flow again. */
  onResume: () => void
  onSendNow: (id: string) => void
  /** True after an explicit halt: entries wait until resumed / sent / edited. */
  parked: boolean
}

const entryPreview = (entry: QueuedPromptEntry, c: Translations['composer']) =>
  entry.text.trim() || (entry.attachments.length > 0 ? c.attachmentOnly : c.emptyTurn)

export function QueuePanel({ busy, editingId, entries, onDelete, onEdit, onResume, onSendNow, parked }: QueuePanelProps) {
  const { t } = useI18n()
  const c = t.composer

  if (entries.length === 0) {
    return null
  }

  return (
    // Keyed on the park flag: StatusSection owns its collapse state from
    // defaultCollapsed, so remount on park/unpark. A Stop must EXPAND the
    // panel — the halted prompts' only presence is here, and leaving them
    // behind a collapsed "N queued" pill is how they read as vanished.
    <StatusSection
      accessory={
        parked ? (
          <Tip label={c.queueResumeTip}>
            <Button
              className="text-muted-foreground/75 hover:text-foreground/90"
              onClick={onResume}
              size="micro"
              type="button"
              variant="text"
            >
              {c.queueResume}
            </Button>
          </Tip>
        ) : undefined
      }
      defaultCollapsed={!parked}
      icon={
        <Codicon
          className="text-muted-foreground/70"
          name={parked ? 'debug-pause' : 'layers'}
          size="0.8rem"
        />
      }
      key={parked ? 'parked' : 'flowing'}
      label={parked ? c.queuedPaused(entries.length) : c.queued(entries.length)}
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
