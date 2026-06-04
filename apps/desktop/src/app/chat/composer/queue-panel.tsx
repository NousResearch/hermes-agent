import { useState } from 'react'

import { Button } from '@/components/ui/button'
import { DisclosureCaret } from '@/components/ui/disclosure-caret'
import { ArrowUp, Pencil, Trash2 } from '@/lib/icons'
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

const entryPreview = (entry: QueuedPromptEntry) =>
  entry.text.trim() || (entry.attachments.length > 0 ? 'Attachment-only turn' : 'Empty turn')

export function QueuePanel({ busy, editingId, entries, onDelete, onEdit, onSendNow }: QueuePanelProps) {
  const [collapsed, setCollapsed] = useState(false)

  if (entries.length === 0) {
    return null
  }

  return (
    <div className="rounded-2xl border border-border/65 bg-[color-mix(in_srgb,var(--dt-card)_70%,transparent)] py-0.5 shadow-[0_0_0_1px_color-mix(in_srgb,var(--dt-card)_30%,transparent)_inset]">
      <button
        className="flex w-full items-center gap-1.5 px-2.5 py-1 text-left text-[0.72rem] font-medium text-muted-foreground/92 transition-colors hover:text-foreground/90"
        onClick={() => setCollapsed(open => !open)}
        type="button"
      >
        <DisclosureCaret className="shrink-0" open={!collapsed} size="0.875rem" />
        <span className="truncate">{entries.length} Queued</span>
      </button>

      {!collapsed && (
        <div className="space-y-0.5 px-1.5 pb-0.5">
          {entries.map(entry => {
            const isEditing = editingId === entry.id
            const attachmentsCount = entry.attachments.length

            return (
              <div
                className={cn(
                  'group/queue-row flex items-center gap-1.5 rounded-lg border border-transparent px-1.5 py-1',
                  'transition-colors duration-300 ease-out hover:bg-(--chrome-action-hover) hover:transition-none',
                  isEditing && 'border-[color-mix(in_srgb,var(--dt-composer-ring)_40%,transparent)] bg-accent/25'
                )}
                key={entry.id}
              >
                <span
                  aria-hidden
                  className="h-3.5 w-3.5 shrink-0 rounded-full border border-foreground/35 bg-transparent"
                />
                <div className="min-w-0 flex-1">
                  <p className="truncate text-[0.73rem] leading-4 text-foreground/92">{entryPreview(entry)}</p>
                  {(attachmentsCount > 0 || isEditing) && (
                    <div className="mt-0.5 flex items-center gap-1.5 text-[0.64rem] text-muted-foreground/75">
                      {attachmentsCount > 0 && (
                        <span>
                          {attachmentsCount} attachment{attachmentsCount === 1 ? '' : 's'}
                        </span>
                      )}
                      {isEditing && (
                        <span className="text-[color-mix(in_srgb,var(--dt-composer-ring)_78%,var(--muted-foreground))]">
                          Editing in composer
                        </span>
                      )}
                    </div>
                  )}
                </div>
                <div
                  className={cn(
                    'flex shrink-0 items-center gap-0 transition-opacity',
                    isEditing
                      ? 'opacity-100'
                      : 'opacity-0 group-hover/queue-row:opacity-100 group-focus-within/queue-row:opacity-100'
                  )}
                >
                  <Button
                    aria-label="Edit queued turn"
                    className="h-5 w-5 rounded-md"
                    disabled={Boolean(editingId) && !isEditing}
                    onClick={() => onEdit(entry)}
                    size="icon-xs"
                    title="Edit queued turn"
                    type="button"
                    variant="ghost"
                  >
                    <Pencil size={11} />
                  </Button>
                  <Button
                    aria-label="Send queued turn now"
                    className="h-5 w-5 rounded-md"
                    disabled={busy || isEditing}
                    onClick={() => onSendNow(entry.id)}
                    size="icon-xs"
                    title="Send queued turn now"
                    type="button"
                    variant="ghost"
                  >
                    <ArrowUp size={11} />
                  </Button>
                  <Button
                    aria-label="Delete queued turn"
                    className="h-5 w-5 rounded-md"
                    onClick={() => onDelete(entry.id)}
                    size="icon-xs"
                    title="Delete queued turn"
                    type="button"
                    variant="ghost"
                  >
                    <Trash2 size={11} />
                  </Button>
                </div>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
