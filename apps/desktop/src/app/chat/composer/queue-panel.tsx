import { closestCenter, DndContext, type DragEndEvent } from '@dnd-kit/core'
import { arrayMove, SortableContext, useSortable, verticalListSortingStrategy } from '@dnd-kit/sortable'
import { useMemo } from 'react'

import { StatusRow } from '@/components/chat/status-row'
import { StatusSection } from '@/components/chat/status-section'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { Tip } from '@/components/ui/tooltip'
import { type Translations, useI18n } from '@/i18n'
import { CornerDownLeft, GripVertical, iconSize, Pencil, SteeringWheel, Trash2 } from '@/lib/icons'
import { cn } from '@/lib/utils'
import { isSteerableEntry, type QueuedPromptEntry } from '@/store/composer-queue'

interface QueuePanelProps {
  busy: boolean
  editingId: null | string
  entries: QueuedPromptEntry[]
  onDelete: (id: string) => void
  onEdit: (entry: QueuedPromptEntry) => void
  /** Lift a park (explicit Stop/Esc halt) and let the queue flow again. */
  onResume: () => void
  onReorderQueue: (ids: string[]) => void
  onSendNow: (id: string) => void
  /** Deliver an entry as a mid-turn redirect (no interrupt). Absent when the
   *  host has no steer path — the affordance hides rather than dead-clicks. */
  onSteerNow?: (id: string) => void
  /** True after an explicit halt: entries wait until resumed / sent / edited. */
  parked: boolean
}

const entryPreview = (entry: QueuedPromptEntry, c: Translations['composer']) =>
  (entry.displayText ?? entry.text).trim() || (entry.attachments.length > 0 ? c.attachmentOnly : c.emptyTurn)

export function QueuePanel({
  busy,
  editingId,
  entries,
  onDelete,
  onEdit,
  onResume,
  onReorderQueue,
  onSendNow,
  onSteerNow,
  parked
}: QueuePanelProps) {
  const { t } = useI18n()
  const c = t.composer

  const itemIds = useMemo(() => entries.map(entry => entry.id), [entries])

  const handleDragEnd = (event: DragEndEvent) => {
    const { active, over } = event

    if (!over || active.id === over.id) {
      return
    }

    const from = itemIds.indexOf(String(active.id))
    const to = itemIds.indexOf(String(over.id))

    if (from >= 0 && to >= 0) {
      onReorderQueue(arrayMove(itemIds, from, to))
    }
  }

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
      icon={<Codicon className="text-muted-foreground/70" name={parked ? 'debug-pause' : 'layers'} size="0.8rem" />}
      key={parked ? 'parked' : 'flowing'}
      label={parked ? c.queuedPaused(entries.length) : c.queued(entries.length)}
    >
      <DndContext collisionDetection={closestCenter} onDragEnd={handleDragEnd}>
        <SortableContext items={itemIds} strategy={verticalListSortingStrategy}>
          {entries.map(entry => (
            <SortableQueueRow
              busy={busy}
              c={c}
              editingId={editingId}
              entry={entry}
              key={entry.id}
              onDelete={onDelete}
              onEdit={onEdit}
              onSendNow={onSendNow}
              onSteerNow={onSteerNow}
            />
          ))}
        </SortableContext>
      </DndContext>
    </StatusSection>
  )
}

interface SortableQueueRowProps {
  busy: boolean
  c: Translations['composer']
  editingId: null | string
  entry: QueuedPromptEntry
  onDelete: (id: string) => void
  onEdit: (entry: QueuedPromptEntry) => void
  onSendNow: (id: string) => void
  onSteerNow?: (id: string) => void
}

function SortableQueueRow({
  busy,
  c,
  editingId,
  entry,
  onDelete,
  onEdit,
  onSendNow,
  onSteerNow
}: SortableQueueRowProps) {
  const { attributes, isDragging, listeners, setNodeRef, transform, transition } = useSortable({ id: entry.id })
  const isEditing = editingId === entry.id
  const attachmentsCount = entry.attachments.length
  // Steer only surfaces where it can actually deliver: a live turn to
  // redirect and an entry the redirect can carry (text-only, no slash).
  const canSteer = busy && Boolean(onSteerNow) && isSteerableEntry(entry)

  const style = {
    transform: transform ? `translate3d(0px, ${transform.y}px, 0)` : undefined,
    transition: isDragging ? undefined : transition,
    willChange: isDragging ? 'transform' : undefined
  }

  return (
    <StatusRow
      className={cn(
        'border border-transparent',
        isEditing && 'border-[color-mix(in_srgb,var(--dt-composer-ring)_40%,transparent)] bg-accent/25',
        isDragging && 'z-10 cursor-grabbing opacity-60'
      )}
      ref={setNodeRef}
      style={style}
      trailing={
        <>
          <Tip label={c.queueDrag}>
            <Button
              aria-label={c.queueDrag}
              className="size-5 cursor-grab rounded-md"
              data-reorder-handle
              size="icon-xs"
              type="button"
              variant="ghost"
              {...attributes}
              {...listeners}
            >
              <GripVertical className={iconSize.xs} />
            </Button>
          </Tip>
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
          {canSteer && (
            <Tip label={c.queueSteer}>
              <Button
                aria-label={c.queueSteer}
                className="size-5 rounded-md"
                disabled={isEditing}
                onClick={() => onSteerNow?.(entry.id)}
                size="icon-xs"
                type="button"
                variant="ghost"
              >
                <SteeringWheel className={iconSize.xs} />
              </Button>
            </Tip>
          )}
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
              <CornerDownLeft className={iconSize.xs} />
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
}
