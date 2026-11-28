import { KeyboardSensor, PointerSensor, useSensor, useSensors } from '@dnd-kit/core'
import { arrayMove, sortableKeyboardCoordinates } from '@dnd-kit/sortable'
import { type ReactNode, useState } from 'react'

import { ReorderableList, useSortableBindings } from '@/app/chat/sidebar/reorderable-list'
import { StatusRow } from '@/components/chat/status-row'
import { StatusSection } from '@/components/chat/status-section'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { Tip } from '@/components/ui/tooltip'
import { type Translations, useI18n } from '@/i18n'
import { CornerDownLeft, iconSize, Pencil, SteeringWheel } from '@/lib/icons'
import { cn } from '@/lib/utils'
import { isSteerableEntry, type QueuedPromptEntry } from '@/store/composer-queue'

import { canMoveQueuedEntry } from './queue-reorder'

interface QueuePanelProps {
  busy: boolean
  editingId: null | string
  entries: QueuedPromptEntry[]
  onDelete: (id: string) => void
  /** Acknowledge a turn the gateway lost across a restart (`unknown` row) so
   *  the queued turns behind it flow again. Absent on non-canonical hosts. */
  onDiscardLost?: (id: string) => void
  onEdit: (entry: QueuedPromptEntry) => void
  /** Lift a park (explicit Stop/Esc halt) and let the queue flow again. */
  onResume: () => void
  /** Persist a new queue order (drag or ▲/▼): every id, in the order the user left them. Absent:
   *  the queue renders without reorder affordances. */
  onReorder?: (ids: string[]) => void
  onSendNow: (id: string) => void
  /** Deliver an entry as a mid-turn redirect (no interrupt). Absent when the
   *  host has no steer path — the affordance hides rather than dead-clicks. */
  onSteerNow?: (id: string) => void
  /** True after an explicit halt: entries wait until resumed / sent / edited. */
  parked: boolean
}

const entryPreview = (entry: QueuedPromptEntry, c: Translations['composer']) =>
  entry.displayKind === 'hidden'
    ? c.hiddenQueued
    : (entry.displayText ?? entry.text).trim() || (entry.attachments.length > 0 ? c.attachmentOnly : c.emptyTurn)

/** A preview long enough (or multiline enough) that two lines may still hide
 *  part of it — the entry gets an in-place expand/collapse toggle (#45664). */
const shouldOfferExpandedPreview = (preview: string) => preview.length > 140 || preview.includes('\n')

export function QueuePanel({
  busy,
  editingId,
  entries,
  onDelete,
  onDiscardLost,
  onEdit,
  onReorder,
  onResume,
  onSendNow,
  onSteerNow,
  parked
}: QueuePanelProps) {
  const { t } = useI18n()
  const c = t.composer
  const [expandedIds, setExpandedIds] = useState<ReadonlySet<string>>(() => new Set())

  // Same activation as the sidebar's lists: a press must travel before it lifts a row, so a click
  // on the grip (or a tiny tremor) never reorders by accident.
  const sensors = useSensors(
    useSensor(PointerSensor, { activationConstraint: { distance: 4 } }),
    useSensor(KeyboardSensor, { coordinateGetter: sortableKeyboardCoordinates })
  )

  const toggleExpanded = (id: string) => {
    setExpandedIds(current => {
      const next = new Set(current)

      if (next.has(id)) {
        next.delete(id)
      } else {
        next.add(id)
      }

      return next
    })
  }

  if (entries.length === 0) {
    return null
  }

  const ids = entries.map(entry => entry.id)
  const reorderable = Boolean(onReorder) && entries.length > 1

  const moveBy = (index: number, direction: -1 | 1) => onReorder?.(arrayMove(ids, index, index + direction))

  const renderRow = (entry: QueuedPromptEntry, sortable?: SortableRowBindings) => {
    const isEditing = editingId === entry.id
    const index = ids.indexOf(entry.id)
    const movable = reorderable && Boolean(sortable) && canMoveQueuedEntry(entry)
    const attachmentsCount = entry.attachments.length

    // Steer only surfaces where it can actually deliver: a live turn to
    // redirect and an entry the redirect can carry (text-only, no slash).
    // A server row can be steered only when its owner serves the verb for it.
    const canSteer =
      busy &&
      Boolean(onSteerNow) &&
      (entry.serverStatus ? Boolean(entry.serverControls?.includes('steer')) : isSteerableEntry(entry))

    // The owner died mid-turn and recovered this row as `unknown`: the
    // FIFO behind it is paused until someone acknowledges the loss.
    const lost = entry.serverStatus === 'unknown'
    const preview = entryPreview(entry, c)
    const canExpand = shouldOfferExpandedPreview(preview)
    const isExpanded = expandedIds.has(entry.id)

    return (
      <StatusRow
        className={cn(
          isEditing &&
            'ring-1 ring-inset ring-[color-mix(in_srgb,var(--dt-composer-ring)_40%,transparent)] bg-accent/25'
        )}
        dismiss={{ label: c.queueDelete, onDismiss: () => onDelete(entry.id) }}
        key={entry.id}
        leading={
          movable && sortable ? (
            <Tip label={c.queueReorder}>
              <span
                aria-label={c.queueReorder}
                className="flex cursor-grab touch-none items-center text-muted-foreground/70 hover:text-foreground/90 active:cursor-grabbing"
                data-reorder-handle=""
                data-slot="queue-reorder-handle"
                {...sortable.dragHandleProps}
              >
                <Codicon name="gripper" size="0.8rem" />
              </span>
            </Tip>
          ) : (
            <Codicon className="text-muted-foreground/70" name="comment" size="0.8rem" />
          )
        }
        trailing={
          <>
            {movable && (
              <>
                <Tip label={c.queueMoveUp}>
                  <Button
                    aria-label={c.queueMoveUp}
                    className="size-5 rounded-md"
                    disabled={index <= 0}
                    onClick={() => moveBy(index, -1)}
                    size="icon-xs"
                    type="button"
                    variant="ghost"
                  >
                    <Codicon name="arrow-up" size={iconSize.xs} />
                  </Button>
                </Tip>
                <Tip label={c.queueMoveDown}>
                  <Button
                    aria-label={c.queueMoveDown}
                    className="size-5 rounded-md"
                    disabled={index >= ids.length - 1}
                    onClick={() => moveBy(index, 1)}
                    size="icon-xs"
                    type="button"
                    variant="ghost"
                  >
                    <Codicon name="arrow-down" size={iconSize.xs} />
                  </Button>
                </Tip>
              </>
            )}
            {canExpand && (
              <Tip label={isExpanded ? c.queueCollapse : c.queueExpand}>
                <Button
                  aria-expanded={isExpanded}
                  aria-label={isExpanded ? c.queueCollapse : c.queueExpand}
                  className="size-5 rounded-md"
                  onClick={() => toggleExpanded(entry.id)}
                  size="icon-xs"
                  type="button"
                  variant="ghost"
                >
                  <Codicon
                    className={cn('transition-transform', isExpanded && 'rotate-180')}
                    name="chevron-down"
                    size={iconSize.xs}
                  />
                </Button>
              </Tip>
            )}
            {renderEntryActions({
              busy,
              c,
              editingId,
              entry,
              isEditing,
              lost,
              onDiscardLost,
              onEdit,
              onSendNow,
              onSteerNow,
              canSteer
            })}
          </>
        }
        trailingVisible={isEditing || lost}
      >
        <div className="min-w-0 flex-1">
          <p
            className={cn(
              'text-[0.73rem] leading-4 text-foreground/92',
              isExpanded ? 'max-h-40 overflow-y-auto whitespace-pre-wrap pr-1' : 'line-clamp-2 break-words'
            )}
          >
            {preview}
          </p>
          {(attachmentsCount > 0 || isEditing || lost) && (
            <div className="mt-0.5 flex items-center gap-1.5 text-[0.64rem] text-muted-foreground/75">
              {lost && <span data-slot="queue-lost-note">{c.queueLostNote}</span>}
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

  // A lost (`unknown`) turn pauses the FIFO until someone acknowledges it: its card and Discard
  // must be on screen without expanding the section, and the header names the pause (#134889 NF2).
  const lostEntries = entries.filter(entry => entry.serverStatus === 'unknown')
  const paused = parked || lostEntries.length > 0

  return (
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
      icon={<Codicon className="text-muted-foreground/70" name={paused ? 'debug-pause' : 'layers'} size="0.8rem" />}
      label={paused ? c.queuedPaused(entries.length) : c.queued(entries.length)}
      preview={lostEntries.length > 0 ? lostEntries.map(entry => renderRow(entry)) : undefined}
    >
      {reorderable && onReorder ? (
        <ReorderableList ids={ids} onReorder={onReorder} sensors={sensors}>
          {entries.map(entry => (
            <SortableQueueRow disabled={!canMoveQueuedEntry(entry)} id={entry.id} key={entry.id}>
              {sortable => renderRow(entry, sortable)}
            </SortableQueueRow>
          ))}
        </ReorderableList>
      ) : (
        entries.map(entry => renderRow(entry))
      )}
    </StatusSection>
  )
}

type SortableRowBindings = ReturnType<typeof useSortableBindings>

/** One queue row lifted into the panel's sortable list. A row that cannot move (a lost turn, a
 *  server row its owner offers no `move` for) stays a drop position but never lifts. */
function SortableQueueRow({
  children,
  disabled,
  id
}: {
  children: (sortable: SortableRowBindings) => ReactNode
  disabled: boolean
  id: string
}) {
  const sortable = useSortableBindings(id, { disabled })

  return (
    <div className={cn(sortable.dragging && 'relative z-10 opacity-90')} ref={sortable.ref} style={sortable.style}>
      {children(sortable)}
    </div>
  )
}

interface EntryActionsArgs {
  busy: boolean
  c: Translations['composer']
  canSteer: boolean
  editingId: null | string
  entry: QueuedPromptEntry
  isEditing: boolean
  lost: boolean
  onDiscardLost?: (id: string) => void
  onEdit: (entry: QueuedPromptEntry) => void
  onSendNow: (id: string) => void
  onSteerNow?: (id: string) => void
}

/** Per-row trailing actions: discard for a lost row, else edit/steer/send for a local one. A
 *  server row gets only the edit/steer its owner serves, never send-next: reordering it would
 *  break the owner's one FIFO. */
function renderEntryActions({
  busy,
  c,
  canSteer,
  editingId,
  entry,
  isEditing,
  lost,
  onDiscardLost,
  onEdit,
  onSendNow,
  onSteerNow
}: EntryActionsArgs) {
  return lost && onDiscardLost ? (
    <Tip label={c.queueLostDiscardTip}>
      <Button
        aria-label={c.queueLostDiscard}
        className="h-5 rounded-md px-1.5 text-[0.66rem]"
        data-slot="queue-lost-discard"
        onClick={() => onDiscardLost(entry.id)}
        size="micro"
        type="button"
        variant="text"
      >
        {c.queueLostDiscard}
      </Button>
    </Tip>
  ) : (
    <>
      {(!entry.serverStatus || entry.serverControls?.includes('replace')) && (
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
      )}
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
      {!entry.serverStatus && (
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
      )}
    </>
  )
}
