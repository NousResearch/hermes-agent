import { useState } from 'react'

import { StatusRow } from '@/components/chat/status-row'
import { StatusSection } from '@/components/chat/status-section'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { Tip } from '@/components/ui/tooltip'
import { type Translations, useI18n } from '@/i18n'
import { CornerDownLeft, iconSize, Pencil, SteeringWheel } from '@/lib/icons'
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
  onSendNow: (id: string) => void
  /** Deliver an entry as a mid-turn redirect (no interrupt). Absent when the
   *  host has no steer path — the affordance hides rather than dead-clicks. */
  onSteerNow?: (id: string) => void
  /** True after an explicit halt: entries wait until resumed / sent / edited. */
  parked: boolean
  /** Whether the running turn carries Daybreak: only entries asking for the same program steer it. */
  runningDaybreak?: boolean
  requestedDaybreak?: boolean
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
  onEdit,
  onResume,
  onSendNow,
  onSteerNow,
  parked,
  requestedDaybreak,
  runningDaybreak = false
}: QueuePanelProps) {
  const { t } = useI18n()
  const c = t.composer
  const [expandedIds, setExpandedIds] = useState<ReadonlySet<string>>(() => new Set())

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
      icon={<Codicon className="text-muted-foreground/70" name={parked ? 'debug-pause' : 'layers'} size="0.8rem" />}
      label={parked ? c.queuedPaused(entries.length) : c.queued(entries.length)}
    >
      {entries.map(entry => {
        const isEditing = editingId === entry.id
        const attachmentsCount = entry.attachments.length
        // Steer only surfaces where it can actually deliver: a live turn to
        // redirect and an entry the redirect can carry (text-only, no slash).
        const canSteer = busy && Boolean(onSteerNow) && isSteerableEntry(entry, runningDaybreak, requestedDaybreak)
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
            leading={<Codicon className="text-muted-foreground/70" name="comment" size="0.8rem" />}
            trailing={
              <>
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
              </>
            }
            trailingVisible={isEditing}
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
