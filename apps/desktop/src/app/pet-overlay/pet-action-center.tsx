import { useCallback, useEffect, useRef, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { Input } from '@/components/ui/input'
import { Textarea } from '@/components/ui/textarea'
import { useI18n } from '@/i18n'
import { cn } from '@/lib/utils'
import type { PetActionCenterLiveStatus, PetActionCenterState } from '@/store/pet-action-center'
import type { PetActionCenterApprovalChoice, PetActionCenterControl } from '@/store/pet-overlay'
import { sendPetOverlayControl } from '@/store/pet-overlay'

import { LiveStatusStrip } from './live-status-strip'
import { type LiveTextAction, LiveTurnActions, type LiveTurnItem } from './live-turn-actions'

// ── Types ─────────────────────────────────────────────────────────────────

type Item = NonNullable<PetActionCenterState['items'][number]>
type ApprovalItem = Extract<Item, { kind: 'approval' }>
type ClarifyItem = Extract<Item, { kind: 'clarify' }>
type ActionCenterStrings = ReturnType<typeof useI18n>['t']['pet']['actionCenter']

interface PetActionCenterProps {
  state: PetActionCenterState
  /** Fired when the panel opens or closes so the parent can manage overlay
   *  focus/click-through centrally. Never fired for incoming state changes
   *  that don't involve an explicit user click. */
  onOpenChange?: (open: boolean) => void
}

// ── Helpers ───────────────────────────────────────────────────────────────

function isApprovalItem(item: Item): item is ApprovalItem {
  return item.kind === 'approval'
}

function isClarifyItem(item: Item): item is ClarifyItem {
  return item.kind === 'clarify'
}

function isLiveTurnItem(item: Item): item is LiveTurnItem {
  return item.kind === 'live-turn'
}

function liveStatusFor(item: Item): PetActionCenterLiveStatus | null {
  if (isLiveTurnItem(item)) {
    return {
      activityKind: item.activityKind,
      activityName: item.activityName,
      connectionState: item.connectionState,
      queuedCount: item.queuedCount,
      status: item.status,
      turnStartedAt: item.turnStartedAt
    }
  }

  return item.liveStatus ?? null
}

function selectedItem(state: PetActionCenterState): Item | null {
  const { items, selectedItemId } = state

  if (selectedItemId) {
    const found = items.find(i => i.id === selectedItemId)

    if (found) {
      return found
    }
  }

  return items[0] ?? null
}

const FOCUSABLE_SELECTOR = [
  'a[href]',
  'button:not([disabled])',
  'input:not([disabled])',
  'select:not([disabled])',
  'textarea:not([disabled])',
  '[tabindex]:not([tabindex="-1"])'
].join(',')

function isVisibleWithin(element: HTMLElement, dialog: HTMLElement): boolean {
  let current: HTMLElement | null = element

  while (current && dialog.contains(current)) {
    const style = window.getComputedStyle(current)

    if (current.hidden || current.getAttribute('aria-hidden') === 'true' || style.display === 'none' || style.visibility === 'hidden') {
      return false
    }

    current = current.parentElement
  }

  return true
}

function dialogFocusableElements(dialog: HTMLElement): HTMLElement[] {
  return Array.from(dialog.querySelectorAll<HTMLElement>(FOCUSABLE_SELECTOR)).filter(
    element => element.tabIndex >= 0 && isVisibleWithin(element, dialog)
  )
}

// ── Component ─────────────────────────────────────────────────────────────

/**
 * Gateway-less action center for the pet overlay. Renders a trigger button
 * when collapsed; a click opens an inline panel that surfaces pending
 * approvals and clarify requests. All controls flow through
 * `sendPetOverlayControl` — no gateway imports, no profile/session/route
 * values leave the component.
 *
 * The panel never auto-opens or auto-focuses on new events. Only an explicit
 * user click on the trigger opens it. Closing the panel restores focus to the
 * trigger.
 */
export function PetActionCenter({ state, onOpenChange }: PetActionCenterProps) {
  const { t } = useI18n()
  const ac = t.pet.actionCenter

  const [panelOpen, setPanelOpen] = useState(false)
  const triggerRef = useRef<HTMLButtonElement | null>(null)
  const dialogRef = useRef<HTMLDivElement | null>(null)
  const focusFrameRef = useRef<number | null>(null)

  const cancelScheduledFocus = useCallback(() => {
    if (focusFrameRef.current !== null) {
      cancelAnimationFrame(focusFrameRef.current)
      focusFrameRef.current = null
    }
  }, [])

  const scheduleFocus = useCallback(
    (focus: () => void) => {
      cancelScheduledFocus()
      focusFrameRef.current = requestAnimationFrame(() => {
        focusFrameRef.current = null
        focus()
      })
    },
    [cancelScheduledFocus]
  )

  useEffect(() => cancelScheduledFocus, [cancelScheduledFocus])

  useEffect(() => {
    if (panelOpen) {
      scheduleFocus(() => dialogRef.current?.focus())
    }
  }, [panelOpen, scheduleFocus])

  // ── Selected item (derived from state, with fallback) ──────────────────
  const item = selectedItem(state)

  // ── Nested ephemeral state ──────────────────────────────────────────────
  type NestedMode = 'none' | 'confirm-always' | 'deny-reason'
  const [nestedMode, setNestedMode] = useState<NestedMode>('none')
  const [denyReason, setDenyReason] = useState('')
  const [clarifyChoice, setClarifyChoice] = useState<string | null>(null)
  const [clarifyText, setClarifyText] = useState('')
  const [liveDraft, setLiveDraft] = useState('')
  const nestedSourceRef = useRef<HTMLButtonElement | null>(null)
  const denyInputRef = useRef<HTMLInputElement | null>(null)
  const alwaysCancelRef = useRef<HTMLButtonElement | null>(null)
  const restoreNestedFocusRef = useRef(false)
  const pendingDraftActionRef = useRef<{ action: LiveTextAction; itemId: string } | null>(null)

  // Reset nested state when the selected item changes or disappears.
  const itemId = item?.id ?? null
  const itemIdRef = useRef(itemId)
  useEffect(() => {
    if (itemIdRef.current !== itemId) {
      itemIdRef.current = itemId
      nestedSourceRef.current = null
      restoreNestedFocusRef.current = false
      setNestedMode('none')
      setDenyReason('')
      setClarifyChoice(null)
      setClarifyText('')
      setLiveDraft('')
      pendingDraftActionRef.current = null
    }
  }, [itemId])

  // Nested layers are always opened by an explicit button click. Focus their
  // safe first control after the transition, and only restore the remembered
  // source when the user explicitly cancels that layer.
  useEffect(() => {
    if (nestedMode === 'deny-reason') {
      scheduleFocus(() => denyInputRef.current?.focus())

      return
    }

    if (nestedMode === 'confirm-always') {
      scheduleFocus(() => alwaysCancelRef.current?.focus())

      return
    }

    if (restoreNestedFocusRef.current) {
      restoreNestedFocusRef.current = false
      const source = nestedSourceRef.current

      if (source?.isConnected) {
        scheduleFocus(() => {
          if (source.isConnected) {
            source.focus()
          }
        })
      }
    }
  }, [nestedMode, scheduleFocus])

  // ── Action status ──────────────────────────────────────────────────────
  const action = state.action
  const isSubmitting = action?.status === 'submitting'
  const isSteerRejected = action?.status === 'steer-rejected' && action.itemId === itemId

  useEffect(() => {
    const pending = pendingDraftActionRef.current

    if (!pending || pending.itemId !== item?.id || action?.itemId !== item.id || action.status === 'submitting') {
      return
    }

    if (
      (action.status === 'success' && pending.action === 'send') ||
      (action.status === 'steered' && pending.action === 'steer') ||
      (action.status === 'queued' && pending.action === 'queue')
    ) {
      setLiveDraft('')
    }

    pendingDraftActionRef.current = null
  }, [action?.itemId, action?.status, item?.id, item?.kind])

  // ── Control dispatch ────────────────────────────────────────────────────
  function send(control: PetActionCenterControl) {
    sendPetOverlayControl(control)
  }

  // ── Panel open/close ────────────────────────────────────────────────────
  function openPanel() {
    setPanelOpen(true)
    onOpenChange?.(true)
  }

  function closePanel() {
    setPanelOpen(false)
    nestedSourceRef.current = null
    restoreNestedFocusRef.current = false
    setNestedMode('none')
    setDenyReason('')
    setClarifyChoice(null)
    setClarifyText('')
    onOpenChange?.(false)
    // Restore focus to trigger if still connected.
    scheduleFocus(() => triggerRef.current?.focus())
  }

  // ── Keyboard ─────────────────────────────────────────────────────────────
  function onDialogKeyDown(e: React.KeyboardEvent) {
    if (e.key === 'Tab') {
      const dialog = dialogRef.current

      if (!dialog) {
        return
      }

      const focusable = dialogFocusableElements(dialog)
      const activeIndex = focusable.indexOf(document.activeElement as HTMLElement)

      const next = e.shiftKey
        ? activeIndex <= 0
          ? focusable.at(-1)
          : null
        : activeIndex === -1 || activeIndex === focusable.length - 1
          ? focusable[0]
          : null

      if (next) {
        e.preventDefault()
        next.focus()
      } else if (focusable.length === 0) {
        e.preventDefault()
        dialog.focus()
      }

      return
    }

    if (e.key === 'Escape') {
      e.stopPropagation()

      if (nestedMode !== 'none') {
        cancelNested()
      } else {
        closePanel()
      }
    }
  }

  // ── Approval actions ─────────────────────────────────────────────────────
  function sendApproval(choice: PetActionCenterApprovalChoice, reason?: string) {
    if (!item) {
      return
    }

    send(
      reason !== undefined
        ? { type: 'action-center-approval', itemId: item.id, choice, reason }
        : { type: 'action-center-approval', itemId: item.id, choice }
    )
  }

  function onApproveOnce() {
    sendApproval('approve-once')
  }

  function onApproveSession() {
    sendApproval('approve-session')
  }

  function onApproveAlwaysClick(e: React.MouseEvent<HTMLButtonElement>) {
    nestedSourceRef.current = e.currentTarget
    restoreNestedFocusRef.current = false
    setNestedMode('confirm-always')
  }

  function onApproveAlwaysConfirm() {
    restoreNestedFocusRef.current = false
    sendApproval('approve-always')
    setNestedMode('none')
  }

  function onDenyClick(e: React.MouseEvent<HTMLButtonElement>) {
    nestedSourceRef.current = e.currentTarget
    restoreNestedFocusRef.current = false
    setNestedMode('deny-reason')
  }

  function onDenySubmit() {
    restoreNestedFocusRef.current = false
    sendApproval('deny', denyReason || undefined)
    setNestedMode('none')
    setDenyReason('')
  }

  function cancelNested() {
    restoreNestedFocusRef.current = true
    setNestedMode('none')
    setDenyReason('')
  }

  // ── Clarify actions ──────────────────────────────────────────────────────
  function sendClarify(answer: string) {
    if (!item) {
      return
    }

    send({ type: 'action-center-clarify', itemId: item.id, answer })
  }

  function onClarifyChoiceClick(choice: string) {
    setClarifyChoice(choice)
    setClarifyText('')
  }

  function onClarifyTextChange(e: React.ChangeEvent<HTMLTextAreaElement>) {
    setClarifyText(e.target.value)
    setClarifyChoice(null)
  }

  function onClarifyTextKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.nativeEvent.isComposing) {
      return
    }

    if (e.key === 'Enter' && !e.shiftKey && canSubmitClarify && !isSubmitting) {
      e.preventDefault()
      submitClarify()
    }
  }

  const canSubmitClarify = clarifyChoice !== null || clarifyText.trim() !== ''

  function submitClarify() {
    if (!canSubmitClarify || isSubmitting) {
      return
    }

    const answer = clarifyChoice ?? clarifyText.trim()
    sendClarify(answer)
  }

  function onClarifySkip() {
    sendClarify('')
  }

  // ── Live turn actions ───────────────────────────────────────────────────
  function onLiveTextAction(actionType: LiveTextAction) {
    if (!item || !isLiveTurnItem(item) || isSubmitting) {
      return
    }

    const text = liveDraft.trim()

    if (!text) {
      return
    }

    const controlType = {
      queue: 'action-center-queue',
      send: 'action-center-submit',
      steer: 'action-center-steer'
    } as const

    pendingDraftActionRef.current = { action: actionType, itemId: item.id }
    send({ type: controlType[actionType], itemId: item.id, text })
  }

  function onLiveStop() {
    if (item && isLiveTurnItem(item) && !isSubmitting) {
      send({ type: 'action-center-stop', itemId: item.id })
    }
  }

  function onLiveAcknowledge() {
    if (item && isLiveTurnItem(item) && !isSubmitting) {
      send({ type: 'action-center-acknowledge', itemId: item.id })
    }
  }

  function onOpenInApp() {
    if (item && item.allowedActions.includes('open-in-app') && !isSubmitting) {
      send({ type: 'action-center-open-session', itemId: item.id })
    }
  }

  // ── Navigation ──────────────────────────────────────────────────────────
  function navigate(direction: 1 | -1) {
    const items = state.items

    if (items.length === 0 || !item) {
      return
    }

    const currentIndex = items.findIndex(i => i.id === item.id)

    if (currentIndex === -1) {
      return
    }

    const nextIndex = (currentIndex + direction + items.length) % items.length
    const nextItem = items[nextIndex]

    if (nextItem) {
      send({ type: 'action-center-select', itemId: nextItem.id })
    }
  }

  // ── Prevent pointer events from reaching the pet gesture layer ───────────
  function onPanelPointerDown(e: React.PointerEvent) {
    e.stopPropagation()
  }

  function onPanelPointerUp(e: React.PointerEvent) {
    e.stopPropagation()
  }

  // ── Live region for status announcements ─────────────────────────────────
  // Terminal results survive removal of the acted-on item. Recoverable errors
  // and steer rejection stay attached to their selected item so feedback can
  // never appear to describe a different session after navigation.
  const liveMessage = (() => {
    switch (action?.status) {
      case 'success':
        return ac.success

      case 'stale':
        return ac.stale

      case 'steered':
        return ac.steered

      case 'queued':
        return ac.queued

      case 'stopped':
        return ac.stopped

      case 'acknowledged':
        return ac.acknowledged

      case 'steer-rejected':
        return action.itemId === itemId ? ac.steerRejected : ''

      case 'error':
        return action.itemId === itemId ? ac.errorGeneric : ''

      default:
        return ''
    }
  })()

  const visibleLiveStatus = item ? liveStatusFor(item) : null

  // ── Render ───────────────────────────────────────────────────────────────
  if (state.items.length === 0 && state.secureInputCount === 0 && !panelOpen) {
    return null
  }

  return (
    <div className="flex flex-col items-center gap-1" onPointerDown={onPanelPointerDown} onPointerUp={onPanelPointerUp}>
      {/* Trigger button — collapsed state */}
      {!panelOpen && (
        <Button aria-label={ac.open} onClick={openPanel} ref={triggerRef} size="xs" variant="secondary">
          {ac.open}
        </Button>
      )}

      {/* Panel — open state. Floating-panel language per DESIGN.md: shadow-nous
          + --stroke-nous hairline, rounded-lg, backdrop-blur, all token-driven
          so it inherits the active desktop theme instead of looking like a
          foreign body. */}
      {panelOpen && (
        <div
          aria-label={ac.title}
          aria-modal="true"
          className={cn(
            'flex w-80 max-w-xs flex-col gap-2 rounded-lg border border-(--stroke-nous)',
            'bg-(--ui-bg-elevated)/95 p-3 text-(--foreground) shadow-(--shadow-nous)',
            'backdrop-blur-xl [-webkit-backdrop-filter:blur(0.75rem)]'
          )}
          onKeyDown={onDialogKeyDown}
          ref={dialogRef}
          role="dialog"
          tabIndex={-1}
        >
          {/* Header: count + close */}
          <div className="flex items-center justify-between">
            <span className="text-[0.6875rem] font-semibold text-(--ui-text-primary)">
              {ac.itemCount(state.items.length, state.attentionCount)}
            </span>
            <Button aria-label={ac.close} onClick={closePanel} size="icon-xs" variant="ghost">
              <Codicon name="close" size="0.75rem" />
            </Button>
          </div>

          {/* Secure input hint */}
          {state.secureInputCount > 0 && (
            <div className="text-[0.6875rem] text-(--ui-text-secondary)">
              {ac.secureInputHint(state.secureInputCount)}
            </div>
          )}

          {/* Live region for action status */}
          {liveMessage && (
            <div className="text-[0.6875rem] text-(--ui-text-secondary)" role="status">
              {liveMessage}
            </div>
          )}


          {/* Live status belongs to the selected prompt/session and precedes
              its detail. Prompt-attached snapshots never become a duplicate
              standalone row in this component. */}
          {item && visibleLiveStatus && (
            <LiveStatusStrip
              ac={ac}
              profileLabel={item.profileLabel}
              sessionTitle={item.sessionTitle}
              status={visibleLiveStatus}
              storedSessionId={item.storedSessionId}
            />
          )}

          {/* Reply preview — only for live-turn items with non-empty reply text */}
          {item && isLiveTurnItem(item) && item.reply?.text && (
            <div
              aria-label={ac.replyLabel}
              className="max-h-45 flex flex-col gap-0.5 overflow-y-auto overflow-x-hidden whitespace-pre-wrap break-words pt-1 text-[0.6875rem] text-(--ui-text-secondary) [user-select:text]"
              role="region"
              tabIndex={0}
            >
              <div className="mb-0.5 font-semibold">{ac.replyLabel}</div>
              <div className="border-t border-(--ui-stroke-tertiary) pt-1">{item.reply.text}</div>
            </div>
          )}

          {/* Item detail */}
          {item && isApprovalItem(item) && <ApprovalDetail ac={ac} item={item} />}
          {item && isClarifyItem(item) && <ClarifyDetail ac={ac} item={item} />}

          {/* Navigation */}
          {state.items.length > 1 && nestedMode === 'none' && (
            <div className="flex items-center justify-center gap-1">
              <Button aria-label={ac.previous} onClick={() => navigate(-1)} size="icon-xs" variant="ghost">
                <Codicon name="chevron-left" size="0.875rem" />
              </Button>
              <Button aria-label={ac.next} onClick={() => navigate(1)} size="icon-xs" variant="ghost">
                <Codicon name="chevron-right" size="0.875rem" />
              </Button>
            </div>
          )}

          {/* Approval actions */}
          {item && isApprovalItem(item) && (
            <ApprovalActions
              ac={ac}
              allowedActions={item.allowedActions}
              allowPermanent={item.allowPermanent}
              alwaysCancelRef={alwaysCancelRef}
              command={item.command}
              denyInputRef={denyInputRef}
              denyReason={denyReason}
              isSubmitting={isSubmitting}
              nestedMode={nestedMode}
              onApproveAlwaysClick={onApproveAlwaysClick}
              onApproveAlwaysConfirm={onApproveAlwaysConfirm}
              onApproveOnce={onApproveOnce}
              onApproveSession={onApproveSession}
              onDenyClick={onDenyClick}
              onDenyReasonChange={setDenyReason}
              onDenySubmit={onDenySubmit}
              onNestedCancel={cancelNested}
            />
          )}

          {/* Clarify actions */}
          {item && isClarifyItem(item) && (
            <ClarifyActions
              ac={ac}
              allowedActions={item.allowedActions}
              canSubmit={canSubmitClarify}
              choices={item.choices}
              clarifyChoice={clarifyChoice}
              clarifyText={clarifyText}
              isSubmitting={isSubmitting}
              onChoiceClick={onClarifyChoiceClick}
              onSkip={onClarifySkip}
              onSubmit={submitClarify}
              onTextChange={onClarifyTextChange}
              onTextKeyDown={onClarifyTextKeyDown}
            />
          )}

          {/* Live session actions */}
          {item && isLiveTurnItem(item) && (
            <LiveTurnActions
              ac={ac}
              draft={liveDraft}
              isSteerRejected={isSteerRejected}
              isSubmitting={isSubmitting}
              item={item}
              onAcknowledge={onLiveAcknowledge}
              onDraftChange={setLiveDraft}
              onStop={onLiveStop}
              onTextAction={onLiveTextAction}
            />
          )}

          {/* Exact-session navigation is capability-gated for every item kind. */}
          {item && item.allowedActions.includes('open-in-app') && nestedMode === 'none' && (
            <div className="flex justify-end">
              <Button disabled={isSubmitting} onClick={onOpenInApp} size="inline" variant="textStrong">
                {ac.openInApp}
              </Button>
            </div>
          )}

          {/* No items */}
          {state.items.length === 0 && state.secureInputCount === 0 && (
            <div className="text-center text-[0.6875rem] text-(--ui-text-tertiary)">{ac.noItems}</div>
          )}
        </div>
      )}
    </div>
  )
}

// ── Sub-components ────────────────────────────────────────────────────────

function ApprovalDetail({ ac, item }: { ac: ActionCenterStrings; item: ApprovalItem }) {
  return (
    <div className="flex flex-col gap-1">
      <div className="text-[0.6875rem] font-semibold text-(--ui-text-primary)">{ac.approvalTitle}</div>
      <div className="text-[0.6875rem] text-(--ui-text-secondary)">{item.description}</div>
      <div className="break-all whitespace-pre-wrap font-mono text-[0.625rem] text-(--ui-text-tertiary)">
        {item.command}
      </div>
    </div>
  )
}

function ClarifyDetail({ ac, item }: { ac: ActionCenterStrings; item: ClarifyItem }) {
  return (
    <div className="flex flex-col gap-1">
      <div className="text-[0.6875rem] font-semibold text-(--ui-text-primary)">{ac.clarifyTitle}</div>
      <div className="text-[0.6875rem] text-(--ui-text-secondary)">{item.question}</div>
    </div>
  )
}

interface ApprovalActionsProps {
  ac: ActionCenterStrings
  allowedActions: Item['allowedActions']
  allowPermanent: boolean
  alwaysCancelRef: React.RefObject<HTMLButtonElement | null>
  command: string
  denyInputRef: React.RefObject<HTMLInputElement | null>
  isSubmitting: boolean
  nestedMode: 'none' | 'confirm-always' | 'deny-reason'
  denyReason: string
  onApproveOnce: () => void
  onApproveSession: () => void
  onApproveAlwaysClick: (e: React.MouseEvent<HTMLButtonElement>) => void
  onApproveAlwaysConfirm: () => void
  onDenyClick: (e: React.MouseEvent<HTMLButtonElement>) => void
  onDenyReasonChange: (value: string) => void
  onDenySubmit: () => void
  onNestedCancel: () => void
}

function ApprovalActions({
  ac,
  allowedActions,
  allowPermanent,
  alwaysCancelRef,
  command,
  denyInputRef,
  isSubmitting,
  nestedMode,
  denyReason,
  onApproveOnce,
  onApproveSession,
  onApproveAlwaysClick,
  onApproveAlwaysConfirm,
  onDenyClick,
  onDenyReasonChange,
  onDenySubmit,
  onNestedCancel
}: ApprovalActionsProps) {
  const has = (action: string) => allowedActions.includes(action as Item['allowedActions'][number])

  return (
    <>
      {nestedMode === 'confirm-always' && (
        <div className="flex flex-col gap-1.5">
          <div className="text-[0.6875rem] font-semibold text-(--ui-text-primary)">{ac.alwaysConfirmTitle}</div>
          <div className="text-[0.6875rem] text-(--ui-text-secondary)">{ac.alwaysConfirmDescription(command)}</div>
          <div className="flex justify-end gap-1">
            <Button onClick={onNestedCancel} ref={alwaysCancelRef} size="xs" variant="text">
              {ac.cancel}
            </Button>
            <Button disabled={isSubmitting} onClick={onApproveAlwaysConfirm} size="xs" variant="destructive">
              {ac.alwaysConfirmConfirm}
            </Button>
          </div>
        </div>
      )}

      {nestedMode === 'deny-reason' && (
        <div className="flex flex-col gap-1.5">
          <div className="text-[0.6875rem] font-semibold text-(--ui-text-primary)">{ac.denyReason}</div>
          <Input
            aria-label={ac.denyReason}
            disabled={isSubmitting}
            onChange={e => onDenyReasonChange(e.target.value)}
            placeholder={ac.denyReasonPlaceholder}
            ref={denyInputRef}
            size="sm"
            value={denyReason}
          />
          <div className="flex justify-end gap-1">
            <Button onClick={onNestedCancel} size="xs" variant="text">
              {ac.cancel}
            </Button>
            <Button
              aria-label={ac.denySubmit}
              disabled={isSubmitting}
              onClick={onDenySubmit}
              size="xs"
              variant="destructive"
            >
              {ac.denySubmit}
            </Button>
          </div>
        </div>
      )}

      {/* Keep source buttons mounted while a nested layer is open so focus can
          return to the exact button node that opened it. */}
      <div hidden={nestedMode !== 'none'}>
        <div className="flex flex-wrap gap-1">
          {has('approve-once') && (
            <Button disabled={isSubmitting} onClick={onApproveOnce} size="xs" variant="default">
              {ac.approveOnce}
            </Button>
          )}
          {has('approve-session') && (
            <Button disabled={isSubmitting} onClick={onApproveSession} size="xs" variant="secondary">
              {ac.approveSession}
            </Button>
          )}
          {has('approve-always') && allowPermanent && (
            <Button disabled={isSubmitting} onClick={onApproveAlwaysClick} size="xs" variant="outline">
              {ac.approveAlways}
            </Button>
          )}
          {has('deny') && (
            <Button disabled={isSubmitting} onClick={onDenyClick} size="xs" variant="text">
              {ac.deny}
            </Button>
          )}
        </div>
      </div>
    </>
  )
}

interface ClarifyActionsProps {
  ac: ActionCenterStrings
  allowedActions: Item['allowedActions']
  canSubmit: boolean
  choices: string[] | null
  clarifyChoice: string | null
  clarifyText: string
  isSubmitting: boolean
  onChoiceClick: (choice: string) => void
  onTextChange: (e: React.ChangeEvent<HTMLTextAreaElement>) => void
  onTextKeyDown: (e: React.KeyboardEvent<HTMLTextAreaElement>) => void
  onSkip: () => void
  onSubmit: () => void
}

function ClarifyActions({
  ac,
  allowedActions,
  canSubmit,
  choices,
  clarifyChoice,
  clarifyText,
  isSubmitting,
  onChoiceClick,
  onTextChange,
  onTextKeyDown,
  onSkip,
  onSubmit
}: ClarifyActionsProps) {
  const has = (action: string) => allowedActions.includes(action as Item['allowedActions'][number])

  return (
    <div className="flex flex-col gap-1.5">
      {/* Backend choices */}
      {choices?.map(choice => (
        <Button
          aria-label={choice}
          disabled={isSubmitting}
          key={choice}
          onClick={() => onChoiceClick(choice)}
          size="xs"
          variant={clarifyChoice === choice ? 'default' : 'secondary'}
        >
          {choice}
        </Button>
      ))}

      {/* Other (free text) */}
      <Textarea
        aria-label={ac.other}
        disabled={isSubmitting}
        onChange={onTextChange}
        onKeyDown={onTextKeyDown}
        placeholder={ac.otherPlaceholder}
        rows={2}
        size="sm"
        value={clarifyText}
      />

      {/* Submit + Skip */}
      <div className="flex justify-end gap-1">
        {has('clarify-skip') && (
          <Button aria-label={ac.skip} disabled={isSubmitting} onClick={onSkip} size="xs" variant="ghost">
            {ac.skip}
          </Button>
        )}
        {has('clarify-respond') && (
          <Button disabled={!canSubmit || isSubmitting} onClick={onSubmit} size="xs" variant="default">
            {ac.clarifySubmit}
          </Button>
        )}
      </div>
    </div>
  )
}
