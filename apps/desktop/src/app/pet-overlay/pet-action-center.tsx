import { useCallback, useEffect, useRef, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Textarea } from '@/components/ui/textarea'
import { useI18n } from '@/i18n'
import type { PetActionCenterState } from '@/store/pet-action-center'
import type { PetActionCenterApprovalChoice, PetActionCenterControl } from '@/store/pet-overlay'
import { sendPetOverlayControl } from '@/store/pet-overlay'

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
  const nestedSourceRef = useRef<HTMLButtonElement | null>(null)
  const denyInputRef = useRef<HTMLInputElement | null>(null)
  const alwaysCancelRef = useRef<HTMLButtonElement | null>(null)
  const restoreNestedFocusRef = useRef(false)

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
  const isItemError = action?.status === 'error' && action.itemId === itemId

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
  // Success/stale are terminal states — the item may have already been removed
  // from the list by the time the status arrives. Show the message as long as
  // the action status is success/stale, regardless of whether itemId still
  // matches the current selection. Error remains item-specific; submitting is
  // global because the bridge accepts only one side-effect action at a time.
  const liveMessage = action?.status === 'success' ? ac.success : action?.status === 'stale' ? ac.stale : ''

  // ── Render ───────────────────────────────────────────────────────────────
  if (state.attentionCount === 0 && !panelOpen) {
    return null
  }

  return (
    <div
      onPointerDown={onPanelPointerDown}
      onPointerUp={onPanelPointerUp}
      style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 4 }}
    >
      {/* Trigger button — collapsed state */}
      {!panelOpen && (
        <Button aria-label={ac.open} onClick={openPanel} ref={triggerRef} size="xs" variant="secondary">
          {ac.open}
        </Button>
      )}

      {/* Panel — open state */}
      {panelOpen && (
        <div
          aria-label={ac.title}
          aria-modal="true"
          onKeyDown={onDialogKeyDown}
          ref={dialogRef}
          role="dialog"
          style={{
            background: 'var(--ui-bg-elevated)',
            border: '1px solid var(--stroke-nous)',
            borderRadius: 4,
            boxShadow: 'var(--shadow-nous)',
            color: 'var(--foreground)',
            display: 'flex',
            flexDirection: 'column',
            gap: 8,
            maxWidth: 320,
            padding: 12,
            width: 320
          }}
          tabIndex={-1}
        >
          {/* Header: count + close */}
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <span style={{ fontSize: 11, fontWeight: 600, color: 'var(--ui-text-primary)' }}>
              {ac.count(state.attentionCount)}
            </span>
            <Button aria-label={ac.close} onClick={closePanel} size="icon-xs" variant="ghost">
              ✕
            </Button>
          </div>

          {/* Secure input hint */}
          {state.secureInputCount > 0 && (
            <div style={{ fontSize: 11, color: 'var(--ui-text-secondary)' }}>
              {ac.secureInputHint(state.secureInputCount)}
            </div>
          )}

          {/* Live region for action status */}
          {liveMessage && (
            <div role="status" style={{ fontSize: 11, color: 'var(--ui-text-secondary)' }}>
              {liveMessage}
            </div>
          )}

          {/* Error state (recoverable) */}
          {isItemError && <div style={{ fontSize: 11, color: 'var(--ui-text-secondary)' }}>{ac.errorGeneric}</div>}

          {/* Item detail */}
          {item && isApprovalItem(item) && <ApprovalDetail ac={ac} item={item} />}
          {item && isClarifyItem(item) && <ClarifyDetail ac={ac} item={item} />}

          {/* Navigation */}
          {state.items.length > 1 && nestedMode === 'none' && (
            <div style={{ display: 'flex', gap: 4, justifyContent: 'center' }}>
              <Button aria-label={ac.previous} onClick={() => navigate(-1)} size="xs" variant="ghost">
                ‹
              </Button>
              <Button aria-label={ac.next} onClick={() => navigate(1)} size="xs" variant="ghost">
                ›
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

          {/* No items */}
          {state.items.length === 0 && state.secureInputCount === 0 && (
            <div style={{ fontSize: 11, color: 'var(--ui-text-tertiary)', textAlign: 'center' }}>{ac.noItems}</div>
          )}
        </div>
      )}
    </div>
  )
}

// ── Sub-components ────────────────────────────────────────────────────────

function ApprovalDetail({ ac, item }: { ac: ActionCenterStrings; item: ApprovalItem }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
      <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--ui-text-primary)' }}>{ac.approvalTitle}</div>
      <div style={{ fontSize: 11, color: 'var(--ui-text-secondary)' }}>{item.description}</div>
      <div
        style={{
          fontFamily: 'var(--font-mono, monospace)',
          fontSize: 10,
          color: 'var(--ui-text-tertiary)',
          whiteSpace: 'pre-wrap',
          wordBreak: 'break-all'
        }}
      >
        {item.command}
      </div>
    </div>
  )
}

function ClarifyDetail({ ac, item }: { ac: ActionCenterStrings; item: ClarifyItem }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
      <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--ui-text-primary)' }}>{ac.clarifyTitle}</div>
      <div style={{ fontSize: 11, color: 'var(--ui-text-secondary)' }}>{item.question}</div>
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
        <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
          <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--ui-text-primary)' }}>{ac.alwaysConfirmTitle}</div>
          <div style={{ fontSize: 11, color: 'var(--ui-text-secondary)' }}>{ac.alwaysConfirmDescription(command)}</div>
          <div style={{ display: 'flex', gap: 4, justifyContent: 'flex-end' }}>
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
        <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
          <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--ui-text-primary)' }}>{ac.denyReason}</div>
          <Input
            aria-label={ac.denyReason}
            onChange={e => onDenyReasonChange(e.target.value)}
            placeholder={ac.denyReasonPlaceholder}
            ref={denyInputRef}
            size="sm"
            value={denyReason}
          />
          <div style={{ display: 'flex', gap: 4, justifyContent: 'flex-end' }}>
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
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
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
    <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
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
        onChange={onTextChange}
        onKeyDown={onTextKeyDown}
        placeholder={ac.otherPlaceholder}
        rows={2}
        size="sm"
        value={clarifyText}
      />

      {/* Submit + Skip */}
      <div style={{ display: 'flex', gap: 4, justifyContent: 'flex-end' }}>
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
