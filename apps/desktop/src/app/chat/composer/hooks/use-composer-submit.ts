import { SLASH_COMMAND_RE } from '@hermes/shared'
import { type RefObject, useLayoutEffect, useRef } from 'react'

import { usePaneVisible } from '@/components/pane-shell/pane-visibility'
import { translateNow, useI18n } from '@/i18n'
import { isSlashCommandText } from '@/lib/chat-runtime'
import { isSideTaskSlashCommand } from '@/lib/desktop-slash-commands'
import { triggerHaptic } from '@/lib/haptics'
import type { BusyInputMode } from '@/store/busy-input-mode'
import { hasClarifyRequest, skipClarifyRequest } from '@/store/clarify'
import {
  clearSessionDraft,
  type ComposerAttachment,
  type ComposerTransportPayload,
  freezeComposerTransportPayload,
  isFreshDraftScope
} from '@/store/composer'
import { type ComposerDraftRetry, draftRetry } from '@/store/composer-draft-retry'
import { resetBrowseState } from '@/store/composer-input-history'
import { enqueueQueuedPrompt, type QueuedPromptEntry, serverOwnsComposerQueue } from '@/store/composer-queue'
import { hasConnectionRequest, skipConnectionRequest } from '@/store/connection-request'
import { notify } from '@/store/notifications'
import { answerParkedCard, skipParkedCards, yieldParkedCardsToMessage } from '@/store/parked-card-answers'
import { hasBlockingPromptRequest } from '@/store/prompts'

import { cloneAttachments, type QueueEditState } from '../composer-utils'
import { onComposerSubmitRequest } from '../focus'
import { pathifyRefs } from '../path-refs'
import { composerPlainText } from '../rich-editor'
import { useComposerScope, useComposerSurfaceId } from '../scope'
import type { ChatBarProps } from '../types'

interface UseComposerSubmitArgs {
  activeQueueSessionKey: string | null
  activeQueueSessionKeyRef: RefObject<string | null>
  attachments: ComposerAttachment[]
  busy: boolean
  busyInputMode?: BusyInputMode | null
  compacting?: boolean
  clearDraft: () => void
  disabled: boolean
  draftScopeRef: RefObject<string | null>
  draftRef: RefObject<string>
  drainNextQueued: () => Promise<boolean>
  editorRef: RefObject<HTMLDivElement | null>
  exitQueuedEdit: (action: 'cancel' | 'save') => boolean
  focusInput: () => void
  inputDisabled: boolean
  loadIntoComposer: (text: string, attachments: ComposerAttachment[]) => void
  onCancel: ChatBarProps['onCancel']
  onSteer: ChatBarProps['onSteer']
  onSteerHidden: ChatBarProps['onSteerHidden']
  onSubmit: ChatBarProps['onSubmit']
  queueCurrentDraft: () => boolean | Promise<boolean>
  queueEdit: QueueEditState | null
  queuedPrompts: QueuedPromptEntry[]
  sessionId: string | null | undefined
  setComposerText: (value: string) => void
  stashAt: (scope: string | null, text?: string, attachments?: ComposerAttachment[]) => void
  /** Restored-send identity (see use-composer-draft); absent = every send is a new message. */
  rememberDraftRetry?: (scope: string | null, retry: ComposerDraftRetry) => void
  takeDraftRetry?: (text: string, attachments: ComposerAttachment[]) => ComposerDraftRetry | undefined
}

/**
 * The composer's submit engine — the orchestration seam where the draft and
 * queue meet. `submitDraft` is the one decision tree (queue-edit save · slash-
 * now-while-busy · queue · drain · send · stop); `dispatchSubmit` is the shared
 * send-with-restore primitive (re-loads + re-stashes the draft if the gateway
 * rejects, so nothing is ever lost); `steerDraft` redirects the live turn. Reads
 * the draft + queue APIs; owns no state of its own beyond the stable
 * external-submit listener ref.
 */
export function useComposerSubmit({
  activeQueueSessionKey,
  activeQueueSessionKeyRef,
  attachments,
  busy,
  busyInputMode = 'interrupt',
  compacting = false,
  clearDraft,
  disabled,
  draftScopeRef,
  draftRef,
  drainNextQueued,
  editorRef,
  exitQueuedEdit,
  focusInput,
  inputDisabled,
  loadIntoComposer,
  onCancel,
  onSteer,
  onSteerHidden,
  onSubmit,
  queueCurrentDraft,
  queueEdit,
  queuedPrompts,
  sessionId,
  setComposerText,
  stashAt,
  rememberDraftRetry,
  takeDraftRetry
}: UseComposerSubmitArgs) {
  const paneVisible = usePaneVisible()
  const scope = useComposerScope()
  const surfaceId = useComposerSurfaceId()
  const { t } = useI18n()
  const copy = t.desktop

  // Shared send primitive: fire onSubmit, and if the gateway rejects (accepted
  // === false) or throws, re-stash the draft so the words survive. Repaint it
  // only while the same session still owns the visible composer; a late reject
  // must not publish an old session's text into the newly focused one.
  // `target` carries an explicit destination (a canonical queue admission).
  const dispatchSubmit = (
    text: string,
    attachments?: ComposerAttachment[],
    displayKind?: 'hidden',
    target?: Parameters<ChatBarProps['onSubmit']>[1]
  ) => {
    // A fresh chat's composer is keyed by its per-lifecycle fresh-draft key
    // (`__new__:<uuid>`), but the submit contract spells "no session yet" as
    // null: the create handoff below and the composer drift prong both key off
    // it, and draftKey(null) resolves to that same fresh bucket.
    const submittedScope =
      target?.storedSessionId ?? (isFreshDraftScope(draftScopeRef.current) ? null : draftScopeRef.current)

    let restoreScope = submittedScope
    const submittedAttachments = attachments ?? []

    // Only this operation's explicit session.create handoff may re-home a
    // pre-session submit. A null → stored render can also be user navigation.
    const assignment =
      submittedScope === null
        ? {
            onComposerScopeAssigned: (scope: string) => {
              restoreScope = scope
            }
          }
        : {}

    // The composer owns this send's identity: the caller's (a queue entry), the restored draft's
    // (the explicit retry), else a new one. Equal text never inherits an identity, so a new
    // message that repeats an uncertain send, in this pane or another, is its own turn.
    const submissionId = target?.submission_id ?? crypto.randomUUID()

    const restore = () => {
      stashAt(restoreScope, text, submittedAttachments)
      // Restored with its identity: Enter on this exact draft retries the same send.
      rememberDraftRetry?.(restoreScope, draftRetry(submissionId, text, submittedAttachments))

      if ((isFreshDraftScope(draftScopeRef.current) ? null : draftScopeRef.current) === restoreScope) {
        loadIntoComposer(text, submittedAttachments)
      }
    }

    // A hidden submit is machine text (a setup note, never something the user
    // typed), so a rejection drops it instead of loading it into the draft.
    const rejected = displayKind ? () => {} : restore

    void Promise.resolve(
      onSubmit(text, {
        ...target,
        submission_id: submissionId,
        ...(attachments ? { attachments } : {}),
        composerScope: submittedScope,
        ...assignment,
        ...(displayKind ? { displayKind } : {})
      })
    )
      .then(accepted => void (accepted === false ? rejected() : clearSessionDraft(submittedScope)))
      .catch(rejected)
  }

  // External "submit this prompt" requests (e.g. the review pane's agent-ship
  // button) route through the same send path. Match both the composer target
  // and the exact visible surface captured at click time — every tile stays
  // mounted, and a session can be rendered in more than one pane.
  //
  // Busy: a request from a card the user just clicked must not be dropped
  // because the agent is mid-sentence — that gap is exactly when they click.
  // Steer the live turn (the same stop-and-correct a typed message gets), and
  // if the turn has already ended, or a steer is not possible, queue it so it
  // runs next. This holds for hidden setup notes and for visible messages a
  // button sends on the user's behalf alike.
  const externalSubmitRef = useRef({ busy, dispatchSubmit, onSteer, onSteerHidden })
  externalSubmitRef.current = { busy, dispatchSubmit, onSteer, onSteerHidden }

  useLayoutEffect(
    () =>
      onComposerSubmitRequest(({ surfaceId: requestedSurfaceId, target, text, displayKind }) => {
        if (
          target === scope.target &&
          surfaceId !== null &&
          requestedSurfaceId === surfaceId &&
          paneVisible &&
          !inputDisabled
        ) {
          const current = externalSubmitRef.current

          if (!current.busy) {
            current.dispatchSubmit(text, undefined, displayKind)

            return
          }

          const queueKey = activeQueueSessionKeyRef.current

          // External requests contain only text; the unsent draft and its attachments stay in the composer.
          const enqueue = () =>
            void enqueueQueuedPrompt(queueKey, { text, attachments: [], ...(displayKind ? { displayKind } : {}) })

          // A hidden note never becomes a user turn: it rides session.steer into
          // the model's next tool result, and keeps its kind if it has to queue.
          if (displayKind) {
            if (current.onSteerHidden) {
              void Promise.resolve(current.onSteerHidden(text))
                .then(accepted => {
                  if (!accepted) {
                    enqueue()
                  }
                })
                .catch(enqueue)
            } else {
              enqueue()
            }

            return
          }

          if (
            current.onSteer &&
            !hasBlockingPromptRequest(sessionId) &&
            text.trim() &&
            !SLASH_COMMAND_RE.test(text.trim())
          ) {
            void Promise.resolve(current.onSteer(text))
              .then(accepted => {
                if (!accepted) {
                  enqueue()
                }
              })
              .catch(enqueue)
          } else {
            enqueue()
          }
        }
      }),
    [activeQueueSessionKeyRef, inputDisabled, paneVisible, scope.target, sessionId, surfaceId]
  )

  // Returns false when the submit was refused and must not refocus the input.
  const submitWhileBusy = (text: string, payloadPresent: boolean, blockingPrompt: boolean, busyModeOverride?: BusyInputMode) => {
    // Slash commands should execute immediately even while the agent is
    // busy — they're client-side operations (/yolo, /skin, /new, /help,
    // etc.) or self-contained gateway RPCs (/status, /compress).  onSubmit
    // routes them to executeSlashCommand, which has its own per-command
    // busy guard for commands that genuinely need an idle session (skill
    // /send directives).  Queuing them would make every slash command wait
    // for the current turn to finish, which is how the TUI never behaves.
    if (isSlashCommandText(text)) {
      if (attachments.length) {
        // Slash commands cannot ride alongside attachments — warn the user
        // instead of silently queuing the payload (which would then reach the
        // idle path and be submitted as plain text with no command execution).
        notify({
          kind: 'warning',
          title: copy.slashCommandIgnoredTitle,
          message: copy.slashCommandIgnoredBody
        })

        return false
      }

      const retry = takeDraftRetry?.(text, [])
      triggerHaptic('submit')
      clearDraft()
      dispatchSubmit(text, undefined, undefined, retry && { submission_id: retry.id })
    } else if (!blockingPrompt && !attachments.length && text.trim()) {
      // Queue during compaction, otherwise apply the configured one-message policy.
      // Slash commands above and parked-card answers retain higher priority.
      const effectiveMode = busyModeOverride ?? busyInputMode

      if (compacting || effectiveMode === 'queue') {
        queueCurrentDraft()
      } else if (effectiveMode === 'steer') {
        if (onSteerHidden) {
          steerAtToolBoundary(text)
        } else {
          queueCurrentDraft()
        }
      } else if (effectiveMode !== null) {
        steerDraft(effectiveMode, text)
      } else {
        return false
      }
    } else if (payloadPresent) {
      // Attachments can't ride a redirect (no tool-result image carriage) —
      // queue the whole payload for the next turn. Same for a turn parked on
      // an approval/sudo/secret prompt: a steer can't reach the model while
      // the tool batch is blocked, so the message runs as the next turn.
      queueCurrentDraft()
    } else {
      // Stop button (the only way to reach here while busy with an empty
      // composer — empty Enter is short-circuited in the keydown handler).
      triggerHaptic('cancel')
      void Promise.resolve(onCancel())
    }

    return true
  }

  // 'answered': the draft was a parked card's answer. 'yielded': a card closed for it, so it runs as
  // the next turn.
  const answerParkedCardWithDraft = (text: string, payloadPresent: boolean): 'answered' | 'yielded' | null => {
    // A card parked on this session owns the turn: the agent is blocked inside
    // its tool batch, so a steer or queued follow-up would sit undelivered until
    // the card's own timeout. Plain text answers the card instead (a clarify
    // takes it as its next question's answer) and is not sent again.
    //
    // A slash command or attachments cannot be an answer: the card is skipped
    // and the draft routes normally. An onboarding setup card the message
    // moves past instead closes as unanswered and its turn ends there, so the
    // message runs as the next turn. The skip is fire-and-forget, not awaited:
    // it clears the card synchronously and both RPCs ride the same socket in
    // call order, so the gateway resolves the clarify before it sees the
    // follow-up. Awaiting first would leave the draft live for a tick, long
    // enough for a second Enter to send it twice.
    //
    // /btw and /bg run beside the turn (snapshot / separate session) and touch
    // no parked card. With attachments the draft isn't routed as a slash
    // command, so it falls back to the ordinary-message behavior.
    const isSideQuestion = !attachments.length && isSideTaskSlashCommand(text)
    const routable = payloadPresent && !queueEdit && !isSideQuestion
    const slashCommand = !attachments.length && SLASH_COMMAND_RE.test(text.trim())

    // Like dispatchSubmit's restore: an answer that never reached the backend gives the words back.
    const answeredScope = draftScopeRef.current

    const restore = () => {
      stashAt(answeredScope, text, [])

      if (draftScopeRef.current === answeredScope) {
        loadIntoComposer(text, [])
      }
    }

    if (routable && !attachments.length && !slashCommand && answerParkedCard(sessionId ?? '', text.trim(), restore)) {
      triggerHaptic('submit')
      resetBrowseState(sessionId)
      clearDraft()
      focusInput()

      return 'answered'
    }

    const yielded = routable && !slashCommand && yieldParkedCardsToMessage(sessionId ?? '')

    if (routable && hasClarifyRequest(sessionId)) {
      void skipClarifyRequest(sessionId)
    }

    // Same for a pending connection card: ordinary typing continues the operation.
    if (routable && hasConnectionRequest(sessionId)) {
      void skipConnectionRequest(sessionId)
    }

    // And for the onboarding chat's setup and theme cards.
    if (routable) {
      skipParkedCards(sessionId ?? '')
    }

    return yielded ? 'yielded' : null
  }

  const submitDraft = (busyModeOverride?: BusyInputMode) => {
    if (disabled) {
      return
    }

    // Source the text from the DOM editor, not React state. The AUI composer
    // state (`draft`) and the derived `hasComposerPayload` lag the DOM by a
    // render, so on fast typing or IME composition the final keystroke(s) may
    // not have synced yet — reading state here drops the message (Enter looks
    // like it does nothing; typing a trailing space only "fixes" it because the
    // extra input event forces a state sync). draftRef is updated on every
    // input event; refresh it from the editor once more to also cover an
    // in-flight keystroke that hasn't fired its input event yet.
    const editor = editorRef.current

    if (editor) {
      const domText = composerPlainText(editor)

      if (domText !== draftRef.current) {
        draftRef.current = domText
        setComposerText(domText)
      }
    }

    // A path that never got its committing space (`@apps/desktop/` left by a Tab
    // descend, then Enter) is still the reference the user picked — promote it
    // on the way out so it attaches instead of submitting as inert text.
    const text = pathifyRefs(draftRef.current)
    const payloadPresent = text.trim().length > 0 || attachments.length > 0

    const parked = answerParkedCardWithDraft(text, payloadPresent)

    if (parked === 'answered') {
      return
    }

    // Approval / sudo / secret prompts also park the turn inside a tool batch,
    // but typing CANNOT answer them (no message text approves a command or
    // supplies a password), so there is no skip-and-steer path: a steer would
    // sit undelivered behind the blocked prompt, and stopping the turn to force
    // it through resolves the prompt to empty and ends the turn as "Operation
    // interrupted." — the message looks eaten. Queue the words as the next turn
    // instead; the prompt stays answerable and the queue drains on settle. A
    // setup card that just closed for this message ends its turn the same way.
    const blockingPrompt = !queueEdit && (parked === 'yielded' || hasBlockingPromptRequest(sessionId))

    if (queueEdit) {
      exitQueuedEdit('save')
    } else if (busy) {
      if (!submitWhileBusy(text, payloadPresent, blockingPrompt, busyModeOverride)) {
        return
      }
    } else if (!payloadPresent && queuedPrompts.length > 0) {
      void drainNextQueued()
    } else if (payloadPresent) {
      const submittedAttachments = cloneAttachments(attachments)
      const retry = takeDraftRetry?.(text, submittedAttachments)
      triggerHaptic('submit')
      resetBrowseState(sessionId)
      clearDraft()
      // Keep blob: previews alive for the optimistic bubble; revoke when that
      // consumer is discarded/replaced (not here — clear would race the clone).
      scope.attachments.clear({ retainPreviewUrls: true })
      dispatchSubmit(text, submittedAttachments, undefined, retry && { submission_id: retry.id })
    }

    focusInput()
  }

  // A correction whose RPC rejects (socket dropped, session gone mid-turn)
  // must not vanish with the already-cleared draft. Capture the target scope at
  // dispatch time so a late rejection cannot enqueue into a newly focused chat.
  const queueRejectedCorrection = (
    frozen: ComposerTransportPayload,
    queueKey: string | null,
    restoreScope: string | null
  ) => {
    const hasTerminalTransport = frozen.displayText !== frozen.transportText

    // A sessionless chat may already have a per-lifecycle __new__: key. It is
    // a draft bucket, not a queue that a worker will ever drain.
    if (queueKey && !isFreshDraftScope(queueKey) && serverOwnsComposerQueue(sessionId ?? queueKey)) {
      dispatchSubmit(frozen.transportText, [], undefined, {
        ...(hasTerminalTransport ? { displayText: frozen.displayText } : {}),
        fromQueue: true,
        sessionId: sessionId ?? null,
        storedSessionId: queueKey
      })
    } else if (queueKey && !isFreshDraftScope(queueKey)) {
      enqueueQueuedPrompt(queueKey, {
        text: frozen.displayText,
        attachments: [],
        ...(hasTerminalTransport
          ? { displayText: frozen.displayText, frozenTransport: frozen.transportText }
          : {})
      })
    } else {
      // Store against the draft that actually supplied the words. A delayed
      // rejection after navigation must never overwrite another chat's editor.
      stashAt(restoreScope, frozen.displayText, [])

      if (draftScopeRef.current === restoreScope && activeQueueSessionKeyRef.current === queueKey) {
        loadIntoComposer(frozen.displayText, [])
      }
    }
  }

  const steerDraft = (mode: 'interrupt' | 'steer' = 'interrupt', text = pathifyRefs(draftRef.current).trim()) => {
    if (!onSteer || !text || attachments.length > 0 || SLASH_COMMAND_RE.test(text)) {
      // A slash command is never a queued model prompt, even if this helper is called directly.
      if (text && !SLASH_COMMAND_RE.test(text)) {
        queueCurrentDraft()
      }

      return
    }

    // Freeze `@terminal:` chips the same way idle submit / queue enqueue do.
    // Steer used to forward the bare token only (#77078).
    const frozen = freezeComposerTransportPayload(text)

    if (frozen.missingLabels.length > 0) {
      notify({
        kind: 'warning',
        title: translateNow('composer.terminalSelectionMissingTitle'),
        message: translateNow('composer.terminalSelectionMissingBody')
      })

      return
    }

    const rejectedQueueKey = activeQueueSessionKeyRef.current
    const restoreScope = draftScopeRef.current

    triggerHaptic('submit')
    clearDraft()

    // The draft is already cleared, so a refused or failed redirect must keep
    // the only copy (#68927): the canonical gateway queue when the server owns
    // it, the local queue otherwise, or the composer itself when there is no
    // queue yet (a new chat is busy before its first session exists). Keep the
    // frozen transport for the queue; restoring to the composer keeps the chip
    // form so the user can re-send it as-is.
    const keep = () => queueRejectedCorrection(frozen, rejectedQueueKey, restoreScope)

    // Also recover callbacks that throw synchronously before returning a promise.
    try {
      void Promise.resolve(onSteer(frozen.transportText, mode))
        .then(accepted => {
          if (!accepted) {
            keep()
          }
        })
        .catch(keep)
    } catch {
      keep()
    }
  }

  const steerAtToolBoundary = (text: string) => {
    if (!onSteerHidden || !text || attachments.length > 0 || SLASH_COMMAND_RE.test(text)) {
      queueCurrentDraft()

      return
    }

    // Freeze terminal selections at dispatch, exactly like redirect and queue.
    // Model-facing transport contains the selection; queue fallback persists
    // only its display chip and holds the frozen transport in the runtime map.
    const frozen = freezeComposerTransportPayload(text)

    if (frozen.missingLabels.length > 0) {
      notify({
        kind: 'warning',
        title: translateNow('composer.terminalSelectionMissingTitle'),
        message: translateNow('composer.terminalSelectionMissingBody')
      })

      return
    }

    const rejectedQueueKey = activeQueueSessionKeyRef.current
    const restoreScope = draftScopeRef.current

    triggerHaptic('submit')
    clearDraft()

    const rejected = () => queueRejectedCorrection(frozen, rejectedQueueKey, restoreScope)

    // Preserve synchronous dispatch while recovering from both sync throws
    // and async refusals without losing the draft.
    try {
      void Promise.resolve(onSteerHidden(frozen.transportText))
        .then(accepted => {
          if (!accepted) {
            rejected()
          }
        })
        .catch(rejected)
    } catch {
      rejected()
    }
  }

  const queueDraft = () => {
    if (disabled || !busy) {
      return
    }

    queueCurrentDraft()
    focusInput()
  }

  return { dispatchSubmit, queueDraft, steerDraft, submitDraft }
}
