import { type RefObject, useLayoutEffect, useRef } from 'react'

import { usePaneVisible } from '@/components/pane-shell/pane-visibility'
import { SLASH_COMMAND_RE } from '@/lib/chat-runtime'
import { triggerHaptic } from '@/lib/haptics'
import { hasClarifyRequest, skipClarifyRequest } from '@/store/clarify'
import { clearSessionDraft, type ComposerAttachment } from '@/store/composer'
import { resetBrowseState } from '@/store/composer-input-history'
import { enqueueQueuedPrompt, type QueuedPromptEntry } from '@/store/composer-queue'
import { hasMcpSetupRequest, skipMcpSetupRequest } from '@/store/mcp-setup'
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
  compacting: boolean
  clearDraft: () => void
  disabled: boolean
  draftRef: RefObject<string>
  drainNextQueued: () => Promise<boolean>
  editorRef: RefObject<HTMLDivElement | null>
  exitQueuedEdit: (action: 'cancel' | 'save') => boolean
  focusInput: () => void
  inputDisabled: boolean
  loadIntoComposer: (text: string, attachments: ComposerAttachment[]) => void
  onCancel: ChatBarProps['onCancel']
  onSteer: ChatBarProps['onSteer']
  onSubmit: ChatBarProps['onSubmit']
  queueCurrentDraft: () => boolean
  queueEdit: QueueEditState | null
  queuedPrompts: QueuedPromptEntry[]
  sessionId: string | null | undefined
  setComposerText: (value: string) => void
  stashAt: (scope: string | null, text?: string, attachments?: ComposerAttachment[]) => void
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
  compacting,
  clearDraft,
  disabled,
  draftRef,
  drainNextQueued,
  editorRef,
  exitQueuedEdit,
  focusInput,
  inputDisabled,
  loadIntoComposer,
  onCancel,
  onSteer,
  onSubmit,
  queueCurrentDraft,
  queueEdit,
  queuedPrompts,
  sessionId,
  setComposerText,
  stashAt
}: UseComposerSubmitArgs) {
  const paneVisible = usePaneVisible()
  const scope = useComposerScope()
  const surfaceId = useComposerSurfaceId()

  // Shared send primitive: fire onSubmit, and if the gateway rejects (accepted
  // === false) or throws, re-load + re-stash the draft so the words survive.
  const dispatchSubmit = (text: string, attachments?: ComposerAttachment[], displayKind?: 'hidden') => {
    const submittedScope = activeQueueSessionKeyRef.current
    const submittedAttachments = attachments ?? []

    const restore = () => {
      // Only repaint the live composer while the submitting session is still
      // the active one — the gateway can reject well after the user has
      // switched away, and painting then would drop the rejected text into
      // the OTHER session's composer, where the paint-triggered sync stashes
      // it under the now-active key (#66661). Known trade-off: a stored-key →
      // runtime-id rotation (a submit that triggers a resume) also fails this
      // key comparison, so a rejection of THAT submit isn't repainted — the
      // text still survives in the submitted scope's stash below and is
      // restored the next time that stored session's key is active, matching
      // the stash-side semantics #54527 already chose.
      if (activeQueueSessionKeyRef.current === submittedScope) {
        loadIntoComposer(text, submittedAttachments)
      }

      // Use the scope captured at dispatch, not whatever session is focused
      // now — re-stashing into the currently-focused session would overwrite
      // its draft with the rejected text from a different session (#54527).
      stashAt(submittedScope, text, submittedAttachments)
    }

    void Promise.resolve(
      attachments
        ? onSubmit(text, { attachments, composerScope: submittedScope, ...(displayKind ? { displayKind } : {}) })
        : onSubmit(text, { composerScope: submittedScope, ...(displayKind ? { displayKind } : {}) })
    )
      .then(accepted => void (accepted === false ? restore() : clearSessionDraft(submittedScope)))
      .catch(restore)
  }

  // External "submit this prompt" requests (e.g. the review pane's agent-ship
  // button) route through the same send path. Match both the composer target
  // and the exact visible surface captured at click time — every tile stays
  // mounted, and a session can be rendered in more than one pane.
  const dispatchSubmitRef = useRef(dispatchSubmit)
  dispatchSubmitRef.current = dispatchSubmit

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
          dispatchSubmitRef.current(text, undefined, displayKind)
        }
      }),
    [inputDisabled, paneVisible, scope.target, surfaceId]
  )

  const submitDraft = () => {
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

    // A clarify card parked on this session owns the turn: the agent is blocked
    // inside its tool batch waiting on `clarify.respond`, so a follow-up routed
    // through steer/queue sits undelivered until the clarify's own timeout
    // (default 5 min) — the message looks sent and nothing happens. Typing a
    // real message instead of picking an option IS the answer "none of these":
    // skip the question so the tool returns, then route the words normally.
    //
    // Fire-and-forget, not awaited: the skip clears the card synchronously and
    // both RPCs ride the same socket in call order, so the gateway resolves the
    // clarify before it sees the follow-up. Awaiting first would leave the draft
    // live for a tick — long enough for a second Enter to send it twice.
    if (payloadPresent && !queueEdit && hasClarifyRequest(sessionId)) {
      void skipClarifyRequest(sessionId)
    }

    // Same deal for a pending MCP setup card: the agent is blocked on
    // mcp.setup.respond, so a typed message declines the card and rides on.
    if (payloadPresent && !queueEdit && hasMcpSetupRequest(sessionId)) {
      void skipMcpSetupRequest(sessionId)
    }

    // Approval / sudo / secret prompts also park the turn inside a tool batch,
    // but typing CANNOT answer them (no message text approves a command or
    // supplies a password), so there is no skip-and-steer path: a steer would
    // sit undelivered behind the blocked prompt, and stopping the turn to force
    // it through resolves the prompt to empty and ends the turn as "Operation
    // interrupted." — the message looks eaten. Queue the words as the next turn
    // instead; the prompt stays answerable and the queue drains on settle.
    const blockingPrompt = !queueEdit && hasBlockingPromptRequest(sessionId)

    if (queueEdit) {
      exitQueuedEdit('save')
    } else if (busy) {
      // Slash commands should execute immediately even while the agent is
      // busy — they're client-side operations (/yolo, /skin, /new, /help,
      // etc.) or self-contained gateway RPCs (/status, /compress).  onSubmit
      // routes them to executeSlashCommand, which has its own per-command
      // busy guard for commands that genuinely need an idle session (skill
      // /send directives).  Queuing them would make every slash command wait
      // for the current turn to finish, which is how the TUI never behaves.
      if (!attachments.length && SLASH_COMMAND_RE.test(text.trim())) {
        triggerHaptic('submit')
        clearDraft()
        dispatchSubmit(text)
      } else if (!compacting && !blockingPrompt && !attachments.length && text.trim()) {
        // Cursor-style stop-and-correct: interrupt the live turn and redirect
        // it with this text. redirect() preserves the shown reasoning/work; if
        // the turn already ended, steerDraft re-queues so nothing is lost.
        steerDraft()
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
    } else if (!payloadPresent && queuedPrompts.length > 0) {
      void drainNextQueued()
    } else if (payloadPresent) {
      const submittedAttachments = cloneAttachments(attachments)
      triggerHaptic('submit')
      resetBrowseState(sessionId)
      clearDraft()
      scope.attachments.clear()
      dispatchSubmit(text, submittedAttachments)
    }

    focusInput()
  }

  // Redirect the live turn with a correction. The gateway either restarts the
  // active model request with its displayed context or waits for the current
  // tool boundary. If the turn already ended, queue the words instead.
  const steerDraft = () => {
    const text = draftRef.current.trim()

    // Guard on live editor state, not the render-lagged `canSteer`: a redirect
    // fired on a fast Enter must not be dropped because state hasn't synced.
    if (!onSteer || !text || attachments.length > 0 || SLASH_COMMAND_RE.test(text)) {
      return
    }

    triggerHaptic('submit')
    clearDraft()

    void Promise.resolve(onSteer(text)).then(accepted => {
      if (!accepted && activeQueueSessionKey) {
        enqueueQueuedPrompt(activeQueueSessionKey, { text, attachments: [] })
      }
    })
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
