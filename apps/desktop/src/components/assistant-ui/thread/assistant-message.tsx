import {
  ActionBarPrimitive,
  BranchPickerPrimitive,
  ErrorPrimitive,
  MessagePrimitive,
  useAuiState,
  useMessageRuntime,
  useThreadRuntime
} from '@assistant-ui/react'
import { useStore } from '@nanostores/react'
import { type FC, type ReactNode, useCallback, useContext, useMemo, useState } from 'react'
import { useInRouterContext, useNavigate } from 'react-router'

import { useSessionView } from '@/app/chat/session-view'
import { SETTINGS_ROUTE } from '@/app/routes'
import { dispatchedTo } from '@/components/assistant-ui/thread/agent-delivery'
import { ChangedFilesCard } from '@/components/assistant-ui/thread/changed-files-card'
import {
  contentHasVisibleText,
  messageContentText,
  pickPrimaryPreviewTarget
} from '@/components/assistant-ui/thread/content'
import { MESSAGE_PARTS_COMPONENTS } from '@/components/assistant-ui/thread/message-parts'
import { ReactionPicker } from '@/components/assistant-ui/thread/message-reactions'
import { ResponseMessageIds } from '@/components/assistant-ui/thread/response-group'
import { ResponseLoadingIndicator, TurnActivityIndicator } from '@/components/assistant-ui/thread/status'
import { MessageTimelineTimestamp } from '@/components/assistant-ui/thread/timeline-timestamp'
import { useMessageReactions, useTapbackDoubleClick } from '@/components/assistant-ui/thread/use-message-reactions'
import { AGENT_MESSAGE_RE } from '@/components/assistant-ui/thread/user-message'
import { isApprovalActivity, isCurrentTurnMessage } from '@/components/assistant-ui/tool/approval-activity'
import { TooltipIconButton } from '@/components/assistant-ui/tooltip-icon-button'
import { formatElapsed } from '@/components/chat/activity-timer'
import { PreviewAttachment } from '@/components/chat/preview-attachment'
import { Codicon } from '@/components/ui/codicon'
import { CopyButton } from '@/components/ui/copy-button'
import { useI18n } from '@/i18n'
import { errorRecoveryPlan, type ErrorSurface, formatErrorDiagnostics, isOAuthReauthSurface } from '@/lib/error-surface'
import { errorCardText } from '@/lib/error-surface-copy'
import { triggerHaptic } from '@/lib/haptics'
import {
  AudioLines,
  GitForkIcon,
  KeyRound,
  Loader2Icon,
  RefreshCwIcon,
  SmilePlusIcon,
  Upload,
  VolumeXIcon,
  XIcon
} from '@/lib/icons'
import { extractPreviewTargets } from '@/lib/preview-targets'
import { markAssistantIdSpoken } from '@/lib/spoken-reply'
import { useEnterAnimation } from '@/lib/use-enter-animation'
import { cn } from '@/lib/utils'
import { playSpeechText, stopVoicePlayback } from '@/lib/voice-playback'
import { openFreeTierSignIn } from '@/store/free-tier-sign-in'
import { notifyError } from '@/store/notifications'
import { startManualProviderOAuth } from '@/store/onboarding'
import { $activeGatewayProfile, normalizeProfileKey, requestFreshSession } from '@/store/profile'
import { sessionApprovalRequest } from '@/store/prompts'
import { requestSendDiagnostics } from '@/store/send-diagnostics'
import { $connection, $currentModel, setModelPickerOpen } from '@/store/session'
import { sessionTileDelegate } from '@/store/session-states'
import { notifyThreadEditOpen } from '@/store/thread-scroll'
import { $voicePlayback } from '@/store/voice-playback'

// Stable empty identity for the settled-parts selector — a fresh [] per render
// would re-derive the changed-files card on every message re-render.
const EMPTY_PARTS: readonly unknown[] = []

// PERF: hoisted to module scope so the element OBJECT is identical on every
// render of every assistant message. React bails out of re-rendering a child
// whose element identity is unchanged, so a status flip on the message root
// (pending -> complete and back, N rows per stream flush) can no longer
// descend into the parts subtree at all. Its props were already the module
// constant MESSAGE_PARTS_COMPONENTS, so nothing per-message is captured here.
const MESSAGE_PARTS = <MessagePrimitive.Parts components={MESSAGE_PARTS_COMPONENTS} />

const COMPLETION_TIME_FORMATTER = new Intl.DateTimeFormat(undefined, {
  hour: 'numeric',
  minute: '2-digit'
})

const validUnixSeconds = (value: unknown): value is number =>
  typeof value === 'number' && Number.isFinite(value) && value > 0

const completionDate = (value: unknown): Date | null => {
  if (!validUnixSeconds(value)) {
    return null
  }

  const date = new Date(value * 1000)

  return Number.isNaN(date.getTime()) ? null : date
}

interface MessageActionProps {
  messageId: string
  /** Lazy accessor — reads the live message text at action time. Passing the
   *  text itself as a prop forces the whole footer to re-render on every
   *  streaming delta flush (the text changes ~30×/s), which profiling showed
   *  was a large slice of per-token script time on long transcripts. */
  getMessageText: () => string
  onBranchInNewChat?: (messageId: string) => void
}

interface AssistantMessageProps {
  onBranchInNewChat?: (messageId: string) => void
  onDismissError?: (messageId: string) => void
}

export const AssistantMessage: FC<AssistantMessageProps> = props => {
  // A reply to an inter-agent delivery is part of that exchange, not part of
  // the human conversation — collapse it under a compact notice ("Reply to
  // <sender>", expandable), mirroring the sender-side notice the previous
  // user message already renders as. Grok-bots parity: the transcript shows
  // events; the texts are one click away. Detection: the immediately
  // preceding user message matches AGENT_MESSAGE_RE — UNLESS that delivery
  // answers a `message_agent` dispatch this bot itself sent to the sender
  // earlier in the thread. Seen from the dispatching bot, the inbound row is
  // the teammate's answer and the next assistant message is the report to
  // the human (#114629); folding it hid the substance of the turn behind a
  // "Replied to" row nothing was ever sent through.
  const interAgentSender = useAuiState(s => {
    const messages = s.thread.messages

    for (let i = messages.length - 1; i >= 0; i--) {
      if (messages[i].id !== s.message.id) {
        continue
      }

      for (let j = i - 1; j >= 0; j--) {
        const prev = messages[j] as { content?: unknown; role?: string }

        if (prev.role === 'assistant') {
          return null
        }

        if (prev.role === 'user') {
          const match = AGENT_MESSAGE_RE.exec(messageContentText(prev.content as never).trim())

          if (!match) {
            return null
          }

          const sender = (match[1] || match[3] || 'agent').trim()

          return dispatchedTo(messages.slice(0, j), [match[1], match[2], match[3]]) ? null : sender
        }
      }

      return null
    }

    return null
  })

  // The collapse gate below needs the LIVE running status, but only an
  // inter-agent reply can ever be collapsed. Dispatching on that first keeps
  // the status subscription out of the standard path entirely — the standard
  // message root now re-renders for content, never for a pending flip.
  return interAgentSender ? (
    <InterAgentAssistantMessage {...props} sender={interAgentSender} />
  ) : (
    <AssistantMessageBody {...props} />
  )
}

/** The compact stand-in a settled inter-agent reply collapses to (Grok-bots
 *  parity — the transcript shows the event; the text is one click away). */
const InterAgentCollapsedNotice: FC<{ sender: string }> = ({ sender }) => (
  <div className="flex max-w-[min(86%,44rem)] flex-col gap-0.5 self-center px-2 py-0.5 text-[0.6875rem] leading-5 text-muted-foreground/60">
    <span className="flex items-center justify-center gap-1.5">
      <Codicon className="shrink-0 text-muted-foreground/55" name="arrow-small-right" size="0.8125rem" />
      <span className="wrap-anywhere">Replied to {sender}</span>
    </span>
    <details className="self-center">
      <summary className="cursor-pointer select-none text-center text-muted-foreground/45 hover:text-muted-foreground/70">
        show reply
      </summary>
      <div className="mt-1 max-w-[36rem] rounded-lg border border-(--ui-stroke-tertiary) px-3 py-2 text-left text-[0.75rem] leading-5 text-foreground/85">
        {MESSAGE_PARTS}
      </div>
    </details>
  </div>
)

/**
 * An assistant reply that answers an inter-agent delivery. Owns the only
 * root-level `isRunning` subscription left in this file, and it is confined to
 * the rare inter-agent case: the reply renders collapsed once it settles, so
 * the gate genuinely needs live status. Never collapse while streaming — the
 * user should see progress.
 *
 * The collapse is expressed as a CHILD of the normal body, not as a competing
 * root. Returning a bare MessagePrimitive.Root here for the settled case put a
 * different element type in this position than the running case
 * (AssistantMessageBody), so settling unmounted the whole row and mounted a
 * fresh one — throwing away the DOM the scroll anchor was holding, which can
 * jump the transcript under the reader. One component, one root, children
 * vary: settling is now a prop change React applies in place.
 */
const InterAgentAssistantMessage: FC<AssistantMessageProps & { sender: string }> = ({ sender, ...props }) => {
  const isRunning = useAuiState(s => s.message.status?.type === 'running')

  return (
    <AssistantMessageBody
      {...props}
      collapsedNotice={isRunning ? null : <InterAgentCollapsedNotice sender={sender} />}
    />
  )
}

const AssistantMessageBody: FC<AssistantMessageProps & { collapsedNotice?: null | ReactNode }> = ({
  collapsedNotice = null,
  onBranchInNewChat,
  onDismissError
}) => {
  const messageId = useAuiState(s => s.message.id)
  const messageRuntime = useMessageRuntime()
  const threadRuntime = useThreadRuntime()
  const responseIds = useContext(ResponseMessageIds)
  const responseTail = responseIds.length === 0 || responseIds.at(-1) === messageId
  const { t } = useI18n()

  // PERF: this component must NOT subscribe to the streaming text, and no
  // longer subscribes to the streaming STATUS either. Every selector here
  // returns a value that stays referentially stable across token flushes
  // (booleans, '' while running), so the 30 Hz delta stream only re-renders
  // the markdown part and the tiny status leaves — not the footer, the
  // preview block, or this root.
  const hasVisibleText = useAuiState(s => contentHasVisibleText(s.message.content))
  const sessionId = useStore(useSessionView().$runtimeId)
  const approval = useStore(useMemo(() => sessionApprovalRequest(sessionId), [sessionId]))

  const activityOnly = useAuiState(
    state =>
      isCurrentTurnMessage(state.thread.messages, state.message.id) &&
      state.message.content.some(part => part.type === 'tool-call' && isApprovalActivity(part)) &&
      state.message.content.every(
        part => (part.type === 'tool-call' && isApprovalActivity(part)) || (part.type === 'text' && !part.text.trim())
      )
  )

  // Sealed mid-turn commentary keeps its text but not the footer, so a
  // tool-heavy turn doesn't grow a copy/refresh bar per paragraph (see
  // ChatMessage.interim).
  const isInterim = useAuiState(s => s.message.metadata?.custom?.interim === true)

  // Whole-turn duration and completion timestamp are set once at settlement,
  // so these selectors stay stable across the 30 Hz delta stream.
  const turnDurationS = useAuiState(s => s.message.metadata?.custom?.durationS as number | undefined)
  const turnCompletedAt = useAuiState(s => {
    const value = s.message.metadata?.custom?.timelineCompletedAt

    return completionDate(value) ? (value as number) : undefined
  })

  const getMessageText = useCallback(
    () =>
      responseIds.length
        ? responseIds
            .map(id => messageContentText(threadRuntime.getMessageById(id).getState().content))
            .filter(Boolean)
            .join('\n\n')
        : messageContentText(messageRuntime.getState().content),
    [messageRuntime, responseIds, threadRuntime]
  )

  // useEnterAnimation consults `enabled` ONLY when its callback ref fires,
  // i.e. at mount: the hook parks the value in a ref and returns a
  // useCallback([]) identity, and its own contract is "`enabled` is captured
  // at mount-time only — flipping it later doesn't suddenly play the animation
  // on existing nodes" (see lib/use-enter-animation.ts). So a live
  // subscription here would re-render this root on every pending flip to feed
  // a value the hook already ignores. Capture it once, off the runtime, with
  // no subscription at all.
  const [initiallyRunning] = useState(() => messageRuntime.getState().status?.type === 'running')
  const enterRef = useEnterAnimation(initiallyRunning, `assistant-message:${messageId}`)

  // Double-click the reply to heart it (iMessage). Undefined while reactions
  // are off, so the root carries no listener at all.
  const onDoubleClick = useTapbackDoubleClick(messageId, 'assistant')

  return (
    <MessagePrimitive.Root
      className={cn(
        'group flex w-full min-w-0 max-w-full flex-col gap-0 self-start overflow-hidden',
        collapsedNotice && 'pb-(--conversation-turn-gap)'
      )}
      data-approval-activity-only={approval && activityOnly ? '' : undefined}
      data-role="assistant"
      data-slot="aui_assistant-message-root"
      // Collapsed inter-agent rows never carried the tapback listener; keeping
      // that exact truth table means gating it on the notice rather than on
      // whether the hook returned a handler.
      onDoubleClick={collapsedNotice ? undefined : onDoubleClick}
      ref={enterRef}
    >
      {collapsedNotice ?? (
        <>
          <div
            className="wrap-anywhere min-w-0 max-w-full overflow-hidden text-pretty text-[length:var(--conversation-text-font-size)] leading-(--dt-line-height) text-foreground"
            data-slot="aui_assistant-message-content"
          >
            {/* Todos render in the composer status stack now, not inline. */}
            {MESSAGE_PARTS}
            <AssistantStatusSlot />
            <AssistantPreviewEmbeds />
            <MessagePrimitive.Error>
              <ErrorPrimitive.Root
                className="mt-1.5 flex flex-col gap-1.5 rounded-lg border border-[color-mix(in_srgb,var(--dt-destructive)_35%,transparent)] bg-[color-mix(in_srgb,var(--dt-destructive)_7%,transparent)] px-3 py-2 text-[0.78rem] leading-5 text-[color-mix(in_srgb,var(--dt-destructive)_78%,var(--ui-text-secondary))]"
                role="alert"
              >
                <div className="flex items-start gap-1.5">
                  <div className="min-w-0 flex-1">
                    <ErrorCardHeadline />
                  </div>
                  {onDismissError && (
                    <TooltipIconButton
                      className="-my-0.5 shrink-0 text-current opacity-70 hover:opacity-100"
                      onClick={() => onDismissError(messageId)}
                      side="top"
                      tooltip={t.assistant.thread.dismissError}
                    >
                      <XIcon className="size-3.5" />
                    </TooltipIconButton>
                  )}
                </div>
                <ErrorRecoveryActions />
              </ErrorPrimitive.Root>
            </MessagePrimitive.Error>
          </div>
          <MessageTimelineTimestamp className="px-(--message-text-indent) pt-0.5" suppressIfDuplicatePart />
          {hasVisibleText && !isInterim && responseTail && (
            <AssistantFooter
              completedAt={turnCompletedAt}
              durationS={turnDurationS}
              getMessageText={getMessageText}
              messageId={messageId}
              onBranchInNewChat={onBranchInNewChat}
            />
          )}
          {!hasVisibleText &&
            !isInterim &&
            (turnDurationS !== undefined || turnCompletedAt !== undefined) && (
              <div className="flex min-h-6 items-center px-(--message-text-indent)">
                <TurnCompletionMeta completedAt={turnCompletedAt} durationS={turnDurationS} />
              </div>
            )}
          {/* Last thing in the turn — under the action bar, the way Cursor ends a
          turn on its summary rather than burying it above the controls. */}
          <SettledChangedFiles />
          <StreamingMarker />
        </>
      )}
    </MessagePrimitive.Root>
  )
}

/**
 * PERF leaf: the only subscriber to this message's streaming status inside the
 * message content. Previously `messageStatus` / `isPlaceholder` /
 * `isLastMessage` were read by AssistantMessage itself, so every pending flip
 * re-rendered the whole message subtree — at stream breadth N, N subtrees in a
 * single commit, which is what widened the recalc scope. Reading them here
 * confines the flip to this leaf; the sibling parts subtree is a hoisted
 * constant element and bails out.
 *
 * Behaviour is byte-identical to the old inline expression, including the
 * TAIL-ONLY rule: the activity row belongs to the tail of the thread, period.
 * A stale pending bubble mid-transcript (a turn that ended without its settle
 * event, a steer race) must never show one — a spinner above a later user
 * message reads as the agent answering out of order.
 *
 * The activity row is mounted by the TAIL of the thread and decides for itself
 * whether the turn owes the user a line, so there is deliberately no
 * `isRunning` gate on the mount here. Gating it on this bubble's own `running`
 * status was the hole: a turn that seals a bubble mid-flight (message.interim)
 * or finishes one while the agent keeps going leaves a settled message at the
 * tail, so the row unmounted and the seconds went uncounted while the
 * composer's arc border and Stop button said work was still happening.
 * TurnActivityIndicator subscribes to the status it needs internally, so it is
 * itself a leaf and this stays off the message root either way.
 */
const AssistantStatusSlot: FC = () => {
  // ONE subscription, not one per input. Each useAuiState is a separate store
  // subscription with its own equality check and its own chance to schedule a
  // render, and these inputs always move together on a status flip — so
  // reading them separately just multiplies the wake-ups for a single logical
  // change. The selector collapses them to one stable string, which bails out
  // on every flush that does not actually change what this slot renders.
  const slot = useAuiState(s => {
    if (s.thread.messages[s.thread.messages.length - 1]?.id !== s.message.id) {
      return 'none'
    }

    return s.message.status?.type === 'running' && s.message.content.length === 0 ? 'placeholder' : 'activity'
  })

  if (slot === 'none') {
    return null
  }

  return slot === 'placeholder' ? <ResponseLoadingIndicator /> : <TurnActivityIndicator />
}

/**
 * PERF leaf: owns the settled-text selector that feeds the link previews.
 *
 * This was the last status-dependent read at the message root, and the most
 * expensive one: the selector flips between '' while running and the full
 * `messageContentText(content)` join once settled, so every running <-> settled
 * transition re-ran the join for the whole message AND re-rendered the root.
 * At stream breadth N that is N joins plus N root re-renders per flip. Reading
 * it here confines both to this leaf, which renders nothing at all in the
 * common case.
 *
 * The streaming-side optimization is unchanged and still the point of the ''
 * branch: preview targets only materialize once the turn completes, so while
 * running the selector returns a stable '' and per-token flushes skip the
 * regex scan and the re-render it would cause.
 *
 * Renders exactly what the root used to render at this position — the same
 * wrapper div with the same classes, or nothing when there are no targets —
 * so the DOM is byte-identical either way. A component boundary adds no node
 * of its own, so unlike StreamingMarker this needs no placement care.
 */
const AssistantPreviewEmbeds: FC = () => {
  const completedText = useAuiState(s =>
    s.message.status?.type === 'running' ? '' : messageContentText(s.message.content)
  )

  const previewTargets = useMemo(() => {
    if (!completedText || !/(https?:\/\/|file:\/\/)/i.test(completedText)) {
      return []
    }

    return pickPrimaryPreviewTarget(extractPreviewTargets(completedText))
  }, [completedText])

  if (previewTargets.length === 0) {
    return null
  }

  return (
    <div className="mt-3 flex flex-wrap gap-2">
      {previewTargets.map(target => (
        <PreviewAttachment key={target} source="explicit-link" target={target} />
      ))}
    </div>
  )
}

/**
 * PERF leaf: owns the `settledParts` selector so the tail's settle stops
 * re-rendering the message root. This is the one status-derived selector that
 * returns an OBJECT (`s.message.parts`) rather than a primitive, so it cannot
 * bail out on identity churn — keeping it at the root meant every settle
 * re-rendered the root and everything under it.
 *
 * Cursor's changed-files card only appears once the turn settles: while the
 * agent is still editing, the tool rows narrate each patch and a card that
 * grew a row per write would thrash the transcript. `EMPTY_PARTS` while
 * running keeps this selector referentially stable across the 30 Hz delta
 * stream.
 *
 * It also only rides the LAST turn. The card is a "here's what just landed"
 * summary, not a per-turn artifact: leaving one behind on every reply would
 * stack a wall of stale cards down the transcript. Sending the next message
 * retires it — the working tree it describes is already history by then.
 */
const SettledChangedFiles: FC = () => {
  const settledParts = useAuiState(s => {
    const isLastMessage = s.thread.messages[s.thread.messages.length - 1]?.id === s.message.id

    return s.message.status?.type === 'running' || !isLastMessage ? EMPTY_PARTS : s.message.parts
  })

  return <ChangedFilesCard parts={settledParts} />
}

/**
 * Carries the streaming flag that used to sit on the message root as
 * `data-streaming`.
 *
 * The flag has no CSS behind it (every `[data-streaming='true']` rule targets
 * `[data-slot='code-card']`), but it is not dead: it is the settled-row signal
 * for the short-session hang repro, which derives the settled count by
 * subtracting the number of `[data-message-streaming='true']` markers from the
 * number of message roots, and gates the assistant-response wait on that count
 * growing. At most one marker per row carries the attribute, which is what
 * makes the subtraction exact.
 *
 * Deliberately NOT named `data-streaming`: shiki-highlighter.tsx puts that
 * exact attribute on a deferred `[data-slot='code-card']`, which is a
 * descendant of this root. Once the repro matches on a descendant rather than
 * the root's own attribute, a shared name would make any message holding a
 * still-deferred code card read as "still streaming". A distinct name keeps
 * the signal about the MESSAGE and immune to how deep it sits.
 *
 * On the root it was a per-flip attribute write on the element that owns the
 * whole message subtree, which is the invalidation this prong exists to remove.
 * Three properties make this placement cheap and behaviour-neutral:
 *
 *  - A ROOT-LEVEL sibling, not a child of the message content. The
 *    `:first-child` / `:last-child` margin rules in styles.css match blocks
 *    *inside* `[data-slot='aui_assistant-message-content']`; a node added
 *    there would steal `:last-child` from the status indicator and silently
 *    change the gap between bubbles mid-stream. No rule selects message-root
 *    children by position, so this slot is inert.
 *  - PERMANENTLY MOUNTED, toggling only the attribute. Mounting/unmounting per
 *    flip would be a DOM structure change and dirty its siblings; an attribute
 *    write on a childless node invalidates exactly one element.
 *  - `display: none`, so it costs no layout or paint. `querySelectorAll` and
 *    `:has()` still match it — they read the DOM, not the box tree.
 *
 * Tracks plain `isRunning` (not tail-only), exactly like the old root
 * attribute, so the repro's row accounting is unchanged.
 */
const StreamingMarker: FC = () => {
  const isRunning = useAuiState(s => s.message.status?.type === 'running')

  return (
    <span
      aria-hidden="true"
      className="hidden"
      data-message-streaming={isRunning ? 'true' : undefined}
      data-slot="aui_message-streaming-marker"
    />
  )
}

// ── Layered error card pieces ────────────────────────────────────────────
//
// The gateway stamps failed turns with a structured {layer, code, retryable}
// descriptor (metadata.custom.errorSurface — see agent/error_surface.py).
// These leaves render a plain-language headline + recovery actions, both
// resolved from ONE table keyed on the failure code (lib/error-surface.ts,
// i18n `assistant.thread.errorCodes`); the raw provider/gateway text moves to
// a collapsed "Details" line. Older backends never send the descriptor: the
// headline falls back to a generic title and the action row still offers
// Retry / Open logs / Copy error details, so nothing regresses on version skew.

const useErrorSurface = () => useAuiState(s => s.message.metadata?.custom?.errorSurface as ErrorSurface | undefined)

const useErrorText = () =>
  useAuiState(s => {
    const status = s.message.status as { error?: unknown; type?: string } | undefined

    return status?.type === 'incomplete' && typeof status.error === 'string' ? status.error : ''
  })

const ErrorCardHeadline: FC = () => {
  const { t } = useI18n()
  const surface = useErrorSurface()
  const errorText = useErrorText()
  const { body, title } = errorCardText(t.assistant.thread, surface)

  return (
    <>
      <div className="font-medium">{title}</div>
      <div>{body}</div>
      {errorText && (
        <details className="mt-0.5 min-w-0 text-[0.72rem] opacity-70">
          <summary className="cursor-pointer select-none">{t.assistant.thread.errorDetails}</summary>
          <div className="wrap-anywhere mt-0.5 whitespace-pre-wrap font-mono">{errorText}</div>
        </details>
      )}
    </>
  )
}

// Isolated because useNavigate() THROWS outside a <Router> (bare test
// harnesses, embedded panes render threads router-free). The parent gates
// these children's mount on useInRouterContext(), which is safe anywhere.
const SettingsLinkAction: FC<{ icon?: ReactNode; label: string; to: string }> = ({ icon, label, to }) => {
  const navigate = useNavigate()

  return (
    <button className="aui-error-action" onClick={() => navigate(to)} type="button">
      {icon}
      {label}
    </button>
  )
}

// Settings → Keys deep link for a rejected API key: `?tab=keys` plus
// `&key=<ENV>` when the descriptor names the env var (keys-settings.tsx
// scrolls to and expands that row). Older backends omit `api_key_env`; the
// tab alone is still the right place.
const updateApiKeyRoute = (surface: ErrorSurface | undefined) => {
  const params = new URLSearchParams({ tab: 'keys' })

  if (surface?.apiKeyEnv) {
    params.set('key', surface.apiKeyEnv)
  }

  return `${SETTINGS_ROUTE}?${params.toString()}`
}

// "Edit message" for a safety refusal: opens the preceding user message in
// the edit composer, the same runtime call the bubble's own click performs
// (user-message.tsx ActionBarPrimitive.Edit). Retry would reproduce the
// refusal; changing the words is the only way forward.
const EditPreviousMessageAction: FC<{ label: string }> = ({ label }) => {
  const threadRuntime = useThreadRuntime()

  const previousUserMessageId = useAuiState(s => {
    const messages = s.thread.messages
    const index = messages.findIndex(message => message.id === s.message.id)

    for (let i = index - 1; i >= 0; i--) {
      if (messages[i].role === 'user') {
        return messages[i].id
      }
    }

    return null
  })

  const beginEdit = useCallback(() => {
    if (!previousUserMessageId) {
      return
    }

    triggerHaptic('selection')
    notifyThreadEditOpen()
    threadRuntime.getMessageById(previousUserMessageId).composer.beginEdit()
  }, [previousUserMessageId, threadRuntime])

  if (!previousUserMessageId) {
    return null
  }

  return (
    <button className="aui-error-action" onClick={beginEdit} type="button">
      {label}
    </button>
  )
}

// "Compress conversation" for a context overflow: runs /compress against the
// failed turn's OWN session through the app's slash pipeline (the same
// session.compress RPC path the typed command takes — slash.ts `compress`),
// so it inherits the stale-runtime recovery, transcript replacement and
// progress notice instead of re-implementing them here.
const CompressConversationAction: FC<{ label: string }> = ({ label }) => {
  const { t } = useI18n()
  const view = useSessionView()
  const sessionId = useStore(view.$runtimeId)

  const compress = useCallback(() => {
    const delegate = sessionTileDelegate()

    if (!sessionId || !delegate) {
      notifyError(new Error('slash delegate unavailable'), t.assistant.thread.errorCompressFailed)

      return
    }

    triggerHaptic('submit')
    void delegate.executeSlash('/compress', sessionId).catch(error => {
      notifyError(error, t.assistant.thread.errorCompressFailed)
    })
  }, [sessionId, t.assistant.thread.errorCompressFailed])

  return (
    <button className="aui-error-action" onClick={compress} type="button">
      {label}
    </button>
  )
}

const ErrorRecoveryActions: FC = () => {
  const { t } = useI18n()
  const copy = t.assistant.thread
  const surface = useErrorSurface()
  const errorText = useErrorText()

  // useNavigate() would throw here when no Router is above us; the deep-link
  // children mount only when one is (see SettingsLinkAction).
  const inRouter = useInRouterContext()
  const model = useStore($currentModel)
  const connection = useStore($connection)

  // Open Logs reveals the LOCAL Electron profile's HERMES_HOME/logs. On a
  // remote/cloud connection the failed turn's gateway+agent logs live on the
  // remote box — the local folder only holds Desktop-side transport logs, so
  // the label says "Open Desktop logs" there instead of implying it opens the
  // runtime's logs.
  const remoteConnection = connection?.mode === 'remote'

  // One table decides which buttons this failure gets (lib/error-surface.ts).
  const plan = errorRecoveryPlan(surface)

  // An expired/revoked OAuth grant (HTTP 401 on nous / openai-codex / ...):
  // the one-click fix is re-running that provider's sign-in, which the
  // onboarding overlay already owns end to end (device code → poll →
  // reload.env → model confirm). Scoped to the gateway profile the failed
  // session runs on, so a Bot profile's grant is renewed, not the primary's.
  const gatewayProfile = useStore($activeGatewayProfile)

  const signInAgain = useCallback(() => {
    if (!isOAuthReauthSurface(surface)) {
      return
    }

    triggerHaptic('submit')
    const key = normalizeProfileKey(gatewayProfile)
    startManualProviderOAuth(surface.provider, key === 'default' ? undefined : key)
  }, [gatewayProfile, surface])

  // The free tier's door: the same dialog the status-bar chip and the first-launch
  // intro open. Signing in is free and lifts every free-tier refusal.
  const signInFreeTier = useCallback(() => {
    triggerHaptic('submit')
    openFreeTierSignIn()
  }, [])

  // Reveal a local folder through Electron; `logsRoot` is the profile's
  // HERMES_HOME/logs, and its parent is the Hermes data folder itself (what
  // the user needs to see to free space after a disk-full failure).
  const openLocalDir = useCallback(async (resolve: (logsRoot: string) => string, failedMessage: string) => {
    try {
      const root = await window.hermesDesktop?.logsRoot?.()

      if (!root) {
        notifyError(new Error('logs root unavailable'), failedMessage)

        return
      }

      const result = await window.hermesDesktop?.openDir?.(resolve(root))

      if (result && !result.ok) {
        notifyError(new Error(result.error || 'open failed'), failedMessage)
      }
    } catch (error) {
      notifyError(error, failedMessage)
    }
  }, [])

  const openLogs = useCallback(
    () => openLocalDir(root => root, copy.errorOpenLogsFailed),
    [copy.errorOpenLogsFailed, openLocalDir]
  )

  const openHermesFolder = useCallback(
    () => openLocalDir(root => root.replace(/[\\/]+logs[\\/]*$/, ''), copy.errorOpenHermesFolderFailed),
    [copy.errorOpenHermesFolderFailed, openLocalDir]
  )

  const diagnosticsText = useCallback(
    () =>
      formatErrorDiagnostics({
        errorText,
        model: model || undefined,
        surface
      }),
    [errorText, model, surface]
  )

  const startNewSession = useCallback(() => {
    triggerHaptic('submit')
    requestFreshSession()
  }, [])

  const chooseModel = useCallback(() => {
    triggerHaptic('selection')
    setModelPickerOpen(true)
  }, [])

  const localFolders = Boolean(window.hermesDesktop?.logsRoot)

  return (
    <div className="flex flex-wrap items-center gap-1.5">
      {plan.editMessage && <EditPreviousMessageAction label={copy.editMessage} />}
      {plan.compress && <CompressConversationAction label={copy.errorCompressConversation} />}
      {plan.chooseModel && (
        <button className="aui-error-action" onClick={chooseModel} type="button">
          {copy.errorChooseModel}
        </button>
      )}
      {plan.startNewSession && (
        <button className="aui-error-action" onClick={startNewSession} type="button">
          {copy.errorStartNewSession}
        </button>
      )}
      {plan.signInAgain && isOAuthReauthSurface(surface) && (
        <button className="aui-error-action" onClick={signInAgain} type="button">
          <KeyRound className="size-3" />
          {copy.errorSignInAgain(surface.providerLabel || surface.provider)}
        </button>
      )}
      {plan.signInFreeTier && (
        <button className="aui-error-action" onClick={signInFreeTier} type="button">
          <KeyRound className="size-3" />
          {copy.errorSignInFreeTier}
        </button>
      )}
      {plan.updateApiKey && inRouter && (
        <SettingsLinkAction
          icon={<KeyRound className="size-3" />}
          label={copy.errorUpdateApiKey}
          to={updateApiKeyRoute(surface)}
        />
      )}
      {plan.openHermesFolder && localFolders && (
        <button className="aui-error-action" onClick={() => void openHermesFolder()} type="button">
          {copy.errorOpenHermesFolder}
        </button>
      )}
      {plan.retry && (
        <ActionBarPrimitive.Reload asChild>
          <button className="aui-error-action" onClick={() => triggerHaptic('submit')} type="button">
            <RefreshCwIcon className="size-3" />
            {copy.errorRetry}
          </button>
        </ActionBarPrimitive.Reload>
      )}
      {plan.switchProvider && inRouter && (
        <SettingsLinkAction label={copy.errorSwitchProvider} to={`${SETTINGS_ROUTE}?tab=config:model`} />
      )}
      {localFolders && (
        <button className="aui-error-action" onClick={() => void openLogs()} type="button">
          {remoteConnection ? copy.errorOpenDesktopLogs : copy.errorOpenLogs}
        </button>
      )}
      <button className="aui-error-action" onClick={() => requestSendDiagnostics(diagnosticsText())} type="button">
        <Upload className="size-3" />
        {copy.errorSendDiagnostics}
      </button>
      <CopyButton
        appearance="inline"
        className="aui-error-action"
        label={copy.errorCopyDiagnostics}
        text={diagnosticsText}
      />
    </div>
  )
}

interface TurnCompletionMetaProps {
  className?: string
  completedAt?: number
  durationS?: number
}

const TurnCompletionMeta: FC<TurnCompletionMetaProps> = ({ className, completedAt, durationS }) => {
  const { t } = useI18n()
  const finished = completionDate(completedAt)

  if (durationS === undefined && !finished) {
    return null
  }

  return (
    <span
      className={cn('select-none text-[0.6875rem] leading-5 tabular-nums text-muted-foreground', className)}
      data-slot="aui_turn-completion"
    >
      {durationS !== undefined && (
        <span data-slot="aui_turn-duration" title={t.assistant.thread.turnDuration(formatElapsed(durationS))}>
          ⏱ {formatElapsed(durationS)}
        </span>
      )}
      {durationS !== undefined && finished && ' · '}
      {finished && (
        <time data-slot="aui_turn-finished-at" dateTime={finished.toISOString()}>
          {COMPLETION_TIME_FORMATTER.format(finished)}
        </time>
      )}
    </span>
  )
}

const AssistantActionBar: FC<MessageActionProps & { completedAt?: number; durationS?: number }> = ({
  completedAt,
  durationS,
  messageId,
  getMessageText,
  onBranchInNewChat
}) => {
  const { t } = useI18n()
  const copy = t.assistant.thread

  const [pickerOpen, setPickerOpen] = useState(false)
  const { enabled: reactionsEnabled, react, reactions: shownReactions } = useMessageReactions(messageId, 'assistant')

  const pickEmoji = useCallback(
    (emoji: null | string) => {
      setPickerOpen(false)
      react(emoji)
    },
    [react]
  )

  return (
    <div className="relative flex w-full shrink-0 items-center justify-end gap-1.5">
      <TurnCompletionMeta className="mr-auto px-0.5" completedAt={completedAt} durationS={durationS} />
      <ActionBarPrimitive.Root
        className={
          // NOTE: intentionally NOT `hideWhenRunning`. That prop unmounts the
          // bar while the thread streams, which collapses every completed
          // assistant message's footer by this bar's height and shifts the
          // whole conversation when the turn resolves. The bar is already
          // invisible by default (opacity-0 + pointer-events-none, reveals on
          // hover), so keeping it mounted reserves stable layout height with
          // no visual change during streaming.
          'relative flex flex-row items-center justify-end gap-1.5 py-1.5 opacity-0 pointer-events-none group-hover:pointer-events-auto group-hover:opacity-100 focus-within:pointer-events-auto focus-within:opacity-100'
        }
        data-slot="aui_msg-actions"
      >
        {onBranchInNewChat && (
          <TooltipIconButton
            onClick={() => {
              triggerHaptic('selection')
              onBranchInNewChat(messageId)
            }}
            tooltip={copy.branchNewChat}
          >
            <GitForkIcon className="size-3.5" />
          </TooltipIconButton>
        )}
        <CopyButton appearance="icon" buttonSize="icon" label={copy.copy} text={getMessageText} />
        <ReadAloudButton getText={getMessageText} messageId={messageId} />
        <ActionBarPrimitive.Reload asChild>
          <TooltipIconButton onClick={() => triggerHaptic('submit')} tooltip={copy.refresh}>
            <RefreshCwIcon className="size-3.5" />
          </TooltipIconButton>
        </ActionBarPrimitive.Reload>
      </ActionBarPrimitive.Root>
      {/* ONE slot, Slack-style: the picker trigger and the landed reaction are
          the same element, so reacting never shifts layout. Empty → ☺, hidden
          until hover like its action-bar neighbors (state lives in styles.css
          — the aui_msg-reactions rules outweigh Tailwind utilities here).
          Reacted → the emoji itself, always visible at full strength, and
          clicking it reopens the picker to switch or retract. Outside
          ActionBarPrimitive.Root so a landed reaction doesn't ride the bar's
          hover opacity. */}
      {(reactionsEnabled || shownReactions.length > 0) && (
        <ReactionPicker
          onOpenChange={setPickerOpen}
          onSelect={pickEmoji}
          open={pickerOpen}
          selected={shownReactions.find(reaction => reaction.author === 'user')?.emoji}
        >
          <TooltipIconButton
            data-reacted={shownReactions.length > 0 || undefined}
            data-slot="aui_msg-reactions"
            data-state={pickerOpen ? 'open' : undefined}
            onClick={reactionsEnabled ? () => setPickerOpen(open => !open) : undefined}
            tooltip={copy.react}
          >
            {shownReactions.length > 0 ? (
              <span className="flex items-center gap-0.5 text-[0.8125rem] leading-none">
                {shownReactions.map(reaction => (
                  <span className="reaction-pop" key={`${reaction.author}-${reaction.emoji}`}>
                    {reaction.emoji}
                  </span>
                ))}
              </span>
            ) : (
              <SmilePlusIcon className="size-3.5" />
            )}
          </TooltipIconButton>
        </ReactionPicker>
      )}
    </div>
  )
}

const ReadAloudButton: FC<{ getText: () => string; messageId: string }> = ({ getText, messageId }) => {
  const { t } = useI18n()
  const copy = t.assistant.thread
  const voicePlayback = useStore($voicePlayback)
  const view = useSessionView()
  const sessionId = useStore(view.$runtimeId)

  const readAloudStatus =
    voicePlayback.source === 'read-aloud' && voicePlayback.messageId === messageId ? voicePlayback.status : 'idle'

  const isPreparing = readAloudStatus === 'preparing'
  const isSpeaking = readAloudStatus === 'speaking'
  const anyPlaybackActive = voicePlayback.status !== 'idle'
  const Icon = isPreparing ? Loader2Icon : isSpeaking ? VolumeXIcon : AudioLines
  const tooltip = isPreparing ? copy.preparingAudio : isSpeaking ? copy.stopReading : copy.readAloud

  const read = useCallback(async () => {
    const text = getText()

    if (!text || $voicePlayback.get().status !== 'idle') {
      return
    }

    try {
      await playSpeechText(text, { messageId, source: 'read-aloud' })
      markAssistantIdSpoken(sessionId, view.$messages.get(), messageId)
    } catch (error) {
      notifyError(error, copy.readAloudFailed)
    }
  }, [copy.readAloudFailed, getText, messageId, sessionId, view.$messages])

  return (
    <TooltipIconButton
      disabled={isPreparing || (!isSpeaking && anyPlaybackActive)}
      onClick={() => {
        triggerHaptic('selection')
        void (isSpeaking ? stopVoicePlayback() : read())
      }}
      tooltip={tooltip}
    >
      <Icon className={cn('size-3.5', isPreparing && 'animate-spin')} />
    </TooltipIconButton>
  )
}

const AssistantFooter: FC<MessageActionProps & { completedAt?: number; durationS?: number }> = ({
  completedAt,
  durationS,
  ...props
}) => {
  return (
    <div className="flex min-h-6 flex-col items-end gap-1 pr-(--message-text-indent) pl-(--message-text-indent)">
      <BranchPickerPrimitive.Root
        className="inline-flex h-6 items-center gap-1 text-xs text-muted-foreground"
        hideWhenSingleBranch
      >
        <BranchPickerPrimitive.Previous className="grid size-6 place-items-center rounded-md text-muted-foreground transition-colors hover:bg-accent hover:text-foreground disabled:cursor-default disabled:opacity-35">
          <Codicon name="chevron-left" size="0.875rem" />
        </BranchPickerPrimitive.Previous>
        <span className="tabular-nums">
          <BranchPickerPrimitive.Number /> / <BranchPickerPrimitive.Count />
        </span>
        <BranchPickerPrimitive.Next className="grid size-6 place-items-center rounded-md text-muted-foreground transition-colors hover:bg-accent hover:text-foreground disabled:cursor-default disabled:opacity-35">
          <Codicon name="chevron-right" size="0.875rem" />
        </BranchPickerPrimitive.Next>
      </BranchPickerPrimitive.Root>
      <AssistantActionBar completedAt={completedAt} durationS={durationS} {...props} />
    </div>
  )
}
