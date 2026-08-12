import {
  ActionBarPrimitive,
  BranchPickerPrimitive,
  ErrorPrimitive,
  MessagePrimitive,
  useAuiState,
  useMessageRuntime
} from '@assistant-ui/react'
import { useStore } from '@nanostores/react'
import { type FC, useCallback, useMemo, useState } from 'react'

import { ChangedFilesCard } from '@/components/assistant-ui/thread/changed-files-card'
import {
  contentHasVisibleText,
  messageContentText,
  pickPrimaryPreviewTarget
} from '@/components/assistant-ui/thread/content'
import { MESSAGE_PARTS_COMPONENTS } from '@/components/assistant-ui/thread/message-parts'
import { ReactionPicker } from '@/components/assistant-ui/thread/message-reactions'
import { ResponseLoadingIndicator, StreamStallIndicator } from '@/components/assistant-ui/thread/status'
import { MessageTimelineTimestamp } from '@/components/assistant-ui/thread/timeline-timestamp'
import { useMessageReactions, useTapbackDoubleClick } from '@/components/assistant-ui/thread/use-message-reactions'
import { AGENT_MESSAGE_RE } from '@/components/assistant-ui/thread/user-message'
import { TooltipIconButton } from '@/components/assistant-ui/tooltip-icon-button'
import { formatElapsed } from '@/components/chat/activity-timer'
import { PreviewAttachment } from '@/components/chat/preview-attachment'
import { Codicon } from '@/components/ui/codicon'
import { CopyButton } from '@/components/ui/copy-button'
import { useI18n } from '@/i18n'
import { triggerHaptic } from '@/lib/haptics'
import { AudioLines, GitForkIcon, Loader2Icon, RefreshCwIcon, SmilePlusIcon, VolumeXIcon, XIcon } from '@/lib/icons'
import { extractPreviewTargets } from '@/lib/preview-targets'
import { useEnterAnimation } from '@/lib/use-enter-animation'
import { cn } from '@/lib/utils'
import { playSpeechText, stopVoicePlayback } from '@/lib/voice-playback'
import { notifyError } from '@/store/notifications'
import { $voicePlayback } from '@/store/voice-playback'

// Stable empty identity for the settled-parts selector — a fresh [] per render
// would re-derive the changed-files card on every message re-render.
const EMPTY_PARTS: readonly unknown[] = []

interface MessageActionProps {
  messageId: string
  /** Lazy accessor — reads the live message text at action time. Passing the
   *  text itself as a prop forces the whole footer to re-render on every
   *  streaming delta flush (the text changes ~30×/s), which profiling showed
   *  was a large slice of per-token script time on long transcripts. */
  getMessageText: () => string
  onBranchInNewChat?: (messageId: string) => void
}

export const AssistantMessage: FC<{
  onBranchInNewChat?: (messageId: string) => void
  onDismissError?: (messageId: string) => void
}> = ({ onBranchInNewChat, onDismissError }) => {
  const messageId = useAuiState(s => s.message.id)
  const messageRuntime = useMessageRuntime()
  const { t } = useI18n()

  // A reply to an inter-agent delivery is part of that exchange, not part of
  // the human conversation — collapse it under a compact notice ("Reply to
  // <sender>", expandable), mirroring the sender-side notice the previous
  // user message already renders as. Grok-bots parity: the transcript shows
  // events; the texts are one click away. Detection: the immediately
  // preceding user message matches AGENT_MESSAGE_RE.
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

          return match ? (match[1] || match[3] || 'agent').trim() : null
        }
      }

      return null
    }

    return null
  })

  // PERF: this component must NOT subscribe to the streaming text. Every
  // selector here returns a value that stays referentially stable across
  // token flushes (booleans, status strings, '' while running), so the
  // 30 Hz delta stream only re-renders the markdown part and the tiny
  // StreamStallIndicator leaf — not the footer/preview/root subtree.
  const messageStatus = useAuiState(s => s.message.status?.type)
  const isRunning = messageStatus === 'running'
  const isPlaceholder = useAuiState(s => s.message.status?.type === 'running' && s.message.content.length === 0)
  const hasVisibleText = useAuiState(s => contentHasVisibleText(s.message.content))
  // Sealed mid-turn commentary keeps its text but not the footer, so a
  // tool-heavy turn doesn't grow a copy/refresh bar per paragraph (see
  // ChatMessage.interim).
  const isInterim = useAuiState(s => s.message.metadata?.custom?.interim === true)

  // Whole-turn wall-clock seconds (set once at completion — referentially
  // stable across the 30 Hz delta stream, so this adds no per-token renders).
  const turnDurationS = useAuiState(s => s.message.metadata?.custom?.durationS as number | undefined)

  // The thinking/stall indicator belongs to the TAIL of the thread, period. A
  // stale pending bubble mid-transcript (a turn that ended without its settle
  // event, a steer race) must never show one — a spinner above a later user
  // message reads as the agent answering out of order. Booleans are stable
  // across token flushes, so this selector adds no streaming re-renders.
  const isLastMessage = useAuiState(s => s.thread.messages[s.thread.messages.length - 1]?.id === s.message.id)

  // Preview targets only materialize once the turn completes — while running
  // the selector returns '' (stable), so per-token flushes skip the regex
  // scan and the re-render it would cause.
  const completedText = useAuiState(s =>
    s.message.status?.type === 'running' ? '' : messageContentText(s.message.content)
  )

  const previewTargets = useMemo(() => {
    if (!completedText || !/(https?:\/\/|file:\/\/)/i.test(completedText)) {
      return []
    }

    return pickPrimaryPreviewTarget(extractPreviewTargets(completedText))
  }, [completedText])

  const getMessageText = useCallback(() => messageContentText(messageRuntime.getState().content), [messageRuntime])

  // Cursor's changed-files card only appears once the turn settles: while the
  // agent is still editing, the tool rows narrate each patch and a card that
  // grew a row per write would thrash the transcript. `[]` while running keeps
  // this selector referentially stable across the 30 Hz delta stream.
  //
  // It also only rides the LAST turn. The card is a "here's what just landed"
  // summary, not a per-turn artifact: leaving one behind on every reply would
  // stack a wall of stale cards down the transcript. Sending the next message
  // retires it — the working tree it describes is already history by then.
  const settledParts = useAuiState(s => {
    const isLastMessage = s.thread.messages[s.thread.messages.length - 1]?.id === s.message.id

    return s.message.status?.type === 'running' || !isLastMessage ? EMPTY_PARTS : s.message.parts
  })

  const enterRef = useEnterAnimation(isRunning, `assistant-message:${messageId}`)

  // Double-click the reply to heart it (iMessage). Undefined while reactions
  // are off, so the root carries no listener at all.
  const onDoubleClick = useTapbackDoubleClick(messageId, 'assistant')

  // Reply inside an inter-agent exchange: render collapsed (Grok-bots
  // parity — the transcript shows the event; the text is one click away).
  // Never collapse while streaming: the user should see progress, and the
  // status selectors above stay live either way.
  if (interAgentSender && !isRunning) {
    return (
      <MessagePrimitive.Root
        className="group flex w-full min-w-0 max-w-full flex-col gap-0 self-start overflow-hidden pb-(--conversation-turn-gap)"
        data-role="assistant"
        data-slot="aui_assistant-message-root"
      >
        <div className="flex max-w-[min(86%,44rem)] flex-col gap-0.5 self-center px-2 py-0.5 text-[0.6875rem] leading-5 text-muted-foreground/60">
          <span className="flex items-center justify-center gap-1.5">
            <Codicon className="shrink-0 text-muted-foreground/55" name="arrow-small-right" size="0.8125rem" />
            <span className="wrap-anywhere">Replied to {interAgentSender}</span>
          </span>
          <details className="self-center">
            <summary className="cursor-pointer select-none text-center text-muted-foreground/45 hover:text-muted-foreground/70">
              show reply
            </summary>
            <div className="mt-1 max-w-[36rem] rounded-lg border border-(--ui-stroke-tertiary) px-3 py-2 text-left text-[0.75rem] leading-5 text-foreground/85">
              <MessagePrimitive.Parts components={MESSAGE_PARTS_COMPONENTS} />
            </div>
          </details>
        </div>
      </MessagePrimitive.Root>
    )
  }

  return (
    <MessagePrimitive.Root
      className="group flex w-full min-w-0 max-w-full flex-col gap-0 self-start overflow-hidden"
      data-role="assistant"
      data-slot="aui_assistant-message-root"
      data-streaming={isRunning ? 'true' : undefined}
      onDoubleClick={onDoubleClick}
      ref={enterRef}
    >
      <div
        className="wrap-anywhere min-w-0 max-w-full overflow-hidden text-pretty text-[length:var(--conversation-text-font-size)] leading-(--dt-line-height) text-foreground"
        data-slot="aui_assistant-message-content"
      >
        {/* Todos render in the composer status stack now, not inline. */}
        <MessagePrimitive.Parts components={MESSAGE_PARTS_COMPONENTS} />
        {isLastMessage && (isPlaceholder ? <ResponseLoadingIndicator /> : isRunning && <StreamStallIndicator />)}
        {previewTargets.length > 0 && (
          <div className="mt-3 flex flex-wrap gap-2">
            {previewTargets.map(target => (
              <PreviewAttachment key={target} source="explicit-link" target={target} />
            ))}
          </div>
        )}
        <MessagePrimitive.Error>
          <ErrorPrimitive.Root
            className="mt-1.5 flex items-start gap-1.5 text-[0.78rem] leading-5 text-[color-mix(in_srgb,var(--dt-destructive)_78%,var(--ui-text-secondary))]"
            role="alert"
          >
            <ErrorPrimitive.Message className="min-w-0 flex-1" />
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
          </ErrorPrimitive.Root>
        </MessagePrimitive.Error>
      </div>
      <MessageTimelineTimestamp className="px-(--message-text-indent) pt-0.5" suppressIfDuplicatePart />
      {hasVisibleText && !isInterim && (
        <AssistantFooter
          durationS={turnDurationS}
          getMessageText={getMessageText}
          messageId={messageId}
          onBranchInNewChat={onBranchInNewChat}
        />
      )}
      {/* Last thing in the turn — under the action bar, the way Cursor ends a
          turn on its summary rather than burying it above the controls. */}
      <ChangedFilesCard parts={settledParts} />
    </MessagePrimitive.Root>
  )
}

const AssistantActionBar: FC<MessageActionProps> = ({ messageId, getMessageText, onBranchInNewChat }) => {
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
    } catch (error) {
      notifyError(error, copy.readAloudFailed)
    }
  }, [copy.readAloudFailed, getText, messageId])

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

const AssistantFooter: FC<MessageActionProps & { durationS?: number }> = ({ durationS, ...props }) => {
  const { t } = useI18n()

  return (
  <div className="flex min-h-6 flex-col items-end gap-1 pr-(--message-text-indent) pl-(--message-text-indent)">
    {durationS !== undefined && (
      <span
        className="select-none px-0.5 text-[0.6875rem] leading-5 tabular-nums text-muted-foreground"
        data-slot="aui_turn-duration"
        title={t.assistant.thread.turnDuration(formatElapsed(durationS))}
      >
        ⏱ {formatElapsed(durationS)}
      </span>
    )}
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
    <AssistantActionBar {...props} />
  </div>
  )
}
