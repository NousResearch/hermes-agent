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

import { requestComposerFocus } from '@/app/chat/composer/focus'
import { MarkdownTextContent } from '@/components/assistant-ui/markdown-text'
import {
  contentHasVisibleText,
  messageContentText,
  pickPrimaryPreviewTarget
} from '@/components/assistant-ui/thread/content'
import { MESSAGE_PARTS_COMPONENTS } from '@/components/assistant-ui/thread/message-parts'
import { StreamStallIndicator } from '@/components/assistant-ui/thread/status'
import { formatMessageTimestamp } from '@/components/assistant-ui/thread/timestamp'
import { TooltipIconButton } from '@/components/assistant-ui/tooltip-icon-button'
import { PreviewAttachment } from '@/components/chat/preview-attachment'
import { Codicon } from '@/components/ui/codicon'
import { CopyButton } from '@/components/ui/copy-button'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuTrigger
} from '@/components/ui/dropdown-menu'
import { useI18n } from '@/i18n'
import { triggerHaptic } from '@/lib/haptics'
import { GitBranchIcon, Loader2Icon, Volume2Icon, VolumeXIcon, XIcon } from '@/lib/icons'
import { splitAssistantMessageIntoReplyChunks } from '@/lib/message-chunks'
import { extractPreviewTargets } from '@/lib/preview-targets'
import { useEnterAnimation } from '@/lib/use-enter-animation'
import { cn } from '@/lib/utils'
import { playSpeechText, stopVoicePlayback } from '@/lib/voice-playback'
import {
  $messageRepliesEnabled,
  $messageReplyTarget,
  messageRepliesEnabledForProfile,
  messageTextCanBeRepliedTo,
  startMessageReply
} from '@/store/message-replies'
import { notifyError } from '@/store/notifications'
import { $activeGatewayProfile } from '@/store/profile'
import { $voicePlayback } from '@/store/voice-playback'

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

  // PERF: this component must NOT subscribe to the streaming text. Every
  // selector here returns a value that stays referentially stable across
  // token flushes (booleans, status strings, '' while running), so the
  // 30 Hz delta stream only re-renders the markdown part and the tiny
  // StreamStallIndicator leaf — not the footer/preview/root subtree.
  const messageStatus = useAuiState(s => s.message.status?.type)
  const isRunning = messageStatus === 'running'
  const isPlaceholder = useAuiState(s => s.message.status?.type === 'running' && s.message.content.length === 0)
  const hasVisibleText = useAuiState(s => contentHasVisibleText(s.message.content))

  // Preview targets only materialize once the turn completes — while running
  // the selector returns '' (stable), so per-token flushes skip the regex
  // scan and the re-render it would cause.
  const completedText = useAuiState(s =>
    s.message.status?.type === 'running' ? '' : messageContentText(s.message.content)
  )

  const replyTarget = useStore($messageReplyTarget)
  const activeProfile = useStore($activeGatewayProfile)
  const repliesEnabled = useStore($messageRepliesEnabled) || messageRepliesEnabledForProfile(activeProfile)
  const replyChunks = useMemo(
    () => (!isRunning && repliesEnabled && completedText ? splitAssistantMessageIntoReplyChunks(completedText) : []),
    [completedText, isRunning, repliesEnabled]
  )
  const renderReplyChunks = replyChunks.length > 1
  const replyingToThisMessage =
    replyTarget?.messageId === messageId || Boolean(replyTarget?.messageId.startsWith(`${messageId}:chunk:`))

  const previewTargets = useMemo(() => {
    if (!completedText || !/(https?:\/\/|file:\/\/)/i.test(completedText)) {
      return []
    }

    return pickPrimaryPreviewTarget(extractPreviewTargets(completedText))
  }, [completedText])

  const getMessageText = useCallback(() => messageContentText(messageRuntime.getState().content), [messageRuntime])

  const enterRef = useEnterAnimation(isRunning, `assistant-message:${messageId}`)

  if (isPlaceholder) {
    return null
  }

  return (
    <MessagePrimitive.Root
      className={cn(
        'group flex w-full min-w-0 max-w-full flex-col gap-0 self-start overflow-hidden rounded-xl transition-[box-shadow,outline-color] duration-150',
        replyingToThisMessage &&
          'outline outline-1 outline-[color-mix(in_srgb,var(--dt-composer-ring)_70%,transparent)] shadow-[0_0_0_3px_color-mix(in_srgb,var(--dt-composer-ring)_12%,transparent)]'
      )}
      data-reply-target={replyingToThisMessage ? 'true' : undefined}
      data-role="assistant"
      data-slot="aui_assistant-message-root"
      data-streaming={isRunning ? 'true' : undefined}
      ref={enterRef}
    >
      <div
        className="wrap-anywhere min-w-0 max-w-full overflow-hidden text-pretty text-[length:var(--conversation-text-font-size)] leading-(--dt-line-height) text-foreground"
        data-slot="aui_assistant-message-content"
      >
        {/* Todos render in the composer status stack now, not inline. */}
        {renderReplyChunks ? (
          <div className="flex w-full min-w-0 flex-col gap-3" data-slot="aui_reply-chunks">
            {replyChunks.map(chunk => {
              const chunkMessageId = `${messageId}:chunk:${chunk.index}`
              const selected = replyTarget?.messageId === chunkMessageId

              return (
                <section
                  className={cn(
                    'min-w-0 rounded-xl border border-(--ui-stroke-tertiary) bg-[color-mix(in_srgb,var(--ui-editor-surface-background)_76%,transparent)] px-3 py-2.5 transition-[border-color,box-shadow]',
                    selected &&
                      'border-[color-mix(in_srgb,var(--dt-composer-ring)_58%,transparent)] shadow-[0_0_0_3px_color-mix(in_srgb,var(--dt-composer-ring)_10%,transparent)]'
                  )}
                  data-slot="aui_reply-chunk"
                  key={chunkMessageId}
                >
                  <MarkdownTextContent isRunning={false} text={chunk.text} />
                  <div className="mt-2 flex justify-end">
                    <button
                      className="inline-flex h-6 shrink-0 items-center gap-1 rounded-md border border-[color-mix(in_srgb,var(--dt-composer-ring)_30%,transparent)] bg-[color-mix(in_srgb,var(--dt-composer-ring)_8%,transparent)] px-2 text-[0.7rem] font-medium text-[color-mix(in_srgb,var(--dt-composer-ring)_82%,var(--foreground))] transition-colors hover:bg-[color-mix(in_srgb,var(--dt-composer-ring)_15%,transparent)]"
                      onClick={() => {
                        startMessageReply({ messageId: chunkMessageId, quote: chunk.text })
                        triggerHaptic('selection')
                        requestComposerFocus('main')
                      }}
                      title="Reply to this part"
                      type="button"
                    >
                      <Codicon name="reply" size="0.75rem" />
                      Reply
                    </button>
                  </div>
                </section>
              )
            })}
          </div>
        ) : (
          <MessagePrimitive.Parts components={MESSAGE_PARTS_COMPONENTS} />
        )}
        {isRunning && <StreamStallIndicator />}
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
      {hasVisibleText && !renderReplyChunks && (
        <AssistantFooter getMessageText={getMessageText} messageId={messageId} onBranchInNewChat={onBranchInNewChat} />
      )}
    </MessagePrimitive.Root>
  )
}

const AssistantActionBar: FC<MessageActionProps> = ({ messageId, getMessageText, onBranchInNewChat }) => {
  const { t } = useI18n()
  const copy = t.assistant.thread
  const [menuOpen, setMenuOpen] = useState(false)
  const activeProfile = useStore($activeGatewayProfile)
  const repliesEnabled = useStore($messageRepliesEnabled) || messageRepliesEnabledForProfile(activeProfile)

  const replyToMessage = useCallback(() => {
    const text = getMessageText()

    if (!text.trim()) {
      return
    }

    startMessageReply({ messageId, quote: text })
    triggerHaptic('selection')
    requestComposerFocus('main')
  }, [getMessageText, messageId])

  const showReply = repliesEnabled && messageTextCanBeRepliedTo(getMessageText())

  return (
    <div className="relative flex w-full shrink-0 items-center justify-end gap-2">
      {showReply && (
        <button
          className="inline-flex h-6 shrink-0 items-center gap-1 rounded-md border border-[color-mix(in_srgb,var(--dt-composer-ring)_30%,transparent)] bg-[color-mix(in_srgb,var(--dt-composer-ring)_8%,transparent)] px-2 text-[0.7rem] font-medium text-[color-mix(in_srgb,var(--dt-composer-ring)_82%,var(--foreground))] transition-colors hover:bg-[color-mix(in_srgb,var(--dt-composer-ring)_15%,transparent)]"
          onClick={replyToMessage}
          title="Reply to this message"
          type="button"
        >
          <Codicon name="reply" size="0.75rem" />
          Reply
        </button>
      )}
      <ActionBarPrimitive.Root
        className={cn(
          // NOTE: intentionally NOT `hideWhenRunning`. That prop unmounts the
          // bar while the thread streams, which collapses every completed
          // assistant message's footer by this bar's height and shifts the
          // whole conversation when the turn resolves. The bar is already
          // invisible by default (opacity-0 + pointer-events-none, reveals on
          // hover), so keeping it mounted reserves stable layout height with
          // no visual change during streaming.
          'relative flex flex-row items-center justify-end gap-2 py-1.5 opacity-0 pointer-events-none group-hover:pointer-events-auto group-hover:opacity-100 focus-within:pointer-events-auto focus-within:opacity-100',
          menuOpen && 'pointer-events-auto opacity-100 [&_button]:opacity-100'
        )}
        data-slot="aui_msg-actions"
      >
        <CopyButton appearance="icon" buttonSize="icon" label={copy.copy} text={getMessageText} />
        <ActionBarPrimitive.Reload asChild>
          <TooltipIconButton onClick={() => triggerHaptic('submit')} tooltip={copy.refresh}>
            <Codicon name="refresh" />
          </TooltipIconButton>
        </ActionBarPrimitive.Reload>
        <DropdownMenu onOpenChange={setMenuOpen} open={menuOpen}>
          <DropdownMenuTrigger asChild>
            <TooltipIconButton tooltip={copy.moreActions}>
              <Codicon name="ellipsis" />
            </TooltipIconButton>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="start" onCloseAutoFocus={e => e.preventDefault()} sideOffset={6}>
            <MessageTimestamp />
            <DropdownMenuItem onSelect={() => onBranchInNewChat?.(messageId)}>
              <GitBranchIcon />
              {copy.branchNewChat}
            </DropdownMenuItem>
            <ReadAloudItem getText={getMessageText} messageId={messageId} />
          </DropdownMenuContent>
        </DropdownMenu>
      </ActionBarPrimitive.Root>
    </div>
  )
}

const ReadAloudItem: FC<{ getText: () => string; messageId: string }> = ({ getText, messageId }) => {
  const { t } = useI18n()
  const copy = t.assistant.thread
  const voicePlayback = useStore($voicePlayback)

  const readAloudStatus =
    voicePlayback.source === 'read-aloud' && voicePlayback.messageId === messageId ? voicePlayback.status : 'idle'

  const isPreparing = readAloudStatus === 'preparing'
  const isSpeaking = readAloudStatus === 'speaking'
  const anyPlaybackActive = voicePlayback.status !== 'idle'
  const Icon = isPreparing ? Loader2Icon : isSpeaking ? VolumeXIcon : Volume2Icon

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
    <DropdownMenuItem
      disabled={isPreparing || (!isSpeaking && anyPlaybackActive)}
      onSelect={e => {
        e.preventDefault()
        void (isSpeaking ? stopVoicePlayback() : read())
      }}
    >
      <Icon className={isPreparing ? 'animate-spin' : undefined} />
      {isPreparing ? copy.preparingAudio : isSpeaking ? copy.stopReading : copy.readAloud}
    </DropdownMenuItem>
  )
}

const MessageTimestamp: FC = () => {
  const { t } = useI18n()
  const createdAt = useAuiState(s => s.message.createdAt)
  const label = formatMessageTimestamp(createdAt, t.assistant.thread)

  if (!label) {
    return null
  }

  return <DropdownMenuLabel className="text-xs font-normal text-muted-foreground">{label}</DropdownMenuLabel>
}

const AssistantFooter: FC<MessageActionProps> = props => (
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
    <AssistantActionBar {...props} />
  </div>
)
