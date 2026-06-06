import type { Unstable_TriggerAdapter, Unstable_TriggerItem } from '@assistant-ui/core'
import {
  ActionBarPrimitive,
  BranchPickerPrimitive,
  ComposerPrimitive,
  ErrorPrimitive,
  MessagePrimitive,
  type ToolCallMessagePartProps,
  useAui,
  useAuiState
} from '@assistant-ui/react'
import { useStore } from '@nanostores/react'
import { IconPlayerStopFilled } from '@tabler/icons-react'
import {
  type ClipboardEvent,
  type ComponentProps,
  type FC,
  type FocusEvent,
  type FormEvent,
  type KeyboardEvent,
  type DragEvent as ReactDragEvent,
  type ReactNode,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState
} from 'react'

import { COMPOSER_DROP_ACTIVE_CLASS, COMPOSER_DROP_FADE_CLASS } from '@/app/chat/composer/drop-affordance'
import {
  type ComposerInsertMode,
  focusComposerInput,
  markActiveComposer,
  onComposerFocusRequest,
  onComposerInsertRequest
} from '@/app/chat/composer/focus'
import { useAtCompletions } from '@/app/chat/composer/hooks/use-at-completions'
import { useSlashCompletions } from '@/app/chat/composer/hooks/use-slash-completions'
import { dragHasAttachments, droppedFileInlineRef, insertInlineRefsIntoEditor } from '@/app/chat/composer/inline-refs'
import {
  composerPlainText,
  placeCaretEnd,
  refChipElement,
  renderComposerContents,
  RICH_INPUT_SLOT
} from '@/app/chat/composer/rich-editor'
import { detectTrigger, textBeforeCaret, type TriggerState } from '@/app/chat/composer/text-utils'
import { ComposerTriggerPopover } from '@/app/chat/composer/trigger-popover'
import { extractDroppedFiles, HERMES_PATHS_MIME } from '@/app/chat/hooks/use-composer-actions'
import { ClarifyTool } from '@/components/assistant-ui/clarify-tool'
import { DirectiveContent, hermesDirectiveFormatter } from '@/components/assistant-ui/directive-text'
import { MarkdownText, MarkdownTextContent } from '@/components/assistant-ui/markdown-text'
import { VirtualizedThread } from '@/components/assistant-ui/thread-virtualizer'
import { HoistedTodoPanel, todosFromMessageContent } from '@/components/assistant-ui/todo-tool'
import { ToolFallback, ToolGroupSlot } from '@/components/assistant-ui/tool-fallback'
import { TooltipIconButton } from '@/components/assistant-ui/tooltip-icon-button'
import { UserMessageText } from '@/components/assistant-ui/user-message-text'
import { useElapsedSeconds } from '@/components/chat/activity-timer'
import { ActivityTimerText } from '@/components/chat/activity-timer-text'
import { DisclosureRow } from '@/components/chat/disclosure-row'
import { GeneratedImageProvider, useGeneratedImageContext } from '@/components/chat/generated-image-context'
import { ImageGenerationPlaceholder } from '@/components/chat/image-generation-placeholder'
import { Intro, type IntroProps } from '@/components/chat/intro'
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
import { Loader } from '@/components/ui/loader'
import type { HermesGateway } from '@/hermes'
import { useResizeObserver } from '@/hooks/use-resize-observer'
import { DATA_IMAGE_URL_RE } from '@/lib/embedded-images'
import { triggerHaptic } from '@/lib/haptics'
import { GitBranchIcon, Loader2Icon, Volume2Icon, VolumeXIcon } from '@/lib/icons'
import { extractPreviewTargets } from '@/lib/preview-targets'
import { useEnterAnimation } from '@/lib/use-enter-animation'
import { cn } from '@/lib/utils'
import { playSpeechText, stopVoicePlayback } from '@/lib/voice-playback'
import { notifyError } from '@/store/notifications'
import { $voicePlayback } from '@/store/voice-playback'

type ThreadLoadingState = 'response' | 'session'

interface MessageActionProps {
  messageId: string
  messageText: string
  onBranchInNewChat?: (messageId: string) => void
}

let readAloudAudio: HTMLAudioElement | null = null

function partText(part: unknown): string {
  if (typeof part === 'string') {
    return part
  }

  if (!part || typeof part !== 'object') {
    return ''
  }

  const row = part as { text?: unknown; type?: unknown }

  return (!row.type || row.type === 'text') && typeof row.text === 'string' ? row.text : ''
}

function messageContentText(content: unknown): string {
  if (typeof content === 'string') {
    return content.trim()
  }

  return Array.isArray(content) ? content.map(partText).join('').trim() : ''
}

export const Thread: FC<{
  clampToComposer?: boolean
  cwd?: string | null
  gateway?: HermesGateway | null
  intro?: IntroProps
  loading?: ThreadLoadingState
  onBranchInNewChat?: (messageId: string) => void
  onCancel?: () => Promise<void> | void
  sessionId?: string | null
  sessionKey?: string | null
}> = ({
  clampToComposer = false,
  cwd = null,
  gateway = null,
  intro,
  loading,
  onBranchInNewChat,
  onCancel,
  sessionId = null,
  sessionKey
}) => {
  const messageComponents = useMemo(
    () => ({
      AssistantMessage: () => <AssistantMessage onBranchInNewChat={onBranchInNewChat} />,
      SystemMessage,
      UserEditComposer: () => <UserEditComposer cwd={cwd} gateway={gateway} sessionId={sessionId} />,
      UserMessage: () => <UserMessage onCancel={onCancel} />
    }),
    [cwd, gateway, onBranchInNewChat, onCancel, sessionId]
  )

  const emptyPlaceholder = intro ? (
    <div
      className="flex min-h-0 w-full flex-col items-center justify-center"
      style={{ paddingBottom: 'var(--composer-measured-height)' }}
    >
      <Intro {...intro} />
    </div>
  ) : undefined

  return (
    <GeneratedImageProvider>
      <div className="relative grid h-full min-h-0 max-w-full grid-rows-[minmax(0,1fr)] overflow-hidden bg-transparent contain-[layout_paint]">
        <VirtualizedThread
          clampToComposer={clampToComposer}
          components={messageComponents}
          emptyPlaceholder={emptyPlaceholder}
          loadingIndicator={loading === 'response' ? <ResponseLoadingIndicator /> : null}
          sessionKey={sessionKey}
        />
        {loading === 'session' && <CenteredThreadSpinner />}
      </div>
    </GeneratedImageProvider>
  )
}

function pickPrimaryPreviewTarget(targets: string[]): string[] {
  if (targets.length <= 1) {
    return targets
  }

  const localUrl = targets.find(value => /^https?:\/\/(?:localhost|127\.0\.0\.1|0\.0\.0\.0|\[::1\])/i.test(value))

  return [localUrl || targets[targets.length - 1]]
}

const CenteredThreadSpinner: FC = () => (
  <div
    aria-label="Loading session"
    className="pointer-events-none absolute inset-0 z-1 grid place-items-center"
    role="status"
  >
    <Loader
      aria-hidden="true"
      className="size-12 text-midground/70"
      pathSteps={220}
      role="presentation"
      strokeScale={0.72}
      type="rose-curve"
    />
  </div>
)

const AssistantMessage: FC<{ onBranchInNewChat?: (messageId: string) => void }> = ({ onBranchInNewChat }) => {
  const messageId = useAuiState(s => s.message.id)
  const content = useAuiState(s => s.message.content)
  const messageText = messageContentText(content)
  const hoistedTodos = useMemo(() => todosFromMessageContent(content), [content])

  const previewTargets = useMemo(() => {
    if (!messageText || !/(https?:\/\/|file:\/\/)/i.test(messageText)) {
      return []
    }

    return pickPrimaryPreviewTarget(extractPreviewTargets(messageText))
  }, [messageText])

  const messageStatus = useAuiState(s => s.message.status?.type)
  const isPlaceholder = messageStatus === 'running' && content.length === 0
  const enterRef = useEnterAnimation(messageStatus === 'running', `assistant-message:${messageId}`)

  if (isPlaceholder) {
    return null
  }

  return (
    <MessagePrimitive.Root
      className="group flex w-full min-w-0 max-w-full flex-col gap-0 self-start overflow-hidden"
      data-role="assistant"
      data-slot="aui_assistant-message-root"
      data-streaming={messageStatus === 'running' ? 'true' : undefined}
      ref={enterRef}
    >
      <div
        className="wrap-anywhere min-w-0 max-w-full overflow-hidden text-pretty text-[length:var(--conversation-text-font-size)] leading-(--dt-line-height) text-foreground"
        data-slot="aui_assistant-message-content"
      >
        {hoistedTodos.length > 0 && <HoistedTodoPanel todos={hoistedTodos} />}
        <MessagePrimitive.Parts components={MESSAGE_PARTS_COMPONENTS} />
        {previewTargets.length > 0 && (
          <div className="mt-3 flex flex-wrap gap-2">
            {previewTargets.map(target => (
              <PreviewAttachment key={target} source="explicit-link" target={target} />
            ))}
          </div>
        )}
        <MessagePrimitive.Error>
          <ErrorPrimitive.Root
            className="mt-1.5 text-[0.78rem] leading-5 text-[color-mix(in_srgb,var(--dt-destructive)_78%,var(--ui-text-secondary))]"
            role="alert"
          >
            <ErrorPrimitive.Message />
          </ErrorPrimitive.Root>
        </MessagePrimitive.Error>
      </div>
      {messageText.trim().length > 0 && (
        <AssistantFooter messageId={messageId} messageText={messageText} onBranchInNewChat={onBranchInNewChat} />
      )}
    </MessagePrimitive.Root>
  )
}

const StatusRow: FC<{ children: ReactNode; label: string } & React.ComponentPropsWithoutRef<'div'>> = ({
  children,
  label,
  className,
  ...rest
}) => (
  <div
    aria-label={label}
    aria-live="polite"
    className={cn('flex max-w-full items-center gap-2 self-start text-sm text-muted-foreground/70', className)}
    role="status"
    {...rest}
  >
    {children}
  </div>
)

const ResponseLoadingIndicator: FC = () => {
  const elapsed = useElapsedSeconds()

  return (
    <StatusRow data-slot="aui_response-loading" label="Hermes is loading a response">
      <span aria-hidden="true" className="dither inline-block size-3 rounded-[2px] text-midground/80 animate-pulse" />
      <ActivityTimerText seconds={elapsed} />
    </StatusRow>
  )
}

const ImageGenerateTool: FC<ToolCallMessagePartProps> = ({ result }) => {
  const generatedImage = useGeneratedImageContext()
  const running = result === undefined

  useEffect(() => {
    generatedImage?.setPending(running)
  }, [generatedImage, running])

  if (!running) {
    return null
  }

  return (
    <div className="mt-1.5">
      <ImageGenerationPlaceholder />
    </div>
  )
}

const ChainToolFallback: FC<ToolCallMessagePartProps> = props => {
  // todo parts are hoisted to a dedicated panel above the message content.
  if (props.toolName === 'todo') {
    return null
  }

  if (props.toolName === 'image_generate') {
    return <ImageGenerateTool {...props} />
  }

  if (props.toolName === 'clarify') {
    return <ClarifyTool {...props} />
  }

  return <ToolFallback {...props} />
}

const ThinkingDisclosure: FC<{
  children: ReactNode
  messageRunning?: boolean
  pending?: boolean
  timerKey?: string
}> = ({ children, messageRunning = false, pending = false, timerKey }) => {
  // `null` = no explicit user toggle yet, defer to the streaming default.
  // The default is "auto-open while streaming, auto-collapse when done" so
  // reasoning surfaces a live preview without manual interaction. The first
  // explicit toggle wins from then on.
  const [userOpen, setUserOpen] = useState<boolean | null>(null)
  const elapsed = useElapsedSeconds(pending, timerKey)
  const scrollRef = useRef<HTMLDivElement | null>(null)
  const contentRef = useRef<HTMLDivElement | null>(null)
  const enterRef = useEnterAnimation(messageRunning, timerKey)

  const open = userOpen ?? pending
  const isPreview = pending && userOpen === null

  // While the preview is live, pin the scroll container to the bottom on
  // every content growth so the latest tokens are always visible. Combined
  // with the top mask in styles.css, this reads as text settling in from
  // below while older lines fade out at the top.
  useEffect(() => {
    if (!isPreview) {
      return
    }

    const el = scrollRef.current
    const content = contentRef.current

    if (!el || !content) {
      return
    }

    const pin = () => {
      el.scrollTop = el.scrollHeight
    }

    pin()
    const observer = new ResizeObserver(pin)
    observer.observe(content)

    return () => observer.disconnect()
    // Re-run when the disclosure toggles so the observer attaches to the new
    // DOM after expand/collapse (refs are conditionally rendered on `open`).
  }, [isPreview, open])

  return (
    <div
      className="text-[length:var(--conversation-tool-font-size)] text-(--ui-text-tertiary)"
      data-slot="aui_thinking-disclosure"
      ref={enterRef}
    >
      <DisclosureRow onToggle={() => setUserOpen(!open)} open={open}>
        <span className="flex min-w-0 items-baseline gap-1.5">
          <span
            className={cn(
              'text-[length:var(--conversation-tool-font-size)] font-medium leading-(--conversation-line-height) text-(--ui-text-secondary)',
              pending && 'shimmer text-foreground/55'
            )}
          >
            Thinking
          </span>
          {pending && (
            <ActivityTimerText
              className="text-[length:var(--conversation-caption-font-size)] tabular-nums text-(--ui-text-tertiary)"
              seconds={elapsed}
            />
          )}
        </span>
      </DisclosureRow>
      {open && (
        <div
          className={cn(
            // Body sits flush with the "Thinking" header — no left indent —
            // and inherits the disclosure-level opacity fade defined in
            // styles.css (~0.67 at rest, 1 on hover/focus).
            'mt-0.5 w-full min-w-0 max-w-full overflow-hidden wrap-anywhere pb-1',
            isPreview && 'thinking-preview max-h-40'
          )}
          ref={scrollRef}
        >
          <div ref={contentRef}>{children}</div>
        </div>
      )}
    </div>
  )
}

// Self-gate "Thinking…" on this message's own reasoning parts. Reading
// `thread.isRunning` directly would flicker shimmer/timer on every old
// assistant whenever the external-store runtime clears+reimports its
// repository (one ref-identity bump per streaming delta).
const ReasoningAccordionGroup: FC<{ children?: ReactNode; endIndex: number; startIndex: number }> = ({
  children,
  endIndex,
  startIndex
}) => {
  const messageId = useAuiState(s => s.message.id)
  const messageRunning = useAuiState(s => s.message.status?.type === 'running')

  const pending = useAuiState(
    s =>
      s.thread.isRunning &&
      s.message.status?.type === 'running' &&
      s.message.parts
        .slice(Math.max(0, startIndex))
        .some(p => p?.type === 'reasoning' && p.status?.type !== 'complete')
  )

  return (
    <ThinkingDisclosure messageRunning={messageRunning} pending={pending} timerKey={`reasoning:${messageId}`}>
      {children}
    </ThinkingDisclosure>
  )
}

const ReasoningTextPart: FC<{ text: string; status?: { type: string } }> = ({ text, status }) => {
  const displayText = text.trimStart()
  const messageRunning = useAuiState(s => s.message.status?.type === 'running')
  const isRunning = status?.type === 'running' || messageRunning

  return (
    <MarkdownTextContent
      containerClassName={cn(
        'text-xs leading-relaxed text-muted-foreground/85',
        isRunning && 'shimmer text-muted-foreground/55'
      )}
      containerProps={{ 'data-slot': 'aui_reasoning-text' } as ComponentProps<'div'>}
      isRunning={isRunning}
      text={displayText}
    />
  )
}

// Module-level constant so the `components` prop on `MessagePrimitive.Parts`
// has a stable identity across renders. Without this every AssistantMessage
// render would create a fresh `components` object, invalidating the memo on
// `MessagePrimitivePartByIndex` and forcing every tool/reasoning child to
// re-render on every streaming delta. Memo invalidation alone doesn't
// remount, but combined with the previous ToolFallback group-swap it was a
// big chunk of the per-delta work.
const MESSAGE_PARTS_COMPONENTS = {
  Reasoning: ReasoningTextPart,
  ReasoningGroup: ReasoningAccordionGroup,
  Text: MarkdownText,
  ToolGroup: ToolGroupSlot,
  tools: { Fallback: ChainToolFallback }
} as const

const TIME_FMT = new Intl.DateTimeFormat(undefined, { hour: 'numeric', minute: '2-digit' })

const SHORT_FMT = new Intl.DateTimeFormat(undefined, {
  day: 'numeric',
  hour: 'numeric',
  minute: '2-digit',
  month: 'short'
})

function startOfDay(d: Date): number {
  return new Date(d.getFullYear(), d.getMonth(), d.getDate()).getTime()
}

function formatMessageTimestamp(value: Date | string | number | undefined): string {
  if (!value) {
    return ''
  }

  const date = value instanceof Date ? value : new Date(value)

  if (Number.isNaN(date.getTime())) {
    return ''
  }

  const dayDelta = Math.round((startOfDay(new Date()) - startOfDay(date)) / 86_400_000)

  if (dayDelta === 0) {
    return `Today, ${TIME_FMT.format(date)}`
  }

  if (dayDelta === 1) {
    return `Yesterday, ${TIME_FMT.format(date)}`
  }

  return SHORT_FMT.format(date)
}

const AssistantActionBar: FC<MessageActionProps> = ({ messageId, messageText, onBranchInNewChat }) => {
  const [menuOpen, setMenuOpen] = useState(false)

  return (
    <div className="relative flex w-full shrink-0 justify-end">
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
        <CopyButton appearance="icon" buttonSize="icon" disabled={!messageText} label="Copy" text={messageText} />
        <ActionBarPrimitive.Reload asChild>
          <TooltipIconButton onClick={() => triggerHaptic('submit')} tooltip="Refresh">
            <Codicon name="refresh" />
          </TooltipIconButton>
        </ActionBarPrimitive.Reload>
        <DropdownMenu onOpenChange={setMenuOpen} open={menuOpen}>
          <DropdownMenuTrigger asChild>
            <TooltipIconButton tooltip="More actions">
              <Codicon name="ellipsis" />
            </TooltipIconButton>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="start" onCloseAutoFocus={e => e.preventDefault()} sideOffset={6}>
            <MessageTimestamp />
            <DropdownMenuItem onSelect={() => onBranchInNewChat?.(messageId)}>
              <GitBranchIcon />
              Branch in new chat
            </DropdownMenuItem>
            <ReadAloudItem messageId={messageId} text={messageText} />
          </DropdownMenuContent>
        </DropdownMenu>
      </ActionBarPrimitive.Root>
    </div>
  )
}

const ReadAloudItem: FC<{ messageId: string; text: string }> = ({ messageId, text }) => {
  const voicePlayback = useStore($voicePlayback)

  const readAloudStatus =
    voicePlayback.source === 'read-aloud' && voicePlayback.messageId === messageId ? voicePlayback.status : 'idle'

  const isPreparing = readAloudStatus === 'preparing'
  const isSpeaking = readAloudStatus === 'speaking'
  const anyPlaybackActive = voicePlayback.status !== 'idle'
  const Icon = isPreparing ? Loader2Icon : isSpeaking ? VolumeXIcon : Volume2Icon

  const read = useCallback(async () => {
    if (!text || $voicePlayback.get().status !== 'idle') {
      return
    }

    try {
      await playSpeechText(text, { messageId, source: 'read-aloud' })
    } catch (error) {
      notifyError(error, 'Read aloud failed')
    }
  }, [messageId, text])

  return (
    <DropdownMenuItem
      disabled={isPreparing || (!isSpeaking && (anyPlaybackActive || !text))}
      onSelect={e => {
        e.preventDefault()
        void (isSpeaking ? stopVoicePlayback() : read())
      }}
    >
      <Icon className={isPreparing ? 'animate-spin' : undefined} />
      {isPreparing ? 'Preparing audio...' : isSpeaking ? 'Stop reading' : 'Read aloud'}
    </DropdownMenuItem>
  )
}

const MessageTimestamp: FC = () => {
  const createdAt = useAuiState(s => s.message.createdAt)
  const label = formatMessageTimestamp(createdAt)

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

const EMPTY_ATTACHMENT_REFS: string[] = []

function messageAttachmentRefs(value: unknown): string[] {
  if (!Array.isArray(value)) {
    return EMPTY_ATTACHMENT_REFS
  }

  return value.every(ref => typeof ref === 'string') ? value : EMPTY_ATTACHMENT_REFS
}

function StickyHumanMessageContainer({ children }: { children: ReactNode }) {
  return (
    <div
      className="group/user-message sticky z-40 -mx-4 flex w-[calc(100%+2rem)] min-w-0 max-w-none flex-col items-stretch gap-0 self-end overflow-visible bg-(--ui-chat-surface-background) px-4 pb-(--conversation-turn-gap) pt-2"
      data-role="user"
      data-slot="aui_user-message-root"
    >
      {children}
    </div>
  )
}

// Shared "user bubble" base. Both the read-only message and the inline
// edit composer render the same bubble surface (rounded glass card,
// shadow-composer); they only differ in border weight, cursor, and
// padding-right (the read-only view reserves room for the restore icon).
const USER_BUBBLE_BASE_CLASS =
  'composer-human-message standalone-glass relative flex w-full min-w-0 max-w-full flex-col gap-1.5 overflow-hidden rounded-xl border bg-(--dt-user-bubble) px-3 py-2 text-left shadow-composer'

const USER_ACTION_ICON_BUTTON_CLASS =
  'grid place-items-center rounded-md bg-transparent text-(--ui-text-secondary) transition-colors hover:bg-(--ui-control-active-background) hover:text-foreground disabled:cursor-default disabled:text-(--ui-text-quaternary) disabled:opacity-70'

const USER_ACTION_ICON_SIZE = '0.6875rem'
const StopGlyph = <IconPlayerStopFilled aria-hidden className="size-3.5 -translate-y-px" />

const UserMessage: FC<{
  onCancel?: () => Promise<void> | void
}> = ({ onCancel }) => {
  const messageId = useAuiState(s => s.message.id)
  const content = useAuiState(s => s.message.content)
  const messageText = messageContentText(content)
  const threadRunning = useAuiState(s => s.thread.isRunning)

  const latestUserId = useAuiState(s => {
    for (let i = s.thread.messages.length - 1; i >= 0; i--) {
      const message = s.thread.messages[i] as { id?: string; role?: string }

      if (message.role === 'user') {
        return message.id ?? null
      }
    }

    return null
  })

  const attachmentRefs = useAuiState(s => {
    const custom = (s.message.metadata?.custom ?? {}) as { attachmentRefs?: unknown }

    return messageAttachmentRefs(custom.attachmentRefs)
  })

  // Sticky human bubbles clamp to ~2 lines with a soft fade so a long prompt
  // doesn't dominate the viewport while the response streams underneath; the
  // clamp lifts on hover / focus (see styles.css). We measure the *unclamped*
  // inner wrapper so the ResizeObserver only fires on real content / width
  // changes, not on every frame while the outer max-height animates open.
  const clampInnerRef = useRef<HTMLDivElement | null>(null)
  const [bodyClamped, setBodyClamped] = useState(false)

  const measureClamp = useCallback(() => {
    const inner = clampInnerRef.current
    const outer = inner?.parentElement

    if (!inner || !outer) {
      return
    }

    const styles = getComputedStyle(inner)
    const lineHeight = parseFloat(styles.lineHeight) || 1.5 * parseFloat(styles.fontSize) || 20
    const fullHeight = inner.scrollHeight

    outer.style.setProperty('--human-msg-full', `${fullHeight}px`)
    setBodyClamped(fullHeight > lineHeight * 2 + 1)
  }, [])

  useResizeObserver(measureClamp, clampInnerRef)

  const hasBody = messageText.trim().length > 0
  const isLatestUser = messageId === latestUserId
  const showStop = isLatestUser && threadRunning && Boolean(onCancel)
  const showRestore = !isLatestUser && !threadRunning

  const bubbleClassName = cn(
    USER_BUBBLE_BASE_CLASS,
    'border-(--ui-stroke-tertiary) pr-9 text-[length:var(--conversation-text-font-size)] leading-(--dt-line-height) text-foreground/95 transition-colors',
    !threadRunning && 'cursor-pointer hover:border-(--ui-stroke-secondary)'
  )

  const bubbleContent = (
    <>
      {attachmentRefs.length > 0 && (
        <span className="-mx-1 flex flex-wrap gap-1 border-b border-border/45 pb-1.5">
          <DirectiveContent text={attachmentRefs.join(' ')} />
        </span>
      )}
      {hasBody && (
        // Render the user's text through a minimal markdown pipeline:
        // backtick `code` and ``` fenced ``` blocks, with directive chips
        // (`@file:` etc.) still resolved inside the plain-text spans.
        <div className="sticky-human-clamp" data-clamped={bodyClamped ? 'true' : undefined}>
          <div ref={clampInnerRef}>
            <UserMessageText className="wrap-anywhere" text={messageText} />
          </div>
        </div>
      )}
    </>
  )

  return (
    <MessagePrimitive.Root asChild>
      <StickyHumanMessageContainer>
        <ActionBarPrimitive.Root className="relative w-full max-w-full" data-slot="aui_user-bubble-actions">
          <div className="human-message-with-todos-wrapper flex w-full flex-col gap-0">
            <div className="relative w-full">
              {threadRunning ? (
                <div className={bubbleClassName}>{bubbleContent}</div>
              ) : (
                <ActionBarPrimitive.Edit asChild>
                  <button
                    aria-label="Edit message"
                    className={bubbleClassName}
                    onClick={() => triggerHaptic('selection')}
                    title="Edit message"
                    type="button"
                  >
                    {bubbleContent}
                  </button>
                </ActionBarPrimitive.Edit>
              )}
              {(showStop || showRestore) && (
                <div className="pointer-events-none absolute right-2 bottom-2 z-10 flex items-center justify-center opacity-0 transition-opacity group-hover/user-message:opacity-100 group-focus-within/user-message:opacity-100">
                  {showStop ? (
                    <button
                      aria-label="Stop"
                      className={cn('pointer-events-auto size-5', USER_ACTION_ICON_BUTTON_CLASS)}
                      onClick={event => {
                        event.preventDefault()
                        event.stopPropagation()
                        void onCancel?.()
                      }}
                      title="Stop"
                      type="button"
                    >
                      {StopGlyph}
                    </button>
                  ) : (
                    <span
                      aria-hidden="true"
                      className="flex size-6 items-center justify-center rounded-md text-(--ui-text-tertiary)"
                      title="Editable checkpoint"
                    >
                      <Codicon name="discard" size="0.875rem" />
                    </span>
                  )}
                </div>
              )}
            </div>
            <BranchPickerPrimitive.Root
              className="checkpoint-container flex items-center gap-1 pb-0 pt-1 pl-1.5 text-[0.75rem] leading-none text-(--ui-text-tertiary)"
              hideWhenSingleBranch
            >
              <span aria-hidden className="checkpoint-icon size-1.5 rounded-full border border-current" />
              <BranchPickerPrimitive.Previous
                className="checkpoint-restore-text rounded-sm bg-transparent px-1 opacity-65 hover:opacity-100 disabled:hidden disabled:cursor-default"
                title="Restore previous checkpoint"
              >
                Restore checkpoint
              </BranchPickerPrimitive.Previous>
              <span className="checkpoint-divider opacity-55">
                <BranchPickerPrimitive.Number />/<BranchPickerPrimitive.Count />
              </span>
              <BranchPickerPrimitive.Next
                className="checkpoint-restore-text rounded-sm bg-transparent px-1 opacity-65 hover:opacity-100 disabled:hidden disabled:cursor-default"
                title="Restore next checkpoint"
              >
                Go forward
              </BranchPickerPrimitive.Next>
            </BranchPickerPrimitive.Root>
          </div>
        </ActionBarPrimitive.Root>
      </StickyHumanMessageContainer>
    </MessagePrimitive.Root>
  )
}

const SLASH_STATUS_RE = /^slash:(?<command>\/[^\n]+)\n(?<output>[\s\S]*)$/

const SystemMessage: FC = () => {
  const text = useAuiState(s => messageContentText(s.message.content))

  if (!text) {
    return null
  }

  const slashStatus = text.match(SLASH_STATUS_RE)

  if (slashStatus?.groups) {
    return (
      <MessagePrimitive.Root
        className="max-w-[min(86%,44rem)] self-center px-2 py-0.5 text-center text-[0.6875rem] leading-5 text-muted-foreground/60"
        data-role="system"
        data-slot="aui_system-message-root"
      >
        <span className="font-mono text-muted-foreground/55">{slashStatus.groups.command}</span>
        <span className="mx-1.5 text-muted-foreground/35">·</span>
        <span className="whitespace-pre-wrap">{slashStatus.groups.output.trim()}</span>
      </MessagePrimitive.Root>
    )
  }

  return (
    <MessagePrimitive.Root
      className="max-w-[min(86%,44rem)] self-center px-2 py-0.5 text-center text-[0.6875rem] leading-5 text-muted-foreground/55"
      data-role="system"
      data-slot="aui_system-message-root"
    >
      <span className="whitespace-pre-wrap">{text}</span>
    </MessagePrimitive.Root>
  )
}

interface UserEditComposerProps {
  cwd: string | null
  gateway: HermesGateway | null
  sessionId: string | null
}

const UserEditComposer: FC<UserEditComposerProps> = ({ cwd, gateway, sessionId }) => {
  const aui = useAui()
  const draft = useAuiState(s => s.composer.text)
  const rootRef = useRef<HTMLDivElement | null>(null)
  const editorRef = useRef<HTMLDivElement | null>(null)
  const draftRef = useRef(draft)
  const dragDepthRef = useRef(0)
  const [dragActive, setDragActive] = useState(false)
  const [trigger, setTrigger] = useState<TriggerState | null>(null)
  const [triggerActive, setTriggerActive] = useState(0)
  const [triggerItems, setTriggerItems] = useState<readonly Unstable_TriggerItem[]>([])
  // See index.tsx: set in keydown when the open popover consumes a nav/control
  // key so the matching keyup skips refreshTrigger (timing-immune vs reading
  // `trigger`, which keyup sees as already-null after Escape).
  const triggerKeyConsumedRef = useRef(false)
  const [triggerPlacement, setTriggerPlacement] = useState<'bottom' | 'top'>('top')
  const [focusRequestId, setFocusRequestId] = useState(0)
  const [submitting, setSubmitting] = useState(false)
  const expanded = draft.includes('\n')
  const canSubmit = draft.trim().length > 0
  const at = useAtCompletions({ cwd, gateway, sessionId })
  const slash = useSlashCompletions({ gateway })

  const focusEditor = useCallback(() => {
    const editor = editorRef.current

    focusComposerInput(editor)

    if (editor) {
      placeCaretEnd(editor)
    }

    markActiveComposer('edit')
  }, [])

  const requestEditFocus = useCallback(() => {
    setFocusRequestId(id => id + 1)
  }, [])

  const appendExternalText = useCallback(
    (text: string, mode: ComposerInsertMode) => {
      const value = text.trim()

      if (!value) {
        return
      }

      const base = mode === 'inline' ? draftRef.current.trimEnd() : draftRef.current
      const sep = mode === 'inline' ? (base ? ' ' : '') : base && !base.endsWith('\n') ? '\n\n' : ''
      const next = `${base}${sep}${value}`

      draftRef.current = next
      aui.composer().setText(next)

      const editor = editorRef.current

      if (editor) {
        renderComposerContents(editor, next)
        placeCaretEnd(editor)
      }

      setFocusRequestId(id => id + 1)
    },
    [aui]
  )

  useEffect(() => {
    draftRef.current = draft

    const editor = editorRef.current

    if (
      editor &&
      (editor.childNodes.length === 0 || (document.activeElement !== editor && composerPlainText(editor) !== draft))
    ) {
      renderComposerContents(editor, draft)

      if (document.activeElement === editor) {
        placeCaretEnd(editor)
      }
    }
  }, [draft])

  useEffect(() => {
    focusEditor()
  }, [focusEditor, focusRequestId])

  useEffect(() => {
    const offFocus = onComposerFocusRequest(target => {
      if (target === 'edit') {
        setFocusRequestId(id => id + 1)
      }
    })

    const offInsert = onComposerInsertRequest(({ mode, target, text }) => {
      if (target === 'edit') {
        appendExternalText(text, mode)
      }
    })

    return () => {
      offFocus()
      offInsert()
    }
  }, [appendExternalText])

  const syncDraftFromEditor = useCallback(
    (editor: HTMLDivElement) => {
      const nextDraft = composerPlainText(editor)

      if (nextDraft !== draftRef.current) {
        draftRef.current = nextDraft
        aui.composer().setText(nextDraft)
      }

      return nextDraft
    },
    [aui]
  )

  const refreshTrigger = useCallback(() => {
    const editor = editorRef.current

    if (!editor) {
      return
    }

    const before = textBeforeCaret(editor)
    const detected = detectTrigger(before ?? composerPlainText(editor))

    if (detected) {
      const rect = editor.getBoundingClientRect()
      const spaceAbove = rect.top
      const spaceBelow = window.innerHeight - rect.bottom

      setTriggerPlacement(spaceAbove < 220 && spaceBelow > spaceAbove ? 'bottom' : 'top')
    }

    setTrigger(detected)

    // Only reset the highlight when the trigger actually changed (opened, or
    // the query/kind differs). Re-detecting the *same* trigger — e.g. on a
    // caret move (mouseup) or a stray refresh — must preserve the user's
    // current selection instead of snapping back to the first item.
    if (detected?.kind !== trigger?.kind || detected?.query !== trigger?.query) {
      setTriggerActive(0)
    }
  }, [trigger])

  const closeTrigger = useCallback(() => {
    setTrigger(null)
    setTriggerItems([])
    setTriggerActive(0)
  }, [])

  const triggerAdapter: Unstable_TriggerAdapter | null =
    trigger?.kind === '@' ? at.adapter : trigger?.kind === '/' ? slash.adapter : null

  useEffect(() => {
    if (!trigger || !triggerAdapter?.search) {
      setTriggerItems([])

      return
    }

    setTriggerItems(triggerAdapter.search(trigger.query))
  }, [trigger, triggerAdapter])

  useEffect(() => {
    setTriggerActive(idx => Math.min(idx, Math.max(0, triggerItems.length - 1)))
  }, [triggerItems.length])

  const triggerLoading = trigger?.kind === '@' ? at.loading : trigger?.kind === '/' ? slash.loading : false

  const replaceTriggerWithChip = useCallback(
    (item: Unstable_TriggerItem) => {
      const editor = editorRef.current

      if (!editor || !trigger) {
        return
      }

      const serialized = hermesDirectiveFormatter.serialize(item)
      const starter = serialized.endsWith(':')
      const text = starter || serialized.endsWith(' ') ? serialized : `${serialized} `
      const directive = !starter && serialized.match(/^@([^:]+):(.+)$/)

      const finish = () => {
        draftRef.current = composerPlainText(editor)
        aui.composer().setText(draftRef.current)
        requestEditFocus()
        starter ? window.setTimeout(refreshTrigger, 0) : closeTrigger()
      }

      const sel = window.getSelection()
      const range = sel?.rangeCount ? sel.getRangeAt(0) : null
      const node = range?.startContainer
      const offset = range?.startOffset ?? 0

      if (!sel || !range || node?.nodeType !== Node.TEXT_NODE || offset < trigger.tokenLength) {
        const current = composerPlainText(editor)
        renderComposerContents(editor, `${current.slice(0, Math.max(0, current.length - trigger.tokenLength))}${text}`)
        placeCaretEnd(editor)

        return finish()
      }

      const replaceRange = document.createRange()
      replaceRange.setStart(node, offset - trigger.tokenLength)
      replaceRange.setEnd(node, offset)
      replaceRange.deleteContents()

      if (directive) {
        const chip = refChipElement(directive[1], directive[2])
        const space = document.createTextNode(' ')
        const fragment = document.createDocumentFragment()
        fragment.append(chip, space)
        replaceRange.insertNode(fragment)

        const caret = document.createRange()
        caret.setStart(space, 1)
        caret.collapse(true)
        sel.removeAllRanges()
        sel.addRange(caret)

        return finish()
      }

      document.execCommand('insertText', false, text)
      finish()
    },
    [aui, closeTrigger, refreshTrigger, requestEditFocus, trigger]
  )

  const insertDroppedRefs = useCallback(
    (candidates: ReturnType<typeof extractDroppedFiles>) => {
      const editor = editorRef.current

      if (!editor) {
        return false
      }

      const refs = candidates
        .map(candidate => droppedFileInlineRef(candidate, cwd))
        .filter((ref): ref is string => Boolean(ref))

      const nextDraft = insertInlineRefsIntoEditor(editor, refs)

      if (nextDraft === null) {
        return false
      }

      draftRef.current = nextDraft
      aui.composer().setText(nextDraft)
      requestEditFocus()

      return true
    },
    [aui, cwd, requestEditFocus]
  )

  const resetDragState = useCallback(() => {
    dragDepthRef.current = 0
    setDragActive(false)
  }, [])

  const handleDragEnter = (event: ReactDragEvent<HTMLElement>) => {
    if (!dragHasAttachments(event.dataTransfer, HERMES_PATHS_MIME)) {
      return
    }

    event.preventDefault()
    dragDepthRef.current += 1

    if (!dragActive) {
      setDragActive(true)
    }
  }

  const handleDragOver = (event: ReactDragEvent<HTMLElement>) => {
    if (!dragHasAttachments(event.dataTransfer, HERMES_PATHS_MIME)) {
      return
    }

    event.preventDefault()
    event.dataTransfer.dropEffect = 'copy'
  }

  const handleDragLeave = (event: ReactDragEvent<HTMLElement>) => {
    event.preventDefault()
    dragDepthRef.current = Math.max(0, dragDepthRef.current - 1)

    if (dragDepthRef.current === 0) {
      setDragActive(false)
    }
  }

  const handleDrop = (event: ReactDragEvent<HTMLElement>) => {
    if (!dragHasAttachments(event.dataTransfer, HERMES_PATHS_MIME)) {
      return
    }

    const candidates = extractDroppedFiles(event.dataTransfer)

    if (!candidates.length) {
      return
    }

    event.preventDefault()
    event.stopPropagation()
    resetDragState()

    if (insertDroppedRefs(candidates)) {
      triggerHaptic('selection')
    }
  }

  const handleInput = (event: FormEvent<HTMLDivElement>) => {
    const editor = event.currentTarget

    if (editor.childNodes.length === 1 && editor.firstChild?.nodeName === 'BR') {
      editor.replaceChildren()
    }

    syncDraftFromEditor(editor)
    window.setTimeout(refreshTrigger, 0)
  }

  const handlePaste = (event: ClipboardEvent<HTMLDivElement>) => {
    const pastedText = event.clipboardData.getData('text')

    if (!pastedText || DATA_IMAGE_URL_RE.test(pastedText.trim())) {
      event.preventDefault()

      return
    }

    event.preventDefault()
    document.execCommand('insertText', false, pastedText)
    syncDraftFromEditor(event.currentTarget)
  }

  const submitEdit = (editor: HTMLDivElement) => {
    const nextDraft = syncDraftFromEditor(editor)

    if (submitting || !nextDraft.trim()) {
      return
    }

    setSubmitting(true)
    aui.composer().send()
  }

  const handleEditBlur = useCallback(
    (event: FocusEvent<HTMLDivElement>) => {
      const nextTarget = event.relatedTarget

      if (nextTarget instanceof Node && event.currentTarget.contains(nextTarget)) {
        return
      }

      window.setTimeout(() => {
        const root = rootRef.current
        const active = document.activeElement

        if (submitting || (root && active && root.contains(active))) {
          return
        }

        closeTrigger()
        aui.composer().cancel()
      }, 80)
    },
    [aui, closeTrigger, submitting]
  )

  const handleKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    if (trigger && triggerItems.length > 0) {
      if (event.key === 'ArrowDown') {
        event.preventDefault()
        triggerKeyConsumedRef.current = true
        setTriggerActive(idx => (idx + 1) % triggerItems.length)

        return
      }

      if (event.key === 'ArrowUp') {
        event.preventDefault()
        triggerKeyConsumedRef.current = true
        setTriggerActive(idx => (idx - 1 + triggerItems.length) % triggerItems.length)

        return
      }

      if (event.key === 'Enter' || event.key === 'Tab') {
        event.preventDefault()
        triggerKeyConsumedRef.current = true
        const item = triggerItems[triggerActive]

        if (item) {
          replaceTriggerWithChip(item)
        }

        return
      }

      if (event.key === 'Escape') {
        event.preventDefault()
        triggerKeyConsumedRef.current = true
        closeTrigger()

        return
      }
    }

    if (event.key === 'Escape') {
      event.preventDefault()
      aui.composer().cancel()

      return
    }

    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault()
      submitEdit(event.currentTarget)
    }
  }

  const handleKeyUp = () => {
    // If this keyup belongs to a key the open trigger popover already consumed
    // in keydown (Arrow/Enter/Tab/Escape), skip the refresh. Those keys never
    // edit text, and for Escape the keydown already closed the menu — a refresh
    // here would re-detect the still-present `/` and instantly reopen it. We
    // read a ref set during keydown rather than `trigger`, because by keyup
    // time React has re-rendered and `trigger` may already be null.
    if (triggerKeyConsumedRef.current) {
      triggerKeyConsumedRef.current = false

      return
    }

    window.setTimeout(refreshTrigger, 0)
  }

  return (
    <ComposerPrimitive.Root className="contents" data-slot="aui_edit-composer-root">
      <StickyHumanMessageContainer>
        <div
          className="composer-human-message-container human-execution-message-top relative flex w-full items-start rounded-md bg-(--ui-chat-surface-background)"
          onBlur={handleEditBlur}
          onDragEnter={handleDragEnter}
          onDragLeave={handleDragLeave}
          onDragOver={handleDragOver}
          onDrop={handleDrop}
          ref={rootRef}
        >
          {trigger && (
            <ComposerTriggerPopover
              activeIndex={triggerActive}
              items={triggerItems}
              kind={trigger.kind}
              loading={triggerLoading}
              onHover={setTriggerActive}
              onPick={replaceTriggerWithChip}
              placement={triggerPlacement}
            />
          )}
          <div
            className={cn(
              USER_BUBBLE_BASE_CLASS,
              'ui-prompt-input__container relative border-(--ui-stroke-secondary) data-[expanded=true]:min-h-20',
              COMPOSER_DROP_FADE_CLASS,
              dragActive && COMPOSER_DROP_ACTIVE_CLASS
            )}
            data-expanded={expanded ? 'true' : undefined}
          >
            <div
              aria-label="Edit message"
              autoFocus
              className={cn(
                'ui-prompt-input-editor__input max-h-48 w-full resize-none bg-transparent p-0 pr-7 text-[length:var(--conversation-text-font-size)] leading-(--dt-line-height) text-foreground/95 outline-none',
                'empty:before:content-[attr(data-placeholder)] empty:before:text-muted-foreground/60',
                '**:data-ref-text:cursor-default',
                expanded ? 'min-h-16' : 'min-h-[1.25rem]'
              )}
              contentEditable
              data-placeholder="Edit message"
              data-slot={RICH_INPUT_SLOT}
              onBlur={() => window.setTimeout(closeTrigger, 80)}
              onDragOver={handleDragOver}
              onDrop={handleDrop}
              onFocus={() => markActiveComposer('edit')}
              onInput={handleInput}
              onKeyDown={handleKeyDown}
              onKeyUp={handleKeyUp}
              onMouseUp={refreshTrigger}
              onPaste={handlePaste}
              ref={editorRef}
              role="textbox"
              suppressContentEditableWarning
            />
            <ComposerPrimitive.Input className="sr-only" tabIndex={-1} unstable_focusOnScrollToBottom={false} />
            <button
              aria-label="Send edited message"
              className={cn('absolute right-2 bottom-2 size-5', USER_ACTION_ICON_BUTTON_CLASS)}
              disabled={!canSubmit || submitting}
              onClick={() => {
                const editor = editorRef.current

                if (editor) {
                  submitEdit(editor)
                }
              }}
              title="Send edited message"
              type="button"
            >
              {submitting ? StopGlyph : <Codicon name="arrow-up" size={USER_ACTION_ICON_SIZE} />}
            </button>
          </div>
        </div>
      </StickyHumanMessageContainer>
    </ComposerPrimitive.Root>
  )
}
