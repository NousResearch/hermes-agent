import {
  type ReasoningMessagePartComponent,
  type TextMessagePartProps,
  type ToolCallMessagePartProps,
  useAuiState,
  useMessagePartReasoning
} from '@assistant-ui/react'
import { useStore } from '@nanostores/react'
import { type ComponentProps, type FC, type ReactNode, useEffect, useRef, useState } from 'react'

import { ClarifyTool } from '@/components/assistant-ui/clarify-tool'
import { MarkdownText, MarkdownTextContent } from '@/components/assistant-ui/markdown-text'
import { McpSetupTool } from '@/components/assistant-ui/mcp-setup-tool'
import { AgentDeliveryNotice, deliveryTargetFromCommand } from '@/components/assistant-ui/thread/agent-delivery'
import { TimelineTimestamp } from '@/components/assistant-ui/thread/timeline-timestamp'
import { DelegateTool } from '@/components/assistant-ui/tool/delegate'
import { ToolFallback, ToolGroupSlot } from '@/components/assistant-ui/tool/fallback'
import { formatElapsed, useElapsedSeconds, useMeasuredDuration } from '@/components/chat/activity-timer'
import { ActivityTimerText } from '@/components/chat/activity-timer-text'
import { GeneratedImage } from '@/components/chat/generated-image-result'
import { SCAFFOLD_LABEL_CLASS, SCAFFOLD_META_CLASS, ScaffoldRow } from '@/components/chat/scaffold-row'
import { useI18n } from '@/i18n'
import { generatedImageFromResult } from '@/lib/generated-images'
import { separateGluedReasoningBlocks } from '@/lib/reasoning-blocks'
import { useEnterAnimation } from '@/lib/use-enter-animation'
import { cn } from '@/lib/utils'
import { $reasoningCollapsedByDefault } from '@/store/reasoning-disclosure'

type TimelineToolCallProps = ToolCallMessagePartProps & { completedAt?: number; timestamp?: number }

const ImageGenerateTool: FC<TimelineToolCallProps> = props => {
  const { args, completedAt, result, timestamp } = props
  const aspectRatio = typeof args?.aspect_ratio === 'string' ? args.aspect_ratio : undefined

  // The image card owns successful generations. Failed or malformed results
  // still need the normal tool row: it extracts the error text and gives the
  // user an honest, expandable failure rather than silently dropping the call.
  if (result !== undefined && !generatedImageFromResult(result)) {
    return <ToolFallback {...props} />
  }

  return (
    <div className="mt-1.5">
      <TimelineTimestamp className="mb-0.5 block" completedAt={completedAt} timestamp={timestamp} />
      <GeneratedImage aspectRatio={aspectRatio} result={result} />
    </div>
  )
}

const DelegateToolPart: FC<TimelineToolCallProps> = props => {
  // A call that failed outright dispatched nothing — there are no children to
  // list, only an error. The generic row extracts and expands it properly.
  if (props.isError) {
    return <ToolFallback {...props} />
  }

  return (
    <>
      <TimelineTimestamp className="mb-0.5 block" completedAt={props.completedAt} timestamp={props.timestamp} />
      <DelegateTool args={props.args} result={props.result} toolCallId={props.toolCallId} />
    </>
  )
}

const ChainToolFallback: FC<TimelineToolCallProps> = props => {
  // todo parts are hoisted to a dedicated panel above the message content.
  if (props.toolName === 'todo') {
    return null
  }

  // An inter-agent delivery run through the terminal tool renders as the
  // compact "Messaged X" / "Message from X" notices, not a transcript row
  // (Grok-bots parity; the receiving side already renders notices via
  // AGENT_MESSAGE_RE). Non-delivery terminal calls fall through unchanged.
  if (props.toolName === 'terminal' && !props.isError) {
    const command = typeof props.args?.command === 'string' ? props.args.command : ''

    if (deliveryTargetFromCommand(command)) {
      return <AgentDeliveryNotice {...props} />
    }
  }

  // A reaction's UI is the emoji landing on the bubble (message.reaction
  // event) — a "React To Message" tool block next to it would be the agent
  // narrating its own tapback. Failures still render so they're debuggable.
  if (props.toolName === 'react_to_message' && !props.isError) {
    return null
  }

  if (props.toolName === 'delegate_task') {
    return <DelegateToolPart {...props} />
  }

  if (props.toolName === 'image_generate') {
    return <ImageGenerateTool {...props} />
  }

  if (props.toolName === 'clarify') {
    return (
      <>
        <TimelineTimestamp className="mb-0.5 block" completedAt={props.completedAt} timestamp={props.timestamp} />
        <ClarifyTool {...props} />
      </>
    )
  }

  if (props.toolName === 'setup_mcp') {
    return <McpSetupTool {...props} />
  }

  return <ToolFallback {...props} />
}

type TimelineTextPartProps = TextMessagePartProps & { completedAt?: number; timestamp?: number }

const TimelineMarkdownText: FC<TimelineTextPartProps> = ({ completedAt, timestamp }) => (
  <>
    <TimelineTimestamp className="mb-0.5 block" completedAt={completedAt} timestamp={timestamp} />
    <MarkdownText />
  </>
)

const ThinkingDisclosure: FC<{
  children: ReactNode
  completedAt?: number
  messageRunning?: boolean
  pending?: boolean
  timestamp?: number
  // Required: the block's duration is remembered against this key, so a
  // component that mounts after the block finished can still report it.
  timerKey: string
}> = ({ children, completedAt, messageRunning = false, pending = false, timestamp, timerKey }) => {
  const { t } = useI18n()
  const reasoningCollapsedByDefault = useStore($reasoningCollapsedByDefault)
  // `null` = no explicit user toggle yet. Live reasoning remains visible by
  // default, unless the user opts into the low-jitter collapsed presentation.
  const [userOpen, setUserOpen] = useState<boolean | null>(null)
  const elapsed = useElapsedSeconds(pending, timerKey)
  const thoughtFor = useMeasuredDuration(pending, timerKey)
  const scrollRef = useRef<HTMLDivElement | null>(null)
  const contentRef = useRef<HTMLDivElement | null>(null)
  const enterRef = useEnterAnimation(messageRunning, timerKey)
  // A live preview that later settles must not unmount its body — that is the
  // "turn settled and everything jumped" shift. Latch that we showed one so
  // the clip stays. Groups that mount already complete (earlier thoughts in
  // a still-running turn) never latch, so they stay collapsed.
  const [sawLivePreview, setSawLivePreview] = useState(false)

  if (pending && !sawLivePreview) {
    setSawLivePreview(true)
  }

  // The collapsed-by-default preference outranks the latch: it opts out of
  // live previews entirely, so there is nothing to hold open.
  const showPreview = !reasoningCollapsedByDefault && (pending || sawLivePreview)
  const open = userOpen ?? showPreview
  const isPreview = userOpen === null && showPreview

  // Three ways a finished block can report itself. With a measured duration it
  // says so, unless the timer's whole seconds round it to "0s" — accurate and
  // useless — in which case it just says it was quick. With no duration at all
  // it still has to read as finished; a turn that ended must not go on saying
  // "Thinking".
  let thoughtLabel = t.assistant.thread.thinking

  if (!pending) {
    if (thoughtFor === null) {
      thoughtLabel = t.assistant.thread.thought
    } else if (thoughtFor < 1) {
      thoughtLabel = t.assistant.thread.thoughtBriefly
    } else {
      thoughtLabel = t.assistant.thread.thoughtFor(formatElapsed(thoughtFor))
    }
  }

  // While the preview is live, pin the scroll container to the bottom on
  // every content growth so the latest tokens are always visible.
  useEffect(() => {
    if (!isPreview) {
      return
    }

    const el = scrollRef.current
    const content = contentRef.current

    if (!el || !content) {
      return
    }

    // Height-gated: the observer also fires when the container's WIDTH changes
    // (sidebar sash drag resizes every message), and pinning there forces a
    // scrollHeight read+write per preview per frame. Only actual content
    // growth needs the pin; the height rides the RO entry, reflow-free.
    let lastHeight = -1

    const pin = (entries: readonly ResizeObserverEntry[]) => {
      const height = entries[entries.length - 1]?.borderBoxSize?.[0]?.blockSize ?? -1
      const grew = height < 0 || height > lastHeight
      lastHeight = height

      if (grew) {
        el.scrollTop = el.scrollHeight
      }
    }

    // No sync pin(): the observer's guaranteed initial delivery runs it with
    // layout already clean (still before paint), avoiding a forced reflow.
    const observer = new ResizeObserver(pin)
    observer.observe(content)

    return () => observer.disconnect()
    // Re-run when the disclosure toggles so the observer attaches to the new
    // DOM after expand/collapse (refs are conditionally rendered on `open`).
  }, [isPreview, open])

  return (
    <div
      className="text-[length:var(--conversation-tool-font-size)] text-(--ui-text-tertiary)"
      data-conversation-scaffold=""
      data-slot="aui_thinking-disclosure"
      ref={enterRef}
    >
      <ScaffoldRow
        onToggle={() => setUserOpen(!open)}
        open={open}
        trailing={
          <span className="flex shrink-0 items-center gap-1.5">
            <TimelineTimestamp className={SCAFFOLD_META_CLASS} completedAt={completedAt} timestamp={timestamp} />
            {pending && <ActivityTimerText className={SCAFFOLD_META_CLASS} seconds={elapsed} />}
          </span>
        }
      >
        <span className={cn(SCAFFOLD_LABEL_CLASS, pending && 'shimmer')}>{thoughtLabel}</span>
      </ScaffoldRow>
      {open && (
        <div
          className={cn(
            // Body sits flush with the "Thinking" header — no left indent —
            // and inherits the disclosure-level opacity fade defined in
            // styles.css (~0.67 at rest, 1 on hover/focus). overflow-auto so
            // the max-h-40 preview is a real scroller, not a clip.
            // No overscroll-contain in any state: a body that can't scroll
            // itself (expanded, or content shorter than the cap) must chain
            // the wheel to the thread viewport, and a scrollable preview
            // chains past its own edge once it hits top/bottom — matching
            // every other scroller in the app (settings, code blocks). A
            // contain here is a dead zone or a scroll discontinuity, the
            // same trap #84964 hit in the session list.
            'mt-0.5 w-full min-w-0 max-w-full overflow-auto wrap-anywhere pb-1',
            isPreview && 'max-h-40'
          )}
          data-slot="aui_thinking-body"
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
        .slice(Math.max(0, startIndex), endIndex + 1)
        .some(p => p?.type === 'reasoning' && p.status?.type !== 'complete')
  )

  // A reasoning group with no actual text is pure noise — drop the whole
  // "Thinking" disclosure rather than leave an empty header eating a row. This
  // applies live too: encrypted/spinner-coerced reasoning (Opus reasoning max)
  // never carries visible text, and the bottom-of-thread loader already signals
  // "thinking", so an empty header is never wanted. Real reasoning surfaces the
  // instant its first token lands.
  const hasContent = useAuiState(s =>
    s.message.parts
      .slice(Math.max(0, startIndex), endIndex + 1)
      .some(p => p?.type === 'reasoning' && typeof p.text === 'string' && p.text.trim().length > 0)
  )

  const timestamp = useAuiState(s =>
    s.message.parts.slice(Math.max(0, startIndex), endIndex + 1).reduce<number | undefined>((earliest, part) => {
      const value = part.type === 'reasoning' ? (part as { timestamp?: number }).timestamp : undefined

      return value === undefined ? earliest : earliest === undefined ? value : Math.min(earliest, value)
    }, undefined)
  )

  const completedAt = useAuiState(s =>
    s.message.parts.slice(Math.max(0, startIndex), endIndex + 1).reduce<number | undefined>((latest, part) => {
      const value = part.type === 'reasoning' ? (part as { completedAt?: number }).completedAt : undefined

      return value === undefined ? latest : latest === undefined ? value : Math.max(latest, value)
    }, undefined)
  )

  if (!hasContent) {
    return null
  }

  return (
    // Keyed per block, not per message: the timer registry hands every caller
    // of a key the same origin, so a turn that thinks three separate times used
    // to measure the second and third blocks from the first one's start and
    // report the running total as each block's duration.
    <ThinkingDisclosure
      completedAt={completedAt}
      messageRunning={messageRunning}
      pending={pending}
      timerKey={`reasoning:${messageId}:${startIndex}`}
      timestamp={timestamp}
    >
      {children}
    </ThinkingDisclosure>
  )
}

// Read the part from context, same contract as MarkdownText's
// useMessagePartText — the reasoning-only smoothing wrapper (removed) stalled
// the char-reveal at empty, blanking the widget.
const ReasoningTextPart: ReasoningMessagePartComponent = () => {
  const { status, text } = useMessagePartReasoning()
  const messageRunning = useAuiState(s => s.message.status?.type === 'running')

  return (
    <MarkdownTextContent
      containerClassName="text-xs leading-snug text-muted-foreground/85"
      containerProps={{ 'data-slot': 'aui_reasoning-text' } as ComponentProps<'div'>}
      disableArtifacts
      isRunning={status.type === 'running' || messageRunning}
      text={separateGluedReasoningBlocks(text.trimStart())}
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
export const MESSAGE_PARTS_COMPONENTS = {
  Reasoning: ReasoningTextPart,
  ReasoningGroup: ReasoningAccordionGroup,
  Text: TimelineMarkdownText,
  ToolGroup: ToolGroupSlot,
  tools: { Fallback: ChainToolFallback }
} as const
