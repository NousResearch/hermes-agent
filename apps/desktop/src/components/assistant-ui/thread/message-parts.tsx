import { type ToolCallMessagePartProps, useAuiState } from '@assistant-ui/react'
import { type ComponentProps, type FC, type ReactNode, useEffect, useRef, useState } from 'react'

import { ClarifyTool } from '@/components/assistant-ui/clarify-tool'
import { MarkdownText, MarkdownTextContent } from '@/components/assistant-ui/markdown-text'
import { ToolFallback, ToolGroupSlot } from '@/components/assistant-ui/tool/fallback'
import { useElapsedSeconds } from '@/components/chat/activity-timer'
import { ActivityTimerText } from '@/components/chat/activity-timer-text'
import { DisclosureRow } from '@/components/chat/disclosure-row'
import { GeneratedImage } from '@/components/chat/generated-image-result'
import { useI18n } from '@/i18n'
import { useEnterAnimation } from '@/lib/use-enter-animation'
import { cn } from '@/lib/utils'

const ImageGenerateTool: FC<ToolCallMessagePartProps> = ({ args, result }) => {
  const aspectRatio = typeof args?.aspect_ratio === 'string' ? args.aspect_ratio : undefined

  return (
    <div className="mt-1.5">
      <GeneratedImage aspectRatio={aspectRatio} result={result} />
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
  /** Header label; defaults to the "Thinking" string. The commentary lane
   *  passes "Working" to keep narration visually distinct from reasoning. */
  label?: ReactNode
  messageRunning?: boolean
  pending?: boolean
  slot?: string
  timerKey?: string
}> = ({ children, label, messageRunning = false, pending = false, slot = 'aui_thinking-disclosure', timerKey }) => {
  const { t } = useI18n()
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
      data-slot={slot}
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
            {label ?? t.assistant.thread.thinking}
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

  if (!hasContent) {
    return null
  }

  return (
    <ThinkingDisclosure messageRunning={messageRunning} pending={pending} timerKey={`reasoning:${messageId}`}>
      {children}
    </ThinkingDisclosure>
  )
}

// Codex commentary/analysis narration — the "Working" lane. Reuses the
// disclosure chrome from Thinking but stays a separate surface so genuine
// reasoning and user-facing progress narration never share a header. The
// part arrives as assistant-ui's `data` part (name: "commentary"); its
// status is `running` only while it is the last part of a running message,
// which is exactly the live-streaming window.
const CommentaryDataPart: FC<{ data?: { text?: unknown }; status?: { type: string } }> = ({ data, status }) => {
  const { t } = useI18n()
  const messageId = useAuiState(s => s.message.id)
  const messageRunning = useAuiState(s => s.message.status?.type === 'running')
  const threadRunning = useAuiState(s => s.thread.isRunning)
  const text = typeof data?.text === 'string' ? data.text : ''
  const pending = threadRunning && messageRunning && status?.type === 'running'

  // Empty narration renders no header — same rule as the reasoning group.
  if (!text.trim()) {
    return null
  }

  return (
    <ThinkingDisclosure
      label={t.assistant.thread.working}
      messageRunning={messageRunning}
      pending={pending}
      slot="aui_commentary-disclosure"
      timerKey={`commentary:${messageId}`}
    >
      <MarkdownTextContent
        containerClassName="text-xs leading-snug text-muted-foreground/85"
        containerProps={{ 'data-slot': 'aui_commentary-text' } as ComponentProps<'div'>}
        isRunning={pending}
        text={text.trimStart()}
      />
    </ThinkingDisclosure>
  )
}

const ReasoningTextPart: FC<{ text: string; status?: { type: string } }> = ({ text, status }) => {
  const displayText = text.trimStart()
  const messageRunning = useAuiState(s => s.message.status?.type === 'running')
  const isRunning = status?.type === 'running' || messageRunning

  return (
    <MarkdownTextContent
      containerClassName="text-xs leading-snug text-muted-foreground/85"
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
export const MESSAGE_PARTS_COMPONENTS = {
  data: { by_name: { commentary: CommentaryDataPart } },
  Reasoning: ReasoningTextPart,
  ReasoningGroup: ReasoningAccordionGroup,
  Text: MarkdownText,
  ToolGroup: ToolGroupSlot,
  tools: { Fallback: ChainToolFallback }
} as const
