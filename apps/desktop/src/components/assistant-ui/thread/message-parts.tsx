import { type ToolCallMessagePartProps, useAuiState } from '@assistant-ui/react'
import { type ComponentProps, type FC, type ReactNode, useEffect, useMemo, useRef, useState } from 'react'

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

const NullPart: FC = () => null

type TracePart = {
  isError?: boolean
  result?: unknown
  status?: { type?: string }
  text?: string
  type?: string
}

function resultFlag(result: unknown, key: string): unknown {
  if (!result || typeof result !== 'object') {
    return undefined
  }

  return (result as Record<string, unknown>)[key]
}

function traceSummary(parts: readonly TracePart[]) {
  let reasoning = 0
  let tools = 0
  let failed = 0
  let running = 0

  for (const part of parts) {
    if (part.type === 'reasoning') {
      if (typeof part.text === 'string' && part.text.trim()) {
        reasoning += 1
      }

      if (part.status?.type && part.status.type !== 'complete') {
        running += 1
      }

      continue
    }

    if (part.type !== 'tool-call') {
      continue
    }

    tools += 1

    if (part.result === undefined) {
      running += 1
    }

    if (part.isError || resultFlag(part.result, 'success') === false || resultFlag(part.result, 'error')) {
      failed += 1
    }
  }

  return `${failed}:${reasoning}:${running}:${tools}:${reasoning + tools}`
}

function parseTraceSummary(signature: string) {
  const [failed = 0, reasoning = 0, running = 0, tools = 0, total = 0] = signature.split(':').map(Number)

  return { failed, reasoning, running, tools, total }
}

export const ProcessTraceDisclosure: FC<{
  children: ReactNode
  hasVisibleText: boolean
}> = ({ children, hasVisibleText }) => {
  const { t } = useI18n()
  const messageRunning = useAuiState(s => s.message.status?.type === 'running')
  const summarySignature = useAuiState(s => traceSummary(s.message.parts as TracePart[]))
  const summary = useMemo(() => parseTraceSummary(summarySignature), [summarySignature])
  const [userOpen, setUserOpen] = useState<boolean | null>(null)
  const defaultOpen = messageRunning || !hasVisibleText
  const open = userOpen ?? defaultOpen

  useEffect(() => {
    if (messageRunning) {
      setUserOpen(null)
    }
  }, [messageRunning])

  const summaryText = useMemo(() => {
    const parts = [t.assistant.thread.processTraceStepCount(summary.total)]

    if (summary.failed > 0) {
      parts.push(t.assistant.thread.processTraceErrorCount(summary.failed))
    } else if (summary.running > 0) {
      parts.push(t.assistant.thread.processTraceRunning)
    }

    return parts.join(' · ')
  }, [summary.failed, summary.running, summary.total, t.assistant.thread])

  if (summary.total === 0) {
    return null
  }

  return (
    <div
      className="mb-1.5 text-[length:var(--conversation-tool-font-size)] text-(--ui-text-tertiary)"
      data-slot="process-trace-disclosure"
    >
      <DisclosureRow onToggle={() => setUserOpen(!open)} open={open}>
        <span className="flex min-w-0 items-baseline gap-1.5">
          <span
            className={cn(
              'font-medium leading-(--conversation-line-height) text-(--ui-text-secondary)',
              messageRunning && 'shimmer text-foreground/55'
            )}
          >
            {t.assistant.thread.processTrace}
          </span>
          <span className="text-[length:var(--conversation-caption-font-size)] leading-(--conversation-caption-line-height) text-(--ui-text-tertiary)">
            {summaryText}
          </span>
        </span>
      </DisclosureRow>
      {open && <div className="mt-0.5 grid min-w-0 max-w-full gap-(--tool-row-gap) pb-1 pl-0.5">{children}</div>}
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
            {t.assistant.thread.thinking}
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

// Module-level constants so the `components` prop on `MessagePrimitive.Parts`
// has a stable identity across renders. Without this every AssistantMessage
// render would create a fresh `components` object, invalidating the memo on
// `MessagePrimitivePartByIndex` and forcing every tool/reasoning child to
// re-render on every streaming delta. Memo invalidation alone doesn't
// remount, but combined with the previous ToolFallback group-swap it was a
// big chunk of the per-delta work.
export const MESSAGE_PARTS_COMPONENTS = {
  Reasoning: ReasoningTextPart,
  ReasoningGroup: ReasoningAccordionGroup,
  Text: MarkdownText,
  ToolGroup: ToolGroupSlot,
  tools: { Fallback: ChainToolFallback }
} as const

export const MESSAGE_TRACE_PARTS_COMPONENTS = {
  Reasoning: ReasoningTextPart,
  ReasoningGroup: ReasoningAccordionGroup,
  Text: NullPart,
  ToolGroup: ToolGroupSlot,
  tools: { Fallback: ChainToolFallback }
} as const

export const MESSAGE_RESPONSE_PARTS_COMPONENTS = {
  Reasoning: NullPart,
  ReasoningGroup: NullPart,
  Text: MarkdownText,
  ToolGroup: NullPart,
  tools: { Fallback: NullPart }
} as const
