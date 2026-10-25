/**
 * Work block: the "Group work between replies" setting ($groupWork, #76147).
 *
 * One accordion per stretch of thinking + tool calls between two visible AI
 * texts: a plain summary line ("Thought · Explored 3 files, ran 2 commands")
 * with a chevron, no box. Folded by default; open, the steps hang under it,
 * indented, as a flat list of stock rows (a thinking row, a tool row or file
 * edit that expands in place; edits start folded). Questions, consent cards,
 * subagent cards, the theme picker, images and delivery notices
 * (`staysOnScreen`) split the stretch and stay on screen.
 *
 * Registered as assistant-ui's `ChainOfThought` component, which hands it one
 * range of message parts; each part renders through the stock components.
 */
import { MessagePrimitive, useAuiState } from '@assistant-ui/react'
import { useStore } from '@nanostores/react'
import { type ComponentProps, type FC, type ReactNode, useLayoutEffect, useMemo, useRef } from 'react'

import { setThemeCall } from '@/components/assistant-ui/theme-picker/args'
import { deliveryTargetFromCommand } from '@/components/assistant-ui/thread/agent-delivery'
import { TimelineTimestamp } from '@/components/assistant-ui/thread/timeline-timestamp'
import { selectMessageRunning, toolEntryDisclosureId } from '@/components/assistant-ui/tool/fallback-model'
import { isToolCallPart, summarizeToolRun, type ToolCallLike } from '@/components/assistant-ui/tool/run-summary'
import { SCAFFOLD_LABEL_CLASS, ScaffoldRow } from '@/components/chat/scaffold-row'
import { FadeText } from '@/components/ui/fade-text'
import { useI18n } from '@/i18n'
import { connectorCalls, mcpTargets } from '@/lib/connector-tools'
import { persistStringArray, storedStringArray } from '@/lib/storage'
import { isCardTool, isFileEditTool, isSilentTool } from '@/lib/tool-render-class'
import { useEnterAnimation } from '@/lib/use-enter-animation'
import { cn } from '@/lib/utils'
import { $showReasoning } from '@/store/reasoning-disclosure'
import { $showToolActivity } from '@/store/tool-activity'
import { $groupWork, $toolDisclosureOpen, $toolDisclosureStates, setToolDisclosureOpen } from '@/store/tool-view'

import { splitWorkRange, type WorkPartLike } from './work-block-split'

type PartsComponents = NonNullable<ComponentProps<typeof MessagePrimitive.PartByIndex>['components']>
type WorkComponents = PartsComponents & Required<Pick<PartsComponents, 'ReasoningGroup'>>

type AnyPart = { args?: unknown; text?: unknown; toolName?: string; type: string }
type ToolArgs = Parameters<typeof setThemeCall>[1]
type ToolPartLike = ToolCallLike & { timestamp?: number }

interface BlockState {
  completedAt?: number
  /** Row disclosure ids of the file edits in the stretch. */
  editIds: readonly string[]
  /** Thinking with visible text (what ReasoningGroup would draw). */
  hasThought: boolean
  key: string
  live: boolean
  messageRunning: boolean
  startedAt?: number
  summary: string
  toolCount: number
}

// Parts are immutable (new args or result = new object): classify each version once.
const staysCache = new WeakMap<object, boolean>()

/**
 * Cards that stay on screen: questions for the user, consent cards, subagent
 * fan-outs, the theme picker, generated images (fallback.tsx toolGroupKey +
 * splitRunItems: connector/MCP calls and set_theme through the tool_call
 * bridge count by their args), plus inter-agent delivery notices
 * (ChainToolFallback renders those instead of a terminal row). File edits fold
 * in as steps: the diff opens from its row inside the line.
 */
function staysOnScreen(part: AnyPart): boolean {
  let stays = staysCache.get(part)

  if (stays === undefined) {
    const name = part.toolName ?? ''
    // Streaming calls can lack args until they arrive; the classifiers read fields off it.
    const args = (part.args ?? {}) as ToolArgs
    const command = typeof args.command === 'string' ? args.command : ''

    stays =
      (isCardTool(name) && !isFileEditTool(name)) ||
      connectorCalls(name, args).length > 0 ||
      mcpTargets(name, args).length > 0 ||
      setThemeCall(name, args) !== null ||
      (name === 'terminal' && deliveryTargetFromCommand(command) !== null)
    staysCache.set(part, stays)
  }

  return stays
}

// Every range of a message runs this on every store update; the parts array
// is new only when a part changes, so encode each array once. Structure only
// (type + stays flag), so a text delta changes nothing.
const encodeCache = new WeakMap<readonly AnyPart[], string>()

function encodeParts(parts: readonly AnyPart[]): string {
  let encoded = encodeCache.get(parts)

  if (encoded === undefined) {
    encoded = parts.map(part => `${part.type}:${part.type === 'tool-call' && staysOnScreen(part) ? 1 : 0}`).join('|')
    encodeCache.set(parts, encoded)
  }

  return encoded
}

function decodeParts(key: string): WorkPartLike[] {
  return key.split('|').map(entry => {
    const [type, stays] = entry.split(':')

    return { stays: stays === '1', type }
  })
}

// Mirrors fallback.tsx useToolRun: recompute only when a call arrives,
// resolves, settles, or its args/result object changes.
function useBlockState(start: number, end: number): BlockState {
  const { locale } = useI18n()
  const cache = useRef<null | { signature: string; tools: readonly ToolPartLike[]; value: BlockState }>(null)

  return useAuiState(state => {
    const parts = state.message.parts
    const slice = parts.slice(start, end + 1)
    const tools = slice.filter(isToolCallPart).filter(tool => !isSilentTool(tool.toolName)) as unknown as ToolPartLike[]
    const messageRunning = selectMessageRunning(state)
    // Live while the turn runs and nothing has come after the stretch.
    const live = messageRunning && end >= parts.length - 1

    const hasThought = slice.some(
      part => part.type === 'reasoning' && typeof part.text === 'string' && part.text.trim().length > 0
    )

    const signature = tools
      .map(
        tool =>
          `${tool.toolCallId}:${tool.result === undefined ? 0 : 1}:${tool.isError ? 1 : 0}:${tool.timestamp ?? ''}:${tool.completedAt ?? ''}`
      )
      .concat(String(live), String(messageRunning), String(hasThought), state.message.id, locale)
      .join('|')

    const sameInputs =
      cache.current?.tools.length === tools.length &&
      tools.every((tool, index) => {
        const previous = cache.current!.tools[index]

        return (
          tool.args === previous.args &&
          tool.result === previous.result &&
          tool.toolName === previous.toolName &&
          tool.isError === previous.isError
        )
      })

    if (cache.current?.signature !== signature || !sameInputs) {
      const times = (pick: (tool: ToolPartLike) => number | undefined) =>
        tools.map(pick).filter((value): value is number => value !== undefined)

      const started = times(tool => tool.timestamp)
      const completed = times(tool => tool.completedAt)

      cache.current = {
        signature,
        tools,
        value: {
          completedAt: completed.length ? Math.max(...completed) : undefined,
          editIds: tools
            .filter(tool => isFileEditTool(tool.toolName))
            .map(tool => toolEntryDisclosureId(state.message.id, tool as Parameters<typeof toolEntryDisclosureId>[1])),
          hasThought,
          key: `work-block:${state.message.id}:${tools[0]?.toolCallId ?? start}`,
          live,
          messageRunning,
          startedAt: started.length ? Math.min(...started) : undefined,
          summary: tools.length ? summarizeToolRun(tools, live) : '',
          toolCount: tools.length
        }
      }
    }

    return cache.current.value
  })
}

// The disclosure store is shared with stock rows, so the line remembers the
// folds it seeded (across restarts too) and hands them back when the setting
// goes off: an edit nobody reopened renders stock again, open by default.
const SEEDED_FOLDS_KEY = 'hermes.desktop.workBlock.seededFolds'
const MAX_SEEDED_FOLDS = 240
const seededFolds = new Set(storedStringArray(SEEDED_FOLDS_KEY).slice(-MAX_SEEDED_FOLDS))

/**
 * Code edits inside the line start folded, unlike a loose diff
 * row, which opens by default: seed their row disclosure as closed before they
 * mount, in one store write. An explicit choice already in the store is kept.
 */
function foldEdits(ids: readonly string[]) {
  const states = $toolDisclosureStates.get()
  const missing = ids.filter(id => states[id] === undefined)

  if (missing.length) {
    missing.forEach(id => seededFolds.add(id))
    persistStringArray(SEEDED_FOLDS_KEY, [...seededFolds].slice(-MAX_SEEDED_FOLDS))
    $toolDisclosureStates.set({ ...states, ...Object.fromEntries(missing.map(id => [id, false])) })
  }
}

$groupWork.subscribe(grouped => {
  if (grouped || !seededFolds.size) {
    return
  }

  const states = { ...$toolDisclosureStates.get() }
  const unfolded = [...seededFolds].filter(id => states[id] === false)

  seededFolds.clear()
  persistStringArray(SEEDED_FOLDS_KEY, [])
  unfolded.forEach(id => delete states[id])

  if (unfolded.length) {
    $toolDisclosureStates.set(states)
  }
})

/** One stock row per step: each run of thinking as one ReasoningGroup, each tool call as its own row. */
function renderSteps(components: WorkComponents, parts: readonly WorkPartLike[], start: number, end: number) {
  const { ReasoningGroup } = components
  const rows: ReactNode[] = []
  let thinking: number[] = []

  const flushThinking = () => {
    if (thinking.length) {
      rows.push(
        <ReasoningGroup endIndex={thinking.at(-1)!} key={`thinking-${thinking[0]}`} startIndex={thinking[0]}>
          {thinking.map(index => (
            <MessagePrimitive.PartByIndex components={components} index={index} key={index} />
          ))}
        </ReasoningGroup>
      )
      thinking = []
    }
  }

  for (let index = start; index <= end; index++) {
    if (parts[index]?.type === 'reasoning') {
      thinking.push(index)
    } else {
      flushThinking()
      rows.push(<MessagePrimitive.PartByIndex components={components} index={index} key={index} />)
    }
  }

  flushThinking()

  return rows
}

const WorkBlockLine: FC<{ children: ReactNode; end: number; start: number }> = ({ children, end, start }) => {
  const { t } = useI18n()
  const showToolActivity = useStore($showToolActivity)
  const showReasoning = useStore($showReasoning)

  const { completedAt, editIds, hasThought, key, live, messageRunning, startedAt, summary, toolCount } = useBlockState(
    start,
    end
  )

  const open = useStore($toolDisclosureOpen(key)) ?? false
  const enterRef = useEnterAnimation(messageRunning, key)
  const thought = hasThought && showReasoning
  // Thinking alone already is one accordion, and with tool_progress off the
  // tool rows aren't drawn: no line then, the steps show bare. A lone call
  // folds like a run (#76147), unlike stock ToolRun's `expanded = count < 2`.
  const folds = showToolActivity && toolCount > 0

  // Edits that arrive while the line is open (or a line restored open): seed
  // before paint, so a diff never flashes open.
  useLayoutEffect(() => {
    if (folds && open) {
      foldEdits(editIds)
    }
  }, [folds, open, editIds])

  const label = thought ? `${t.assistant.thread.thought} · ${summary}` : summary

  // No box, no guide rule. One wrapper shape in every
  // state, so the first tool arriving after thinking (no line → line) doesn't
  // remount the live thinking row. Only the header line carries the scaffold
  // fade: each step row carries its own, and a marked container would fade
  // them twice (styles.css: marks never nest).
  return (
    <div
      className="grid min-w-0 max-w-full gap-(--tool-row-gap) text-[length:var(--conversation-tool-font-size)] text-(--ui-text-tertiary)"
      data-slot="tool-block"
      data-work-block={folds ? '' : undefined}
      data-work-block-open={folds && open ? '' : undefined}
      ref={folds ? enterRef : undefined}
    >
      {folds && (
        // A faint resting caret, like thinking headers (styles.css): the line
        // is the only way to reach its steps, so the affordance shows unhovered.
        <div className="[--disclosure-caret-rest:0.4]" data-conversation-scaffold="" data-work-block-header="">
          <ScaffoldRow
            onToggle={() => {
              if (!open) {
                foldEdits(editIds)
              }

              setToolDisclosureOpen(key, !open)
            }}
            open={open}
            trailing={<TimelineTimestamp completedAt={completedAt} timestamp={startedAt} />}
          >
            <FadeText className={cn(SCAFFOLD_LABEL_CLASS, 'truncate')}>
              {live ? <span className="shimmer">{label}</span> : label}
            </FadeText>
          </ScaffoldRow>
        </div>
      )}
      {(!folds || open) && (
        // No height cap: a max-height grid sizes its rows against the cap and
        // squeezes overflow-hidden rows (an open diff) instead of growing.
        <div className={cn('grid min-w-0 max-w-full gap-(--tool-row-gap)', folds && 'ps-4')} data-work-block-body="">
          {children}
        </div>
      )}
    </div>
  )
}

/** The `ChainOfThought` component for `MessagePrimitive.Parts`, rendering parts with `components`. */
export function makeWorkBlock(components: WorkComponents): FC {
  const WorkBlock: FC = () => {
    // The range arrives as a slice of message.parts (same objects), so its
    // first part's position in the message is the range start. During a
    // stream the range can lag the message by an update (indexOf -1): keep the
    // last good range then, or the block unmounts and remounts on every delta.
    const lastRange = useRef('')

    const range = useAuiState(state => {
      const chain = state.chainOfThought.parts
      const start = state.message.parts.indexOf(chain[0] as (typeof state.message.parts)[number])

      if (start >= 0) {
        lastRange.current = `${start}:${start + chain.length - 1}`
      }

      return lastRange.current
    })

    const structure = useAuiState(state => encodeParts(state.message.parts as readonly AnyPart[]))

    const { items, parts } = useMemo(() => {
      const parts = decodeParts(structure)
      const [start, end] = range ? range.split(':').map(Number) : [-1, -1]

      return { items: range ? splitWorkRange(parts, start, end) : [], parts }
    }, [range, structure])

    return items.map(item =>
      item.kind === 'block' ? (
        <WorkBlockLine end={item.end} key={`block-${item.start}`} start={item.start}>
          {renderSteps(components, parts, item.start, item.end)}
        </WorkBlockLine>
      ) : (
        <MessagePrimitive.PartByIndex components={components} index={item.index} key={`part-${item.index}`} />
      )
    )
  }

  return WorkBlock
}
