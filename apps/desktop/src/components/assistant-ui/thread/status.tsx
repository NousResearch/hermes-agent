import { useAuiState } from '@assistant-ui/react'
import { useStore } from '@nanostores/react'
import { type FC, type ReactNode, useEffect, useMemo, useState } from 'react'

import { useSessionView } from '@/app/chat/session-view'
import { activitySignature, toolNarratesWait, TURN_QUIET_S } from '@/components/assistant-ui/thread/turn-activity'
import { toolPresentVerb } from '@/components/assistant-ui/tool/run-summary'
import { useElapsedSeconds } from '@/components/chat/activity-timer'
import { ActivityTimerText } from '@/components/chat/activity-timer-text'
import { SCAFFOLD_LABEL_CLASS } from '@/components/chat/scaffold-row'
import { Codicon } from '@/components/ui/codicon'
import { Loader } from '@/components/ui/loader'
import { StatusPulse } from '@/components/ui/status-pulse'
import { getLocalModelsStatus } from '@/hermes'
import { useI18n } from '@/i18n'
import { cn } from '@/lib/utils'
import { $backgroundResume } from '@/store/background-delegation'
import { sessionCompacting } from '@/store/compaction'
import { $localModelsEnabled } from '@/store/local-models-flag'
import { sessionAwaitingInput } from '@/store/prompts'
import { parseModelLoadWait, sessionProviderWait } from '@/store/provider-wait'
import { $currentModel } from '@/store/session'
import { type DraftingTool, sessionDraftingTool } from '@/store/tool-drafting'
import type { LocalModelLoadProgress } from '@/types/hermes'

// A status line is scaffolding like any other — "Editing" while the model
// drafts a call is the same kind of line as "Explored 3 files" once it has run,
// and reads as one continuous column only if it shares their type and colour.
const StatusRow: FC<{ children: ReactNode; label: string } & React.ComponentPropsWithoutRef<'div'>> = ({
  children,
  label,
  className,
  ...rest
}) => (
  <div
    aria-label={label}
    aria-live="polite"
    className={cn(
      'flex min-w-0 max-w-full items-center gap-1.5 self-start leading-(--conversation-line-height)',
      'text-(--conversation-scaffold-text)',
      className
    )}
    data-conversation-scaffold=""
    role="status"
    {...rest}
  >
    {children}
  </div>
)

// Fixed label while auto-compaction runs — decoupled from backend status text.
const COMPACTION_LABEL = 'Summarizing thread'

const HintText: FC<{ children: ReactNode }> = ({ children }) => (
  <span className={cn(SCAFFOLD_LABEL_CLASS, 'shimmer min-w-0 flex-1 truncate')}>{children}</span>
)

/** Renderer-side load synthesis: poll the local-models status while a turn
 * is busy with NO progress frame from the backend. The backend's wait loop
 * only narrates the MAIN chat request — a model load triggered while the
 * gateway is still initializing, or one consumed by a parallel auxiliary
 * call (title generation autoloads the same model), never gets a frame,
 * and the load looked like nothing was happening. The status route reads
 * the same SSE snapshot, so this bar carries the identical percent. */
function useLocalModelLoad(active: boolean): (LocalModelLoadProgress & { model: string }) | null {
  const model = useStore($currentModel)
  const [progress, setProgress] = useState<(LocalModelLoadProgress & { model: string }) | null>(null)

  // Behind the --local launch flag: without it, no status polling and no
  // load bar (the local server can't be the current provider anyway).
  const enabled = $localModelsEnabled.get()

  useEffect(() => {
    if (!enabled || !active || !model) {
      setProgress(null)

      return
    }

    let cancelled = false
    let timer: number | undefined

    const tick = async () => {
      try {
        const status = await getLocalModelsStatus()
        const entry = status.loading?.[model]

        if (!cancelled) {
          setProgress(entry ? { ...entry, model } : null)
        }
      } catch {
        if (!cancelled) {
          setProgress(null)
        }
      }

      if (!cancelled) {
        timer = window.setTimeout(() => void tick(), 1_500)
      }
    }

    void tick()

    return () => {
      cancelled = true

      if (timer !== undefined) {
        window.clearTimeout(timer)
      }
    }
  }, [enabled, active, model])

  return progress
}

/** Wait hint with a real progress bar for managed-local model loads and
 * prompt processing. The percents come from llama-server itself (per-tensor
 * load callback / live prefill counter, via the gateway's wait frames), so a
 * determinate bar is honest — a 40s cold load or a long prefill reads as
 * visible progress instead of an alarming stall. */
const WaitHint: FC<{ hint: string }> = ({ hint }) => {
  const { t } = useI18n()
  const load = parseModelLoadWait(hint)

  if (!load) {
    return <HintText>{hint}</HintText>
  }

  const label =
    load.kind === 'load' ? t.assistant.thread.loadingLocalModel(load.model) : t.assistant.thread.processingPrompt

  return <ProgressHint label={label} percent={load.percent} />
}

const ProgressHint: FC<{ label: string; percent: null | number }> = ({ label, percent }) => (
  <span className="flex min-w-0 flex-1 items-center gap-2">
    <span className={cn(SCAFFOLD_LABEL_CLASS, 'shimmer min-w-0 shrink truncate')}>{label}</span>
    {percent !== null && (
      <>
        <span className="h-1 w-24 shrink-0 overflow-hidden rounded-full bg-(--ui-bg-tertiary)">
          <span
            className="block h-full rounded-full bg-primary transition-[width] duration-500"
            style={{ width: `${Math.max(2, percent)}%` }}
          />
        </span>
        <span className={cn(SCAFFOLD_LABEL_CLASS, 'shrink-0 tabular-nums')}>{percent}%</span>
      </>
    )}
  </span>
)

/** These indicators render inside whichever transcript mounted them, so every
 *  session-scoped signal comes from that surface's view — a tile must never
 *  show the primary chat's compaction, prompt-wait, or turn timer. */
function useThreadSessionStatus() {
  const view = useSessionView()
  const sessionId = useStore(view.$runtimeId)
  // The same turn-busy the composer's arc border and Stop button read. The
  // message-level `running` flag is a weaker signal: it goes false in the gaps
  // between bubbles (a sealed interim row, a settled turn the backend hasn't
  // finished with), which is exactly when the transcript used to fall silent
  // while the app still said it was working.
  const busy = useStore(view.$busy)
  const turnStartedAt = useStore(view.$turnStartedAt)
  const compacting = useStore(useMemo(() => sessionCompacting(sessionId), [sessionId]))
  const drafting = useStore(useMemo(() => sessionDraftingTool(sessionId), [sessionId]))
  const providerWait = useStore(useMemo(() => sessionProviderWait(sessionId), [sessionId]))
  // A pending clarify / approval / sudo / secret means the turn is paused on the
  // user, not working — so don't resurrect the "thinking" timer while they
  // decide (matches the pet's awaitingInput pose taking priority over busy).
  const awaitingInput = useStore(useMemo(() => sessionAwaitingInput(sessionId), [sessionId]))

  return {
    awaitingInput,
    busy,
    compacting,
    drafting,
    providerWait,
    // Epoch ms this surface's turn began, or undefined between turns. The
    // origin for anything measuring the WHOLE turn rather than one phase of
    // it — including the first seconds of a brand-new chat, where the value is
    // seeded at submit and there is no runtime session to key off yet.
    turnStartedAt: turnStartedAt ?? undefined
  }
}

// Long enough that a tool whose arguments arrive in a few frames never gets to
// strobe a label, short enough that a real wait is named almost immediately.
const DRAFTING_REVEAL_MS = 200

/**
 * What to call the wait, if it deserves a name. Compaction outranks a draft —
 * it's rarer, slower, and explains a transcript that looks like it reset.
 */
function useStatusHint(compacting: boolean, drafting: DraftingTool | null, providerWait: string): string {
  const [revealed, setRevealed] = useState(false)
  const name = drafting?.name ?? ''

  useEffect(() => {
    setRevealed(false)

    if (!name) {
      return
    }

    const id = window.setTimeout(() => setRevealed(true), DRAFTING_REVEAL_MS)

    return () => window.clearTimeout(id)
  }, [name])

  if (compacting) {
    return COMPACTION_LABEL
  }

  if (providerWait) {
    return providerWait
  }

  return revealed && name ? toolPresentVerb(name) : ''
}

export const CenteredThreadSpinner: FC = () => {
  const { t } = useI18n()

  return (
    <div
      aria-label={t.assistant.thread.loadingSession}
      className="pointer-events-none absolute inset-0 z-1 grid place-items-center"
      data-slot="thread-session-loading"
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
}

export const ResponseLoadingIndicator: FC = () => {
  const { t } = useI18n()
  const { compacting, drafting, providerWait, turnStartedAt } = useThreadSessionStatus()
  const elapsed = useElapsedSeconds(true, undefined, turnStartedAt)
  const hint = useStatusHint(compacting, drafting, providerWait)
  // Renderer-synthesized load bar: covers loads the backend's wait loop
  // can't narrate (gateway still initializing, or an auxiliary call — not
  // the main request — triggered the autoload). A real wait frame wins.
  const localLoad = useLocalModelLoad(!hint)

  return (
    <StatusRow data-slot="aui_response-loading" label={hint || t.assistant.thread.loadingResponse}>
      <StatusPulse
        aria-hidden="true"
        className="dither inline-block size-3 rounded-[2px] text-midground/80"
        kind="opacity"
      />
      {hint ? (
        <WaitHint hint={hint} />
      ) : localLoad ? (
        <ProgressHint label={t.assistant.thread.loadingLocalModel(localLoad.model)} percent={localLoad.percent} />
      ) : null}
      <ActivityTimerText seconds={elapsed} />
    </StatusRow>
  )
}

// Parked-background affordance: a top-level delegate_task runs in the
// background, so the parent turn ends and the app goes idle while the subagent
// keeps working and its result re-enters as a fresh turn later. Instead of a
// spinner (reads as "stuck"), reuse the same compact, centered system-note
// chrome as the steer / slash-status lines (SystemMessage above) so it sits in
// the thread like every other meta line. Idle-only (gated upstream). Null when
// nothing is parked.
export const BackgroundResumeNotice: FC = () => {
  const { t } = useI18n()
  const resume = useStore($backgroundResume)

  if (!resume) {
    return null
  }

  const label = resume.activity ?? t.assistant.thread.resumeWhenBackgroundDone(resume.count)

  return (
    <div
      aria-live="polite"
      className="flex max-w-[min(86%,44rem)] items-center gap-1.5 self-center px-2 py-0.5 text-[0.6875rem] leading-5 text-muted-foreground/55"
      data-slot="aui_background-resume"
      role="status"
    >
      <Codicon className="text-muted-foreground/55" name="sync" size="0.75rem" />
      <span className="shimmer min-w-0 truncate">{label}</span>
    </div>
  )
}

// Tail activity row. The pre-first-token spinner goes away once content flows,
// but a turn keeps working through gaps it produces nothing during — between
// one tool result landing and the next call arriving, while the provider
// thinks, while a sealed bubble waits on the next one. The composer's arc
// border and Stop button are lit through all of it; the transcript used to be
// silent for most of it, and those seconds went uncounted.
//
// So this row follows the SAME busy signal the composer does, and times every
// gap from the moment the turn last showed something rather than from its own
// mount. What it doesn't do is double-narrate: a tool call in flight already
// carries its own row and timer.
//
// Subscribes to the activity signal ITSELF (rather than taking it as a prop)
// so that per-token updates re-render only this leaf, not the whole
// AssistantMessage subtree.
export const TurnActivityIndicator: FC = () => {
  const { t } = useI18n()
  const activity = useAuiState(s => activitySignature(s.message.content))

  // Timestamp of the last visible progress, held from the moment the quiet
  // spell qualifies. Holding the timestamp (not a boolean) is what lets the
  // timer read "quiet for 12s" rather than the age of this component, which is
  // the whole turn so far.
  const [quietSince, setQuietSince] = useState<number | undefined>(undefined)
  const { awaitingInput, busy, compacting, drafting, providerWait, turnStartedAt } = useThreadSessionStatus()
  const hint = useStatusHint(compacting, drafting, providerWait)

  // A tool run at the tail already narrates the wait — its summary counts the
  // calls, its ticker names the current one, and it carries its own timer. A
  // second spinner under that adds a line and says nothing new. Silent tools
  // (`todo`, reactions) render nothing, so they narrate nothing.
  const toolNarrating = useAuiState(s => toolNarratesWait(s.message.content))

  // Streaming counts as working too, and it leads busy by a flush on the first
  // turn of a fresh chat — so the row can't wait for the store to catch up.
  const messageRunning = useAuiState(s => s.message.status?.type === 'running')

  // Renderer-synthesized load bar (see ResponseLoadingIndicator).
  const working = busy || messageRunning
  const localLoad = useLocalModelLoad(working && !hint && !toolNarrating)

  useEffect(() => {
    setQuietSince(undefined)
    const seenAt = Date.now()
    const id = window.setTimeout(() => setQuietSince(seenAt), TURN_QUIET_S * 1000)

    return () => window.clearTimeout(id)
  }, [activity])

  // Every second the app claims to be working belongs to something. A named
  // wait says what it is straight away; an unnamed gap has to go quiet for
  // TURN_QUIET_S first, or a run of quick calls would strobe a row between
  // each one. The two exemptions are waits already accounted for elsewhere: a
  // question the user is answering, and a tool call carrying its own timer.
  // A live local-model load is a named wait too — it must not wait out the
  // quiet window (the load IS the story from second one).
  const active =
    working && !awaitingInput && !toolNarrating && (Boolean(hint) || localLoad !== null || quietSince !== undefined)

  // Compaction owns the whole turn, so it keeps counting from the turn's start;
  // anything else counts from the moment the turn last produced something — the
  // gap's own mark, or the draft's, whichever named the wait first.
  const elapsed = useElapsedSeconds(
    active,
    undefined,
    compacting ? turnStartedAt : (quietSince ?? drafting?.since ?? turnStartedAt)
  )

  if (!active) {
    return null
  }

  return (
    <StatusRow data-slot="aui_turn-activity" label={hint || 'Hermes is working'}>
      <StatusPulse
        aria-hidden="true"
        className="dither inline-block size-3 rounded-[2px] text-midground/80"
        kind="opacity"
      />
      {hint ? (
        <WaitHint hint={hint} />
      ) : localLoad ? (
        <ProgressHint label={t.assistant.thread.loadingLocalModel(localLoad.model)} percent={localLoad.percent} />
      ) : null}
      <ActivityTimerText seconds={elapsed} />
    </StatusRow>
  )
}
