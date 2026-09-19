import { useAuiState } from '@assistant-ui/react'
import { useStore } from '@nanostores/react'
import { createContext, type FC, type ReactNode, useContext, useMemo, useState } from 'react'

import { formatElapsed } from '@/components/chat/activity-timer'
import { SCAFFOLD_LABEL_CLASS, ScaffoldRow } from '@/components/chat/scaffold-row'
import { useI18n } from '@/i18n'
import { CheckIcon } from '@/lib/icons'
import { $trajectoryCollapsedByDefault } from '@/store/trajectory-disclosure'

export type TrajectoryPart = {
  completedAt?: unknown
  text?: unknown
  timestamp?: unknown
  type?: string
}

export type TrajectoryPlan = {
  elapsedSeconds: number | null
  stepCount: number
}

const HideTrajectoryGroupsContext = createContext(false)

/** True when the unified summary is collapsed and inner thought/tool groups should unmount. */
export function useHideTrajectoryGroups() {
  return useContext(HideTrajectoryGroupsContext)
}

function asTime(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined
}

function isVisibleReasoning(part: TrajectoryPart): boolean {
  return part.type === 'reasoning' && typeof part.text === 'string' && part.text.trim().length > 0
}

function isToolCall(part: TrajectoryPart): boolean {
  return part.type === 'tool-call'
}

function isVisibleText(part: TrajectoryPart): boolean {
  return part.type === 'text' && typeof part.text === 'string' && part.text.trim().length > 0
}

function foldTime(
  part: TrajectoryPart,
  earliest: number | undefined,
  latest: number | undefined
): { earliest: number | undefined; latest: number | undefined } {
  for (const value of [asTime(part.timestamp), asTime(part.completedAt)]) {
    if (value === undefined) {
      continue
    }

    earliest = earliest === undefined ? value : Math.min(earliest, value)
    latest = latest === undefined ? value : Math.max(latest, value)
  }

  return { earliest, latest }
}

/**
 * Decide whether an assistant turn's preliminary work (reasoning + tool-calls)
 * should fold into one summary. Step count is per reasoning-with-text part plus
 * per tool-call part (fail-open: tools that ChainToolFallback later hides still
 * count). Elapsed time is latest−earliest preliminary timestamp/completedAt,
 * else `fallbackElapsedSeconds`, else omitted.
 */
export function planTrajectoryCollapse(
  parts: readonly TrajectoryPart[],
  options: { fallbackElapsedSeconds?: number; preferenceOn: boolean }
): null | TrajectoryPlan {
  if (!options.preferenceOn) {
    return null
  }

  let stepCount = 0
  let lastPreliminary = -1
  let earliest: number | undefined
  let latest: number | undefined
  let hasFinalText = false

  for (let index = 0; index < parts.length; index++) {
    const part = parts[index]

    if (isVisibleReasoning(part) || isToolCall(part)) {
      stepCount += 1
      lastPreliminary = index
      hasFinalText = false
      ;({ earliest, latest } = foldTime(part, earliest, latest))
    } else if (isVisibleText(part) && lastPreliminary >= 0 && index > lastPreliminary) {
      hasFinalText = true
    }
  }

  if (stepCount === 0 || !hasFinalText) {
    return null
  }

  let elapsedSeconds: number | null = null

  if (earliest !== undefined && latest !== undefined) {
    elapsedSeconds = Math.max(0, Math.round(latest - earliest))
  } else if (typeof options.fallbackElapsedSeconds === 'number' && Number.isFinite(options.fallbackElapsedSeconds)) {
    elapsedSeconds = Math.max(0, Math.round(options.fallbackElapsedSeconds))
  }

  return { elapsedSeconds, stepCount }
}

function fallbackElapsedSeconds(custom: unknown): number | undefined {
  if (!custom || typeof custom !== 'object') {
    return undefined
  }

  const record = custom as { durationS?: unknown; timelineCompletedAt?: unknown; timelineTimestamp?: unknown }

  if (typeof record.durationS === 'number' && Number.isFinite(record.durationS)) {
    return record.durationS
  }

  const started = asTime(record.timelineTimestamp)
  const ended = asTime(record.timelineCompletedAt)

  if (started !== undefined && ended !== undefined) {
    return ended - started
  }

  return undefined
}

function parseSignature(signature: string): null | TrajectoryPlan {
  if (!signature) {
    return null
  }

  const separator = signature.indexOf('\u0000')
  const stepCount = Number(signature.slice(0, separator))
  const elapsedRaw = signature.slice(separator + 1)

  return {
    elapsedSeconds: elapsedRaw === '' ? null : Number(elapsedRaw),
    stepCount
  }
}

/**
 * Wraps `MessagePrimitive.Parts`. When the turn has preliminary work plus a
 * trailing deliverable, renders a single scaffold summary and hides thought /
 * tool groups until the header is expanded. Always mounts the same children
 * identity so streaming text does not remount the parts tree.
 */
export const TrajectoryCollapse: FC<{ children: ReactNode }> = ({ children }) => {
  const { t } = useI18n()
  const preferenceOn = useStore($trajectoryCollapsedByDefault)
  const [userOpen, setUserOpen] = useState<boolean | null>(null)

  const signature = useAuiState(s => {
    const rawParts = s.message.parts
    const parts = (Array.isArray(rawParts) ? rawParts : s.message.content) as readonly TrajectoryPart[]
    const plan = planTrajectoryCollapse(parts, {
      fallbackElapsedSeconds: fallbackElapsedSeconds(s.message.metadata?.custom),
      preferenceOn
    })

    if (!plan) {
      return ''
    }

    return `${plan.stepCount}\u0000${plan.elapsedSeconds ?? ''}`
  })

  const plan = useMemo(() => parseSignature(signature), [signature])
  const open = userOpen ?? false
  const hideGroups = plan != null && !open

  const label = plan
    ? plan.elapsedSeconds === null
      ? t.assistant.thread.completedSteps(plan.stepCount)
      : t.assistant.thread.completedStepsIn(plan.stepCount, formatElapsed(plan.elapsedSeconds))
    : null

  return (
    <HideTrajectoryGroupsContext.Provider value={hideGroups}>
      {label && (
        <div className="mb-1" data-conversation-scaffold="" data-slot="aui_trajectory-collapse">
          <ScaffoldRow onToggle={() => setUserOpen(!open)} open={open}>
            <span className="flex min-w-0 items-center gap-1">
              <CheckIcon aria-hidden="true" className="size-3 shrink-0" />
              <span className={SCAFFOLD_LABEL_CLASS}>{label}</span>
            </span>
          </ScaffoldRow>
        </div>
      )}
      {children}
    </HideTrajectoryGroupsContext.Provider>
  )
}
