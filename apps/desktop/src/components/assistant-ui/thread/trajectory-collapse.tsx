import { useAuiState } from '@assistant-ui/react'
import { useStore } from '@nanostores/react'
import { createContext, type FC, type ReactNode, useContext, useMemo, useState } from 'react'

import { formatElapsed } from '@/components/chat/activity-timer'
import { SCAFFOLD_LABEL_CLASS, ScaffoldRow } from '@/components/chat/scaffold-row'
import { useI18n } from '@/i18n'
import { generatedImageFromResult } from '@/lib/generated-images'
import { CheckIcon } from '@/lib/icons'
import { $trajectoryCollapsedByDefault } from '@/store/trajectory-disclosure'

export type TrajectoryPart = {
  completedAt?: unknown
  result?: unknown
  text?: unknown
  timestamp?: unknown
  toolName?: unknown
  type?: string
}

export type TrajectoryPlan = { elapsedSeconds: number | null; stepCount: number }

const HideTrajectoryGroupsContext = createContext(false)

/** Whether grouped execution rows should remain mounted but be hidden until opened. */
export function useHideTrajectoryGroups() {
  return useContext(HideTrajectoryGroupsContext)
}

const asTime = (value: unknown) => (typeof value === 'number' && Number.isFinite(value) ? value : undefined)
const isVisibleReasoning = (part: TrajectoryPart) =>
  part.type === 'reasoning' && typeof part.text === 'string' && part.text.trim().length > 0
const isVisibleText = (part: TrajectoryPart) =>
  part.type === 'text' && typeof part.text === 'string' && part.text.trim().length > 0
const isToolCall = (part: TrajectoryPart) => part.type === 'tool-call'

function foldTime(part: TrajectoryPart, earliest: number | undefined, latest: number | undefined) {
  for (const value of [asTime(part.timestamp), asTime(part.completedAt)]) {
    if (value !== undefined) {
      earliest = earliest === undefined ? value : Math.min(earliest, value)
      latest = latest === undefined ? value : Math.max(latest, value)
    }
  }
  return { earliest, latest }
}

/**
 * A completed turn folds only when a visible final response follows its last
 * reasoning/tool step. Image results stay visible as deliverables, not hidden
 * inside the execution log.
 */
export function planTrajectoryCollapse(
  parts: readonly TrajectoryPart[],
  options: { fallbackElapsedSeconds?: number; messageComplete: boolean; preferenceOn: boolean }
): null | TrajectoryPlan {
  if (!options.preferenceOn || !options.messageComplete) return null
  if (
    parts.some(part => isToolCall(part) && part.toolName === 'image_generate' && generatedImageFromResult(part.result))
  ) {
    return null
  }

  let earliest: number | undefined
  let hasFinalText = false
  let lastPreliminary = -1
  let latest: number | undefined
  let stepCount = 0

  for (const [index, part] of parts.entries()) {
    if (isVisibleReasoning(part) || isToolCall(part)) {
      stepCount += 1
      lastPreliminary = index
      hasFinalText = false
      ;({ earliest, latest } = foldTime(part, earliest, latest))
    } else if (isVisibleText(part) && lastPreliminary >= 0) {
      hasFinalText = true
    }
  }

  if (!stepCount || !hasFinalText) return null
  const measured = earliest !== undefined && latest !== undefined ? latest - earliest : options.fallbackElapsedSeconds
  return {
    elapsedSeconds:
      typeof measured === 'number' && Number.isFinite(measured) ? Math.max(0, Math.round(measured)) : null,
    stepCount
  }
}

function fallbackElapsedSeconds(custom: unknown): number | undefined {
  if (!custom || typeof custom !== 'object') return undefined
  const record = custom as { durationS?: unknown; timelineCompletedAt?: unknown; timelineTimestamp?: unknown }
  if (typeof record.durationS === 'number' && Number.isFinite(record.durationS)) return record.durationS
  const started = asTime(record.timelineTimestamp)
  const ended = asTime(record.timelineCompletedAt)
  return started !== undefined && ended !== undefined ? ended - started : undefined
}

function parsePlan(signature: string): null | TrajectoryPlan {
  if (!signature) return null
  const [count, elapsed] = signature.split('\u0000')
  return { stepCount: Number(count), elapsedSeconds: elapsed === '' ? null : Number(elapsed) }
}

export const TrajectoryCollapse: FC<{ children: ReactNode }> = ({ children }) => {
  const { t } = useI18n()
  const preferenceOn = useStore($trajectoryCollapsedByDefault)
  const [userOpen, setUserOpen] = useState<boolean | null>(null)
  const signature = useAuiState(state => {
    const rawParts = state.message.parts
    const parts = (Array.isArray(rawParts) ? rawParts : state.message.content) as readonly TrajectoryPart[]
    const plan = planTrajectoryCollapse(parts, {
      fallbackElapsedSeconds: fallbackElapsedSeconds(state.message.metadata?.custom),
      messageComplete: state.message.status?.type === 'complete',
      preferenceOn
    })
    return plan ? `${plan.stepCount}\u0000${plan.elapsedSeconds ?? ''}` : ''
  })
  const plan = useMemo(() => parsePlan(signature), [signature])
  const open = userOpen ?? false
  const label =
    plan &&
    (plan.elapsedSeconds === null
      ? t.assistant.thread.completedSteps(plan.stepCount)
      : t.assistant.thread.completedStepsIn(plan.stepCount, formatElapsed(plan.elapsedSeconds)))

  return (
    <HideTrajectoryGroupsContext.Provider value={plan !== null && !open}>
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
