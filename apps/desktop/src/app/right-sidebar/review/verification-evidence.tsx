import { useStore } from '@nanostores/react'
import { useCallback, useEffect, useRef, useState } from 'react'

import { requestComposerSubmit } from '@/app/chat/composer/focus'
import { useGatewayRequest } from '@/app/gateway/hooks/use-gateway-request'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { useI18n } from '@/i18n'
import { relativeTime } from '@/lib/time'
import { cn } from '@/lib/utils'
import { $activeSessionId, $busy, $currentCwd, $selectedStoredSessionId } from '@/store/session'
import { $workspaceChangeTick } from '@/store/workspace-events'

type VerificationStatus = 'failed' | 'loading' | 'not_applicable' | 'passed' | 'stale' | 'unknown' | 'unverified'

interface VerificationEvidence {
  canonical_command: string
  created_at: string
  exit_code: number
  kind: string
  scope: 'full' | 'targeted'
}

interface VerificationState {
  changed_paths: string[]
  evidence: VerificationEvidence | null
  status: VerificationStatus
}

interface VerificationStatusResponse {
  verification?: {
    changed_paths?: unknown
    evidence?: null | Partial<VerificationEvidence>
    status?: unknown
  }
}

const EMPTY_STATE: VerificationState = { changed_paths: [], evidence: null, status: 'loading' }
const STATUSES = new Set<VerificationStatus>(['failed', 'not_applicable', 'passed', 'stale', 'unknown', 'unverified'])

export function VerificationEvidencePanel() {
  const { requestGateway } = useGatewayRequest()
  const cwd = useStore($currentCwd)
  const storedSessionId = useStore($selectedStoredSessionId)
  const runtimeSessionId = useStore($activeSessionId)
  const busy = useStore($busy)
  const workspaceChangeTick = useStore($workspaceChangeTick)
  const [state, setState] = useState<VerificationState>(EMPTY_STATE)
  const requestSeq = useRef(0)

  // The gateway's verification ledger is keyed by the durable STORED session id,
  // not the runtime sid, so resolve stored-first with the runtime id as the
  // fallback (same rule as currentPreviewSessionId in store/preview.ts).
  const sessionId = storedSessionId || runtimeSessionId

  const refresh = useCallback(async () => {
    const seq = (requestSeq.current += 1)

    if (!cwd || !sessionId) {
      setState({ changed_paths: [], evidence: null, status: 'unverified' })

      return
    }

    try {
      const result = await requestGateway<VerificationStatusResponse>('verification.status', {
        cwd,
        session_id: sessionId
      })

      if (seq === requestSeq.current) {
        setState(normalizeVerification(result.verification))
      }
    } catch {
      if (seq === requestSeq.current) {
        setState({ changed_paths: [], evidence: null, status: 'unknown' })
      }
    }
  }, [cwd, requestGateway, sessionId])

  useEffect(() => {
    // New workspace or session: drop the old evidence and show the loading
    // state. Background refreshes (workspace tick, turn settle) keep the
    // previous content visible until the new response lands instead of
    // flashing the skeleton; the seq guard drops any stale response.
    requestSeq.current += 1
    setState(EMPTY_STATE)
  }, [cwd, sessionId])

  useEffect(() => {
    if (!busy) {
      void refresh()
    }
  }, [busy, refresh, workspaceChangeTick])

  return <VerificationEvidenceCard state={state} />
}

export function VerificationEvidenceCard({ state }: { state: VerificationState }) {
  const { t } = useI18n()
  const copy = t.statusStack.coding
  const status = statusPresentation(state.status, copy)
  const detail = verificationDetail(state, copy)
  const actionLabel = state.status === 'passed' ? copy.verificationRunAgain : copy.verificationRun

  return (
    <section
      aria-label={copy.verification}
      aria-live="polite"
      className="shrink-0 border-t border-(--ui-stroke-secondary) p-2"
      data-suppress-pane-reveal-side=""
    >
      <div className="flex min-w-0 items-center gap-2">
        <Codicon
          className={cn('shrink-0', status.iconClass)}
          name={status.icon}
          size="0.8125rem"
          spinning={state.status === 'loading'}
        />
        <span className="min-w-0 flex-1 truncate text-xs font-medium text-(--ui-text-secondary)">
          {copy.verification}
        </span>
        <span className={cn('rounded-full px-1.5 py-0.5 text-[0.625rem] font-medium', status.badgeClass)}>
          {status.label}
        </span>
      </div>

      <p className="mt-1 truncate text-[0.6875rem] text-(--ui-text-quaternary)" title={detail}>
        {detail}
      </p>

      {state.evidence?.canonical_command && (
        <code
          className="mt-1 block truncate rounded bg-(--ui-control-hover-background) px-1.5 py-1 text-[0.625rem] text-(--ui-text-tertiary)"
          title={state.evidence.canonical_command}
        >
          {state.evidence.canonical_command}
        </code>
      )}

      <Button
        className="mt-2 h-11 w-full justify-center text-xs"
        disabled={state.status === 'loading'}
        onClick={() => requestComposerSubmit(copy.verificationPrompt, { target: 'main' })}
        size="sm"
        variant={state.status === 'passed' ? 'ghost' : 'outline'}
      >
        <Codicon name="run-all" size="0.8125rem" />
        {actionLabel}
      </Button>
    </section>
  )
}

function normalizeVerification(value: VerificationStatusResponse['verification']): VerificationState {
  const receivedStatus =
    typeof value?.status === 'string' && STATUSES.has(value.status as VerificationStatus) ? value.status : 'unknown'

  const evidence = normalizeEvidence(value?.evidence)

  const status =
    (receivedStatus === 'failed' || receivedStatus === 'passed' || receivedStatus === 'stale') && !evidence
      ? 'unknown'
      : receivedStatus

  const changedPaths = Array.isArray(value?.changed_paths)
    ? value.changed_paths.filter((path): path is string => typeof path === 'string')
    : []

  return { changed_paths: changedPaths, evidence, status: status as VerificationStatus }
}

function normalizeEvidence(value: null | Partial<VerificationEvidence> | undefined): VerificationEvidence | null {
  if (
    !value ||
    typeof value.canonical_command !== 'string' ||
    typeof value.created_at !== 'string' ||
    typeof value.exit_code !== 'number' ||
    typeof value.kind !== 'string' ||
    (value.scope !== 'full' && value.scope !== 'targeted')
  ) {
    return null
  }

  return {
    canonical_command: value.canonical_command,
    created_at: value.created_at,
    exit_code: value.exit_code,
    kind: value.kind,
    scope: value.scope
  }
}

function statusPresentation(
  status: VerificationStatus,
  copy: ReturnType<typeof useI18n>['t']['statusStack']['coding']
) {
  if (status === 'passed') {
    return {
      badgeClass: 'bg-emerald-500/10 text-emerald-600 dark:text-emerald-400',
      icon: 'check' as const,
      iconClass: 'text-emerald-500',
      label: copy.verificationPassed
    }
  }

  if (status === 'failed') {
    return {
      badgeClass: 'bg-destructive/10 text-destructive',
      icon: 'error' as const,
      iconClass: 'text-destructive',
      label: copy.verificationFailed
    }
  }

  if (status === 'stale') {
    return {
      badgeClass: 'bg-amber-500/10 text-amber-600 dark:text-amber-400',
      icon: 'warning' as const,
      iconClass: 'text-amber-500',
      label: copy.verificationStale
    }
  }

  return {
    badgeClass: 'bg-(--ui-control-hover-background) text-(--ui-text-tertiary)',
    icon: status === 'loading' ? ('loading' as const) : ('circle-outline' as const),
    iconClass: 'text-(--ui-text-quaternary)',
    label:
      status === 'loading'
        ? copy.verificationChecking
        : status === 'unverified'
          ? copy.verificationUnverified
          : copy.verificationUnavailable
  }
}

function verificationDetail(
  state: VerificationState,
  copy: ReturnType<typeof useI18n>['t']['statusStack']['coding']
): string {
  if (state.status === 'loading') {
    return copy.verificationCheckingDetail
  }

  if (state.status === 'unverified') {
    return copy.verificationNoEvidence
  }

  if (state.status === 'unknown' || state.status === 'not_applicable') {
    return copy.verificationUnavailableDetail
  }

  if (state.status === 'stale') {
    return state.changed_paths.length > 0
      ? copy.verificationPathsChanged(state.changed_paths.length)
      : copy.verificationChangesAfterCheck
  }

  const evidence = state.evidence

  if (!evidence) {
    return copy.verificationUnavailableDetail
  }

  const createdAt = Date.parse(evidence.created_at)
  const age = Number.isNaN(createdAt) ? '' : relativeTime(createdAt)

  const prefix =
    state.status === 'failed'
      ? copy.verificationExit(evidence.exit_code)
      : evidence.scope === 'full'
        ? copy.verificationFull
        : copy.verificationTargeted

  return age ? `${prefix} · ${age}` : prefix
}
