import { type ReactNode, useCallback, useEffect, useMemo, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import {
  deleteLearnData,
  getLearnStatus,
  pauseLearn,
  resumeLearn,
  reviewLearnSuggestions,
  startLearn,
  stopLearn,
  updateLearnConfig
} from '@/hermes'
import {
  Brain,
  Check,
  Clock,
  Loader2,
  Lock,
  NotebookTabs,
  Pause,
  Play,
  RefreshCw,
  Save,
  SlidersHorizontal,
  Sparkles,
  Square,
  Trash2
} from '@/lib/icons'
import { notify, notifyError } from '@/store/notifications'
import type { LearnConfigUpdate, LearnStatus } from '@/types/hermes'

import { Pill } from './primitives'

const PLANNED_MODES = [
  {
    title: 'Ask first',
    description: 'Hermes asks before saving memories, drafting skills, or proposing automations.'
  },
  {
    title: 'Auto-draft',
    description: 'Hermes may prepare draft memories, skills, and job suggestions for review.'
  },
  {
    title: 'Learn mode',
    description: 'Opt-in metadata analysis for recurring workflow opportunities. Nothing becomes active automatically.'
  },
  {
    title: 'Teach me this workflow',
    description: 'Explicit bounded observation of one workflow to help draft a reusable skill.'
  }
]

const GUARDRAILS = [
  'Learn only starts after explicit opt-in from this panel.',
  'Learned workflows stay inactive until the user explicitly approves them.',
  'Default collection is metadata-only and profile-scoped.',
  'Computer Use-style observation belongs only in explicit Teach Mode sessions.'
]

const STATE_LABELS: Record<LearnStatus['state'], string> = {
  paused: 'Paused',
  running: 'Running',
  stopped: 'Stopped'
}

const MODE_LABELS: Record<LearnStatus['mode'], string> = {
  learn: 'Learn mode',
  off: 'Off'
}

function parseList(value: string): string[] {
  return value
    .split(',')
    .map(item => item.trim())
    .filter(Boolean)
}

function InfoCard({
  icon: Icon,
  title,
  children
}: {
  icon: typeof Sparkles
  title: string
  children: ReactNode
}) {
  return (
    <div className="rounded-lg bg-background/55 p-3">
      <div className="mb-1.5 flex items-center gap-2 text-sm font-medium">
        <Icon className="size-4 text-muted-foreground" />
        {title}
      </div>
      <div className="text-[0.72rem] leading-relaxed text-muted-foreground">{children}</div>
    </div>
  )
}

export function LearnPanel({ onStatusChange }: { onStatusChange?: (status: LearnStatus) => void }) {
  const [status, setStatus] = useState<LearnStatus | null>(null)
  const [loading, setLoading] = useState(true)
  const [busyAction, setBusyAction] = useState<string | null>(null)
  const [allowlistText, setAllowlistText] = useState('')
  const [denylistText, setDenylistText] = useState('')
  const [retentionDaysText, setRetentionDaysText] = useState('14')

  const refresh = useCallback(async () => {
    setLoading(true)

    try {
      const next = await getLearnStatus()
      setStatus(next)
      onStatusChange?.(next)
    } catch (error) {
      notifyError(error, 'Could not load Learn status')
    } finally {
      setLoading(false)
    }
  }, [onStatusChange])

  useEffect(() => {
    void refresh()
  }, [refresh])

  useEffect(() => {
    if (!status) {
      return
    }

    setAllowlistText(status.allowlist.join(', '))
    setDenylistText(status.denylist.join(', '))
    setRetentionDaysText(String(status.retention_days))
  }, [status])

  const statusLabel = status ? STATE_LABELS[status.state] : loading ? 'Loading' : 'Unknown'
  const modeLabel = status ? MODE_LABELS[status.mode] : 'Off'
  const eventCount = status?.collected_event_count ?? 0
  const canPause = status?.state === 'running'
  const canResume = status?.state === 'paused'
  const canStop = status?.state === 'running' || status?.state === 'paused'
  const isBusy = busyAction !== null || loading
  const actionDisabled = busyAction !== null
  const statusTone = useMemo(() => (status?.state === 'running' ? 'primary' : 'muted'), [status?.state])

  async function runAction(actionName: string, action: () => Promise<LearnStatus>, successMessage: string) {
    setBusyAction(actionName)

    try {
      const next = await action()
      setStatus(next)
      onStatusChange?.(next)
      notify({ kind: 'success', message: successMessage })
    } catch (error) {
      notifyError(error, `Could not ${actionName.toLowerCase()} Learn`)
    } finally {
      setBusyAction(null)
    }
  }

  async function reviewSuggestions() {
    setBusyAction('Review')

    try {
      const result = await reviewLearnSuggestions()
      notify({
        kind: 'success',
        message:
          result.created_count === 1
            ? 'Learn created 1 suggestion'
            : `Learn created ${result.created_count} suggestions`
      })
    } catch (error) {
      notifyError(error, 'Could not review Learn suggestions')
    } finally {
      setBusyAction(null)
    }
  }

  async function saveControls() {
    const retention_days = Number.parseInt(retentionDaysText, 10)

    if (!Number.isFinite(retention_days) || retention_days < 1) {
      notifyError(new Error('Retention days must be a positive number.'), 'Could not save Learn controls')

      return
    }

    const config: LearnConfigUpdate = {
      allowlist: parseList(allowlistText),
      denylist: parseList(denylistText),
      retention_days
    }

    setBusyAction('Save controls')

    try {
      const next = await updateLearnConfig(config)
      setStatus(next)
      onStatusChange?.(next)
      notify({ kind: 'success', message: 'Learn controls saved' })
    } catch (error) {
      notifyError(error, 'Could not save Learn controls')
    } finally {
      setBusyAction(null)
    }
  }

  function confirmDeleteData() {
    if (!window.confirm('Delete collected Learn data for this Hermes profile?')) {
      return
    }

    void runAction('Delete', deleteLearnData, 'Learn data deleted')
  }

  return (
    <div className="mt-3 grid gap-3 rounded-xl bg-background/60 p-3">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="flex flex-wrap items-center gap-2">
            <Brain className="size-4 text-muted-foreground" />
            <span className="text-sm font-medium">Learn</span>
            <Pill>{modeLabel}</Pill>
            <Pill tone={statusTone}>{statusLabel}</Pill>
          </div>
          <p className="mt-1 max-w-2xl text-[0.72rem] leading-relaxed text-muted-foreground">
            Learn converts approved signals into reviewable workflow opportunities and automation suggestions. It never
            enables a learned workflow without explicit approval.
          </p>
        </div>
        <Button
          aria-label="Refresh Learn status"
          disabled={isBusy}
          onClick={() => void refresh()}
          size="sm"
          type="button"
          variant="outline"
        >
          {loading ? <Loader2 className="size-3.5 animate-spin" /> : <RefreshCw className="size-3.5" />}
          Refresh
        </Button>
      </div>

      <div className="flex flex-wrap gap-2">
        {status ? (
          <Button
            aria-label="Start Learn"
            disabled={actionDisabled || status.state === 'running'}
            onClick={() => void runAction('Start', () => startLearn('learn'), 'Learn started')}
            size="sm"
            type="button"
          >
            <Play className="size-3.5" />
            Start Learn
          </Button>
        ) : (
          <Button disabled size="sm" type="button" variant="secondary">
            <Loader2 className="size-3.5 animate-spin" />
            Loading Learn
          </Button>
        )}
        {canPause && (
          <Button
            aria-label="Pause Learn"
            disabled={actionDisabled}
            onClick={() => void runAction('Pause', pauseLearn, 'Learn paused')}
            size="sm"
            type="button"
            variant="secondary"
          >
            <Pause className="size-3.5" />
            Pause Learn
          </Button>
        )}
        {canResume && (
          <Button
            aria-label="Resume Learn"
            disabled={actionDisabled}
            onClick={() => void runAction('Resume', resumeLearn, 'Learn resumed')}
            size="sm"
            type="button"
            variant="secondary"
          >
            <Play className="size-3.5" />
            Resume Learn
          </Button>
        )}
        <Button
          aria-label="Stop Learn"
          disabled={actionDisabled || !canStop}
          onClick={() => void runAction('Stop', stopLearn, 'Learn stopped')}
          size="sm"
          type="button"
          variant="outline"
        >
          <Square className="size-3.5" />
          Stop Learn
        </Button>
        <Button
          aria-label="Review suggestions"
          disabled={actionDisabled || !status}
          onClick={() => void reviewSuggestions()}
          size="sm"
          type="button"
          variant="secondary"
        >
          <Sparkles className="size-3.5" />
          Review suggestions
        </Button>
        <Button
          aria-label="Delete Learn data"
          disabled={actionDisabled || !status}
          onClick={confirmDeleteData}
          size="sm"
          type="button"
          variant="outline"
        >
          <Trash2 className="size-3.5" />
          Delete data
        </Button>
      </div>

      <div className="grid gap-2 md:grid-cols-2">
        <InfoCard icon={Sparkles} title="Purpose">
          Wrap approved local metadata behind one review surface for high-value, low-risk workflow opportunities.
        </InfoCard>
        <InfoCard icon={Lock} title="Safety invariant">
          Learn does not automate from observation. It only prepares proposals for explicit review.
        </InfoCard>
      </div>

      <div className="grid gap-2 rounded-lg bg-background/55 p-3">
        <div className="flex items-center gap-2 text-sm font-medium">
          <Clock className="size-4 text-muted-foreground" />
          Current profile
        </div>
        <div className="grid gap-1.5 text-[0.72rem] leading-relaxed text-muted-foreground sm:grid-cols-2">
          <div>
            <span className="text-foreground">{eventCount}</span> collected metadata events
          </div>
          <div>
            <span className="text-foreground">{status?.retention_days ?? 14}</span> day retention window
          </div>
          <div className="min-w-0 sm:col-span-2">
            Storage:{' '}
            <span className="break-all font-mono text-[0.68rem]">
              {status?.storage_path ?? 'Profile-local Learn storage'}
            </span>
          </div>
        </div>
      </div>

      <div className="grid gap-3 rounded-lg bg-background/55 p-3">
        <div className="flex items-center gap-2 text-sm font-medium">
          <SlidersHorizontal className="size-4 text-muted-foreground" />
          Collection controls
        </div>
        <div className="grid gap-2 md:grid-cols-[1fr_1fr_8rem]">
          <label className="grid gap-1 text-[0.72rem] font-medium" htmlFor="learn-allowlist">
            Allowed apps or domains
            <Input
              disabled={actionDisabled || !status}
              id="learn-allowlist"
              onChange={event => setAllowlistText(event.currentTarget.value)}
              placeholder="code.exe, chrome.exe"
              size="sm"
              type="text"
              value={allowlistText}
            />
          </label>
          <label className="grid gap-1 text-[0.72rem] font-medium" htmlFor="learn-denylist">
            Blocked apps or domains
            <Input
              disabled={actionDisabled || !status}
              id="learn-denylist"
              onChange={event => setDenylistText(event.currentTarget.value)}
              placeholder="slack.exe, bank.example"
              size="sm"
              type="text"
              value={denylistText}
            />
          </label>
          <label className="grid gap-1 text-[0.72rem] font-medium" htmlFor="learn-retention-days">
            Retention days
            <Input
              disabled={actionDisabled || !status}
              id="learn-retention-days"
              min={1}
              onChange={event => setRetentionDaysText(event.currentTarget.value)}
              size="sm"
              type="number"
              value={retentionDaysText}
            />
          </label>
        </div>
        <div className="flex justify-end">
          <Button
            aria-label="Save Learn controls"
            disabled={actionDisabled || !status}
            onClick={() => void saveControls()}
            size="sm"
            type="button"
            variant="secondary"
          >
            {busyAction === 'Save controls' ? (
              <Loader2 className="size-3.5 animate-spin" />
            ) : (
              <Save className="size-3.5" />
            )}
            Save Learn controls
          </Button>
        </div>
      </div>

      <div className="grid gap-2 rounded-lg bg-background/55 p-3">
        <div className="flex items-center gap-2 text-sm font-medium">
          <NotebookTabs className="size-4 text-muted-foreground" />
          Planned modes
        </div>
        <div className="grid gap-2 md:grid-cols-2">
          {PLANNED_MODES.map(mode => (
            <div className="rounded-md bg-muted/20 p-2" key={mode.title}>
              <div className="text-[0.78rem] font-medium">{mode.title}</div>
              <p className="mt-0.5 text-[0.7rem] leading-relaxed text-muted-foreground">{mode.description}</p>
            </div>
          ))}
        </div>
      </div>

      <div className="grid gap-2 rounded-lg bg-background/55 p-3">
        <div className="flex items-center gap-2 text-sm font-medium">
          <Clock className="size-4 text-muted-foreground" />
          Guardrails
        </div>
        <div className="grid gap-1.5">
          {GUARDRAILS.map(item => (
            <div className="flex gap-2 text-[0.72rem] leading-relaxed text-muted-foreground" key={item}>
              <Check className="mt-0.5 size-3.5 shrink-0" />
              <span>{item}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
