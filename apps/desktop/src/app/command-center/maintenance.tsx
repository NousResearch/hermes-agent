import { useCallback, useEffect, useState } from 'react'

import { PageLoader } from '@/components/page-loader'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import {
  type ActionResponse,
  type CuratorStatusResponse,
  type DebugShareResponse,
  getActionStatus,
  getCuratorStatus,
  getMemoryStatus,
  type MemoryStatusResponse,
  resetMemory,
  runBackup,
  runCurator,
  runDebugShare,
  runDoctor,
  runSecurityAudit,
  setCuratorPaused
} from '@/hermes'
import { useI18n } from '@/i18n'
import { AlertCircle } from '@/lib/icons'
import { cn } from '@/lib/utils'
import { upsertDesktopActionTask } from '@/store/activity'
import { notify, notifyError } from '@/store/notifications'
import type { ActionStatusResponse } from '@/types/hermes'

const ACTION_POLL_MS = 1200
const ACTION_POLL_LIMIT = 240 // ~5 minutes of polling before giving up.

function formatBytes(size: number): string {
  if (size <= 0) {
    return ''
  }

  if (size >= 1024 * 1024) {
    return `${(size / (1024 * 1024)).toFixed(1)} MB`
  }

  if (size >= 1024) {
    return `${(size / 1024).toFixed(1)} KB`
  }

  return `${size} B`
}

/** Maintenance panel — desktop parity for `hermes doctor` / `security audit` /
 *  `backup` / `debug share` / `curator` / `memory` (the dashboard System page's
 *  ops section). Spawn-based actions tail their logs inline via the shared
 *  /api/actions status endpoint. */
export function MaintenancePanel() {
  const { t } = useI18n()
  const mm = t.commandCenter.maintenance

  const [actionName, setActionName] = useState<null | string>(null)
  const [actionStatus, setActionStatus] = useState<ActionStatusResponse | null>(null)
  const [curator, setCurator] = useState<CuratorStatusResponse | null>(null)
  const [curatorBusy, setCuratorBusy] = useState(false)
  const [memory, setMemory] = useState<MemoryStatusResponse | null>(null)
  const [memoryBusy, setMemoryBusy] = useState(false)
  const [share, setShare] = useState<DebugShareResponse | null>(null)
  const [sharing, setSharing] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    let cancelled = false

    getCuratorStatus()
      .then(next => !cancelled && setCurator(next))
      .catch(() => {})
    getMemoryStatus()
      .then(next => !cancelled && setMemory(next))
      .catch(() => {})

    return () => void (cancelled = true)
  }, [])

  // Tail the most recently launched spawn action.
  useEffect(() => {
    if (!actionName) {
      return
    }

    let cancelled = false
    let polls = 0
    let timer: null | number = null

    const poll = async () => {
      try {
        const status = await getActionStatus(actionName, 200)

        if (cancelled) {
          return
        }

        setActionStatus(status)
        upsertDesktopActionTask(status)
        polls += 1

        if (status.running && polls < ACTION_POLL_LIMIT) {
          timer = window.setTimeout(() => void poll(), ACTION_POLL_MS)
        }
      } catch {
        // Status endpoint hiccup — stop tailing; the activity rail still has the task.
      }
    }

    void poll()

    return () => {
      cancelled = true

      if (timer !== null) {
        window.clearTimeout(timer)
      }
    }
  }, [actionName])

  const launch = useCallback(
    async (label: string, start: () => Promise<ActionResponse>) => {
      setError('')

      try {
        const started = await start()
        setActionStatus(null)
        setActionName(started.name)
        notify({ kind: 'success', title: mm.actionStarted(label), message: '' })
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err))
        notifyError(err, mm.actionFailed(label))
      }
    },
    [mm]
  )

  const shareDebug = useCallback(async () => {
    setSharing(true)
    setShare(null)
    setError('')

    try {
      setShare(await runDebugShare())
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
      notifyError(err, mm.debugShareFailed)
    } finally {
      setSharing(false)
    }
  }, [mm])

  const toggleCurator = useCallback(async () => {
    if (!curator) {
      return
    }

    setCuratorBusy(true)

    try {
      const next = !curator.paused
      await setCuratorPaused(next)
      setCurator({ ...curator, paused: next })
    } catch (err) {
      notifyError(err, mm.actionFailed(mm.curator))
    } finally {
      setCuratorBusy(false)
    }
  }, [curator, mm])

  const doResetMemory = useCallback(
    async (target: 'all' | 'memory' | 'user', label: string) => {
      if (!window.confirm(mm.resetConfirm(label))) {
        return
      }

      setMemoryBusy(true)

      try {
        const result = await resetMemory(target)
        notify({ kind: 'success', title: mm.resetDone(result.deleted.join(', ') || label), message: '' })
        setMemory(await getMemoryStatus())
      } catch (err) {
        notifyError(err, mm.resetFailed)
      } finally {
        setMemoryBusy(false)
      }
    },
    [mm]
  )

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-5 overflow-y-auto pb-2">
      {error && (
        <span className="inline-flex items-center gap-1 text-[length:var(--conversation-caption-font-size)] text-destructive">
          <AlertCircle className="size-3.5" />
          {error}
        </span>
      )}

      <section>
        <SectionLabel>{mm.runOps}</SectionLabel>
        <OpRow
          description={mm.doctorDesc}
          disabled={actionStatus?.running === true}
          label={mm.doctor}
          onRun={() => void launch(mm.doctor, runDoctor)}
        />
        <OpRow
          description={mm.securityAuditDesc}
          disabled={actionStatus?.running === true}
          label={mm.securityAudit}
          onRun={() => void launch(mm.securityAudit, runSecurityAudit)}
        />
        <OpRow
          description={mm.backupDesc}
          disabled={actionStatus?.running === true}
          label={mm.backup}
          onRun={() => void launch(mm.backup, runBackup)}
        />
        <OpRow
          description={mm.debugShareDesc}
          disabled={sharing}
          label={sharing ? mm.debugShareRunning : mm.debugShare}
          onRun={() => void shareDebug()}
        />

        {share && Object.keys(share.urls).length > 0 && (
          <div className="mt-2 rounded-lg border border-(--ui-stroke-tertiary) bg-(--ui-bg-quinary) p-3">
            <div className="mb-1.5 text-[0.68rem] font-semibold uppercase tracking-[0.12em] text-muted-foreground">
              {mm.debugShareLinks}
            </div>
            {Object.entries(share.urls).map(([key, url]) => (
              <div className="flex items-center justify-between gap-2 py-1" key={key}>
                <span className="min-w-0 truncate font-mono text-[0.7rem]">
                  {key}: {url}
                </span>
                <Button
                  onClick={() => {
                    void window.hermesDesktop.writeClipboard(url)
                    notify({ durationMs: 1500, kind: 'success', message: mm.linkCopied })
                  }}
                  size="xs"
                  variant="text"
                >
                  {mm.copyLink}
                </Button>
              </div>
            ))}
          </div>
        )}

        {actionStatus && (
          <div className="mt-2">
            <div className="mb-1.5 flex items-center gap-2 text-[0.68rem] font-semibold uppercase tracking-[0.12em] text-muted-foreground">
              {mm.viewLog}
              {actionStatus.running && <span className="normal-case tracking-normal">{mm.running}</span>}
            </div>
            <pre
              className="max-h-48 overflow-auto whitespace-pre-wrap wrap-break-word rounded-lg border border-(--ui-stroke-tertiary) bg-(--ui-bg-quinary) p-3 font-mono text-[0.65rem] leading-relaxed text-(--ui-text-tertiary)"
              data-selectable-text="true"
            >
              {actionStatus.lines.join('\n')}
            </pre>
          </div>
        )}
      </section>

      <section>
        <SectionLabel>{mm.curator}</SectionLabel>
        {!curator ? (
          <PageLoader className="min-h-16" label={mm.curator} />
        ) : (
          <div className="flex items-center justify-between gap-3 py-2">
            <div className="min-w-0">
              <div className="flex items-center gap-2">
                <span className="text-[length:var(--conversation-text-font-size)] font-medium">{mm.curator}</span>
                <Badge
                  className={cn(
                    !curator.enabled
                      ? 'bg-(--ui-bg-quinary) text-(--ui-text-tertiary)'
                      : curator.paused
                        ? 'bg-amber-500/15 text-amber-400'
                        : 'bg-emerald-500/15 text-emerald-400'
                  )}
                >
                  {!curator.enabled ? mm.curatorDisabled : curator.paused ? mm.curatorPaused : mm.curatorActive}
                </Badge>
              </div>
              <div className="mt-0.5 text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
                {mm.curatorDesc}
                {' · '}
                {curator.last_run_at ? mm.curatorLastRun(curator.last_run_at) : mm.curatorNeverRan}
              </div>
            </div>
            <div className="flex shrink-0 items-center gap-1.5">
              {curator.enabled && (
                <Button disabled={curatorBusy} onClick={() => void toggleCurator()} size="xs" variant="text">
                  {curator.paused ? mm.resume : mm.pause}
                </Button>
              )}
              <Button
                disabled={actionStatus?.running === true}
                onClick={() => void launch(mm.curator, runCurator)}
                size="xs"
                variant="textStrong"
              >
                {mm.runNow}
              </Button>
            </div>
          </div>
        )}
      </section>

      <section>
        <SectionLabel>{mm.memoryData}</SectionLabel>
        {!memory ? (
          <PageLoader className="min-h-16" label={mm.memoryData} />
        ) : (
          <div>
            <div className="py-1 text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
              {mm.memoryDataDesc}
              {' · '}
              {mm.memoryProvider(memory.active || mm.builtinMemory)}
            </div>
            <MemoryFileRow
              busy={memoryBusy}
              label={mm.memoryFile}
              onReset={() => void doResetMemory('memory', mm.memoryFile)}
              resetLabel={mm.resetMemory}
              size={memory.builtin_files.memory}
              sizeLabel={memory.builtin_files.memory > 0 ? formatBytes(memory.builtin_files.memory) : mm.empty}
            />
            <MemoryFileRow
              busy={memoryBusy}
              label={mm.userFile}
              onReset={() => void doResetMemory('user', mm.userFile)}
              resetLabel={mm.resetUser}
              size={memory.builtin_files.user}
              sizeLabel={memory.builtin_files.user > 0 ? formatBytes(memory.builtin_files.user) : mm.empty}
            />
          </div>
        )}
      </section>
    </div>
  )
}

function SectionLabel({ children }: { children: string }) {
  return (
    <div className="mb-1.5 text-[0.625rem] font-medium uppercase tracking-[0.08em] text-(--ui-text-tertiary)">
      {children}
    </div>
  )
}

function OpRow({
  description,
  disabled,
  label,
  onRun
}: {
  description: string
  disabled?: boolean
  label: string
  onRun: () => void
}) {
  return (
    <div className="flex items-center justify-between gap-3 py-2">
      <div className="min-w-0">
        <div className="text-[length:var(--conversation-text-font-size)] font-medium">{label}</div>
        <div className="mt-0.5 text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
          {description}
        </div>
      </div>
      <Button disabled={disabled} onClick={onRun} size="xs" variant="textStrong">
        {label}
      </Button>
    </div>
  )
}

function MemoryFileRow({
  busy,
  label,
  onReset,
  resetLabel,
  size,
  sizeLabel
}: {
  busy: boolean
  label: string
  onReset: () => void
  resetLabel: string
  size: number
  sizeLabel: string
}) {
  return (
    <div className="flex items-center justify-between gap-3 py-2">
      <div className="min-w-0">
        <span className="text-[length:var(--conversation-text-font-size)] font-medium">{label}</span>
        <span className="ml-2 text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
          {sizeLabel}
        </span>
      </div>
      <Button
        className="text-destructive hover:text-destructive"
        disabled={busy || size <= 0}
        onClick={onReset}
        size="xs"
        variant="text"
      >
        {resetLabel}
      </Button>
    </div>
  )
}
