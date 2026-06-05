import { useStore } from '@nanostores/react'
import { useEffect, useState } from 'react'

import { Button } from '@/components/ui/button'
import { writeClipboardText } from '@/components/ui/copy-button'
import { Dialog, DialogContent, DialogDescription, DialogTitle } from '@/components/ui/dialog'
import { ErrorState } from '@/components/ui/error-state'
import type { DesktopUpdateCommit, DesktopUpdateStage, DesktopUpdateStatus } from '@/global'
import { buildCommitChangelog, type CommitGroup } from '@/lib/commit-changelog'
import { AlertCircle, Check, CheckCircle2, Copy, Loader2, Sparkles, Terminal } from '@/lib/icons'
import { cn } from '@/lib/utils'
import {
  $updateApply,
  $updateChecking,
  $updateOverlayOpen,
  $updateStatus,
  applyUpdates,
  checkUpdates,
  resetUpdateApplyState,
  setUpdateOverlayOpen,
  type UpdateApplyState
} from '@/store/updates'

const STAGE_LABELS: Record<DesktopUpdateStage, string> = {
  idle: 'Getting ready…',
  prepare: 'Getting ready…',
  fetch: 'Downloading…',
  pull: 'Almost there…',
  pydeps: 'Finishing up…',
  restart: 'Restarting Hermes…',
  manual: 'Update from your terminal',
  error: 'Update paused'
}

function totalItems(groups: readonly CommitGroup[]) {
  return groups.reduce((sum, g) => sum + g.items.length, 0)
}

export function UpdatesOverlay() {
  const open = useStore($updateOverlayOpen)
  const status = useStore($updateStatus)
  const checking = useStore($updateChecking)
  const apply = useStore($updateApply)

  useEffect(() => {
    if (open && !status && !checking) {
      void checkUpdates()
    }
  }, [checking, open, status])

  const behind = status?.behind ?? 0

  const phase: 'idle' | 'applying' | 'manual' | 'error' =
    apply.stage === 'manual'
      ? 'manual'
      : apply.applying || apply.stage === 'restart'
        ? 'applying'
        : apply.stage === 'error'
          ? 'error'
          : 'idle'

  const handleClose = (next: boolean) => {
    if (phase === 'applying') {
      return
    }

    setUpdateOverlayOpen(next)

    if (!next && (apply.stage === 'error' || apply.stage === 'restart' || apply.stage === 'manual')) {
      resetUpdateApplyState()
    }
  }

  const handleInstall = () => {
    void applyUpdates()
  }

  return (
    <Dialog onOpenChange={handleClose} open={open}>
      <DialogContent
        className="max-w-sm overflow-hidden border-border/70 p-0 gap-0"
        showCloseButton={phase !== 'applying'}
      >
        {phase === 'applying' && <ApplyingView apply={apply} />}

        {phase === 'manual' && (
          <ManualView command={apply.command ?? 'hermes update'} onDone={() => handleClose(false)} />
        )}

        {phase === 'error' && (
          <ErrorView message={apply.message} onDismiss={() => handleClose(false)} onRetry={handleInstall} />
        )}

        {phase === 'idle' && (
          <IdleView
            behind={behind}
            checking={checking}
            commits={status?.commits ?? []}
            onInstall={handleInstall}
            onLater={() => handleClose(false)}
            onRetryCheck={() => void checkUpdates()}
            status={status}
          />
        )}
      </DialogContent>
    </Dialog>
  )
}

function IdleView({
  behind,
  checking,
  commits,
  onInstall,
  onLater,
  onRetryCheck,
  status
}: {
  behind: number
  checking: boolean
  commits: readonly DesktopUpdateCommit[]
  onInstall: () => void
  onLater: () => void
  onRetryCheck: () => void
  status: DesktopUpdateStatus | null
}) {
  if (!status && checking) {
    return (
      <CenteredStatus icon={<Loader2 className="size-6 animate-spin text-primary" />} title="Looking for updates…" />
    )
  }

  if (!status) {
    return (
      <CenteredStatus
        action={
          <Button onClick={onRetryCheck} size="sm">
            Try again
          </Button>
        }
        icon={<AlertCircle className="size-6 text-muted-foreground" />}
        title="Couldn’t check for updates"
      />
    )
  }

  if (!status.supported) {
    return (
      <CenteredStatus
        body={status.message ?? 'This version of Hermes can’t update itself from inside the app.'}
        icon={<AlertCircle className="size-6 text-muted-foreground" />}
        title="Update not available"
      />
    )
  }

  if (status.error) {
    return (
      <CenteredStatus
        action={
          <Button disabled={checking} onClick={onRetryCheck} size="sm">
            Try again
          </Button>
        }
        body="Check your connection and try again."
        icon={<AlertCircle className="size-6 text-muted-foreground" />}
        title="Couldn’t check for updates"
      />
    )
  }

  if (behind === 0) {
    return (
      <CenteredStatus
        body="You’re running the latest version."
        icon={<CheckCircle2 className="size-7 text-emerald-600 dark:text-emerald-400" />}
        title="You’re all set"
      />
    )
  }

  const groups = buildCommitChangelog(commits)
  const shownItems = totalItems(groups)
  const remaining = Math.max(0, behind - shownItems)

  return (
    <div className="grid gap-5 px-6 pb-6 pt-7 pr-8">
      <div className="flex flex-col items-center gap-3 text-center">
        <span className="flex size-14 items-center justify-center rounded-2xl bg-primary/10 text-primary">
          <Sparkles className="size-7" />
        </span>

        <DialogTitle className="text-center text-xl">New update available</DialogTitle>
        <DialogDescription className="text-center text-sm">
          A new version of Hermes is ready to install.
        </DialogDescription>
      </div>

      <div className="grid gap-3 rounded-xl border border-border/70 bg-muted/20 px-4 py-3">
        {groups.map(group => (
          <div key={group.id}>
            <p className="text-[0.625rem] font-semibold uppercase tracking-wide text-muted-foreground">{group.label}</p>
            <ul className="mt-1.5 grid gap-1.5 text-xs text-foreground">
              {group.items.map(item => (
                <li className="flex items-start gap-2" key={item}>
                  <span aria-hidden className="mt-1.5 inline-block size-1 shrink-0 rounded-full bg-primary" />
                  <span className="leading-snug">{item}</span>
                </li>
              ))}
            </ul>
          </div>
        ))}
      </div>

      <div className="grid gap-2">
        <Button className="font-semibold" onClick={onInstall} size="lg">
          Update now
        </Button>
        <button
          className="text-center text-sm font-medium text-muted-foreground transition-colors hover:text-foreground"
          onClick={onLater}
          type="button"
        >
          Maybe later
        </button>
      </div>

      {remaining > 0 && (
        <p className="text-center text-xs text-muted-foreground">
          + {remaining} more change{remaining === 1 ? '' : 's'} included.
        </p>
      )}
    </div>
  )
}

function ManualView({ command, onDone }: { command: string; onDone: () => void }) {
  const [copied, setCopied] = useState(false)

  const handleCopy = () => {
    void writeClipboardText(command).then(() => {
      setCopied(true)
      window.setTimeout(() => setCopied(false), 1800)
    })
  }

  return (
    <div className="grid gap-5 px-6 pb-6 pt-7 pr-8">
      <div className="flex flex-col items-center gap-3 text-center">
        <span className="flex size-14 items-center justify-center rounded-2xl bg-primary/10 text-primary">
          <Terminal className="size-7" />
        </span>

        <DialogTitle className="text-center text-xl">Update from your terminal</DialogTitle>
        <DialogDescription className="text-center text-sm">
          You installed Hermes from the command line, so updates run there too. Paste this into your terminal:
        </DialogDescription>
      </div>

      <button
        className="group flex w-full items-center justify-between gap-3 rounded-xl border border-border/70 bg-muted/30 px-4 py-3 text-left transition-colors hover:border-border hover:bg-muted/50"
        onClick={handleCopy}
        type="button"
      >
        <code className="select-all font-mono text-sm text-foreground">
          <span className="text-muted-foreground">$ </span>
          {command}
        </code>
        <span className="flex shrink-0 items-center gap-1 text-xs font-medium text-muted-foreground transition-colors group-hover:text-foreground">
          {copied ? (
            <>
              <Check className="size-3.5 text-emerald-600 dark:text-emerald-400" />
              Copied
            </>
          ) : (
            <>
              <Copy className="size-3.5" />
              Copy
            </>
          )}
        </span>
      </button>

      <p className="text-center text-xs text-muted-foreground">
        Hermes will pick up the new version next time you launch it.
      </p>

      <Button className="font-semibold" onClick={onDone} size="lg" variant="outline">
        Done
      </Button>
    </div>
  )
}

function ApplyingView({ apply }: { apply: UpdateApplyState }) {
  const label = STAGE_LABELS[apply.stage] ?? 'Updating Hermes…'

  const percent =
    typeof apply.percent === 'number' && Number.isFinite(apply.percent)
      ? Math.max(2, Math.min(100, Math.round(apply.percent)))
      : null

  return (
    <div className="grid gap-5 px-6 pb-6 pt-7">
      <div className="flex flex-col items-center gap-3 text-center">
        <span className="relative flex size-14 items-center justify-center rounded-2xl bg-primary/10 text-primary">
          <Loader2 className="size-7 animate-spin" />
        </span>

        <DialogTitle className="text-center text-xl">{label}</DialogTitle>
        <DialogDescription className="text-center text-sm">
          The Hermes updater will take over in its own window and reopen Hermes when it&rsquo;s done.
        </DialogDescription>
      </div>

      <div className="h-2 overflow-hidden rounded-full bg-muted">
        <div
          className={cn(
            'h-full rounded-full bg-primary transition-[width] duration-300 ease-out',
            percent === null && 'w-1/3 animate-pulse'
          )}
          style={percent !== null ? { width: `${percent}%` } : undefined}
        />
      </div>

      <p className="text-center text-xs text-muted-foreground">Hermes will close to apply the update.</p>
    </div>
  )
}

function ErrorView({ message, onDismiss, onRetry }: { message: string; onDismiss: () => void; onRetry: () => void }) {
  return (
    <ErrorState
      className="px-6 pb-6 pt-7 pr-8"
      description={
        <DialogDescription className="max-w-prose text-center text-sm leading-5 text-muted-foreground">
          {message || 'No worries — nothing was lost. You can try again now.'}
        </DialogDescription>
      }
      title={
        <DialogTitle className="text-center text-xl font-semibold tracking-tight">Update didn’t finish</DialogTitle>
      }
    >
      <Button className="font-semibold" onClick={onRetry} size="lg">
        Try again
      </Button>
      <Button onClick={onDismiss} variant="text">
        Not now
      </Button>
    </ErrorState>
  )
}

function CenteredStatus({
  action,
  body,
  icon,
  title
}: {
  action?: React.ReactNode
  body?: string
  icon: React.ReactNode
  title: string
}) {
  return (
    <div className="grid gap-4 px-6 pb-6 pt-8 pr-8">
      <div className="flex flex-col items-center gap-3 text-center">
        <span className="flex size-14 items-center justify-center rounded-2xl bg-muted/40">{icon}</span>

        <DialogTitle className="text-center text-lg">{title}</DialogTitle>
        {body && <DialogDescription className="text-center text-sm">{body}</DialogDescription>}
      </div>

      {action && <div className="flex justify-center">{action}</div>}
    </div>
  )
}
