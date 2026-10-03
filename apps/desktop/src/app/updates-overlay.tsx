import { useStore } from '@nanostores/react'
import { useEffect, useState } from 'react'

import { BrandMark } from '@/components/brand-mark'
import { SyncStatusCard } from '@/components/sync-status-card'
import { Button } from '@/components/ui/button'
import { CopyButton, writeClipboardText } from '@/components/ui/copy-button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogTitle,
  preventCloseButtonAutoFocus
} from '@/components/ui/dialog'
import { ErrorIcon, ErrorState } from '@/components/ui/error-state'
import { Loader } from '@/components/ui/loader'
import { Progress } from '@/components/ui/progress'
import { UpdateStatusCard, VersionHero } from '@/components/update-status'
import { VersionDetails } from '@/components/version-details'
import type {
  DesktopUpdateCommit,
  DesktopUpdateStage,
  DesktopUpdateStatus,
  DesktopVersionInfo,
  UpdaterMechanismClient
} from '@/global'
import { runDebugShare, type DebugShareResponse } from '@/hermes'
import { useI18n } from '@/i18n'
import { buildCommitChangelog, type CommitGroup, formatFullChangelogText } from '@/lib/commit-changelog'
import { AlertCircle, Check, Copy, Terminal } from '@/lib/icons'
import { resolveUpdateCopy, type UpdateTarget } from '@/lib/update-copy'
import { cn } from '@/lib/utils'
import {
  $backendUpdateApply,
  $backendUpdateChecking,
  $backendUpdateStatus,
  $desktopVersion,
  $updateApply,
  $updateChecking,
  $updateOverlayOpen,
  $updateOverlayTarget,
  $updateStatus,
  applyBackendUpdate,
  applyUpdates,
  checkBackendUpdates,
  checkUpdates,
  dismissDiscontinuedNotice,
  resetUpdateApplyState,
  setUpdateOverlayOpen,
  type UpdateApplyState
} from '@/store/updates'

import { DiscontinuedNotice } from './retirement-view'

function totalItems(groups: readonly CommitGroup[]) {
  return groups.reduce((sum, g) => sum + g.items.length, 0)
}

export function UpdatesOverlay() {
  const open = useStore($updateOverlayOpen)
  const target = useStore($updateOverlayTarget)

  const clientStatus = useStore($updateStatus)
  const clientChecking = useStore($updateChecking)
  const clientApply = useStore($updateApply)
  const backendStatus = useStore($backendUpdateStatus)
  const backendChecking = useStore($backendUpdateChecking)
  const backendApply = useStore($backendUpdateApply)
  const desktopVersion = useStore($desktopVersion)

  const isBackend = target === 'backend'
  const status = isBackend ? backendStatus : clientStatus
  const checking = isBackend ? backendChecking : clientChecking
  const apply = isBackend ? backendApply : clientApply
  const check = isBackend ? checkBackendUpdates : checkUpdates
  const install = isBackend ? applyBackendUpdate : applyUpdates

  useEffect(() => {
    if (open && !status && !checking) {
      void check()
    }
  }, [check, checking, open, status])

  const behind = status?.behind ?? 0
  const updateAvailable = status?.updateAvailable || behind > 0

  const phase: 'idle' | 'applying' | 'manual' | 'guiSkew' | 'error' =
    apply.stage === 'manual'
      ? 'manual'
      : apply.stage === 'guiSkew'
        ? 'guiSkew'
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

    if (
      !next &&
      (apply.stage === 'error' || apply.stage === 'restart' || apply.stage === 'manual' || apply.stage === 'guiSkew')
    ) {
      resetUpdateApplyState()
    }
  }

  const handleInstall = () => {
    void install()
  }

  return (
    <Dialog onOpenChange={handleClose} open={open}>
      <DialogContent
        bodyClassName="overflow-hidden p-0 gap-0"
        className="max-w-sm"
        onOpenAutoFocus={preventCloseButtonAutoFocus}
        showCloseButton={phase !== 'applying'}
      >
        {phase === 'applying' && (
          <ApplyingView apply={apply} isBackend={isBackend} statusMechanism={status?.mechanism} />
        )}

        {phase === 'manual' && (
          <ManualView
            command={apply.command ?? null}
            isBackend={isBackend}
            message={apply.message}
            onDone={() => handleClose(false)}
          />
        )}

        {phase === 'guiSkew' && <GuiSkewView message={apply.message} onDone={() => handleClose(false)} />}

        {phase === 'error' ? (
          <ErrorView message={apply.message} onDismiss={() => handleClose(false)} onRetry={handleInstall} />
        ) : null}

        {phase === 'idle' && !isBackend && status?.retirement?.state === 'discontinued' && (
          <DiscontinuedNotice
            onDismiss={() => {
              dismissDiscontinuedNotice(status.retirement!)
              handleClose(false)
            }}
            retirement={status.retirement}
          />
        )}
        {phase === 'idle' && (isBackend || !status?.retirement) && (
          <IdleView
            behind={behind}
            checking={checking}
            commits={status?.commits ?? []}
            onInstall={handleInstall}
            onLater={() => handleClose(false)}
            onRetryCheck={() => void check()}
            status={status}
            target={target}
            updateAvailable={updateAvailable}
            version={desktopVersion}
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
  status,
  target,
  updateAvailable,
  version
}: {
  behind: number
  checking: boolean
  commits: readonly DesktopUpdateCommit[]
  onInstall: () => void
  onLater: () => void
  onRetryCheck: () => void
  status: DesktopUpdateStatus | null
  target: UpdateTarget
  updateAvailable: boolean
  version: DesktopVersionInfo | null
}) {
  const { t } = useI18n()
  const u = t.updates

  if (!status && checking) {
    return (
      <CenteredStatus
        icon={<Loader className="size-12" label={u.checking} type="lemniscate-bloom" />}
        title={u.checking}
      />
    )
  }

  if (!status) {
    return (
      <CenteredStatus
        action={
          <Button onClick={onRetryCheck} size="sm">
            {u.tryAgain}
          </Button>
        }
        icon={<ErrorIcon />}
        title={u.checkFailedTitle}
      />
    )
  }

  const details = version ? <VersionDetails version={version} /> : null

  if (status.mechanism === 'app-installer' && status.error && !status.updateAvailable) {
    return (
      <div className="grid gap-4 px-6 pb-6 pt-1 pr-8">
        <VersionHero
          renderHeading={heading => (
            <DialogTitle className="text-lg font-semibold tracking-tight">{heading}</DialogTitle>
          )}
          version={version}
        />
        <div className="flex flex-col items-center gap-2 text-center">
          <AlertCircle className="size-6 text-muted-foreground" />
          <p className="text-sm font-medium">{u.checkUnknownTitleAppInstaller}</p>
          <p className="max-w-prose text-sm leading-5 text-muted-foreground">{u.checkUnknownBodyAppInstaller}</p>
        </div>
        <Button onClick={onRetryCheck} size="sm">
          {u.tryAgain}
        </Button>
        {details}
      </div>
    )
  }

  if (!status.supported || status.error || !updateAvailable) {
    return (
      <div className="grid gap-4 px-6 pb-6 pt-1 pr-8">
        <VersionHero
          renderHeading={heading => (
            <DialogTitle className="text-lg font-semibold tracking-tight">{heading}</DialogTitle>
          )}
          version={version}
        />
        <UpdateStatusCard target={target} />
        <SyncStatusCard />
        {details}
      </div>
    )
  }

  const groups = buildCommitChangelog(commits, {
    labels: {
      new: u.changeLogNew,
      fixed: u.changeLogFixed,
      faster: u.changeLogFaster,
      improved: u.changeLogImproved,
      other: u.changeLogOther
    },
    fallback: { label: u.changeLogFallbackLabel, item: u.changeLogFallbackItem }
  })

  const shownItems = totalItems(groups)
  const remaining = Math.max(0, behind - shownItems)

  const { title, body } = resolveUpdateCopy({
    target,
    shownItems,
    channel: status?.channel === 'stable' ? 'stable' : 'main',
    latestTag: status?.latestTag ?? null,
    mechanism: status?.mechanism,
    copy: u
  })

  const handleCopyFullLog = () => formatFullChangelogText(commits, behind, status.branch)

  return (
    <div className="grid gap-5 px-6 pb-6 pt-7 pr-8">
      <div className="flex flex-col items-center gap-3 text-center">
        <BrandMark className="size-16" />
        <DialogTitle className="text-center text-xl">{title}</DialogTitle>
        <DialogDescription className="text-center text-sm">{body}</DialogDescription>
      </div>

      <div className="grid gap-3">
        {groups.map((group, index) => (
          <div key={group.id}>
            <div className="flex items-center justify-between gap-2">
              <p className="text-[0.625rem] font-semibold text-muted-foreground">{group.label}</p>
              {index === 0 && commits.length > 0 && (
                <CopyButton
                  appearance="icon"
                  buttonSize="icon-xs"
                  className="-my-1 size-5 shrink-0 text-muted-foreground/70 hover:text-foreground"
                  iconClassName="size-3"
                  label={u.copyFullLog}
                  side="left"
                  text={handleCopyFullLog}
                />
              )}
            </div>
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
          {u.updateNow}
        </Button>
        <Button className="font-medium" onClick={onLater} type="button" variant="text">
          {u.maybeLater}
        </Button>
      </div>

      {remaining > 0 && <p className="text-center text-xs text-muted-foreground">{u.moreChanges(remaining)}</p>}
      <SyncStatusCard />
    </div>
  )
}

function ManualView({
  command,
  isBackend,
  message,
  onDone
}: {
  command: string | null
  isBackend: boolean
  message?: string
  onDone: () => void
}) {
  const { t } = useI18n()
  const u = t.updates
  const [copied, setCopied] = useState(false)
  const guidance: string | undefined = message && message !== command ? message : undefined

  const handleCopy = () => {
    if (!command) {
      return
    }

    void writeClipboardText(command).then(() => {
      setCopied(true)
      window.setTimeout(() => setCopied(false), 1800)
    })
  }

  if (!command) {
    return (
      <div className="grid gap-5 px-6 pb-6 pt-7 pr-8">
        <div className="flex flex-col items-center gap-3 text-center">
          <Terminal className="size-8 text-primary" />
          <DialogTitle className="text-center text-xl">
            {isBackend ? u.manualUnavailableTitle : u.manualTitle}
          </DialogTitle>
          <DialogDescription className="text-center text-sm">{message || u.manualPickedUp}</DialogDescription>
        </div>
        <Button className="font-semibold" onClick={onDone} size="lg" variant="secondary">
          {u.done}
        </Button>
      </div>
    )
  }

  return (
    <div className="grid gap-5 px-6 pb-6 pt-7 pr-8">
      <div className="flex flex-col items-center gap-3 text-center">
        <Terminal className="size-8 text-primary" />
        <DialogTitle className="text-center text-xl">{u.manualTitle}</DialogTitle>
        <DialogDescription className="text-center text-sm">
          {guidance ?? (isBackend ? u.manualBodyBackend : u.manualBody)}
        </DialogDescription>
      </div>

      <button
        className={cn(
          'group flex w-full items-center justify-between gap-3 rounded-md border px-4 py-3 text-left transition-colors',
          copied ? 'border-primary/50' : 'border-(--stroke-nous) hover:border-(--ui-stroke-secondary)'
        )}
        onClick={handleCopy}
        type="button"
      >
        <code className="min-w-0 flex-1 truncate select-all font-mono text-sm text-foreground">
          <span className="select-none text-muted-foreground">$ </span>
          {command}
        </code>
        <span
          className={cn(
            'flex shrink-0 items-center gap-1 text-xs font-medium transition-colors',
            copied ? 'text-primary' : 'text-muted-foreground group-hover:text-foreground'
          )}
        >
          {copied ? <Check className="size-3.5" /> : <Copy className="size-3.5" />}
          {copied ? u.copied : u.copy}
        </span>
      </button>

      {!guidance && (
        <p className="text-center text-xs text-muted-foreground">
          {isBackend ? u.manualPickedUpBackend : u.manualPickedUp}
        </p>
      )}

      <Button className="font-semibold" onClick={onDone} size="lg" variant="secondary">
        {u.done}
      </Button>
    </div>
  )
}

function GuiSkewView({ message, onDone }: { message?: string; onDone: () => void }) {
  const { t } = useI18n()
  const u = t.updates

  return (
    <div className="grid gap-5 px-6 pb-6 pt-7 pr-8">
      <div className="flex flex-col items-center gap-3 text-center">
        <AlertCircle className="size-8 text-amber-500" />
        <DialogTitle className="text-center text-xl">{u.guiSkewTitle}</DialogTitle>
        <DialogDescription className="max-w-prose text-center text-sm leading-5 text-muted-foreground">
          {message || u.guiSkewBody}
        </DialogDescription>
      </div>
      <Button className="font-semibold" onClick={onDone} size="lg" variant="secondary">
        {u.done}
      </Button>
    </div>
  )
}

function ApplyingView({
  apply,
  isBackend,
  statusMechanism
}: {
  apply: UpdateApplyState
  isBackend: boolean
  statusMechanism?: UpdaterMechanismClient
}) {
  const { t } = useI18n()
  const u = t.updates
  const label = u.stages[apply.stage as DesktopUpdateStage] ?? u.stages.idle
  const isWindowsPackage = statusMechanism === 'app-installer' || statusMechanism === 'microsoft-store'
  const body = isWindowsPackage ? u.applyingBodyAppInstaller : isBackend ? u.applyingBodyBackend : u.applyingBody
  const currentMessage = apply.message.trim()
  const recentLog = apply.log.slice(-4)
  const percent =
    typeof apply.percent === 'number' && Number.isFinite(apply.percent)
      ? Math.max(2, Math.min(100, Math.round(apply.percent)))
      : null

  return (
    <div className="grid gap-5 px-6 pb-6 pt-7">
      <div className="flex flex-col items-center gap-3 text-center">
        <Loader className="size-16" label={label} type="lemniscate-bloom" />
        <DialogTitle className="text-center text-xl">{label}</DialogTitle>
        <DialogDescription className="text-center text-sm">{body}</DialogDescription>
        {currentMessage ? (
          <p className="max-w-lg break-words text-center text-xs leading-5 text-muted-foreground">{currentMessage}</p>
        ) : null}
      </div>
      <Progress
        aria-label={label}
        indeterminate={percent === null}
        size="lg"
        value={percent === null ? 0 : percent / 100}
      />
      {recentLog.length > 1 ? (
        <div className="max-h-24 overflow-hidden rounded-md border border-border/70 bg-muted/35 px-3 py-2 text-left font-mono text-[11px] leading-4 text-muted-foreground">
          {recentLog.map((entry, index) => (
            <div className="truncate" key={`${entry.at}-${index}`}>
              {entry.message}
            </div>
          ))}
        </div>
      ) : null}
      <p className="text-center text-xs text-muted-foreground">{u.applyingClose}</p>
    </div>
  )
}

function ErrorView({ message, onDismiss, onRetry }: { message: string; onDismiss: () => void; onRetry: () => void }) {
  const { t } = useI18n()
  const u = t.updates
  const maintenance = t.commandCenter.maintenance
  const [share, setShare] = useState<DebugShareResponse | null>(null)
  const [sharing, setSharing] = useState(false)
  const [shareError, setShareError] = useState<string | null>(null)

  const handleDebugShare = async () => {
    setSharing(true)
    setShare(null)
    setShareError(null)

    try {
      const result = await runDebugShare()
      setShare(result)
      if (!result.ok || Object.keys(result.failures).length > 0) {
        setShareError(maintenance.debugShareFailed)
      }
    } catch (error) {
      setShareError(error instanceof Error ? error.message : String(error))
    } finally {
      setSharing(false)
    }
  }

  return (
    <ErrorState
      className="px-6 pb-6 pt-7 pr-8"
      description={
        <DialogDescription className="max-w-prose text-center text-sm leading-5 text-muted-foreground">
          {message || u.errorBody}
        </DialogDescription>
      }
      title={<DialogTitle className="text-center text-xl font-semibold tracking-tight">{u.errorTitle}</DialogTitle>}
    >
      <Button className="font-semibold" onClick={onRetry} size="lg">
        {u.tryAgain}
      </Button>
      <div className="grid gap-2">
        <Button disabled={sharing} onClick={() => void handleDebugShare()} size="lg" variant="secondary">
          {sharing ? <Loader className="size-4" label={maintenance.debugShareRunning} /> : null}
          {sharing ? maintenance.debugShareRunning : maintenance.debugShare}
        </Button>
        <p className="text-center text-xs leading-4 text-muted-foreground">{maintenance.debugShareDesc}</p>
      </div>
      {shareError ? (
        <div
          className="rounded-md border border-destructive/30 bg-destructive/10 px-3 py-2 text-xs text-destructive"
          role="alert"
        >
          {maintenance.debugShareFailed}: {shareError}
        </div>
      ) : null}
      {share && Object.keys(share.urls).length > 0 ? (
        <div
          aria-live="polite"
          className="grid gap-1 rounded-lg border border-(--ui-stroke-tertiary) bg-(--ui-bg-quinary) p-3 text-left"
        >
          <div className="mb-1 text-[0.68rem] font-semibold uppercase tracking-[0.12em] text-muted-foreground">
            {maintenance.debugShareLinks}
          </div>
          <div className="max-h-36 overflow-y-auto">
            {Object.entries(share.urls).map(([label, url]) => (
              <div className="flex min-w-0 items-center gap-2 py-1" key={label}>
                <span className="w-20 shrink-0 truncate text-xs text-muted-foreground">{label}</span>
                <code className="min-w-0 flex-1 truncate font-mono text-xs" title={url}>
                  {url}
                </code>
                <CopyButton
                  appearance="icon"
                  buttonSize="icon-xs"
                  haptic={false}
                  label={maintenance.copyLink}
                  text={url}
                />
              </div>
            ))}
          </div>
          {Object.keys(share.failures).length > 0 ? (
            <p className="mt-1 text-xs text-muted-foreground" role="status">
              {maintenance.debugShareFailed}: {Object.keys(share.failures).join(', ')}
            </p>
          ) : null}
        </div>
      ) : null}
      <Button onClick={onDismiss} variant="text">
        {u.notNow}
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
        {icon}
        <DialogTitle className="text-center text-lg">{title}</DialogTitle>
        {body && <DialogDescription className="text-center text-sm">{body}</DialogDescription>}
      </div>
      {action && <div className="flex justify-center">{action}</div>}
    </div>
  )
}
