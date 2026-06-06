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
  idle: '준비 중…',
  prepare: '준비 중…',
  fetch: '다운로드 중…',
  pull: '거의 다 되었습니다…',
  pydeps: '마무리 중…',
  restart: 'Hermes 재시작 중…',
  manual: '터미널에서 업데이트',
  error: '업데이트 일시 중지됨'
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
      <CenteredStatus icon={<Loader2 className="size-6 animate-spin text-primary" />} title="업데이트 확인 중…" />
    )
  }

  if (!status) {
    return (
      <CenteredStatus
        action={
          <Button onClick={onRetryCheck} size="sm">
            다시 시도
          </Button>
        }
        icon={<AlertCircle className="size-6 text-muted-foreground" />}
        title="업데이트를 확인할 수 없습니다"
      />
    )
  }

  if (!status.supported) {
    return (
      <CenteredStatus
        body={status.message ?? '이 버전의 Hermes는 앱 내에서 자체적으로 업데이트할 수 없습니다.'}
        icon={<AlertCircle className="size-6 text-muted-foreground" />}
        title="업데이트를 사용할 수 없음"
      />
    )
  }

  if (status.error) {
    return (
      <CenteredStatus
        action={
          <Button disabled={checking} onClick={onRetryCheck} size="sm">
            다시 시도
          </Button>
        }
        body="네트워크 연결을 확인하고 다시 시도하세요."
        icon={<AlertCircle className="size-6 text-muted-foreground" />}
        title="업데이트를 확인할 수 없습니다"
      />
    )
  }

  if (behind === 0) {
    return (
      <CenteredStatus
        body="최신 버전을 사용 중입니다."
        icon={<CheckCircle2 className="size-7 text-emerald-600 dark:text-emerald-400" />}
        title="준비 완료"
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

        <DialogTitle className="text-center text-xl">새로운 업데이트 사용 가능</DialogTitle>
        <DialogDescription className="text-center text-sm">
          Hermes의 새 버전을 설치할 준비가 되었습니다.
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
          지금 업데이트
        </Button>
        <button
          className="text-center text-sm font-medium text-muted-foreground transition-colors hover:text-foreground"
          onClick={onLater}
          type="button"
        >
          나중에
        </button>
      </div>

      {remaining > 0 && (
        <p className="text-center text-xs text-muted-foreground">
          + {remaining}개의 변경 사항이 더 포함되어 있습니다.
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

        <DialogTitle className="text-center text-xl">터미널에서 업데이트</DialogTitle>
        <DialogDescription className="text-center text-sm">
          명령줄에서 Hermes를 설치했으므로 업데이트도 터미널에서 실행됩니다. 터미널에 다음 명령어를 붙여넣으세요:
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
              복사됨
            </>
          ) : (
            <>
              <Copy className="size-3.5" />
              복사
            </>
          )}
        </span>
      </button>

      <p className="text-center text-xs text-muted-foreground">
        다음 번에 Hermes를 실행할 때 새 버전이 적용됩니다.
      </p>

      <Button className="font-semibold" onClick={onDone} size="lg" variant="outline">
        완료
      </Button>
    </div>
  )
}

function ApplyingView({ apply }: { apply: UpdateApplyState }) {
  const label = STAGE_LABELS[apply.stage] ?? 'Hermes 업데이트 중…'

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
          Hermes 업데이터가 자체 창에서 실행되며 완료되면 Hermes를 다시 엽니다.
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

      <p className="text-center text-xs text-muted-foreground">업데이트를 적용하기 위해 Hermes가 닫힙니다.</p>
    </div>
  )
}

function ErrorView({ message, onDismiss, onRetry }: { message: string; onDismiss: () => void; onRetry: () => void }) {
  return (
    <ErrorState
      className="px-6 pb-6 pt-7 pr-8"
      description={
        <DialogDescription className="max-w-prose text-center text-sm leading-5 text-muted-foreground">
          {message || '걱정 마세요. 손실된 데이터는 없습니다. 지금 다시 시도할 수 있습니다.'}
        </DialogDescription>
      }
      title={
        <DialogTitle className="text-center text-xl font-semibold tracking-tight">업데이트를 완료하지 못했습니다</DialogTitle>
      }
    >
      <Button className="font-semibold" onClick={onRetry} size="lg">
        다시 시도
      </Button>
      <Button onClick={onDismiss} variant="text">
        나중에
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
