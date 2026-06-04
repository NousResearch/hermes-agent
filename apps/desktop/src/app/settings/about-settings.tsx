import { useStore } from '@nanostores/react'
import { useEffect, useState } from 'react'

import { Button } from '@/components/ui/button'
import { CheckCircle2, ExternalLink, Loader2, RefreshCw, Sparkles } from '@/lib/icons'
import { cn } from '@/lib/utils'
import {
  $desktopVersion,
  $updateApply,
  $updateChecking,
  $updateStatus,
  checkUpdates,
  openUpdatesWindow,
  refreshDesktopVersion
} from '@/store/updates'

import { ListRow, SectionHeading, SettingsContent } from './primitives'

const RELEASE_NOTES_URL = 'https://github.com/NousResearch/hermes-agent/releases'

function relativeTime(ms: number | undefined) {
  if (!ms) {
    return '从未'
  }

  const diff = Date.now() - ms

  if (diff < 60_000) {
    return '刚刚'
  }

  if (diff < 3_600_000) {
    return `${Math.round(diff / 60_000)} 分钟前`
  }

  if (diff < 86_400_000) {
    return `${Math.round(diff / 3_600_000)} 小时前`
  }

  return `${Math.round(diff / 86_400_000)} 天前`
}

export function AboutSettings() {
  const version = useStore($desktopVersion)
  const status = useStore($updateStatus)
  const apply = useStore($updateApply)
  const checking = useStore($updateChecking)
  const [justChecked, setJustChecked] = useState(false)

  // The version atom is loaded once at app boot, which makes About show a
  // stale number after a self-update (the running binary is current, the
  // displayed string is not). Re-read on mount so opening About always
  // reflects the running build.
  useEffect(() => {
    void refreshDesktopVersion()
  }, [])

  const behind = status?.behind ?? 0
  const supported = status?.supported !== false
  const applying = apply.applying || apply.stage === 'restart'

  const handleCheck = async () => {
    setJustChecked(false)
    const next = await checkUpdates()
    setJustChecked(Boolean(next))
  }

  let statusLine: string
  let statusTone: 'idle' | 'available' | 'error' = 'idle'

  if (!supported) {
    statusLine = status?.message ?? "此构建无法从应用内部自行更新。"
    statusTone = 'error'
  } else if (status?.error) {
    statusLine = "无法连接到更新服务器。"
    statusTone = 'error'
  } else if (applying) {
    statusLine = '正在安装更新。'
    statusTone = 'available'
  } else if (behind > 0) {
    statusLine = `有新更新可用（包含 ${behind} 个更改）。`
    statusTone = 'available'
  } else if (status) {
    statusLine = "您已在使用最新版本。"
  } else {
    statusLine = '点击"检查更新"以查找更新。'
  }

  return (
    <SettingsContent>
      <div className="flex flex-col items-center gap-3 pt-6 pb-2 text-center">
        <span className="flex size-16 items-center justify-center rounded-2xl bg-primary/10 text-primary">
          <Sparkles className="size-8" />
        </span>
        <div>
          <h2 className="text-lg font-semibold tracking-tight">Hermes 桌面版</h2>
          <p className="mt-1 text-xs text-muted-foreground">
            {version?.appVersion ? `版本 ${version.appVersion}` : '版本不可用'}
          </p>
        </div>
      </div>

      <div className="mx-auto mt-4 w-full max-w-2xl">
        <SectionHeading icon={RefreshCw} title="更新" />

        <div
          className={cn(
            'rounded-xl border px-4 py-3 text-sm',
            statusTone === 'available' && 'border-primary/30 bg-primary/5 text-foreground',
            statusTone === 'error' && 'border-destructive/35 bg-destructive/5 text-destructive',
            statusTone === 'idle' && 'border-border/70 bg-muted/20 text-foreground'
          )}
        >
          <div className="flex items-start gap-2">
            {statusTone === 'available' ? (
              <Sparkles className="mt-0.5 size-4 shrink-0 text-primary" />
            ) : statusTone === 'error' ? null : (
              <CheckCircle2 className="mt-0.5 size-4 shrink-0 text-emerald-600 dark:text-emerald-400" />
            )}
            <div className="min-w-0">
              <p className="font-medium">{statusLine}</p>
              <p className="mt-1 text-xs text-muted-foreground">
                Last checked {relativeTime(status?.fetchedAt)}
                {justChecked && !checking ? ' · 刚刚' : ''}
              </p>
            </div>
          </div>

          <div className="mt-3 flex flex-wrap items-center gap-2">
            <Button
              disabled={checking || applying || !supported}
              onClick={() => void handleCheck()}
              size="sm"
              variant="outline"
            >
              {checking ? <Loader2 className="size-3 animate-spin" /> : <RefreshCw className="size-3" />}
              {checking ? '检查中…' : '检查更新'}
            </Button>

            {behind > 0 && supported && !applying && (
              <Button onClick={() => openUpdatesWindow()} size="sm">
                查看新内容
              </Button>
            )}

            <Button
              asChild
              className="ml-auto text-xs text-muted-foreground hover:text-foreground"
              size="sm"
              variant="ghost"
            >
              <a
                href={RELEASE_NOTES_URL}
                onClick={event => {
                  event.preventDefault()
                  void window.hermesDesktop?.openExternal?.(RELEASE_NOTES_URL)
                }}
                rel="noreferrer"
                target="_blank"
              >
                <ExternalLink className="size-3" />
                发布说明
              </a>
            </Button>
          </div>
        </div>

        <ListRow
          description="Hermes 会在后台自动检查更新，并在准备好时通知您。"
          hint={`分支 ${status?.branch ?? 'unknown'} · 提交 ${status?.currentSha?.slice(0, 7) ?? 'unknown'}`}
          title="自动更新"
        />
      </div>
    </SettingsContent>
  )
}
