import { useCallback, useEffect, useState } from 'react'

import { PageLoader } from '@/components/page-loader'
import { Button } from '@/components/ui/button'
import {
  type ActionStatusResponse,
  type BackupArchive,
  downloadBackup,
  getActionStatus,
  listBackups,
  runBackup
} from '@/hermes'
import { useI18n } from '@/i18n'
import { AlertCircle, Archive, Download } from '@/lib/icons'
import { upsertDesktopActionTask } from '@/store/activity'
import { notifyError } from '@/store/notifications'

const ACTION_POLL_MS = 1200
const ACTION_POLL_LIMIT = 240

function formatFileSize(bytes: number): string {
  if (bytes <= 0) {
    return '0 B'
  }

  const units = ['B', 'KB', 'MB', 'GB']
  const i = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1)

  return `${(bytes / 1024 ** i).toFixed(i > 0 ? 1 : 0)} ${units[i]}`
}

/** Backups panel — lists existing backup archives (created via the Maintenance
 *  tab's "Create backup" action, or `hermes backup` on the CLI) and lets the
 *  user download or kick off a fresh one without leaving the desktop app. */
export function BackupsPanel() {
  const { t } = useI18n()
  const bb = t.commandCenter.backups

  const [backups, setBackups] = useState<BackupArchive[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [actionName, setActionName] = useState<null | string>(null)
  const [actionStatus, setActionStatus] = useState<ActionStatusResponse | null>(null)
  const [downloadingPath, setDownloadingPath] = useState<null | string>(null)

  const refresh = useCallback(async () => {
    setLoading(true)
    setError('')

    try {
      const response = await listBackups()
      setBackups(response.backups)
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    void refresh()
  }, [refresh])

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
        } else if (!status.running) {
          void refresh()
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
  }, [actionName, refresh])

  const runBackupAction = useCallback(async () => {
    setError('')

    try {
      const started = await runBackup()
      setActionStatus(null)
      setActionName(started.name)
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
      notifyError(err, bb.failed)
    }
  }, [bb])

  const downloadArchive = useCallback(
    async (archive: BackupArchive) => {
      setDownloadingPath(archive.path)
      setError('')

      try {
        const result = await downloadBackup(archive.path)

        if (!result.ok && result.error) {
          setError(result.error)
          notifyError(new Error(result.error), bb.download)
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err))
        notifyError(err, bb.download)
      } finally {
        setDownloadingPath(null)
      }
    },
    [bb]
  )

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-4">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0 text-[length:var(--conversation-text-font-size)] font-medium">{bb.description}</div>
        <Button disabled={actionStatus?.running === true} onClick={() => void runBackupAction()} size="xs" variant="textStrong">
          {actionStatus?.running ? bb.running : bb.run}
        </Button>
      </div>

      {actionStatus && !actionStatus.running && (
        <div className="text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
          {actionStatus.exit_code === 0 ? bb.done : bb.failed}
        </div>
      )}

      {error && (
        <span className="inline-flex items-center gap-1 text-[length:var(--conversation-caption-font-size)] text-destructive">
          <AlertCircle className="size-3.5" />
          {error}
        </span>
      )}

      <div className="min-h-0 flex-1 overflow-y-auto pt-2">
        <div className="mb-2 text-[0.625rem] font-medium uppercase tracking-[0.08em] text-(--ui-text-tertiary)">
          {bb.directory}
        </div>
        {loading ? (
          <PageLoader className="min-h-24" label={bb.directory} />
        ) : backups.length === 0 ? (
          <div className="py-2 text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
            {bb.empty}
          </div>
        ) : (
          <ul>
            {backups.map(archive => (
              <li className="flex items-center gap-2 py-1.5" key={archive.path}>
                <Archive className="size-3.5 shrink-0 text-(--ui-text-tertiary)" />
                <span className="min-w-0 flex-1 truncate font-mono text-[0.7rem]">{archive.name}</span>
                <span className="shrink-0 text-[0.65rem] text-(--ui-text-tertiary)">
                  {formatFileSize(archive.size)}
                </span>
                <Button
                  aria-label={bb.download}
                  className="shrink-0 text-(--ui-text-tertiary) hover:text-foreground"
                  disabled={downloadingPath === archive.path}
                  onClick={() => void downloadArchive(archive)}
                  size="icon-xs"
                  title={bb.download}
                  variant="ghost"
                >
                  <Download className="size-3" />
                </Button>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  )
}
