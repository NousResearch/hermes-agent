import { useStore } from '@nanostores/react'
import { FileText, RefreshCw } from 'lucide-react'
import { useEffect, useState, type CSSProperties } from 'react'

import { Button } from '../components/button'
import { updateFailureRetryAction } from '../lib/update-failure-retry'
import {
  $liveUpdater,
  $logPath,
  $mode,
  type BootstrapStateModel,
  openLogDir,
  refreshLiveUpdater,
  startInstall,
  startUpdate,
  stopLiveUpdater
} from '../store'

interface FailureProps {
  bootstrap: BootstrapStateModel
}

/*
 * Failure screen. Same hero treatment as Welcome/Success — the wordmark
 * carries the brand, so we keep it across every terminal state.
 *
 * Update failures that cite a live marker PID do not re-enter the lock.
 * The screen names that updater and offers to stop it, or to wait.
 */
export default function Failure({ bootstrap }: FailureProps) {
  const logPath = useStore($logPath)
  const mode = useStore($mode)
  const liveUpdater = useStore($liveUpdater)
  const [stopNote, setStopNote] = useState<string | null>(null)
  const isUpdate = mode === 'update'
  const decision = updateFailureRetryAction(isUpdate ? liveUpdater : null)
  const blockedPid = decision.kind === 'stop_or_wait' ? decision.pid : null
  const stopLabel = decision.kind === 'stop_or_wait' ? decision.stopLabel : null
  const detail =
    decision.kind === 'stop_or_wait'
      ? decision.waitMessage
      : (bootstrap.error ??
        (isUpdate
          ? 'Something went wrong during the update.'
          : 'Something went wrong during installation.'))

  useEffect(() => {
    if (!isUpdate) {return}
    void refreshLiveUpdater()
  }, [isUpdate, bootstrap.error])

  async function onRetryUpdate() {
    const owner = await refreshLiveUpdater()
    const next = updateFailureRetryAction(owner)

    // Still alive: stay here. Do not call startUpdate, which acquires.
    if (next.kind === 'stop_or_wait') {return}
    await startUpdate()
  }

  async function onStopUpdater() {
    setStopNote(null)
    const message = await stopLiveUpdater()
    const owner = await refreshLiveUpdater()
    const next = updateFailureRetryAction(owner)

    if (message && next.kind === 'stop_or_wait') {
      setStopNote(message)
    }
  }

  return (
    <div className="hermes-fade-in flex h-full flex-col items-center justify-center gap-6 px-12 py-10">
      <div className="w-full max-w-2xl min-w-0 text-center">
        <p
          className="fit-text mx-auto mb-4 w-full font-['Collapse'] font-bold uppercase leading-[0.9] tracking-[0.08em] text-destructive mix-blend-plus-lighter dark:text-destructive/90"
          style={
            {
              '--fit-text-line-height': '0.9',
              '--fit-text-max': '5rem',
              '--fit-text-min': '2.25rem'
            } as CSSProperties
          }
        >
          <span>
            <span>{isUpdate ? 'Update didn\u2019t finish' : 'Install didn\u2019t finish'}</span>
          </span>
          <span aria-hidden="true">{isUpdate ? 'Update didn\u2019t finish' : 'Install didn\u2019t finish'}</span>
        </p>

        <p className="m-0 mx-auto max-w-xl text-center text-sm leading-normal tracking-tight text-muted-foreground">
          {detail}
        </p>
      </div>

      <div className="flex items-center gap-3">
        {stopLabel ? (
          <Button className="gap-1.5" onClick={() => void onStopUpdater()}>
            {stopLabel}
          </Button>
        ) : null}
        <Button
          className="gap-1.5"
          onClick={() => void (isUpdate ? onRetryUpdate() : startInstall())}
          variant={stopLabel ? 'outline' : 'default'}
        >
          <RefreshCw />
          {isUpdate ? 'Retry update' : 'Retry install'}
        </Button>
        <Button className="gap-1.5" onClick={() => void openLogDir()} variant="text">
          <FileText />
          Open logs
        </Button>
      </div>

      {blockedPid !== null ? (
        <p className="max-w-lg text-center text-xs text-muted-foreground/70">
          Updater PID {blockedPid} is still running. Wait for it to finish, or stop it.
          Retry will not start another update while that PID is alive.
        </p>
      ) : null}

      {stopNote ? (
        <p className="max-w-lg text-center text-xs text-muted-foreground/70">{stopNote}</p>
      ) : null}

      {logPath && (
        <p className="max-w-lg text-center text-xs text-muted-foreground/70">
          Log: <code className="font-mono">{logPath}</code>
        </p>
      )}
    </div>
  )
}
