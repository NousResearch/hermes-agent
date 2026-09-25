import { useIsMutating } from '@tanstack/react-query'
import { type ReactElement, useState } from 'react'

import { Button } from '@/components/ui/button'
import { CheckCircle2, Download, Loader2, Pause, StopFilled, Zap } from '@/lib/icons'
import { cn } from '@/lib/utils'
import { localModelsKey, runningRuntimeInstall, startLocalRuntimeInstall } from '@/store/local-runtime-jobs'
import type { LocalModelsStatus, LocalRuntimeJob } from '@/types/hermes'

import { LocalModelDownloadActions, LocalModelDownloadProgress } from './local-model-download-progress'
import { type LocalModelsActionScope, setServerRunning, useLocalModelsActionScope } from './local-models-actions'
import { ListRow, Pill, SettingsSection } from './primitives'

export interface LocalModelsRuntimeSectionProps {
  status: LocalModelsStatus
  jobs: readonly LocalRuntimeJob[]
  lastError: LocalRuntimeJob | undefined
}

export function LocalModelsRuntimeSection({ status, jobs, lastError }: LocalModelsRuntimeSectionProps): ReactElement {
  const scope: LocalModelsActionScope = useLocalModelsActionScope()
  const { client, copy, owner } = scope
  const installStarting: boolean = useIsMutating({ mutationKey: localModelsKey(owner, 'install') }) > 0
  const [serverBusy, setServerBusy] = useState<boolean>(false)
  const rJob: LocalRuntimeJob | null = runningRuntimeInstall(jobs)

  // Up to date = the authority (status) says the configured tag is what's
  // serving. Shown whenever true — not only right after an update.
  const updateApplied = status.runtime_installed && !status.update_available && status.tag === status.configured_tag

  async function handleServer(action: 'start' | 'stop'): Promise<void> {
    setServerBusy(true)

    try {
      await setServerRunning(scope, action)
    } finally {
      setServerBusy(false)
    }
  }

  return (
    <SettingsSection
      aside={
        status.runtime_installed ? (
          <Pill tone="primary">
            {status.server_running ? copy.serverRunning : copy.runtimeReady(status.runtime_backend ?? '')}
          </Pill>
        ) : undefined
      }
      icon={Zap}
      meta={status.tag}
      title={copy.runtimeTitle}
    >
      {status.runtime_installed ? (
        <ListRow
          action={
            status.server_running ? (
              <Button
                className={cn(serverBusy && '[&_svg]:animate-spin')}
                disabled={serverBusy}
                onClick={() => void handleServer('stop')}
                size="sm"
                variant="outline"
              >
                {serverBusy ? <Loader2 /> : <StopFilled />}
                {copy.stopServer}
              </Button>
            ) : (
              <Button
                className={cn(serverBusy && '[&_svg]:animate-spin')}
                disabled={serverBusy}
                onClick={() => void handleServer('start')}
                size="sm"
                variant="outline"
              >
                {serverBusy ? <Loader2 /> : <Zap />}
                {copy.startServer}
              </Button>
            )
          }
          description={
            status.server_running
              ? copy.runtimeRunningDetail
              : copy.runtimeInstalledDetail(status.tag, status.runtime_backend ?? 'cpu')
          }
          title={copy.runtimeInstalled}
        />
      ) : rJob ? (
        <ListRow
          action={<LocalModelDownloadActions job={rJob} owner={owner} />}
          below={<LocalModelDownloadProgress job={rJob} />}
          description={rJob.detail || copy.installing}
          title={
            <span className="inline-flex items-center gap-2">
              {rJob.status === 'paused' ? (
                <Pause className="size-3.5" />
              ) : (
                <Loader2 className="size-3.5 animate-spin" />
              )}
              {copy.installing}
            </span>
          }
        />
      ) : (
        <ListRow
          action={
            <Button disabled={installStarting} onClick={() => void startLocalRuntimeInstall(owner, client)} size="sm">
              <Download />
              {copy.installAction}
            </Button>
          }
          description={copy.installDetail}
          title={copy.installTitle}
        />
      )}

      {status.update_available && !rJob && (
        <ListRow
          action={
            <Button disabled={installStarting} onClick={() => void startLocalRuntimeInstall(owner, client)} size="sm">
              <Download />
              {copy.updateAction}
            </Button>
          }
          description={copy.updateDetail(status.configured_tag, status.tag)}
          title={copy.updateTitle}
        />
      )}

      {rJob && status.runtime_installed && (
        <ListRow
          action={<LocalModelDownloadActions job={rJob} owner={owner} />}
          below={<LocalModelDownloadProgress job={rJob} />}
          description={rJob.detail || copy.updating}
          title={
            <span className="inline-flex items-center gap-2">
              {rJob.status === 'paused' ? (
                <Pause className="size-3.5" />
              ) : (
                <Loader2 className="size-3.5 animate-spin" />
              )}
              {copy.updating}
            </span>
          }
        />
      )}

      {updateApplied && (
        <ListRow
          description={copy.upToDateDetail(status.tag, status.runtime_backend ?? 'cpu')}
          title={
            <span className="inline-flex items-center gap-2">
              <CheckCircle2 className="size-4 text-emerald-600 dark:text-emerald-400" />
              {copy.upToDateTitle}
            </span>
          }
        />
      )}

      {lastError?.kind === 'runtime-install' && <p className="text-[0.75rem] text-destructive">{lastError.error}</p>}
    </SettingsSection>
  )
}
