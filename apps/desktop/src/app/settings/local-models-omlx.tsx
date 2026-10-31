import { type ReactElement, useState } from 'react'

import { activateOmlxModel, downloadOmlxModel, installOmlx, type OmlxStatus, setOmlxServer } from '@/api/omlx'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Cpu, Download, Loader2 } from '@/lib/icons'
import {
  isCurrentLocalModelsOwner,
  localModelsRequestScope,
  refreshLocalModels,
  useLocalRuntimeJobs,
  watchLocalRuntimeJobs
} from '@/store/local-runtime-jobs'
import { notifyError } from '@/store/notifications'

import { LocalModelDownloadActions, LocalModelDownloadProgress } from './local-model-download-progress'
import { isActiveStatus, useLocalModelsActionScope } from './local-models-actions'
import { ListRow, Pill, SettingsSection } from './primitives'
import { ActiveProfileNote } from './profile-scope'

export function OmlxRuntimeSection({ status }: { status: OmlxStatus }): ReactElement {
  const { owner, client, copy } = useLocalModelsActionScope()
  const [busy, setBusy] = useState(false)
  const [repo, setRepo] = useState('')
  const jobs = useLocalRuntimeJobs(owner, value => value).filter(job => job.target.startsWith('oMLX'))
  const activeJobs = jobs.filter(job => isActiveStatus(job.status))
  const disabled = busy || activeJobs.length > 0
  const lastError = jobs.find(job => job.status === 'error')

  async function run(action: () => Promise<unknown>): Promise<void> {
    setBusy(true)

    try {
      await action()
      watchLocalRuntimeJobs(owner, client)
      refreshLocalModels(owner, client)
    } catch (err) {
      if (isCurrentLocalModelsOwner(owner)) {
        notifyError(err, copy.omlxActionFailed)
      }
    } finally {
      setBusy(false)
    }
  }

  return (
    <>
      <ActiveProfileNote className="mb-5" />
      <SettingsSection
        aside={
          status.installed ? (
            <Pill tone="primary">{status.running ? copy.serverRunning : copy.downloaded}</Pill>
          ) : undefined
        }
        icon={Cpu}
        meta={status.version ?? undefined}
        title="oMLX"
      >
        <ListRow
          action={
            status.installed ? (
              <Button
                disabled={disabled}
                onClick={() =>
                  void run(() => setOmlxServer(status.running ? 'stop' : 'start', localModelsRequestScope(owner)))
                }
                size="sm"
                variant="outline"
              >
                {busy && <Loader2 className="animate-spin" />}
                {status.running ? copy.stopServer : copy.startServer}
              </Button>
            ) : (
              <Button
                disabled={disabled || !status.supported}
                onClick={() => void run(() => installOmlx(localModelsRequestScope(owner)))}
                size="sm"
              >
                <Download />
                {copy.installAction}
              </Button>
            )
          }
          description={copy.omlxDetail}
          title={status.installed ? copy.omlxInstalled : copy.omlxInstall}
        />
        {activeJobs.map(job => (
          <ListRow
            action={<LocalModelDownloadActions job={job} owner={owner} />}
            below={<LocalModelDownloadProgress job={job} />}
            description={job.detail}
            key={job.job_id}
            title={job.target}
          />
        ))}
        {(status.error || lastError?.error) && (
          <p className="text-sm text-destructive" role="alert">
            {status.error || lastError?.error}
          </p>
        )}
      </SettingsSection>
      <SettingsSection icon={Cpu} title={copy.modelsTitle}>
        {status.models.length === 0 && <p className="text-sm text-muted-foreground">{copy.omlxNoModels}</p>}
        {status.models.map(model => (
          <ListRow
            action={
              model.active ? (
                <Pill tone="primary">{copy.activePill}</Pill>
              ) : (
                <Button
                  disabled={disabled || !status.installed}
                  onClick={() => void run(() => activateOmlxModel(model.id, localModelsRequestScope(owner)))}
                  size="sm"
                  variant="outline"
                >
                  {copy.useAction}
                </Button>
              )
            }
            description={model.active ? copy.activeDetail : undefined}
            key={model.id}
            title={model.id}
          />
        ))}
        <form
          onSubmit={event => {
            event.preventDefault()
            void run(() => downloadOmlxModel(repo.trim(), localModelsRequestScope(owner)))
          }}
        >
          <label className="mb-2 block text-sm font-medium" htmlFor="omlx-model-repo">
            {copy.omlxRepoLabel}
          </label>
          <div className="flex gap-2">
            <Input
              autoCapitalize="none"
              autoComplete="off"
              disabled={disabled || !status.installed}
              id="omlx-model-repo"
              onChange={event => setRepo(event.target.value)}
              placeholder="mlx-community/…"
              value={repo}
            />
            <Button disabled={disabled || !status.installed || !repo.trim()} size="sm" type="submit">
              <Download />
              {copy.omlxDownload}
            </Button>
          </div>
          <p className="mt-2 text-sm text-muted-foreground">{copy.omlxRepoHint}</p>
        </form>
      </SettingsSection>
    </>
  )
}
