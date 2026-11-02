import { type ReactElement, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Tip } from '@/components/ui/tooltip'
import { Check, CheckCircle2, Cpu, Download, Eject, Loader2, Trash2 } from '@/lib/icons'
import { cn } from '@/lib/utils'
import { runningDownloadFor } from '@/store/local-runtime-jobs'
import type { LocalCatalogModel, LocalModelPlacement, LocalModelsStatus, LocalRuntimeJob } from '@/types/hermes'

import { downloadStatusText, LocalModelDownloadActions, ProgressBar } from './local-model-download-progress'
import {
  activateModel,
  deleteModel,
  downloadCatalogModel,
  ejectModel,
  isActiveStatus,
  type LocalModelsActionScope,
  useLocalModelsActionScope
} from './local-models-actions'
import { ListRow, Pill } from './primitives'

type SideloadedModel = LocalModelsStatus['models'][number]

interface Residency {
  isLoaded: boolean
  isLoadingNow: boolean
  livePlacement: LocalModelPlacement | undefined
}

function residencyOf(status: LocalModelsStatus, modelId: null | string | undefined): Residency {
  const residency: string | undefined = modelId ? status.loaded_models[modelId] : undefined

  return {
    isLoaded: residency === 'loaded' || residency === 'ready',
    isLoadingNow: residency === 'loading',
    livePlacement: modelId ? status.placement?.[modelId] : undefined
  }
}

function runningActivation(jobs: readonly LocalRuntimeJob[], modelId: null | string | undefined): boolean {
  return jobs.some(
    (job: LocalRuntimeJob): boolean =>
      job.kind === 'model-activate' && job.status === 'running' && job.model_id === modelId
  )
}

function anyActivationRunning(jobs: readonly LocalRuntimeJob[]): boolean {
  return jobs.some((job: LocalRuntimeJob): boolean => job.kind === 'model-activate' && job.status === 'running')
}

interface ResidencyPillsProps {
  residency: Residency
}

function ResidencyPills({ residency }: ResidencyPillsProps): ReactElement {
  const { copy } = useLocalModelsActionScope()
  const { isLoaded, isLoadingNow, livePlacement } = residency

  return (
    <>
      {isLoaded && livePlacement && (
        <Tip label={livePlacement.spilled ? copy.placementSpilledTip : copy.placementResidentTip}>
          <Pill tone={livePlacement.spilled ? 'warn' : 'success'}>
            <Cpu className="mr-1 size-3" />
            {livePlacement.granted_window_label ?? livePlacement.window_label ?? ''}
            {' · '}
            {livePlacement.spilled ? copy.placementSpilled : copy.placementResident}
          </Pill>
        </Tip>
      )}
      {isLoaded && !livePlacement && <Pill>{copy.loadedPill}</Pill>}

      {isLoadingNow && (
        <Pill>
          <Loader2 className="mr-1 size-3 animate-spin" />
          {copy.loadingPill}
        </Pill>
      )}
    </>
  )
}

interface ModelIdProps {
  modelId: string
}

function EjectModelButton({ modelId }: ModelIdProps): ReactElement {
  const scope: LocalModelsActionScope = useLocalModelsActionScope()

  return (
    <Tip label={scope.copy.ejectTip}>
      <Button onClick={() => void ejectModel(scope, modelId)} size="icon" variant="ghost">
        <Eject />
      </Button>
    </Tip>
  )
}

function DeleteModelButton({ modelId }: ModelIdProps): ReactElement {
  const scope: LocalModelsActionScope = useLocalModelsActionScope()
  const [deleting, setDeleting] = useState<boolean>(false)

  async function handleDelete(): Promise<void> {
    if (!window.confirm(scope.copy.deleteConfirm(modelId))) {
      return
    }

    setDeleting(true)

    try {
      await deleteModel(scope, modelId)
    } finally {
      setDeleting(false)
    }
  }

  return (
    <Tip label={scope.copy.deleteAction}>
      <Button
        className={cn(deleting && '[&_svg]:animate-spin')}
        onClick={() => void handleDelete()}
        size="icon"
        variant="ghost"
      >
        {deleting ? <Loader2 /> : <Trash2 />}
      </Button>
    </Tip>
  )
}

export interface CatalogModelRowProps {
  model: LocalCatalogModel
  status: LocalModelsStatus
  jobs: readonly LocalRuntimeJob[]
}

export function CatalogModelRow({ model, status, jobs }: CatalogModelRowProps): ReactElement {
  const scope: LocalModelsActionScope = useLocalModelsActionScope()
  const { copy, owner } = scope
  const dJob: LocalRuntimeJob | null = runningDownloadFor(jobs, model.id)

  const anyDownloadRunning = jobs.some(j => j.kind === 'model-download' && isActiveStatus(j.status))

  const activateTarget = model.downloaded_model_id ?? model.model_id
  const isActive = Boolean(activateTarget && status.active_model_id === activateTarget)
  const residency: Residency = residencyOf(status, activateTarget)
  const { isLoaded, isLoadingNow } = residency
  const activating: boolean = runningActivation(jobs, activateTarget)

  return (
    <ListRow
      action={
        model.downloaded ? (
          <div className="flex items-center justify-end gap-2">
            <ResidencyPills residency={residency} />

            {isActive ? (
              <Tip label={copy.activeDetail}>
                <Pill tone="primary">
                  <Check className="mr-1 size-3" />
                  {copy.activePill}
                </Pill>
              </Tip>
            ) : (
              <Button
                className={cn(activating && '[&_svg]:animate-spin')}
                disabled={anyActivationRunning(jobs)}
                onClick={() => void activateModel(scope, activateTarget ?? null, model.display_name)}
                size="sm"
              >
                {activating ? <Loader2 /> : <Check />}
                {activating ? copy.activating : copy.useAction}
              </Button>
            )}

            {isLoaded && <EjectModelButton modelId={activateTarget ?? model.id} />}

            <DeleteModelButton modelId={model.downloaded_model_id ?? model.id} />
          </div>
        ) : dJob ? (
          <LocalModelDownloadActions job={dJob} owner={owner} />
        ) : (
          <Button
            disabled={!model.fits || anyDownloadRunning || !status.runtime_installed}
            onClick={() => void downloadCatalogModel(scope, model)}
            size="sm"
            variant="outline"
          >
            <Download />
            {copy.downloadAction(model.size_label)}
          </Button>
        )
      }
      below={
        dJob ? (
          <div className="mt-2 grid gap-1">
            <ProgressBar paused={dJob.status === 'paused'} percent={dJob.percent} />

            <p className="text-[0.68rem] text-muted-foreground">{downloadStatusText(dJob, copy)}</p>
          </div>
        ) : undefined
      }
      description={
        <>
          {model.description}

          <span className="mt-1.5 flex flex-wrap items-center gap-1.5">
            {/* Memory: the traffic light. Green = runs fully on
                the GPU; amber = spills to system RAM (works,
                slower); red = doesn't fit this machine at all.
                Detail prose lives in the tooltip. */}
            {!model.fits ? (
              <Tip label={model.fit_detail ?? model.fit_summary}>
                <Pill tone="destructive">
                  <Cpu className="mr-1 size-3" />
                  {copy.pillTooBig}
                </Pill>
              </Tip>
            ) : model.spilled ? (
              <Tip label={model.quant_reason ?? model.fit_summary}>
                <Pill tone="warn">
                  <Cpu className="mr-1 size-3" />
                  {copy.pillUsesRam}
                </Pill>
              </Tip>
            ) : (
              <Tip label={model.quant_reason ?? model.fit_summary}>
                <Pill tone="success">
                  <Cpu className="mr-1 size-3" />
                  {copy.pillFitsGpu}
                </Pill>
              </Tip>
            )}

            {/* Context: one pill. Green 'Full X context' only when
                the model earned its complete window resident on the
                GPU — a big context served from system RAM is slow,
                and a green badge there would sell exactly the wrong
                model, so a spilled full window goes gray. Anything
                starting below its native window gets one quiet
                'Up to' pill instead of a start/grow pair. */}
            {model.fits &&
              model.start_window_label &&
              (model.start_window && model.start_window >= model.native_context ? (
                <Tip label={copy.pillFullContextTip}>
                  <Pill tone={model.spilled ? 'muted' : 'success'}>
                    {copy.pillFullContext(model.native_context_label)}
                  </Pill>
                </Tip>
              ) : (
                <Tip label={copy.pillGrowsTip}>
                  <Pill>{copy.pillUpTo(model.native_context_label)}</Pill>
                </Tip>
              ))}

            {!model.fits && <Pill>{copy.pillUpTo(model.native_context_label)}</Pill>}

            {model.vision && <Pill>{copy.pillVision}</Pill>}
          </span>

          {isActive && !isLoaded && !isLoadingNow && status.server_running && (
            <span className="mt-0.5 block text-(--ui-text-tertiary)">{copy.activeNotLoaded}</span>
          )}
        </>
      }
      title={
        <span className="inline-flex items-center gap-2">
          {model.display_name}

          {model.recommended &&
            (model.recommended_reason ? (
              // The why, straight from the resolver: the tooltip is
              // the branch that picked this model, so the shown
              // rationale can never drift from the actual decision.
              <Tip label={copy.recommendedReason[model.recommended_reason]}>
                <Pill tone="primary">{copy.recommended}</Pill>
              </Tip>
            ) : (
              <Pill tone="primary">{copy.recommended}</Pill>
            ))}
        </span>
      }
    />
  )
}

export interface SideloadedModelRowProps {
  model: SideloadedModel
  status: LocalModelsStatus
  jobs: readonly LocalRuntimeJob[]
}

export function SideloadedModelRow({ model: m, status, jobs }: SideloadedModelRowProps): ReactElement {
  const scope: LocalModelsActionScope = useLocalModelsActionScope()
  const { copy } = scope
  const isActive = status.active_model_id === m.id
  const residency: Residency = residencyOf(status, m.id)
  const activating: boolean = runningActivation(jobs, m.id)

  return (
    <ListRow
      action={
        <div className="flex items-center justify-end gap-2">
          <ResidencyPills residency={residency} />

          {isActive ? (
            <Pill tone="primary">
              <CheckCircle2 className="mr-1 size-3" />
              {copy.activePill}
            </Pill>
          ) : (
            <Button
              className={cn(activating && '[&_svg]:animate-spin')}
              disabled={anyActivationRunning(jobs)}
              onClick={() => void activateModel(scope, m.id, m.id)}
              size="sm"
            >
              {activating ? <Loader2 /> : <Check />}
              {copy.useAction}
            </Button>
          )}

          {residency.isLoaded && <EjectModelButton modelId={m.id} />}

          <DeleteModelButton modelId={m.id} />
        </div>
      }
      description={<span>{copy.addedByYou}</span>}
      title={
        <span className="inline-flex items-center gap-2">
          <span className="truncate font-mono text-[0.8rem]">{m.id}</span>

          <span className="text-[0.68rem] font-normal text-muted-foreground">{m.size_label}</span>
        </span>
      }
    />
  )
}
