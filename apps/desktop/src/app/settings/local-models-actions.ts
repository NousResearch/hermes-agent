import { type QueryClient, useQueryClient } from '@tanstack/react-query'

import {
  activateLocalModel,
  deleteLocalModel,
  downloadLocalModel,
  ejectLocalModel,
  quickstartLocalModels,
  setLocalServer
} from '@/hermes'
import { useI18n } from '@/i18n'
import type { Translations } from '@/i18n/types'
import {
  isCurrentLocalModelsOwner,
  localModelsNotificationTitle,
  type LocalModelsOwner,
  localModelsRequestScope,
  refreshLocalModels,
  watchLocalRuntimeJobs
} from '@/store/local-runtime-jobs'
import { notify, notifyError } from '@/store/notifications'
import type { LocalCatalogModel, LocalRuntimeJob } from '@/types/hermes'

import { useScopedLocalModelsOwner } from './local-models-owner'

export type LocalModelsCopy = Translations['settings']['localModels']

export interface LocalModelsActionScope {
  owner: LocalModelsOwner
  client: QueryClient
  copy: LocalModelsCopy
}

export function useLocalModelsActionScope(): LocalModelsActionScope {
  const owner: LocalModelsOwner = useScopedLocalModelsOwner()
  const client: QueryClient = useQueryClient()
  const { t } = useI18n()

  return { owner, client, copy: t.settings.localModels }
}

// Still on its way (or parked mid-way): rows/hero stay visible while
// paused — a paused download vanishing reads as progress loss.
export function isActiveStatus(status: LocalRuntimeJob['status']): boolean {
  return status === 'paused' || status === 'running'
}

// The actions below never reject: a failure toasts only while its owner is
// still the one on screen, so a late error cannot land on the next connection.

export async function runQuickstart({ owner, client, copy }: LocalModelsActionScope): Promise<void> {
  try {
    await quickstartLocalModels(undefined, localModelsRequestScope(owner))
    watchLocalRuntimeJobs(owner, client)
  } catch (err) {
    if (isCurrentLocalModelsOwner(owner)) {
      notifyError(err, copy.quickstartFailed)
    }
  }
}

export async function downloadCatalogModel(
  { owner, client, copy }: LocalModelsActionScope,
  model: LocalCatalogModel
): Promise<void> {
  try {
    const res = await downloadLocalModel(model.id, localModelsRequestScope(owner))

    if (res.already_downloaded || !res.job_id) {
      refreshLocalModels(owner, client)

      return
    }

    watchLocalRuntimeJobs(owner, client)
  } catch (err) {
    if (isCurrentLocalModelsOwner(owner)) {
      notifyError(err, copy.downloadFailed(model.display_name))
    }
  }
}

export async function activateModel(
  { owner, client, copy }: LocalModelsActionScope,
  target: null | string,
  displayName: string
): Promise<void> {
  if (!target) {
    return
  }

  try {
    await activateLocalModel(target, localModelsRequestScope(owner))
    watchLocalRuntimeJobs(owner, client)
  } catch (err) {
    if (isCurrentLocalModelsOwner(owner)) {
      notifyError(err, copy.activateFailed(displayName))
    }
  }
}

export async function ejectModel({ owner, client, copy }: LocalModelsActionScope, modelId: string): Promise<void> {
  try {
    await ejectLocalModel(modelId, localModelsRequestScope(owner))
    notify({ durationMs: 3_000, kind: 'success', message: copy.ejected, title: localModelsNotificationTitle(owner) })
    refreshLocalModels(owner, client)
  } catch (err) {
    if (isCurrentLocalModelsOwner(owner)) {
      notifyError(err, copy.ejectFailed)
    }
  }
}

export async function setServerRunning(
  { owner, client, copy }: LocalModelsActionScope,
  action: 'start' | 'stop'
): Promise<void> {
  try {
    await setLocalServer(action, localModelsRequestScope(owner))
    notify({
      durationMs: 3_500,
      kind: 'success',
      message: action === 'stop' ? copy.serverStopped : copy.serverStarted,
      title: localModelsNotificationTitle(owner)
    })
    refreshLocalModels(owner, client)
  } catch (err) {
    if (isCurrentLocalModelsOwner(owner)) {
      notifyError(err, action === 'stop' ? copy.serverStopFailed : copy.serverStartFailed)
    }
  }
}

export async function deleteModel({ owner, client, copy }: LocalModelsActionScope, target: string): Promise<void> {
  try {
    await deleteLocalModel(target, localModelsRequestScope(owner))
    notify({
      durationMs: 2_500,
      kind: 'success',
      message: copy.deleted(target),
      title: localModelsNotificationTitle(owner)
    })
    refreshLocalModels(owner, client)
  } catch (err) {
    if (isCurrentLocalModelsOwner(owner)) {
      notifyError(err, copy.deleteFailed)
    }
  }
}
