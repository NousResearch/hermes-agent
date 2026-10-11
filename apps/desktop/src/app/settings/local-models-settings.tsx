import { useIsMutating, useQuery } from '@tanstack/react-query'
import { type ReactElement, useEffect, useRef, useState } from 'react'
import { useNavigate } from 'react-router'

import { NEW_CHAT_ROUTE } from '@/app/routes'
import {
  localModelsCatalogOptions,
  localModelsHardwareOptions,
  localModelsKey,
  type LocalModelsOwner,
  useLocalModelsOwner,
  useLocalModelsStatus,
  useLocalRuntimeJobs
} from '@/store/local-runtime-jobs'
import type { LocalRuntimeJob } from '@/types/hermes'

import { isActiveStatus, supersededByDone } from './local-models-actions'
import { LocalModelsBrowseSection } from './local-models-browse'
import { LocalModelsHardwareSection } from './local-models-hardware-section'
import { LocalModelsModelsSection } from './local-models-models-section'
import { LocalModelsOwnerProvider, useScopedLocalModelsOwner } from './local-models-owner'
import { LocalModelsQuickstart } from './local-models-quickstart'
import { LocalModelsRuntimeSection } from './local-models-runtime-section'
import { SettingsContent, SettingsSkeleton } from './primitives'
import { ActiveProfileNote } from './profile-scope'

export function LocalModelsSettings(): ReactElement {
  const owner: LocalModelsOwner = useLocalModelsOwner()

  return (
    <LocalModelsOwnerProvider key={JSON.stringify(localModelsKey(owner))} value={owner}>
      <ScopedLocalModelsSettings />
    </LocalModelsOwnerProvider>
  )
}

function ScopedLocalModelsSettings(): ReactElement {
  const owner: LocalModelsOwner = useScopedLocalModelsOwner()
  const installStarting: boolean = useIsMutating({ mutationKey: localModelsKey(owner, 'install') }) > 0
  const { data: status } = useLocalModelsStatus(owner, true, true)
  const { data: hardware } = useQuery(localModelsHardwareOptions(owner))
  const { data: catalog } = useQuery(localModelsCatalogOptions(owner))
  // Quickstart escape hatch: true once the user asks for the full pane
  // (model list, HF browser) instead of the one-button setup card.
  const [configure, setConfigure] = useState<boolean>(false)

  const jobs: readonly LocalRuntimeJob[] = useLocalRuntimeJobs(
    owner,
    (value: readonly LocalRuntimeJob[]): readonly LocalRuntimeJob[] => value
  )

  // Setup flows end at the action, not the settings pane: when quickstart
  // finishes while the user is still HERE watching it, land them on a new
  // chat with the model ready to try. Unmount cancels the intent — a user
  // who navigated away mid-download keeps their place (no focus theft).
  // (Lives above the loading return: hooks run unconditionally.)
  const navigate = useNavigate()
  const seenQuickstarts = useRef(new Set<string>())

  const runningQuickstart = jobs.find(j => j.kind === 'quickstart' && isActiveStatus(j.status))

  useEffect(() => {
    // Event detection, not value mirroring: the ref only remembers which
    // job ids THIS mount saw running, so a 'done' already in the list on
    // mount (stale history) never triggers a navigation.
    const seen = seenQuickstarts.current

    for (const j of jobs) {
      if (j.kind !== 'quickstart') {
        continue
      }

      if (j.status === 'running') {
        seen.add(j.job_id)
      } else if (j.status === 'done' && seen.has(j.job_id)) {
        seen.delete(j.job_id)
        navigate(NEW_CHAT_ROUTE)
      }
    }
  }, [jobs, navigate])

  if (!status || !catalog) {
    return <SettingsSkeleton sections={[{ rows: 2 }, { rows: 4 }]} />
  }

  // The latest error that is still current: an error superseded by a
  // newer SUCCESSFUL job of the same operation (retry install after a
  // transient verify failure, redownload of the same model after a broken
  // file) is stale history — rendering it next to the green 'up to date'
  // row reads as a live failure and never heals (#102616). The retry key is
  // the operation identity (kind + target for multi-target kinds), not the
  // kind alone, so a success for a DIFFERENT model never retires another
  // model's still-current error. A newer running job settles the question
  // when it finishes, so it does not clear the error yet.
  const lastError = jobs.find(j => j.status === 'error' && !supersededByDone(jobs, j))

  // Until something is servable (runtime + at least one model), the pane
  // leads with the quickstart hero. A running quickstart pins this view so
  // its progress has a home even after a remount.
  const qJob = runningQuickstart ?? null

  const needsSetup = !status.runtime_installed || status.models.length === 0
  // The setup hero is reserved for an automatic recommendation. A
  // spilled model remains visible below, but setup must not silently choose it.
  const heroModel = catalog.find(c => c.recommended && c.fits) ?? null

  // An active runtime install/update or model download needs the FULL pane
  // (its row lives there with its controls) — the setup hero must not hide
  // it on a remount.
  const otherActiveJob = jobs.some(
    j => (j.kind === 'runtime-install' || j.kind === 'model-download') && isActiveStatus(j.status)
  )

  const failedInstall: boolean = jobs.some(
    (job: LocalRuntimeJob): boolean => job.kind === 'runtime-install' && job.status === 'error'
      // Same recency contract as lastError: a failed install superseded by
      // a completed retry is history, not a reason to leave the hero.
      && !supersededByDone(jobs, job)
  )

  if ((qJob || (needsSetup && !configure && heroModel)) && !otherActiveJob && !installStarting && !failedInstall) {
    return (
      <LocalModelsQuickstart
        heroModel={heroModel}
        job={qJob}
        lastError={lastError}
        onConfigure={() => setConfigure(true)}
      />
    )
  }

  return (
    <SettingsContent>
      <ActiveProfileNote className="mb-5" />
      <LocalModelsRuntimeSection jobs={jobs} lastError={lastError} status={status} />
      <LocalModelsHardwareSection hardware={hardware} />
      <LocalModelsModelsSection catalog={catalog} jobs={jobs} lastError={lastError} status={status} />
      <LocalModelsBrowseSection />
    </SettingsContent>
  )
}
