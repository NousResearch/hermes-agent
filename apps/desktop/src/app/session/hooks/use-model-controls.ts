import type { ModelOptionsResult } from '@hermes/shared'
import { type QueryClient } from '@tanstack/react-query'
import { useCallback, useRef } from 'react'

import type { ModelSelection } from '@/app/shell/model-menu-panel'
import { getGlobalModelInfo } from '@/hermes'
import { useI18n } from '@/i18n'
import { isBusySessionModelSwitch } from '@/lib/gateway-rpc'
import { surfaceModelSwitchConfirm } from '@/lib/guarded-model-switch'
import { modelOptionsQueryKey } from '@/lib/model-options'
import { notifyError } from '@/store/notifications'
import { $activeGatewayProfile } from '@/store/profile'
import {
  $activeSessionId,
  $connection,
  $currentModel,
  $currentProvider,
  getComposerSelectionGeneration,
  getCurrentModelSource,
  markComposerSelectionManual,
  setCurrentModel,
  setCurrentModelSource,
  setCurrentProvider
} from '@/store/session'
import { $sessionStates, sessionTileDelegate } from '@/store/session-states'

interface ModelControlsOptions {
  cacheOwnerConnectionId?: string
  cacheProfile?: string
  queryClient: QueryClient
  requestGateway: <T = unknown>(method: string, params?: Record<string, unknown>) => Promise<T>
}

interface ModelSwitchResponse {
  confirm_message?: string
  confirm_required?: boolean
  deferred?: boolean
}

interface FailedPrimaryMutation {
  owner: string
  previousGeneration: number
  previousModel: string
  previousProvider: string
  previousSource: ReturnType<typeof getCurrentModelSource>
}

export function useModelControls({
  cacheOwnerConnectionId,
  cacheProfile,
  queryClient,
  requestGateway
}: ModelControlsOptions) {
  const { t } = useI18n()
  const copy = t.desktop
  const failedPrimaryMutationsRef = useRef(new Map<number, FailedPrimaryMutation>())
  const profileRefreshEpochRef = useRef(0)
  const restoredPrimaryGenerationRef = useRef<null | { global: number; operation: number; owner: string }>(null)

  // All callbacks here read reactive session state from the store (.get())
  // rather than capturing it as a prop. The actions bag in wiring.tsx mutates
  // in place to keep a stable identity, so memoized surfaces capture these
  // callbacks once and never re-evaluate — a captured prop would be stale
  // forever. The store read is always current.
  const updateModelOptionsCache = useCallback(
    (
      sessionId: null | string,
      provider: string,
      model: string,
      includeGlobal: boolean,
      profile = cacheProfile || $activeGatewayProfile.get(),
      ownerConnectionId = cacheOwnerConnectionId
    ) => {
      const patch = (prev: ModelOptionsResult | undefined) => {
        // Selection state can update before the catalog query has resolved.
        // Keep that optimistic cache structurally complete; the composer
        // interprets a response without `providers` as an empty catalog.
        const providers = prev?.providers?.length
          ? prev.providers
          : provider && model
            ? [{ models: [model], name: provider, slug: provider }]
            : []

        return { ...prev, provider, model, providers }
      }

      queryClient.setQueryData<ModelOptionsResult>(modelOptionsQueryKey(profile, sessionId, ownerConnectionId), patch)

      if (includeGlobal) {
        queryClient.setQueryData<ModelOptionsResult>(modelOptionsQueryKey(profile, null, ownerConnectionId), patch)
      }
    },
    [cacheOwnerConnectionId, cacheProfile, queryClient]
  )

  // Settings → Model writes the profile default, which the backend applies to
  // new sessions only. Keep a live session's renderer state and session-scoped
  // model-options cache authoritative instead of briefly painting the saved
  // default as if the active agent had switched. Marking the composer as
  // default-derived still lets the next fresh draft reseed from profile config.
  const applySavedMainModel = useCallback(
    (provider: string, model: string) => {
      const liveSessionId = $activeSessionId.get()

      setCurrentModelSource('default')

      if (!liveSessionId) {
        setCurrentProvider(provider)
        setCurrentModel(model)
      }

      // A null session id is the profile-global model-options key. Never patch
      // the live session key here: only config.set --session may change it.
      updateModelOptionsCache(null, provider, model, false)
    },
    [updateModelOptionsCache]
  )

  // Seed the composer's model state from the profile default. `force` reseeds
  // for a profile swap (the new profile has its own default); otherwise this
  // only fills an EMPTY selection so a user's pick (plain UI state in
  // $currentModel) survives the lifecycle refreshes that fire on boot / fresh
  // draft / session events. A live session owns the footer, so skip entirely.
  const refreshCurrentModel = useCallback(async (force = false) => {
    // A forced profile swap opens a new intent epoch; an older in-flight
    // response for a previous profile must stand down when it resolves.
    if (force) {
      profileRefreshEpochRef.current += 1
    }

    const profileRefreshEpoch = profileRefreshEpochRef.current
    const profile = $activeGatewayProfile.get()

    try {
      if ($activeSessionId.get()) {
        return
      }

      // A manual pick is sticky. It is never diffed against the catalog: rows
      // are hints, and a custom slug the row lacks is still the user's choice
      // (the gateway validates it on switch).
      const keepManualPick = () => !force && Boolean($currentModel.get()) && getCurrentModelSource() === 'manual'

      if (keepManualPick()) {
        return
      }

      // Snapshot the selection generation before awaiting so a picker click
      // that lands while getGlobalModelInfo is in flight wins over this older
      // default — value comparisons alone miss re-selecting the same row.
      const selectionGeneration = getComposerSelectionGeneration()
      const result = await getGlobalModelInfo(profile)

      if (
        profileRefreshEpochRef.current !== profileRefreshEpoch ||
        $activeSessionId.get() ||
        getComposerSelectionGeneration() !== selectionGeneration ||
        keepManualPick()
      ) {
        return
      }

      if (typeof result.model === 'string') {
        setCurrentModel(result.model)
      }

      if (typeof result.provider === 'string') {
        setCurrentProvider(result.provider)
      }

      if (typeof result.model === 'string' || typeof result.provider === 'string') {
        setCurrentModelSource('default')
      }
    } catch {
      // The delayed session.info event still updates this once the agent is ready.
    }
  }, [])

  // Returns whether the switch was applied so callers can await it before
  // applying follow-up changes. `true` means applied (or deferred/busy-queued
  // for the next turn). `false` means NOT applied — either pending
  // confirmation (warning with Confirm action already shown, pill rolled back),
  // stale (a newer selection or foreground owns any follow-up),
  // or a real failure (error toast). Callers must NOT treat `false` as a
  // generic failure: for `pending` the gateway intentionally returned
  // `confirm_required` and no error should be surfaced.
  // The composer model is plain UI state: with no live session it's just
  // stored (and shipped on the next session.create); with one it's scoped to
  // that session via config.set. It NEVER writes the profile default — that
  // lives in Settings → Model — so picking a model here can't silently mutate
  // global config.
  //
  // `selection.sessionId` targets a specific surface (tile). When omitted, the
  // primary `$activeSessionId` is used (overlay / legacy callers). A tile
  // switch must not touch the primary globals — and must not be blocked by a
  // busy primary turn.
  const selectModel = useCallback(
    async (selection: ModelSelection): Promise<boolean> => {
      const primaryRuntimeId = $activeSessionId.get()
      const liveSessionId = 'sessionId' in selection ? (selection.sessionId ?? null) : primaryRuntimeId
      const touchesPrimary = !liveSessionId || liveSessionId === primaryRuntimeId

      const prevModel = touchesPrimary ? $currentModel.get() : ($sessionStates.get()[liveSessionId!]?.model ?? '')

      const prevProvider = touchesPrimary
        ? $currentProvider.get()
        : ($sessionStates.get()[liveSessionId!]?.provider ?? '')

      const prevSource = getCurrentModelSource()
      const liveGatewayProfile = cacheProfile || $activeGatewayProfile.get()
      const activeProfile = $activeGatewayProfile.get()
      const connectionId = $connection.get()?.connectionId
      const mutationOwner = JSON.stringify([activeProfile, connectionId, primaryRuntimeId])
      let selectionGeneration = getComposerSelectionGeneration()
      const restoredGeneration = restoredPrimaryGenerationRef.current

      const previousSelectionGeneration =
        touchesPrimary &&
        restoredGeneration?.global === selectionGeneration &&
        restoredGeneration.owner === mutationOwner
          ? restoredGeneration.operation
          : selectionGeneration

      let expectedModel = selection.model
      let expectedProvider = selection.provider

      // ponytail: reuse the composer intent token; values alone miss a same-row reselect.
      const isStale = () => {
        const globalGeneration = getComposerSelectionGeneration()
        const restored = restoredPrimaryGenerationRef.current
        const effectiveGeneration = restored?.global === globalGeneration ? restored.operation : globalGeneration

        return touchesPrimary
          ? $activeSessionId.get() !== primaryRuntimeId ||
              $activeGatewayProfile.get() !== activeProfile ||
              $connection.get()?.connectionId !== connectionId ||
              effectiveGeneration !== selectionGeneration ||
              $currentModel.get() !== expectedModel ||
              $currentProvider.get() !== expectedProvider
          : false
      }

      const recordFailedPrimaryMutation = () => {
        if (touchesPrimary) {
          failedPrimaryMutationsRef.current.set(selectionGeneration, {
            owner: mutationOwner,
            previousGeneration: previousSelectionGeneration,
            previousModel: prevModel,
            previousProvider: prevProvider,
            previousSource: prevSource
          })
        }
      }

      const paintSelection = () => {
        expectedModel = selection.model
        expectedProvider = selection.provider

        if (touchesPrimary) {
          restoredPrimaryGenerationRef.current = null
          setCurrentModel(selection.model)
          setCurrentProvider(selection.provider)
          markComposerSelectionManual()
          selectionGeneration = getComposerSelectionGeneration()
        } else if (liveSessionId) {
          // Optimistic tile paint — session.info will confirm; rollback on error.
          sessionTileDelegate()?.updateSession(liveSessionId, state => ({
            ...state,
            model: selection.model,
            provider: selection.provider
          }))
        }
      }

      const cacheSelection = (provider: string, model: string) => {
        updateModelOptionsCache(liveSessionId, provider, model, touchesPrimary && !liveSessionId, liveGatewayProfile)
      }

      const rollbackSelection = (cascadeFailedAncestors = true) => {
        if (isStale()) {
          return
        }

        let previousGeneration = previousSelectionGeneration
        let previousModel = prevModel
        let previousProvider = prevProvider
        let previousSource = prevSource

        if (touchesPrimary && cascadeFailedAncestors) {
          failedPrimaryMutationsRef.current.delete(selectionGeneration)

          for (let failed = failedPrimaryMutationsRef.current.get(previousGeneration); failed;) {
            failedPrimaryMutationsRef.current.delete(previousGeneration)

            if (failed.owner !== mutationOwner) {
              break
            }

            previousGeneration = failed.previousGeneration
            previousModel = failed.previousModel
            previousProvider = failed.previousProvider
            previousSource = failed.previousSource
            failed = failedPrimaryMutationsRef.current.get(previousGeneration)
          }

          restoredPrimaryGenerationRef.current = {
            global: getComposerSelectionGeneration(),
            operation: previousGeneration,
            owner: mutationOwner
          }
        }

        expectedModel = previousModel
        expectedProvider = previousProvider

        if (touchesPrimary) {
          setCurrentModel(previousModel)
          setCurrentProvider(previousProvider)
          setCurrentModelSource(previousSource)
        } else if (liveSessionId) {
          sessionTileDelegate()?.updateSession(liveSessionId, state => ({
            ...state,
            model: prevModel,
            provider: prevProvider
          }))
        }

        cacheSelection(previousProvider, previousModel)
      }

      paintSelection()
      cacheSelection(selection.provider, selection.model)

      // No live session yet: the pick is pure UI state. session.create reads
      // $currentModel/$currentProvider and applies it as that session's override.
      if (!liveSessionId) {
        return true
      }

      // The PRIMARY profile's main agent lets the gateway decide persistence
      // (resolve_persist_behavior): session-only by default, persisted when
      // model.persist_switch_by_default is true or when no default has ever
      // been configured (the first-ever pick, so resolve_provider never falls
      // through to a leftover OPENAI_API_KEY env var — #86414). A plain pick
      // no longer silently rewrites config.yaml (#90235); Settings → Model
      // remains the explicit "set as default" door.
      //
      // Two things stay --session, deliberately:
      //  - a SECONDARY chat tile: picking a model there must not rewrite the
      //    profile default (the cross-session-contamination guard).
      //  - MoA (mixture-of-agents) presets: a transient orchestration choice
      //    that must never become the persisted global gateway default.
      const isSessionOnlyPreset = (selection.provider || '').toLowerCase() === 'moa'
      const scope = touchesPrimary && !isSessionOnlyPreset ? '' : ' --session'

      const requestSwitch = (confirmExpensiveModel = false) =>
        requestGateway<ModelSwitchResponse>('config.set', {
          session_id: liveSessionId,
          key: 'model',
          value: `${selection.model} --provider ${selection.provider}${scope}`,
          ...(confirmExpensiveModel ? { confirm_expensive_model: true } : {})
        })

      const finishSwitch = (result: ModelSwitchResponse | undefined) => {
        if (isStale()) {
          return
        }

        if (touchesPrimary) {
          failedPrimaryMutationsRef.current.clear()
          restoredPrimaryGenerationRef.current = null
        }

        // A pick made DURING a turn is queued by the gateway and applied at the
        // next turn start (`deferred`). Re-fetching now would answer with the
        // model still running and repaint the old name over the user's choice —
        // the switch publishes session.info when it lands, and that is what
        // re-syncs every surface.
        if (!result?.deferred) {
          void queryClient.invalidateQueries({
            queryKey: modelOptionsQueryKey(liveGatewayProfile, liveSessionId, cacheOwnerConnectionId)
          })
        }
      }

      try {
        const result = await requestSwitch()

        if (isStale()) {
          if (result?.confirm_required) {
            recordFailedPrimaryMutation()
          }

          return false
        }

        if (result?.confirm_required) {
          rollbackSelection(false)
          // ONE shared applier for guarded switches (#95293): the same
          // confirm flow the Bots editor routes through — never fork this
          // logic per surface.
          // Not awaited: `selectModel` answers "was the switch applied NOW",
          // and that answer only exists once the user answers the dialog.
          void surfaceModelSwitchConfirm({
            confirmMessage: result.confirm_message,
            failureMessage: copy.modelSwitchFailed,
            finish: finishSwitch,
            isStale: () =>
              isStale() ||
              (!touchesPrimary &&
                (!liveSessionId ||
                  $sessionStates.get()[liveSessionId]?.model !== expectedModel ||
                  $sessionStates.get()[liveSessionId]?.provider !== expectedProvider)),
            model: selection.model,
            repaint: () => {
              paintSelection()
              cacheSelection(selection.provider, selection.model)
            },
            requestConfirmed: async () => {
              try {
                const confirmed = await requestSwitch(true)

                if (confirmed?.confirm_required) {
                  recordFailedPrimaryMutation()
                }

                return confirmed
              } catch (err) {
                recordFailedPrimaryMutation()
                throw err
              }
            },
            rollback: rollbackSelection
          })

          return false
        }

        finishSwitch(result)

        return true
      } catch (err) {
        if (isBusySessionModelSwitch(err)) {
          return isStale() ? false : true
        }

        recordFailedPrimaryMutation()

        if (isStale()) {
          return false
        }

        // An OLDER gateway refuses a mid-turn switch outright (4009) instead of
        // deferring it. Don't punish the user for a backend they haven't
        // updated: keep the pick painted as the composer's selection, which is
        // what the NEXT turn runs anyway. Current gateways never take this
        // path — they answer `deferred`.
        rollbackSelection()
        notifyError(err, copy.modelSwitchFailed)

        return false
      }
    },
    [cacheOwnerConnectionId, cacheProfile, copy.modelSwitchFailed, queryClient, requestGateway, updateModelOptionsCache]
  )

  return { applySavedMainModel, refreshCurrentModel, selectModel }
}
