import { translateNow } from '@/i18n'
import { surfaceModelSwitchConfirm } from '@/lib/guarded-model-switch'
import { modelOptionsQueryKey } from '@/lib/model-options'
import { releaseQueuedLaunchPick } from '@/store/composer-queued-pick'
import { activeGatewayConnectionId, requestGatewayForAgent } from '@/store/gateway'
import { notifyError } from '@/store/notifications'
import { normalizeProfileKey } from '@/store/profile'
import {
  $currentModel,
  $currentProvider,
  composerScopeForOwner,
  getComposerSelectionScope,
  setCurrentModelTransient,
  setCurrentProviderTransient
} from '@/store/session'
import { secondaryProfileOwnerForEvent } from '@/store/session-event-provenance'
import { knownOwnerForSession } from '@/store/session-states'

import { applySessionModelReceipt, commitSessionModel, type ModelSwitchResponse } from '../utils'

import { matchesReceiptOwner } from './session-receipt-owner'
import type { GatewayEventContext } from './types'

/**
 * A model pick made during a turn ("applies from your next message") resolves at the next turn
 * boundary. When it does not apply, the owner says so on `session.updated`:
 *  - `refused` (with a reason): show the named refusal and paint the committed model again;
 *  - `confirmation_required`: a guarded target asks, through the same dialog as an idle pick.
 * Applied picks announce themselves through session.info like any switch.
 */
/** Paint the committed selection back over a pick that did not apply. Only for the active profile's
 *  own session, and only while the composer still shows that pick (a newer pick or a navigation wins). */
function repaintCommitted(
  { deps, sessionId }: Pick<GatewayEventContext, 'deps' | 'sessionId'>,
  target: unknown,
  model: string,
  provider: string,
  requestedProvider?: string
) {
  let restored = false

  const next = deps.updateSessionState(sessionId!, state => {
    // Unversioned legacy runtimes painted their queued pick directly into `model`.
    // A versioned model with no queue is committed; a late refusal cannot roll it back.
    const pending = state.queuedModel?.model ?? (state.modelRevision === undefined ? state.model : undefined)

    const pendingProvider = state.queuedModel?.provider ?? state.provider

    if (
      typeof target !== 'string' ||
      pending !== target ||
      (requestedProvider && requestedProvider !== pendingProvider)
    ) {
      return state
    }

    restored = true

    return { ...state, model, ...(provider ? { provider } : {}), queuedModel: undefined }
  })

  if (
    restored &&
    deps.activeSessionIdRef.current === sessionId &&
    $currentModel.get() === target &&
    (!requestedProvider || $currentProvider.get() === requestedProvider)
  ) {
    setCurrentModelTransient(model)

    if (provider) {
      setCurrentProviderTransient(provider)
    }
  }

  return next
}

// The queued-pick record lives with the owner whose composer saved it: another owner's event names its own.
function queuedPickScope(ownSource: boolean, event: GatewayEventContext['event']): null | string {
  return event.connectionId
    ? composerScopeForOwner(event.connectionId, event.profile ?? '')
    : ownSource
      ? getComposerSelectionScope()
      : null
}

export function handleDeferredPickEvent(ctx: GatewayEventContext): boolean {
  const { deps, event, payload, sessionId, fromActiveSource } = ctx
  const update = payload as Record<string, unknown> | undefined

  // Receipt broadcasts are not in the generated event map; the dispatch fence passes them through.
  if ((event.type as string) !== 'session.updated' || update?.operation !== 'model' || !sessionId) {
    return false
  }

  // Another profile's session can share this id (imports): its outcome is not this composer's.
  const ownSource = fromActiveSource()
  const profile = event.profile ?? deps.activeGatewayProfile

  const invalidate = () =>
    void deps.queryClient.invalidateQueries({ queryKey: modelOptionsQueryKey(profile, sessionId) })

  const scope = queuedPickScope(ownSource, event)
  const secondary = secondaryProfileOwnerForEvent(event)

  const ownsRuntime = () => {
    const primary = deps.activeSessionIdRef.current === sessionId

    // A runtime id alone cannot distinguish two foreground owners.
    if (primary && !fromActiveSource()) {
      return false
    }

    return matchesReceiptOwner(ctx) ?? fromActiveSource()
  }

  // Canonical gateways broadcast the commit as session.updated, even without a new turn.
  if (
    (update.status === undefined || update.status === 'applied') &&
    update.session_id === sessionId &&
    typeof update.revision === 'number' &&
    typeof update.model === 'string' &&
    update.model &&
    typeof update.provider === 'string' &&
    update.provider
  ) {
    const { model, provider, revision } = update

    if (ownsRuntime()) {
      deps.updateSessionState(sessionId, state => commitSessionModel(state, model, provider, revision))
    }

    return true
  }

  if (update.status === 'refused') {
    const model = typeof update.model === 'string' ? update.model : ''

    if (model && ownsRuntime()) {
      const state = repaintCommitted(
        ctx,
        update.target_model,
        model,
        typeof update.provider === 'string' ? update.provider : ''
      )

      if (
        state?.modelRevision !== undefined &&
        !state.queuedModel &&
        state.model === update.target_model &&
        state.model !== model
      ) {
        return true // This target has since committed. Keep its saved launch selection as well.
      }
    }

    releaseQueuedLaunchPick(sessionId, update.target_model, scope)
    invalidate()
    notifyError(new Error(String(update.reason ?? 'refused')), translateNow('desktop.modelSwitchFailed'))

    return true
  }

  if (update.status === 'confirmation_required' && typeof update.confirm === 'string') {
    const pick = (update.payload ?? {}) as { model?: string; provider?: string; reasoning?: string }
    const connectionId = event.connectionId ?? (secondary ? null : activeGatewayConnectionId())
    const target = String(update.target_model ?? pick.model ?? '')
    const requested = pick.model || target
    const committed = typeof update.committed_model === 'string' ? update.committed_model : ''

    if (!ownsRuntime()) {
      return true
    }

    const readRuntime = () => deps.sessionStateByRuntimeIdRef?.current.get(sessionId)
    const pending = readRuntime()?.queuedModel

    if (pending && (pending.model !== requested || (pick.provider && pending.provider !== pick.provider))) {
      return true
    }

    if (committed) {
      repaintCommitted(
        ctx,
        requested,
        committed,
        typeof update.committed_provider === 'string' ? update.committed_provider : '',
        pick.provider
      )
    }

    const before = {
      model: $currentModel.get(),
      provider: $currentProvider.get(),
      scope: getComposerSelectionScope(),
      sessionId: deps.activeSessionIdRef.current
    }

    const runtime = readRuntime()
    const runtimeOwner = knownOwnerForSession(sessionId)

    const ownerScope = (owner: typeof runtimeOwner) =>
      typeof owner === 'string'
        ? normalizeProfileKey(owner)
        : owner
          ? composerScopeForOwner(owner.connectionId, owner.profile)
          : null

    const runtimeIsStale = () => {
      const now = readRuntime()

      return (
        ownerScope(knownOwnerForSession(sessionId)) !== ownerScope(runtimeOwner) ||
        now?.modelRevision !== runtime?.modelRevision ||
        now?.model !== runtime?.model ||
        now?.provider !== runtime?.provider ||
        now?.queuedModel !== runtime?.queuedModel
      )
    }

    const isStale = () => {
      if (!ownsRuntime() || runtimeIsStale()) {
        return true
      }

      // Only a question opened on the primary composer belongs to its selection.
      return (
        before.sessionId === sessionId &&
        (getComposerSelectionScope() !== before.scope ||
          deps.activeSessionIdRef.current !== before.sessionId ||
          $currentModel.get() !== before.model ||
          $currentProvider.get() !== before.provider)
      )
    }

    void surfaceModelSwitchConfirm({
      confirmMessage: typeof update.confirm_message === 'string' ? update.confirm_message : undefined,
      failureMessage: translateNow('desktop.modelSwitchFailed'),
      // A newer pick replaced this one while the dialog was open: confirming it would undo that pick.
      isStale,
      model: target,
      // The whole selection the owner asked about (its reasoning level included) with the owner's
      // token: the same canonical model mutation the TUI resends.
      requestConfirmed: () =>
        requestGatewayForAgent<ModelSwitchResponse>(connectionId, profile, 'session.model', {
          confirm: update.confirm,
          payload: pick,
          session_id: sessionId
        }),
      finish: result => {
        if (result?.status === 'refused' || result?.ok === false) {
          throw new Error(result.reason || translateNow('desktop.modelSwitchFailed'))
        }

        if (result?.model && (result.provider || pick.provider) && !isStale()) {
          deps.updateSessionState(sessionId, state =>
            applySessionModelReceipt(state, result, { model: target, provider: pick.provider || '' })
          )
        }

        invalidate()
      }
    }).then(applied => {
      if (!applied) {
        // Navigation only retires the question, not the original owner's saved launch pick.
        // Keep the captured runtime/queue identity fence: even a same-route replacement wins.
        // No repaint here; releasing the captured scope cannot change another owner's selection.
        if (!runtimeIsStale()) {
          releaseQueuedLaunchPick(sessionId, requested, scope, pick.provider)
        }

        invalidate()
      }
    })

    return true
  }

  return false
}
