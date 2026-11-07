import { translateNow } from '@/i18n'
import { surfaceModelSwitchConfirm } from '@/lib/guarded-model-switch'
import { modelOptionsQueryKey } from '@/lib/model-options'
import { activeGatewayConnectionId, requestGatewayForAgent } from '@/store/gateway'
import { notifyError } from '@/store/notifications'
import { $currentModel, $currentProvider, setCurrentModelTransient, setCurrentProviderTransient } from '@/store/session'

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
  provider: string
) {
  deps.updateSessionState(sessionId!, state => ({ ...state, model, ...(provider ? { provider } : {}) }))

  if (deps.activeSessionIdRef.current === sessionId && $currentModel.get() === target) {
    setCurrentModelTransient(model)

    if (provider) {
      setCurrentProviderTransient(provider)
    }
  }
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

  if (update.status === 'refused') {
    const model = typeof update.model === 'string' ? update.model : ''

    if (model && ownSource) {
      repaintCommitted(ctx, update.target_model, model, typeof update.provider === 'string' ? update.provider : '')
    }

    invalidate()
    notifyError(new Error(String(update.reason ?? 'refused')), translateNow('desktop.modelSwitchFailed'))

    return true
  }

  if (update.status === 'confirmation_required' && typeof update.confirm === 'string') {
    const pick = (update.payload ?? {}) as { model?: string; provider?: string; reasoning?: string }
    const connectionId = event.connectionId ?? activeGatewayConnectionId()
    const before = { model: $currentModel.get(), provider: $currentProvider.get() }
    const target = String(update.target_model ?? pick.model ?? '')
    const committed = typeof update.committed_model === 'string' ? update.committed_model : ''

    void surfaceModelSwitchConfirm({
      confirmMessage: typeof update.confirm_message === 'string' ? update.confirm_message : undefined,
      failureMessage: translateNow('desktop.modelSwitchFailed'),
      // A newer pick replaced this one while the dialog was open: confirming it would undo that pick.
      isStale: () => $currentModel.get() !== before.model || $currentProvider.get() !== before.provider,
      model: target,
      // The whole selection the owner asked about (its reasoning level included) with the owner's
      // token: the same canonical model mutation the TUI resends.
      requestConfirmed: () =>
        requestGatewayForAgent(connectionId, profile, 'session.model', {
          confirm: update.confirm,
          payload: pick,
          session_id: sessionId
        }),
      finish: invalidate
    }).then(applied => {
      // Declined, or the confirmed resend failed: the committed model is still the session's.
      if (!applied && committed && ownSource) {
        repaintCommitted(
          ctx,
          target,
          committed,
          typeof update.committed_provider === 'string' ? update.committed_provider : ''
        )
        invalidate()
      }
    })

    return true
  }

  return false
}
