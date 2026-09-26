import { type ProfileScope, setModelAssignment } from '@/hermes'
import { translateNow } from '@/i18n'
import { dismissNotification, notify } from '@/store/notifications'
import type { ModelAssignmentRequest, ModelAssignmentResponse } from '@/types/hermes'

/**
 * Selection-guard warning as a confirm toast. Resolves true on Confirm, false
 * on dismiss. The desktop has no blocking confirm API; this is the same
 * notify-with-action pattern the in-session model picker uses.
 */
function confirmModelWarning(message: string): Promise<boolean> {
  const id = `model-warning-confirm-${Date.now()}`

  return new Promise(resolve => {
    let settled = false

    const finish = (value: boolean) => {
      if (settled) {
        return
      }

      settled = true
      dismissNotification(id)
      resolve(value)
    }

    notify({
      id,
      kind: 'warning',
      title: translateNow('modelAssignment.confirmTitle'),
      message: message || translateNow('modelAssignment.confirmDetail'),
      detail: translateNow('modelAssignment.confirmDetail'),
      action: {
        label: translateNow('modelAssignment.confirmAction'),
        onClick: () => finish(true)
      },
      onDismiss: () => finish(false)
    })
  })
}

export async function setMainModelAssignment(
  request: Omit<ModelAssignmentRequest, 'scope'>,
  scopeProfile?: ProfileScope,
  options?: { skipConfirmPrompt?: boolean }
): Promise<ModelAssignmentResponse> {
  // Only pass the extra arg when a scope override exists, so unscoped callers
  // keep the exact legacy call shape.
  const assign = (body: Omit<ModelAssignmentRequest, 'scope'>) =>
    scopeProfile == null
      ? setModelAssignment({ ...body, scope: 'main' })
      : setModelAssignment({ ...body, scope: 'main' }, scopeProfile)

  let result = await assign(request)

  // Backend demands an explicit ack before persisting a model that trips a
  // selection guard (expensive / data-training tiers like *-contributor).
  // Settings used to throw confirm_message as a red error, so Apply could
  // never persist. Prompt, then retry with confirm_expensive_model.
  if (result.confirm_required) {
    if (request.confirm_expensive_model || options?.skipConfirmPrompt) {
      // Already acked, or headless onboarding (nothing mounted to click).
      // Fail closed instead of recursing / dangling a prompt.
      throw new Error(result.confirm_message?.trim() || translateNow('modelAssignment.saveFailed'))
    }

    const accepted = await confirmModelWarning(result.confirm_message?.trim() ?? '')

    if (!accepted) {
      throw new Error(translateNow('modelAssignment.declined'))
    }

    result = await assign({ ...request, confirm_expensive_model: true })

    if (result.confirm_required || result.ok !== true) {
      throw new Error(result.confirm_message?.trim() || translateNow('modelAssignment.saveFailed'))
    }
  } else if (result.ok !== true) {
    throw new Error(result.confirm_message?.trim() || translateNow('modelAssignment.saveFailed'))
  }

  return result
}
