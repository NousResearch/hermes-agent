import { getApiRequestProfile, setModelAssignment } from '@/hermes'
import { translateNow } from '@/i18n'
import { requestCronReview } from '@/store/cron'
import {
  beginCronModelImpactAssignment,
  getCronModelImpactScope,
  invalidateCronModelImpactScopeState,
  onCronModelImpactScopeInvalidated
} from '@/store/cron-model-impact-scope'
import { dismissNotification, notify } from '@/store/notifications'
import type {
  CronModelDriftAxis,
  CronModelImpact,
  CronModelImpactJob,
  ModelAssignmentRequest,
  ModelAssignmentResponse
} from '@/types/hermes'

export const CRON_MODEL_IMPACT_NOTIFICATION_ID = 'cron-model-impact'

const MAX_JOBS = 50
const MAX_ID_CODE_POINTS = 256
const MAX_NAME_CODE_POINTS = 120
const ALLOWED_AXES = new Set<CronModelDriftAxis>(['provider', 'model'])

function profileIdentity(): string {
  return getApiRequestProfile()?.trim() || 'default'
}

function codePointLength(value: string): number {
  return [...value].length
}

function hasControlCharacters(value: string): boolean {
  return /\p{C}/u.test(value)
}

function validJob(value: unknown): value is CronModelImpactJob {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return false
  }

  const job = value as Partial<CronModelImpactJob>

  if (
    typeof job.id !== 'string' ||
    job.id.trim() !== job.id ||
    !job.id ||
    codePointLength(job.id) > MAX_ID_CODE_POINTS ||
    hasControlCharacters(job.id) ||
    typeof job.name !== 'string' ||
    job.name.trim() !== job.name ||
    !job.name ||
    codePointLength(job.name) > MAX_NAME_CODE_POINTS ||
    hasControlCharacters(job.name) ||
    !Array.isArray(job.drifted_axes) ||
    job.drifted_axes.length < 1 ||
    job.drifted_axes.length > 2 ||
    new Set(job.drifted_axes).size !== job.drifted_axes.length ||
    !job.drifted_axes.every(axis => ALLOWED_AXES.has(axis))
  ) {
    return false
  }

  return true
}

export function parseCronModelImpact(value: unknown): CronModelImpact | null {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return null
  }

  const impact = value as Partial<CronModelImpact>

  if (
    typeof impact.available !== 'boolean' ||
    typeof impact.guard_enabled !== 'boolean' ||
    !Number.isSafeInteger(impact.affected_count) ||
    (impact.affected_count ?? -1) < 0 ||
    typeof impact.truncated !== 'boolean' ||
    !Array.isArray(impact.jobs) ||
    impact.jobs.length > MAX_JOBS ||
    !impact.jobs.every(validJob)
  ) {
    return null
  }

  const count = impact.affected_count as number
  const ids = impact.jobs.map(job => job.id)

  if (
    new Set(ids).size !== ids.length ||
    (!impact.truncated && count !== impact.jobs.length) ||
    (impact.truncated && (impact.jobs.length !== MAX_JOBS || count <= impact.jobs.length))
  ) {
    return null
  }

  return impact as CronModelImpact
}

function currentResponseScope(profile: string, connection: string, generation: number): boolean {
  const scope = getCronModelImpactScope()

  return profileIdentity() === profile && scope.connection === connection && scope.generation === generation
}

function currentActionScope(profile: string, connection: string): boolean {
  return profileIdentity() === profile && getCronModelImpactScope().connection === connection
}

function detailFor(impact: CronModelImpact): string {
  const visible = impact.jobs.slice(0, 3).map(job => job.name)
  const remaining = impact.affected_count - visible.length

  return remaining > 0 ? translateNow('cron.modelImpact.detailMore', visible.join(', '), remaining) : visible.join(', ')
}

function publishImpact(impact: CronModelImpact, profile: string, connection: string, generation: number): void {
  if (!impact.available) {
    return
  }

  if (!impact.guard_enabled || impact.affected_count === 0) {
    dismissNotification(CRON_MODEL_IMPACT_NOTIFICATION_ID)

    return
  }

  notify({
    id: CRON_MODEL_IMPACT_NOTIFICATION_ID,
    kind: 'warning',
    title: translateNow('cron.modelImpact.title'),
    message: translateNow('cron.modelImpact.message', impact.affected_count),
    detail: detailFor(impact),
    action: {
      label: translateNow('cron.modelImpact.review'),
      onClick: () => {
        if (currentActionScope(profile, connection)) {
          requestCronReview()
        }
      }
    }
  })
}

export async function setMainModelAssignment(
  request: Omit<ModelAssignmentRequest, 'scope'>,
  scopeProfile?: null | string,
  options?: { skipConfirmPrompt?: boolean }
): Promise<ModelAssignmentResponse> {
  const { connection, generation } = beginCronModelImpactAssignment()
  const profile = profileIdentity()

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
      throw new Error(result.confirm_message?.trim() || translateNow('cron.modelImpact.saveFailed'))
    }

    const accepted = await confirmModelWarning(result.confirm_message?.trim() ?? '')

    if (!accepted) {
      throw new Error(translateNow('cron.modelImpact.declined'))
    }

    result = await assign({ ...request, confirm_expensive_model: true })

    if (result.confirm_required || result.ok !== true) {
      throw new Error(result.confirm_message?.trim() || translateNow('cron.modelImpact.saveFailed'))
    }
  } else if (result.ok !== true) {
    throw new Error(result.confirm_message?.trim() || translateNow('cron.modelImpact.saveFailed'))
  }

  // A scoped assignment targets ANOTHER profile's backend: its cron impact
  // belongs to that profile, and the review action would open the ACTIVE
  // profile's cron view — skip the warning rather than mis-route it.
  if (scopeProfile != null) {
    return result
  }

  if (!currentResponseScope(profile, connection, generation)) {
    return result
  }

  // Missing means an older backend. It is not evidence that an existing impact
  // has gone away, so leave the current warning untouched.
  if (result.cron_model_impact !== undefined) {
    const impact = parseCronModelImpact(result.cron_model_impact)

    if (impact) {
      publishImpact(impact, profile, connection, generation)
    }
  }

  return result
}

export function invalidateCronModelImpactScope(options: { clearNotification?: boolean } = {}): void {
  if (options.clearNotification === false) {
    beginCronModelImpactAssignment()

    return
  }

  invalidateCronModelImpactScopeState()
}

// Scope changes originating outside this module (profile/backend switches)
// clear any warning that belongs to the old runtime.
onCronModelImpactScopeInvalidated(() => dismissNotification(CRON_MODEL_IMPACT_NOTIFICATION_ID))

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
      title: translateNow('cron.modelImpact.confirmTitle'),
      message: message || translateNow('cron.modelImpact.confirmDetail'),
      detail: translateNow('cron.modelImpact.confirmDetail'),
      action: {
        label: translateNow('cron.modelImpact.confirmAction'),
        onClick: () => finish(true)
      },
      onDismiss: () => finish(false)
    })
  })
}
