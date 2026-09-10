/**
 * Per-create overrides, translated to `session.create` params and folded over
 * the ones `desktopSessionCreateParams` derived from the visible selection.
 *
 * Reasoning effort rides alone here, with no model pin: the guided onboarding
 * chat wants `minimal` on whatever model the backend already resolved for the
 * profile. A model override would be a different kind of thing — the
 * composer's model and provider are a PAIR, so overriding one without the
 * other mints a session pointing a provider at a model it does not serve — and
 * no caller needs one.
 */
export interface SessionCreateOverrides {
  reasoningEffort?: string
  title?: string
}

export interface SessionCreateOverrideParams {
  reasoning_effort?: string
  title?: string
}

export function sessionCreateOverrideParams(
  overrides: SessionCreateOverrides | undefined
): SessionCreateOverrideParams {
  const params: SessionCreateOverrideParams = {}

  if (overrides?.title) {
    params.title = overrides.title
  }

  if (overrides?.reasoningEffort) {
    params.reasoning_effort = overrides.reasoningEffort
  }

  return params
}
