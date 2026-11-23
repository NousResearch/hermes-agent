/** Pure create normalization shared by renderer protocol and native durable delivery.
 * Identity generation belongs to the caller: a durable replay always supplies its persisted ID. */
export function canonicalCreateParams(input: Record<string, unknown>): Record<string, unknown> {
  const { reasoning_effort: effort, fast, service_tier: picked, room_plumbing: roomPlumbing, ...params } = input

  const tier =
    typeof picked === 'string' && picked ? picked : fast === true ? 'priority' : fast === false ? 'normal' : undefined

  if (roomPlumbing === true) {
    params.follow_profile_config = true
  }

  if (typeof effort === 'string' && effort) {
    params.reasoning = effort
  }

  if (tier) {
    params.service_tier = tier
  }

  if (typeof params.cwd_explicit !== 'boolean') {
    delete params.cwd_explicit
  }

  const allowed = new Set([
    'request_id',
    'source',
    'cwd',
    'model',
    'provider',
    'toolsets',
    'reasoning',
    'profile',
    'cols',
    'title',
    'hidden',
    'follow_profile_config',
    'service_tier',
    'cwd_explicit'
  ])

  const unsupported = Object.keys(params).filter(key => !allowed.has(key))

  if (unsupported.length) {
    throw new Error(`Canonical gateway does not support explicit session options: ${unsupported.join(', ')}`)
  }

  return Object.fromEntries(
    Object.entries(params).filter(
      ([key, value]) =>
        [
          'request_id',
          'cwd',
          'model',
          'provider',
          'toolsets',
          'reasoning',
          'title',
          'hidden',
          'follow_profile_config',
          'service_tier',
          'cwd_explicit'
        ].includes(key) ||
        (key === 'profile' && typeof value === 'string' && value !== '')
    )
  )
}
