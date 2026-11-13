/** Closed, content-free IA counters. Mirrored by shared_metrics_desktop_contract.py. */
export const IA_ACTIONS = {
  rail: ['chat', 'profiles', 'capabilities', 'artifacts', 'cron', 'plugin'],
  drawer: ['settings'],
  session: ['opened_pinned', 'opened_agent', 'opened_search'],
  section: ['pinned_expanded', 'pinned_collapsed', 'agent_expanded', 'agent_collapsed'],
  search: ['opened', 'used', 'results', 'empty', 'partial_failure', 'selected', 'cleared', 'early'],
  pin_count: ['zero', 'one', 'two', 'three', 'overflow'],
  profile_pin_count: ['zero', 'one', 'two', 'three', 'overflow'],
  pin: ['pinned_request', 'agent_request', 'search_request', 'pinned', 'unpinned', 'failed'],
  new_session: [
    'sidebar',
    'command',
    'agent_row',
    'picker_opened',
    'picker_on_new',
    'specific',
    'default',
    'dismissed'
  ],
  gateway: ['popover_open', 'popover_connecting', 'popover_offline', 'system', 'logs', 'reconnect', 'offline_agent']
} as const

export const IA_AREAS = [
  'chat',
  'profiles',
  'capabilities',
  'artifacts',
  'cron',
  'settings',
  'command-center',
  'plugin',
  'other'
] as const
export type IaArea = (typeof IA_AREAS)[number]
export const IA_DWELL_BUCKETS = ['lt5s', '5s_to30s', '30s_to5m', 'gte5m'] as const

export type IaAction =
  | {
      [K in keyof typeof IA_ACTIONS]: `ia.${K}.${(typeof IA_ACTIONS)[K][number]}`
    }[keyof typeof IA_ACTIONS]
  | `ia.view.${IaArea}`
  | `ia.backtrack.${IaArea}`
  | `ia.transition.${IaArea}_to_${IaArea}`
  | `ia.dwell.${IaArea}_${(typeof IA_DWELL_BUCKETS)[number]}`

export const IA_ACTION_IDS = [
  ...Object.entries(IA_ACTIONS).flatMap(([group, events]) => events.map(event => `ia.${group}.${event}`)),
  ...IA_AREAS.flatMap(area => [
    `ia.view.${area}`,
    `ia.backtrack.${area}`,
    ...IA_AREAS.filter(next => next !== area).map(next => `ia.transition.${area}_to_${next}`),
    ...IA_DWELL_BUCKETS.map(bucket => `ia.dwell.${area}_${bucket}`)
  ])
]
