import type {
  WorldActionKind,
  WorldCondition,
  WorldEvent,
  WorldScope,
  WorldSeverity,
  WorldSourceRef
} from './world-events'

export type NpcActivityState =
  | 'idle'
  | 'walking'
  | 'working'
  | 'carrying'
  | 'inspecting'
  | 'repairing'
  | 'talking'
  | 'waiting'
  | 'panicking'
  | 'celebrating'
  | 'resting'
  | 'returning'

export type NpcPersonality = 'bold' | 'cautious' | 'curious' | 'methodical' | 'protective' | 'social'

export interface NpcActivity {
  actor: WorldSourceRef
  personality: NpcPersonality
  state: NpcActivityState
  animationTags: string[]
  target?: WorldSourceRef
  groundedDialogue?: string
}

export interface WorldPresentation {
  sceneTag: string
  scope: WorldScope
  animationTags: string[]
  npcActivities: NpcActivity[]
  participants: WorldSourceRef[]
  actionKinds: WorldActionKind[]
  cosmetic: { soundTag?: string; cameraBeat?: string; intensity: 0 | 1 | 2 | 3 }
}

interface SceneSpec {
  animationTags: string[]
  cameraBeat?: string
  intensity: 0 | 1 | 2 | 3
  npcState: NpcActivityState
  sceneTag: string
  soundTag?: string
}

const SCENES: Record<string, SceneSpec> = {
  'task.blocked': {
    animationTags: ['task.blocked', 'crisis.fire', 'repair', 'extinguish'],
    cameraBeat: 'crisis_reveal',
    intensity: 2,
    npcState: 'panicking',
    sceneTag: 'crisis.fire.local',
    soundTag: 'alert.local'
  },
  'task.block_loop': {
    animationTags: ['task.blocked', 'crisis.earthquake', 'repair', 'route.disrupted'],
    cameraBeat: 'district_shake',
    intensity: 3,
    npcState: 'panicking',
    sceneTag: 'crisis.earthquake.district',
    soundTag: 'alert.siren'
  },
  'worker.crashed': {
    animationTags: ['worker.crashed', 'station.damaged', 'repair'],
    cameraBeat: 'worker_discovery',
    intensity: 2,
    npcState: 'inspecting',
    sceneTag: 'crisis.station.damaged',
    soundTag: 'alert.worker'
  },
  'worker.gave_up': {
    animationTags: ['worker.gave_up', 'worksite.abandoned', 'repair'],
    intensity: 2,
    npcState: 'waiting',
    sceneTag: 'crisis.worksite.abandoned',
    soundTag: 'alert.warning'
  },
  'worker.timed_out': {
    animationTags: ['worker.timed_out', 'clock.frozen', 'queue.responders'],
    intensity: 2,
    npcState: 'waiting',
    sceneTag: 'crisis.transport.stalled',
    soundTag: 'alert.warning'
  },
  'task.running': {
    animationTags: ['task.running', 'travel', 'task.work'],
    intensity: 0,
    npcState: 'working',
    sceneTag: 'activity.task.running'
  },
  'task.ready': {
    animationTags: ['task.ready', 'prepare.tools'],
    intensity: 0,
    npcState: 'walking',
    sceneTag: 'activity.task.ready'
  },
  'task.waiting': {
    animationTags: ['task.waiting', 'queue.waiting'],
    intensity: 0,
    npcState: 'waiting',
    sceneTag: 'activity.task.waiting'
  },
  'task.in_review': {
    animationTags: ['task.review', 'inspect'],
    cameraBeat: 'inspection_arrival',
    intensity: 1,
    npcState: 'inspecting',
    sceneTag: 'activity.task.inspection'
  },
  'pr.review_findings': {
    animationTags: ['pr.review.findings', 'structure.damaged', 'repair', 'talking'],
    cameraBeat: 'review_damage',
    intensity: 2,
    npcState: 'talking',
    sceneTag: 'crisis.structure.damaged',
    soundTag: 'alert.warning'
  },
  'pr.merge_conflict': {
    animationTags: ['pr.merge.conflict', 'construction.collision', 'argue'],
    cameraBeat: 'construction_argument',
    intensity: 2,
    npcState: 'talking',
    sceneTag: 'crisis.construction.conflict',
    soundTag: 'alert.warning'
  },
  'pr.approved': {
    animationTags: ['pr.approved', 'inspection.passed', 'prepare.release'],
    intensity: 1,
    npcState: 'celebrating',
    sceneTag: 'milestone.review.approved'
  },
  'pr.merged_draft': {
    animationTags: ['pr.merged', 'construction.milestone'],
    intensity: 1,
    npcState: 'celebrating',
    sceneTag: 'milestone.merge.draft'
  },
  'pr.merged_stable': {
    animationTags: ['pr.merged', 'celebration.citywide', 'fireworks', 'dance'],
    cameraBeat: 'citywide_celebration',
    intensity: 3,
    npcState: 'celebrating',
    sceneTag: 'celebration.citywide',
    soundTag: 'celebration.city'
  },
  'release.succeeded': {
    animationTags: ['release.succeeded', 'infrastructure.online', 'celebration'],
    intensity: 2,
    npcState: 'celebrating',
    sceneTag: 'milestone.release.online'
  },
  'release.started': {
    animationTags: ['release.started', 'launch.sequence', 'inspect'],
    intensity: 1,
    npcState: 'inspecting',
    sceneTag: 'activity.release.launching'
  },
  'release.failed': {
    animationTags: ['release.failed', 'launch.malfunction', 'repair'],
    cameraBeat: 'launch_failure',
    intensity: 3,
    npcState: 'panicking',
    sceneTag: 'crisis.launch.failed',
    soundTag: 'alert.siren'
  },
  'gateway.disconnected': {
    animationTags: ['gateway.disconnected', 'communications.blackout', 'signal.interference'],
    cameraBeat: 'city_blackout',
    intensity: 3,
    npcState: 'waiting',
    sceneTag: 'crisis.blackout.city',
    soundTag: 'alert.siren'
  },
  'gateway.connected': {
    animationTags: ['gateway.connected', 'communications.online'],
    intensity: 1,
    npcState: 'celebrating',
    sceneTag: 'recovery.comms.online'
  },
  'gateway.degraded': {
    animationTags: ['gateway.degraded', 'communications.static', 'inspect'],
    intensity: 2,
    npcState: 'inspecting',
    sceneTag: 'crisis.comms.degraded',
    soundTag: 'alert.warning'
  },
  'auth.failed': {
    animationTags: ['auth.failed', 'checkpoint.locked'],
    intensity: 2,
    npcState: 'waiting',
    sceneTag: 'crisis.access.denied',
    soundTag: 'alert.warning'
  },
  'approval.required': {
    animationTags: ['approval.required', 'checkpoint.locked', 'inspect'],
    intensity: 1,
    npcState: 'inspecting',
    sceneTag: 'activity.approval.required'
  },
  'approval.granted': {
    animationTags: ['approval.granted', 'checkpoint.open', 'return'],
    intensity: 1,
    npcState: 'returning',
    sceneTag: 'recovery.approval.open'
  },
  'approval.rejected': {
    animationTags: ['approval.rejected', 'checkpoint.locked', 'talking'],
    intensity: 1,
    npcState: 'talking',
    sceneTag: 'activity.approval.rejected',
    soundTag: 'alert.warning'
  },
  'agent.active': {
    animationTags: ['agent.active', 'travel'],
    intensity: 0,
    npcState: 'walking',
    sceneTag: 'activity.agent.active'
  },
  'agent.idle': {
    animationTags: ['agent.idle', 'resting'],
    intensity: 0,
    npcState: 'resting',
    sceneTag: 'activity.agent.idle'
  },
  'agent.warning': {
    animationTags: ['agent.warning', 'alert.marker'],
    intensity: 1,
    npcState: 'inspecting',
    sceneTag: 'alert.warning'
  },
  'agent.error': {
    animationTags: ['agent.error', 'alert.marker'],
    intensity: 2,
    npcState: 'inspecting',
    sceneTag: 'alert.error'
  },
  'credits.depleted': {
    animationTags: ['credits.depleted', 'power.rationing', 'queue.paused'],
    intensity: 3,
    npcState: 'waiting',
    sceneTag: 'crisis.power.paused',
    soundTag: 'alert.siren'
  },
  'credits.low': {
    animationTags: ['credits.low', 'power.rationing', 'inspect'],
    intensity: 2,
    npcState: 'inspecting',
    sceneTag: 'crisis.power.rationing',
    soundTag: 'alert.warning'
  },
  'credits.reset': {
    animationTags: ['credits.reset', 'power.restored', 'celebration.local'],
    intensity: 2,
    npcState: 'celebrating',
    sceneTag: 'recovery.power.restored'
  },
  'task.completed': {
    animationTags: ['task.completed', 'construction.finished', 'celebration.local'],
    intensity: 1,
    npcState: 'celebrating',
    sceneTag: 'milestone.task.completed'
  },
  'task.recovered': {
    animationTags: ['task.recovered', 'repair', 'extinguish', 'route.restored'],
    cameraBeat: 'recovery_relief',
    intensity: 2,
    npcState: 'repairing',
    sceneTag: 'recovery.route.restored'
  },
  'task.removed': {
    animationTags: ['task.removed', 'worksite.decommissioned'],
    intensity: 1,
    npcState: 'returning',
    sceneTag: 'activity.worksite.closed'
  }
}

const FALLBACK_SCENE: SceneSpec = {
  animationTags: ['alert', 'fallback.generic'],
  intensity: 1,
  npcState: 'inspecting',
  sceneTag: 'alert.unclassified'
}

function hash(value: string): number {
  let result = 2_166_136_261

  for (let index = 0; index < value.length; index += 1) {
    result ^= value.charCodeAt(index)
    result = Math.imul(result, 16_777_619)
  }

  return result >>> 0
}

export function stableEventSeed(event: Pick<WorldEvent, 'id' | 'kind' | 'source'>): number {
  return hash(`${event.source}\0${event.id}\0${event.kind}`)
}

function sceneFor(event: WorldEvent): SceneSpec {
  const base = SCENES[event.kind] ?? FALLBACK_SCENE

  if (event.kind === 'task.blocked' && (event.scope === 'district' || event.severity === 'critical')) {
    return { ...base, intensity: 3, sceneTag: 'crisis.fire.district', soundTag: 'alert.siren' }
  }

  return base
}

function animationTagsFor(tags: readonly string[], assetTags: ReadonlySet<string>): string[] {
  if (assetTags.size === 0) {
    return [...tags]
  }

  const resolved = tags.filter(tag => assetTags.has(tag))

  return resolved.length === tags.length ? resolved : [...resolved, 'fallback.generic']
}

function uniqueRefs(refs: readonly WorldSourceRef[]): WorldSourceRef[] {
  const seen = new Set<string>()

  return refs.filter(ref => {
    const key = `${ref.board ?? ''}\0${ref.taskId ?? ''}\0${ref.agentId ?? ''}\0${ref.prId ?? ''}`

    if (seen.has(key)) {
      return false
    }

    seen.add(key)

    return true
  })
}

function dialogueFor(event: WorldEvent, state: NpcActivityState): string | undefined {
  if (event.detail) {
    return event.detail
  }

  if (state === 'celebrating') {
    return event.title
  }

  return undefined
}

export function resolveNpcPersonality(
  role: string | undefined,
  actor: WorldSourceRef,
  event: Pick<WorldEvent, 'id' | 'kind' | 'source'>
): NpcPersonality {
  const normalized = role?.trim().toLowerCase() ?? ''

  if (normalized.includes('review') || normalized.includes('qa') || normalized.includes('audit')) {
    return 'methodical'
  }

  if (normalized.includes('research') || normalized.includes('science')) {
    return 'curious'
  }

  if (normalized.includes('security') || normalized.includes('ops') || normalized.includes('incident')) {
    return 'protective'
  }

  if (normalized.includes('release') || normalized.includes('deploy')) {
    return 'bold'
  }

  if (normalized.includes('support') || normalized.includes('community') || normalized.includes('social')) {
    return 'social'
  }

  const variants: NpcPersonality[] = ['methodical', 'curious', 'protective', 'social', 'cautious', 'bold']
  const seed = stableEventSeed({ ...event, id: `${event.id}:${actor.agentId ?? actor.taskId ?? ''}` })

  return variants[seed % variants.length]
}

export function resolveNpcActivity(
  event: WorldEvent,
  actor: WorldSourceRef,
  state: NpcActivityState,
  animationTags: readonly string[],
  role?: string
): NpcActivity {
  return {
    actor,
    personality: resolveNpcPersonality(role, actor, event),
    state,
    animationTags: [...animationTags],
    ...(event.sourceRef ? { target: event.sourceRef } : {}),
    ...(dialogueFor(event, state) ? { groundedDialogue: dialogueFor(event, state) } : {})
  }
}

export function resolveWorldPresentation(
  event: WorldEvent,
  conditions: readonly WorldCondition[],
  assetTags: ReadonlySet<string> = new Set()
): WorldPresentation {
  const spec = sceneFor(event)
  const tags = animationTagsFor(spec.animationTags, assetTags)

  const related = conditions
    .filter(condition => condition.sourceRef?.taskId && condition.sourceRef.taskId === event.sourceRef?.taskId)
    .map(condition => condition.sourceRef!)

  const participants = uniqueRefs([...(event.sourceRef ? [event.sourceRef] : []), ...related])
  const primaryActor = participants[0] ?? { agentId: `world:${stableEventSeed(event)}` }
  const role = typeof event.facts.role === 'string' ? event.facts.role : undefined
  const npcActivities = [resolveNpcActivity(event, primaryActor, spec.npcState, tags, role)]

  if (event.kind === 'pr.merged_stable') {
    npcActivities.push(
      resolveNpcActivity(event, { agentId: `celebration:${stableEventSeed(event)}` }, 'celebrating', [
        'celebration.citywide',
        'dance'
      ])
    )
  }

  if (event.kind === 'task.blocked' || event.kind === 'task.block_loop') {
    npcActivities.push(
      resolveNpcActivity(event, { agentId: `responders:${stableEventSeed(event)}` }, 'repairing', [
        'repair',
        'extinguish'
      ])
    )
  }

  return {
    sceneTag: spec.sceneTag,
    scope: event.scope,
    animationTags: tags,
    npcActivities,
    participants,
    actionKinds: event.actionKinds,
    cosmetic: {
      ...(spec.cameraBeat ? { cameraBeat: spec.cameraBeat } : {}),
      ...(spec.soundTag ? { soundTag: spec.soundTag } : {}),
      intensity: spec.intensity
    }
  }
}

export function severityLabel(severity: WorldSeverity): string {
  return severity
}
