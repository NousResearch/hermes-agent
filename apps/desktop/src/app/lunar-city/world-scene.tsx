import { useMemo } from 'react'

import type { WorldProjection } from '@/store/lunar-city'

import type { DialogueSubject } from './dialogue-tray'
import { resolveWorldAnimation } from './world-animation'
import { LUNAR_CITY_ASSET_MANIFEST } from './world-assets'
import type { WorldCondition, WorldEvent } from './world-events'
import { resolveWorldPresentation } from './world-presentation'

interface WorldSceneProps {
  onSelectSubject?: (subject: DialogueSubject) => void
  projection: WorldProjection
}

function conditionEvent(condition: WorldCondition): WorldEvent {
  return {
    actionKinds:
      condition.kind === 'task.blocked' || condition.kind === 'worker.stale'
        ? ['inspect_blocker', 'inspect']
        : ['inspect'],
    facts: condition.facts,
    id: condition.id,
    kind: condition.kind,
    occurredAt: Date.now(),
    receivedAt: Date.now(),
    scope: condition.scope,
    severity: condition.severity,
    source: condition.source,
    sourceRef: condition.sourceRef,
    title: condition.title,
    ...(condition.detail ? { detail: condition.detail } : {}),
    transition: true
  }
}

export function WorldScene({ onSelectSubject, projection }: WorldSceneProps) {
  const conditionPresentations = useMemo(
    () =>
      projection.conditions.map(condition => ({
        condition,
        presentation: resolveWorldPresentation(conditionEvent(condition), projection.conditions)
      })),
    [projection.conditions]
  )

  const eventPresentations = useMemo(
    () =>
      projection.recentEvents.map(event => ({
        event,
        presentation: resolveWorldPresentation(event, projection.conditions)
      })),
    [projection.conditions, projection.recentEvents]
  )

  return (
    <section
      aria-label="Lunar City scene"
      className="relative min-h-[30rem] overflow-hidden rounded-3xl border border-cyan-300/20 bg-[#090d18] p-4 text-white shadow-2xl"
    >
      <div
        aria-hidden
        className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_50%_0%,rgba(36,115,164,0.28),transparent_55%),linear-gradient(145deg,#111a2b,#060811)]"
      />
      <div className="relative grid gap-4 md:grid-cols-[minmax(0,1fr)_20rem]">
        <div className="min-h-[26rem] rounded-2xl border border-white/10 bg-black/20 p-4">
          <div className="flex items-center justify-between gap-3">
            <div>
              <p className="text-[0.65rem] font-semibold uppercase tracking-[0.18em] text-cyan-300">
                Live world projection
              </p>
              <h2 className="mt-1 text-xl font-semibold">Hermes colony network</h2>
            </div>
            <span
              className={`rounded-full px-2 py-1 text-[0.65rem] uppercase tracking-wide ${projection.stale ? 'bg-amber-400/15 text-amber-200' : 'bg-emerald-400/15 text-emerald-200'}`}
            >
              {projection.stale ? 'Stale source' : 'Connected'}
            </span>
          </div>

          <div
            aria-label="Baseline 3D scene"
            className="mt-4 overflow-hidden rounded-xl border border-cyan-300/20 bg-black/30"
          >
            <img
              alt="Lunar City Blender baseline with grounded roads and concave terrain"
              className="block aspect-[16/9] w-full object-cover"
              src={`${import.meta.env.BASE_URL}lunar-city/lunar-city-baseline.png`}
            />
            <div className="flex flex-wrap items-center justify-between gap-2 px-3 py-2 text-[0.65rem] text-slate-300">
              <span>
                {LUNAR_CITY_ASSET_MANIFEST.assets.length} authored baseline assets ·{' '}
                {LUNAR_CITY_ASSET_MANIFEST.renderProfile}
              </span>
              <span className="flex flex-wrap gap-3">
                <a
                  className="text-cyan-200 underline decoration-cyan-200/40 underline-offset-2 hover:text-white"
                  download="lunar-city-baseline.glb"
                  href={`${import.meta.env.BASE_URL}lunar-city/lunar-city-baseline.glb`}
                >
                  Download 3D scene
                </a>
                <a
                  className="text-violet-200 underline decoration-violet-200/40 underline-offset-2 hover:text-white"
                  download="profile-class-roster.json"
                  href={`${import.meta.env.BASE_URL}${LUNAR_CITY_ASSET_MANIFEST.profileManifest}`}
                >
                  Download class roster
                </a>
              </span>
            </div>
          </div>

          <div className="mt-5 grid gap-3 sm:grid-cols-2">
            {conditionPresentations.map(({ condition, presentation }) => (
              <button
                className="rounded-xl border border-white/10 bg-white/5 p-3 text-left transition hover:border-cyan-200/50 hover:bg-white/10"
                data-scene={presentation.sceneTag}
                data-source-id={condition.id}
                data-testid={`world-scene-${presentation.sceneTag}`}
                key={condition.id}
                onClick={() => onSelectSubject?.({ condition, title: condition.title })}
                type="button"
              >
                <span className="text-[0.65rem] uppercase tracking-wide text-amber-200">{presentation.sceneTag}</span>
                <strong className="mt-1 block text-sm">{condition.title}</strong>
                <span className="mt-1 block text-xs text-slate-300">
                  {condition.kind} · {condition.severity}
                </span>
                <span className="mt-2 block text-[0.65rem] text-slate-400">
                  {presentation.animationTags.join(' · ')}
                </span>
                <span aria-label="NPC activity" className="mt-3 block space-y-1 border-t border-white/10 pt-2">
                  {presentation.npcActivities.map(activity => (
                    (() => {
                      const animation = resolveWorldAnimation(activity, presentation)

                      return (
                    <span
                      className="flex items-center justify-between gap-2 text-[0.65rem] text-slate-300"
                      data-animation-clip={animation.clip}
                      data-animation-intensity={animation.intensity}
                      data-animation-loop={animation.loop}
                      data-animation-tags={activity.animationTags.join(',')}
                      data-personality={activity.personality}
                      data-testid={`world-npc-${condition.id}-${activity.state}`}
                      key={`${condition.id}:${activity.actor.agentId ?? activity.actor.taskId ?? activity.state}`}
                    >
                      <span>
                        {activity.state} · {activity.personality}
                      </span>
                      {activity.groundedDialogue && (
                        <span className="truncate text-slate-400">“{activity.groundedDialogue}”</span>
                      )}
                    </span>
                      )
                    })()
                  ))}
                </span>
              </button>
            ))}
          </div>

          {eventPresentations.length === 0 && conditionPresentations.length === 0 && (
            <div className="mt-12 rounded-xl border border-dashed border-white/15 p-8 text-center text-sm text-slate-300">
              The colony is quiet. New Hermes work will appear here as it happens.
            </div>
          )}

          {eventPresentations.length > 0 && (
            <div className="mt-5 border-t border-white/10 pt-4">
              <p className="text-[0.65rem] uppercase tracking-[0.18em] text-slate-400">Recent world events</p>
              <div className="mt-2 flex flex-wrap gap-2">
                {eventPresentations.slice(-8).map(({ event, presentation }) => (
                  <button
                    className="rounded-lg border border-white/10 bg-black/20 px-2 py-1.5 text-left text-xs hover:border-cyan-200/50"
                    data-scene={presentation.sceneTag}
                    data-source-id={event.id}
                    data-testid={`world-scene-${presentation.sceneTag}`}
                    key={event.id}
                    onClick={() => onSelectSubject?.({ event, title: event.title })}
                    type="button"
                  >
                    <span className="font-mono text-cyan-200">{event.kind}</span>
                    <span className="ml-1 text-slate-300">{event.title}</span>
                    <span className="ml-2 text-[0.65rem] text-slate-400">
                      {presentation.npcActivities
                        .map(activity => `${activity.state} · ${activity.personality}`)
                        .join(' / ')}
                    </span>
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>

        <div className="space-y-3">
          <div className="rounded-2xl border border-white/10 bg-black/20 p-4">
            <p className="text-[0.65rem] uppercase tracking-[0.18em] text-slate-400">Colony status</p>
            <div className="mt-3 grid grid-cols-2 gap-2 text-center">
              <div className="rounded-lg bg-white/5 p-2">
                <strong className="block text-lg">{projection.conditions.length}</strong>
                <span className="text-[0.65rem] text-slate-400">conditions</span>
              </div>
              <div className="rounded-lg bg-white/5 p-2">
                <strong className="block text-lg">{projection.recentEvents.length}</strong>
                <span className="text-[0.65rem] text-slate-400">events</span>
              </div>
            </div>
            {projection.sourceError && <p className="mt-3 text-xs text-amber-200">{projection.sourceError}</p>}
          </div>

          {projection.transitions.length > 0 && (
            <div aria-label="World recap" className="rounded-2xl border border-violet-300/20 bg-violet-400/10 p-4">
              <p className="text-[0.65rem] uppercase tracking-[0.18em] text-violet-200">While you were away</p>
              <p className="mt-2 text-sm text-slate-200">
                {projection.transitions.length} Hermes transition{projection.transitions.length === 1 ? '' : 's'} need
                your attention.
              </p>
            </div>
          )}
        </div>
      </div>
    </section>
  )
}
