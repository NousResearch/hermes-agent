import { loadWorldManifest } from '../manifest'
import type { EntityKey, LeaderAnimationState, LeaderId, LunarCityWorldHandle, QualityTier } from '../model'
import { createLunarCityWorld } from '../world/create-world'
import type { LeaderLifeMode } from '../world/leader-life'

import { mountSystemsDemo } from './systems-demo'

const canvas = document.querySelector<HTMLCanvasElement>('#city')!
const status = document.querySelector<HTMLElement>('#status')!
const loading = document.querySelector<HTMLElement>('#loading')!
const selection = document.querySelector<HTMLElement>('#selection')!
const leaders = document.querySelector<HTMLSelectElement>('#leaders')!
const animation = document.querySelector<HTMLSelectElement>('#animation')!
const retry = document.querySelector<HTMLButtonElement>('#retry')!
const interior = document.querySelector<HTMLButtonElement>('#interior')!
const interiorNote = document.querySelector<HTMLElement>('#interior-note')!
const leaderLife = document.querySelector<HTMLSelectElement>('#leader-life')!
const leaderLifeNote = document.querySelector<HTMLElement>('#leader-life-note')!
let releaseSystemsDemo: (() => void) | undefined
let world: LunarCityWorldHandle | undefined
let previewCharacter: { id: string; worker: boolean } | undefined
let interiorBuilding: string | undefined
let interiorEnabled = false

function selectInteriorBuilding(key?: string) {
  world?.setInteriorBuilding?.()
  interiorEnabled = false
  const match = world?.getInteriorBuildings?.().find(plan => key === `lunar-city:model:${encodeURIComponent(plan.id)}`)
  interiorBuilding = match?.id
  interior.disabled = !match
  interior.textContent = 'See inside'
  interior.setAttribute('aria-pressed', 'false')
  interiorNote.textContent = match ? `${match.title}: prototype interior · shell fit pending review.` : ''
}

interior.onclick = () => {
  if (!interiorBuilding) {return}
  const opened = world?.setInteriorBuilding?.(interiorEnabled ? undefined : interiorBuilding) ?? false
  interiorEnabled = opened
  interior.textContent = opened ? 'Show exterior' : 'See inside'
  interior.setAttribute('aria-pressed', String(opened))
}

function setPreviewAnimation(state: string) {
  if (!previewCharacter) {
    return
  }

  if (previewCharacter.worker) {
    world?.setReviewWorkerAnimation?.(previewCharacter.id, state)
  } else {
    world?.setLeaderAnimation(previewCharacter.id as LeaderId, state as LeaderAnimationState)
  }
}

function selectAnimationOwner(key?: string) {
  const id = key?.split(':').at(-1)
  setPreviewAnimation('idle')
  previewCharacter = id ? { id, worker: key!.includes(':review-worker:') } : undefined
  const life = key?.startsWith('lunar-city:leader:') ? world?.getLeaderLife?.(id as LeaderId) : undefined
  leaderLife.disabled = !life?.canWalk
  leaderLifeNote.textContent = key?.startsWith('lunar-city:leader:')
    ? !life ? 'Local movement unavailable: no safe home placement.'
      : !life.canWalk ? 'Locomotion unavailable in this imported model: a usable walk clip is required.'
        : 'Local visual activity only · does not start backend work.'
    : ''
  leaderLife.value = life ? life.manualOverride ? life.mode : 'automatic' : ''
  animation.options.length = 0

  const clips = id
    ? previewCharacter?.worker
      ? world?.reviewWorkerClips?.get(id)
      : world?.leaderStateClips?.get(id)
    : undefined

  animation.add(new Option(clips && Object.keys(clips).length ? 'Choose an imported clip…' : 'No rig clips', ''))

  for (const [state, clip] of Object.entries(clips ?? {})) {
    if (clip) {
      animation.add(new Option(state, state))
    }
  }

  animation.disabled = animation.options.length === 1
}

let paused = window.matchMedia('(prefers-reduced-motion: reduce)').matches

async function start() {
  document.documentElement.dataset.worldReady = 'false'
  retry.hidden = true
  loading.hidden = false

  try {
    const url = './lunar-city/v2-review/world-manifest.v2.json'
    const manifest = await loadWorldManifest(url)
    releaseSystemsDemo?.()
    world?.destroy()
    world = await createLunarCityWorld(
      canvas,
      manifest,
      intent => {
        if (intent.kind === 'select-focus') {
          world?.dispatchCamera({ kind: 'focus', entityKey: intent.entityKey, follow: false })
          selection.textContent = intent.entityKey.split(':').at(-1) ?? ''
          selectAnimationOwner(intent.entityKey)
          selectInteriorBuilding(intent.entityKey)
        }
      },
      undefined,
      url
    )
    world.setQuality('balanced')
    world.setReducedMotion(paused)
    selectInteriorBuilding()
    leaders.options.length = 1

    for (const asset of manifest.reviewLeaderAssets ?? []) {
      leaders.add(new Option(asset.id, `lunar-city:leader:${asset.id}`))
    }

    for (const asset of manifest.reviewWorkerAssets ?? []) {
      leaders.add(new Option(`Worker: ${asset.id}`, `lunar-city:review-worker:${asset.id}`))
    }

    const interiorIds = new Set(world.getInteriorBuildings?.().map(plan => plan.id))

    for (const model of manifest.models.filter(model => model.uri.includes('review-building-') || interiorIds.has(model.id))) {
      leaders.add(new Option(`Building: ${model.id} (review)`, `lunar-city:model:${model.id}`))
    }

    status.textContent = 'Review only · Metre scale · Materials, geometry and entrances pending acceptance'
    loading.hidden = true
    releaseSystemsDemo = mountSystemsDemo(world, manifest)
    document.documentElement.dataset.worldReady = 'true'
  } catch (error) {
    status.textContent = 'City could not load'
    loading.textContent = error instanceof Error ? error.message : String(error)
    retry.hidden = false
  }
}

leaders.onchange = () => {
  if (leaders.value) {
    world?.dispatchCamera({ kind: 'focus', entityKey: leaders.value as EntityKey, follow: false })
  }

  selection.textContent = leaders.selectedOptions[0]?.textContent ?? ''
  selectAnimationOwner(leaders.value || undefined)
  selectInteriorBuilding(leaders.value || undefined)
}

document.querySelector<HTMLButtonElement>('#overview')!.onclick = () => {
  world?.dispatchCamera({ kind: 'return-to-city' })
  selection.textContent = ''
  leaders.value = ''
  selectAnimationOwner()
  selectInteriorBuilding()
}

document.querySelector<HTMLSelectElement>('#quality')!.onchange = event =>
  {const quality=(event.target as HTMLSelectElement).value as QualityTier; world?.setQuality(quality);window.dispatchEvent(new CustomEvent('lunar-review-quality',{detail:quality}))}

document.querySelector<HTMLButtonElement>('#motion')!.onclick = event => {
  paused = !paused
  window.dispatchEvent(new CustomEvent('lunar-review-motion',{detail:paused}))
  world?.setReducedMotion(paused)
  ;(event.currentTarget as HTMLButtonElement).setAttribute('aria-pressed', String(paused))
}

animation.onchange = () => {
  if (animation.value) {
    setPreviewAnimation(animation.value)
  }
}

leaderLife.onchange = () => {
  if (previewCharacter && !previewCharacter.worker && leaderLife.value) {
    world?.setLeaderLifeMode?.(previewCharacter.id as LeaderId, leaderLife.value as LeaderLifeMode | 'automatic')
  }
}

retry.onclick = () => void start()
window.addEventListener('pagehide', () => { releaseSystemsDemo?.(); world?.destroy() }, { once: true })
void start()

export function reviewState() {
  return { camera: world?.getCameraState(), metrics: world?.getPerfSnapshot?.(), ready: !!world }
}
