// Avatar clip map — mirrors FlowTec Commander's clipMap.ts for Hermes Desktop
// The baked Commander mascot GLB has these exact clips and morphs.

export const AVATAR_GLB_URL = new URL('./commander_avatar.glb', import.meta.url).href

export type AvatarClipName =
  | 'idle'
  | 'idle_soft'
  | 'talk'
  | 'think'
  | 'look'
  | 'point'
  | 'wave'
  | 'agree'
  | 'reject'
  | 'surprise'
  | 'celebrate'
  | 'dismiss'

export type AvatarEmotionName =
  | 'neutral'
  | 'happy'
  | 'smile'
  | 'thinking'
  | 'surprised'
  | 'sad'
  | 'angry'
  | 'focus'
  | 'talk'
  | 'helpful'
  | 'excited'

export type AvatarRuntimeEvent =
  | 'idle'
  | 'scan'        // look + focus
  | 'working'     // think + thinking
  | 'step'        // point + focus
  | 'speaking'    // talk + talk
  | 'notify'      // surprise + surprised
  | 'error'       // reject + sad
  | 'success'     // agree + happy

export const LOOPING_CLIPS: AvatarClipName[] = [
  'idle',
  'idle_soft',
  'talk',
  'think',
  'look',
]

export function isLoopingClip(name: AvatarClipName): boolean {
  return LOOPING_CLIPS.includes(name)
}

export const FACIAL_MORPHS: AvatarEmotionName[] = [
  'neutral',
  'happy',
  'smile',
  'thinking',
  'surprised',
  'sad',
  'angry',
  'focus',
  'talk',
  'helpful',
  'excited',
]

const EVENT_CLIP: Record<AvatarRuntimeEvent, AvatarClipName> = {
  idle: 'idle',
  scan: 'look',
  working: 'think',
  step: 'point',
  speaking: 'talk',
  notify: 'surprise',
  error: 'reject',
  success: 'agree',
}

const EVENT_EMOTION: Record<AvatarRuntimeEvent, AvatarEmotionName> = {
  idle: 'neutral',
  scan: 'focus',
  working: 'thinking',
  step: 'focus',
  speaking: 'talk',
  notify: 'surprised',
  error: 'sad',
  success: 'happy',
}

export function clipForEvent(event: AvatarRuntimeEvent): AvatarClipName {
  return EVENT_CLIP[event] ?? 'idle'
}

export function emotionForEvent(event: AvatarRuntimeEvent): AvatarEmotionName {
  return EVENT_EMOTION[event] ?? 'neutral'
}

export function applyEmotion(
  mesh: THREE.Mesh,
  emotion: AvatarEmotionName,
  weight: number
): void {
  const dict = mesh.morphTargetDictionary
  const influences = mesh.morphTargetInfluences

  if (!dict || !influences) return

  const idx = dict[emotion]
  if (idx === undefined) return

  // Set all to 0, then the target to weight
  for (let i = 0; i < influences.length; i++) {
    influences[i] = 0
  }
  influences[idx] = weight
}

// Re-export THREE types we need
import * as THREE from 'three'
export { THREE }