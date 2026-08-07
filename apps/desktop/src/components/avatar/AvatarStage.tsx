// AvatarStage — mounts the baked Commander mascot and drives it for Hermes Desktop.
// Mirrors FlowTec Commander's AvatarStage.tsx (commit cd3412f).

import { useCallback, useEffect, useMemo, useRef } from 'react'
import { useFrame, useThree } from '@react-three/fiber'
import { useGLTF, useAnimations } from '@react-three/drei'
import type { Group, Mesh } from 'three'
import { LoopOnce, LoopRepeat } from 'three'

import {
  AVATAR_GLB_URL,
  applyEmotion,
  clipForEvent,
  emotionForEvent,
  isLoopingClip,
  type AvatarClipName,
  type AvatarEmotionName,
  type AvatarRuntimeEvent,
} from './clipMap'

/** Crossfade between body clips, in seconds. */
const FADE_SECONDS = 0.5

/** How fast a facial morph reaches full weight. */
const EMOTION_LERP_PER_SECOND = 4

export interface AvatarStageProps {
  /** Metres. The mascot is baked at 1.6 units tall. */
  scale?: number
  position?: [number, number, number]
  /** Simple state to drive the avatar (idle, listening, thinking, etc.) */
  state?: AvatarRuntimeEvent
}

/** Avatar state machine — simple version for Hermes Desktop */
export type AvatarState =
  | 'idle'
  | 'listening'
  | 'thinking'
  | 'speaking'
  | 'success'
  | 'error'
  | 'working'

const STATE_TO_EVENT: Record<AvatarState, AvatarRuntimeEvent> = {
  idle: 'idle',
  listening: 'scan',
  thinking: 'working',
  speaking: 'speaking',
  success: 'success',
  error: 'error',
  working: 'working',
}

const STATE_TO_BASE_CLIP: Record<AvatarState, AvatarClipName> = {
  idle: 'idle',
  listening: 'look',
  thinking: 'think',
  speaking: 'talk',
  success: 'idle',
  error: 'idle_soft',
  working: 'think',
}

export function AvatarStage({
  scale = 1,
  position = [0, -0.8, 0],
  state = 'idle',
}: AvatarStageProps) {
  const group = useRef<Group>(null)
  const { invalidate } = useThree()
  const { scene, animations } = useGLTF(AVATAR_GLB_URL)

  // Filter out Mixamo NLA strips the bake leaks (same as Commander)
  const clips = useMemo(
    () => animations.filter((clip) => !clip.name.includes('mixamo.com|Layer0')),
    [animations],
  )

  const { actions, mixer } = useAnimations(clips, group)

  // The SkinnedMesh carrying the facial shape keys.
  const faceMesh = useMemo(() => {
    let found: Mesh | null = null
    scene.traverse((child) => {
      const mesh = child as Mesh
      if (!found && mesh.morphTargetDictionary && mesh.morphTargetInfluences) {
        found = mesh
      }
    })
    return found
  }, [scene])

  const currentClip = useRef<AvatarClipName | null>(null)
  const currentState = useRef<AvatarState | null>(null)
  const currentEmotion = useRef<AvatarEmotionName>('neutral')
  const emotionWeight = useRef(0)

  /** Crossfade to `name`, looping or not. Returns false if the clip is absent. */
  const playClip = useCallback(
    (name: AvatarClipName): boolean => {
      const next = actions[name]
      if (!next) return false
      if (currentClip.current === name) return true

      const prev = currentClip.current ? actions[currentClip.current] : null
      const looping = isLoopingClip(name)

      next
        .reset()
        .setLoop(looping ? LoopRepeat : LoopOnce, looping ? Infinity : 1)
        .fadeIn(FADE_SECONDS)
        .play()
      // Never clamp: a held final pose is indistinguishable from a hung app.
      // One-shots hand back to the base clip via the `finished` listener below.
      next.clampWhenFinished = false

      if (prev && prev !== next) prev.fadeOut(FADE_SECONDS)
      currentClip.current = name
      return true
    },
    [actions],
  )

  // Start in idle so the avatar is alive before the first event arrives.
  useEffect(() => {
    playClip('idle')
    invalidate()
    return () => {
      mixer.stopAllAction()
    }
  }, [playClip, mixer, invalidate])

  // A one-shot gesture has run its course — settle into the state's base clip
  // instead of freezing on the last frame.
  useEffect(() => {
    function onFinished() {
      const state = currentState.current ?? 'idle'
      currentClip.current = null // force the crossfade even if names collide
      playClip(STATE_TO_BASE_CLIP[state])
      invalidate()
    }
    mixer.addEventListener('finished', onFinished)
    return () => {
      mixer.removeEventListener('finished', onFinished)
    }
  }, [mixer, playClip, invalidate])

  useFrame((_, delta) => {
    // --- body: gesture on transition, base clip while the state persists ---
    if (state !== currentState.current) {
      currentState.current = state
      const event = STATE_TO_EVENT[state as AvatarState]
      const gesture = clipForEvent(event)
      if (!playClip(gesture)) playClip(STATE_TO_BASE_CLIP[state as AvatarState])
    }

    // --- face ---
    const event = STATE_TO_EVENT[state as AvatarState]
    const wantedEmotion = emotionForEvent(event)
    if (wantedEmotion !== currentEmotion.current) {
      currentEmotion.current = wantedEmotion
      emotionWeight.current = 0
    }
    if (faceMesh && emotionWeight.current < 1) {
      emotionWeight.current = Math.min(
        1,
        emotionWeight.current + delta * EMOTION_LERP_PER_SECOND,
      )
      applyEmotion(faceMesh, currentEmotion.current, emotionWeight.current)
    }

    // NOTE: do NOT call mixer.update() here. drei's useAnimations already runs
    // it in its own useFrame — a second call doubles the delta and every clip
    // plays at 2x speed.

    invalidate()
  })

  return (
    <group ref={group} position={position} scale={scale} dispose={null}>
      <primitive object={scene} />
    </group>
  )
}

useGLTF.preload(AVATAR_GLB_URL)