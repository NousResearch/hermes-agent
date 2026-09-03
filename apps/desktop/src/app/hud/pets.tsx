/**
 * Pixel pets on the HUD. Each agent has a pet (Settings → HUD → pet per
 * agent): Hank or Mina (the bundled pixel characters, see LICENSE-ART.md), the agent's
 * Bot Mode avatar as a bobbing badge, or none. The pet on the bar is the
 * pet of the agent you are talking to; in room mode every member's pet
 * comes out.
 *
 * They act on the turn. The same activity signals the floating pet already
 * consumes (store/pet → derivePetState) drive them: a send makes the pet
 * stop and quote your prompt, tools make it pace with a "working…" bubble,
 * the reply makes it hop and say "done", an error makes it droop. Idle, it
 * patrols the strip of headroom above the bar, turns at the edges and stops
 * to look at the pointer when it comes near. Pointer-transparent end to end.
 */

import { useStore } from '@nanostores/react'
import { useEffect, useMemo, useRef } from 'react'

import hankSprite from '@/components/pet/assets/hud/hud-pet-hank.png'
import minaSprite from '@/components/pet/assets/hud/hud-pet-mina.png'
import { useI18n } from '@/i18n'
import type { HudLaunchOptions, HudPetChoice } from '@/lib/hud-prefs'
import { $hudLaunchOptions, $hudPrefs, $hudRoom, $hudRoomFeed } from '@/store/hud'
import { $petActivity, derivePetState, type PetState } from '@/store/pet'
import { $activeGatewayProfile, normalizeProfileKey } from '@/store/profile'
import { $messages } from '@/store/session'

import {
  initialPetWalk,
  PET_LOOK_RADIUS,
  PET_WALK_SPEED,
  petBob,
  type PetWalkMode,
  type PetWalkState,
  stepPetWalk
} from './pet-walk'

/** Headroom the shell reserves above the bar while pets are on, CSS px. */
export const HUD_PET_HEADROOM = 60

type HudAgent = HudLaunchOptions['agents'][number]

const SPRITES: Record<'hank' | 'mina', { height: number; src: string; width: number }> = {
  hank: { height: 56, src: hankSprite, width: 34 },
  mina: { height: 56, src: minaSprite, width: 38 }
}

/** Which pet an agent gets when Settings says nothing: the default profile
 *  gets Hank, the next profile gets Mina, everyone else walks as their own
 *  avatar (or a role glyph when they have none). */
export function defaultPetChoice(profile: string, index: number): HudPetChoice {
  if (normalizeProfileKey(profile) === 'default') {
    return 'hank'
  }

  return index <= 1 ? 'mina' : 'avatar'
}

export function petChoiceFor(
  profile: string,
  index: number,
  petByAgent: Record<string, HudPetChoice> | undefined
): HudPetChoice {
  const key = normalizeProfileKey(profile)
  const chosen = petByAgent?.[key]

  return chosen ?? defaultPetChoice(key, index)
}

interface HudPet {
  agent: HudAgent | undefined
  choice: Exclude<HudPetChoice, 'none'>
  key: string
  width: number
}

/** Which pets are out: the active agent's, or every member's in a room. */
function usePets(): HudPet[] {
  const prefs = useStore($hudPrefs)
  const { agents } = useStore($hudLaunchOptions)
  const active = normalizeProfileKey(useStore($activeGatewayProfile))
  const room = useStore($hudRoom)
  const feed = useStore($hudRoomFeed)

  return useMemo(() => {
    const byKey = new Map(agents.map(agent => [normalizeProfileKey(agent.profile), agent] as const))
    const profiles = room && feed?.groupId === room && feed.members.length ? feed.members : [active]
    const out: HudPet[] = []

    profiles.forEach(profile => {
      const key = normalizeProfileKey(profile)
      const index = Math.max(0, agents.findIndex(agent => normalizeProfileKey(agent.profile) === key))
      const choice = petChoiceFor(key, index, prefs?.petByAgent)

      if (choice === 'none' || out.some(pet => pet.key === key)) {
        return
      }

      out.push({ agent: byKey.get(key), choice, key, width: choice === 'avatar' ? 48 : SPRITES[choice].width })
    })

    return out
  }, [active, agents, feed, prefs?.petByAgent, room])
}

function walkMode(state: PetState): PetWalkMode {
  switch (state) {
    case 'run':
      return 'pace'

    case 'review':

    case 'waiting':

    case 'wave':

    case 'jump':

    case 'failed':
      return 'stand'

    default:
      return 'patrol'
  }
}

/** The last thing the user said in this session, shortened for a bubble. */
function lastPromptSnippet(): string {
  const messages = $messages.get()

  for (let i = messages.length - 1; i >= 0; i -= 1) {
    const message = messages[i]

    if (message.role !== 'user') {
      continue
    }

    const text = message.parts
      .map(part => (part as { text?: unknown }).text)
      .filter((text): text is string => typeof text === 'string')
      .join(' ')
      .replace(/\s+/g, ' ')
      .trim()

    return text.length > 42 ? `${text.slice(0, 41)}…` : text
  }

  return ''
}

export function HudPets() {
  const prefs = useStore($hudPrefs)
  const on = prefs?.pets !== false
  const pets = usePets()
  const activity = useStore($petActivity)
  const petState = derivePetState(activity)
  const { t } = useI18n()
  const h = t.hud
  const stripRef = useRef<HTMLDivElement | null>(null)
  const spriteRefs = useRef<Map<string, HTMLElement>>(new Map())
  const statesRef = useRef<Map<string, PetWalkState>>(new Map())
  const modeRef = useRef<PetWalkMode>('patrol')
  modeRef.current = walkMode(petState)

  const bubble = useMemo(() => {
    switch (petState) {
      case 'failed':
        return h.petError

      case 'jump':

      case 'wave':
        return h.petDone

      case 'waiting':
        return h.petWaiting

      case 'run':
        return h.petWorking

      case 'review':
        return lastPromptSnippet() || h.petThinking

      default:
        return ''
    }
  }, [h, petState])

  useEffect(() => {
    const strip = stripRef.current

    if (!on || !strip || pets.length === 0) {
      return
    }

    const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    const states = statesRef.current

    // Keep positions for pets that stay; place newcomers spread along the strip.
    for (const key of [...states.keys()]) {
      if (!pets.some(pet => pet.key === key)) {
        states.delete(key)
      }
    }

    pets.forEach((pet, index) => {
      if (!states.has(pet.key)) {
        const x = Math.max(0, strip.clientWidth * ((index + 1) / (pets.length + 1)) - pet.width / 2)
        states.set(pet.key, initialPetWalk(x, index % 2 === 0 ? 1 : -1))
      }
    })

    let pointerX: number | null = null
    let last = performance.now()
    let frame = 0

    const onMove = (event: MouseEvent) => {
      const rect = strip.getBoundingClientRect()
      pointerX = event.clientY <= rect.bottom + 80 ? event.clientX - rect.left : null
    }

    const onLeave = () => {
      pointerX = null
    }

    const paint = (now: number) => {
      for (const pet of pets) {
        const state = states.get(pet.key)
        const el = spriteRefs.current.get(pet.key)

        if (!state || !el) {
          continue
        }

        const mode = modeRef.current
        let bob = petBob(state)
        let tilt = 0

        if (petState === 'jump' || petState === 'wave') {
          bob = Math.abs(Math.sin(now / 140)) * 8
        } else if (petState === 'failed') {
          tilt = 12 * state.dir
        } else if (mode === 'pace') {
          bob = Math.abs(Math.sin(now / 70)) * 3
        }

        el.style.transform = `translate(${Math.round(state.x)}px, ${-bob}px) rotate(${tilt}deg) scaleX(${state.dir})`
      }
    }

    const tick = (now: number) => {
      const dt = Math.min(0.05, (now - last) / 1000)
      last = now
      const stripWidth = strip.clientWidth

      for (const pet of pets) {
        const state = states.get(pet.key)

        if (!state) {
          continue
        }

        states.set(
          pet.key,
          stepPetWalk(state, dt, {
            stripWidth,
            width: pet.width,
            pointerX,
            speed: PET_WALK_SPEED,
            lookRadius: PET_LOOK_RADIUS,
            mode: modeRef.current,
            random: Math.random
          })
        )
      }

      paint(now)
      frame = requestAnimationFrame(tick)
    }

    window.addEventListener('mousemove', onMove)
    window.addEventListener('mouseleave', onLeave)

    if (reduced) {
      paint(performance.now())
    } else {
      frame = requestAnimationFrame(tick)
    }

    return () => {
      cancelAnimationFrame(frame)
      window.removeEventListener('mousemove', onMove)
      window.removeEventListener('mouseleave', onLeave)
    }
    // petState is read through modeRef inside the loop; the loop restarts only
    // when the set of pets changes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [on, pets])

  if (!on || pets.length === 0) {
    return null
  }

  return (
    <div aria-hidden data-hud-pets ref={stripRef} style={{ height: HUD_PET_HEADROOM }}>
      {pets.map(pet => (
        <div
          data-hud-pet
          data-hud-pet-state={petState}
          key={pet.key}
          ref={el => {
            if (el) {
              spriteRefs.current.set(pet.key, el)
            } else {
              spriteRefs.current.delete(pet.key)
            }
          }}
          style={{ width: pet.width }}
        >
          {bubble ? <span data-hud-pet-bubble>{bubble}</span> : null}
          {pet.choice === 'avatar' ? (
            pet.agent?.image ? (
              // The agent's own pixel avatar, walking. No disc: it is the character.
              <img alt="" draggable={false} height={52} src={pet.agent.image} style={{ width: 'auto' }} />
            ) : (
              <span className="grid size-10 place-items-center rounded-full border-2 border-white/80 bg-[rgb(12_14_18/0.85)] text-xl">
                {pet.agent?.emoji ?? '🤖'}
              </span>
            )
          ) : (
            <img
              alt=""
              draggable={false}
              height={SPRITES[pet.choice].height}
              src={SPRITES[pet.choice].src}
              width={SPRITES[pet.choice].width}
            />
          )}
        </div>
      ))}
    </div>
  )
}
