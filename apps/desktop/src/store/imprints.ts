import { atom } from 'nanostores'

import { type ImprintValence, listImprints, recordImprint } from '@/hermes'
import { notifyError } from '@/store/notifications'

// Imprints — one-tap 👍/👎 on Hermes' replies that Hermes remembers as a
// preference (no reply, no model call). This store holds two things:
//   • $imprintsEnabled — mirrors the canonical `memory.imprints_enabled` config
//     key, so the composer buttons and the Settings switch never disagree
//     (same one-source-of-truth pattern as voice-prefs' auto_tts).
//   • $imprints — which reply currently carries which thumb, so reopening a
//     session restores the buttons' state.

export const $imprintsEnabled = atom<boolean>(true)

export const $imprints = atom<Record<string, ImprintValence>>({})

function writeImprint(messageId: string, valence: ImprintValence | undefined): void {
  const next = { ...$imprints.get() }

  if (valence) {
    next[messageId] = valence
  } else {
    delete next[messageId]
  }

  $imprints.set(next)
}

/** Seed the enabled flag from a loaded config payload (mount / refresh). */
export function applyImprintsEnabledFromConfig(
  config: { memory?: { imprints_enabled?: unknown } | null } | null | undefined
) {
  const raw = config?.memory?.imprints_enabled
  // Absent key means "default on" — match the backend default.
  $imprintsEnabled.set(raw === undefined || raw === null ? true : Boolean(raw))
}

/** Load which replies already carry a 👍/👎 for the active profile. */
export async function hydrateImprints(): Promise<void> {
  try {
    const { enabled, imprints } = await listImprints()
    $imprintsEnabled.set(Boolean(enabled))
    const next: Record<string, ImprintValence> = {}

    for (const entry of imprints) {
      next[entry.message_id] = entry.valence
    }

    $imprints.set(next)
  } catch {
    // A failed hydrate just leaves the buttons in their default (unset) state;
    // it must never block a session from loading.
  }
}

/** Current thumb for a message, or undefined. */
export function imprintFor(messageId: string): ImprintValence | undefined {
  return $imprints.get()[messageId]
}

/**
 * Tap a thumb. Tapping the active thumb again clears it; tapping the other one
 * flips it. Optimistic — the UI updates instantly and reverts if the write
 * fails.
 */
export async function toggleImprint(
  messageId: string,
  valence: ImprintValence,
  opts: { excerpt?: string; sessionId?: string } = {}
): Promise<void> {
  const previous = $imprints.get()[messageId]
  const next: ImprintValence | 'none' = previous === valence ? 'none' : valence

  writeImprint(messageId, next === 'none' ? undefined : next)

  try {
    await recordImprint({ messageId, valence: next, excerpt: opts.excerpt, sessionId: opts.sessionId })
  } catch (error) {
    // Revert to the pre-tap state.
    writeImprint(messageId, previous)
    notifyError(error, 'Could not save that reaction')
  }
}
