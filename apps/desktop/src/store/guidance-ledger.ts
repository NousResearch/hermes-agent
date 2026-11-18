/**
 * What the redesign's discovery surfaces have already settled for this person: one mark per topic.
 *
 * This module is the ONLY place that knows where those marks live. Today that is one machine-level
 * UI preference (`lib/ui-preferences/keys.ts`), so a splash seen in the desktop app or in one
 * browser is seen in every browser that opens this machine's Webapp. The marks are meant to move to
 * the account; that move replaces this module's storage and nothing that calls it.
 */

import { Codecs, persistentAtom } from '@/lib/persisted'

/** `seen`: shown and acknowledged. `not_applicable`: settled without showing (a new install never
 *  sees the rename splash), so a later visit does not reconsider it. */
export type GuidanceMark = 'not_applicable' | 'seen'

export type GuidanceTopic = 'renameSplash'

type GuidanceLedger = Partial<Record<GuidanceTopic, GuidanceMark>>

export const GUIDANCE_LEDGER_KEY = 'hermes.desktop.guidance.v1'

const TOPICS: ReadonlySet<string> = new Set<GuidanceTopic>(['renameSplash'])
const MARKS: ReadonlySet<string> = new Set<GuidanceMark>(['not_applicable', 'seen'])

function sanitize(value: unknown): GuidanceLedger {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return {}
  }

  return Object.fromEntries(
    Object.entries(value).filter(([topic, mark]) => TOPICS.has(topic) && typeof mark === 'string' && MARKS.has(mark))
  )
}

export const $guidanceLedger = persistentAtom<GuidanceLedger>(GUIDANCE_LEDGER_KEY, {}, Codecs.json(sanitize))

export function guidanceMark(topic: GuidanceTopic): GuidanceMark | undefined {
  return $guidanceLedger.get()[topic]
}

/** Records a topic's mark. `seen` is final: nothing downgrades it back to `not_applicable`. */
export function markGuidance(topic: GuidanceTopic, mark: GuidanceMark): void {
  const ledger = $guidanceLedger.get()

  if (ledger[topic] === mark || ledger[topic] === 'seen') {
    return
  }

  $guidanceLedger.set({ ...ledger, [topic]: mark })
}
