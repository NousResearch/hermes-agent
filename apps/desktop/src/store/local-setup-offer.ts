/**
 * The local-setup offer: "this machine can run a model locally" for every
 * machine that qualifies, delivered by state transitions, never by a timer.
 *
 * Three surfaces read this one record:
 *
 * - The handoff tour ends on the model pill (`signpost.ts`) when the machine
 *   qualifies. Every guided user who reaches the handoff sees it.
 * - A card above the primary composer after the first finished turn once
 *   onboarding is over (handoff or skip). It renders only while that session
 *   is idle, so an automatic follow-up turn hides it and the next idle shows it.
 * - A "Run locally" row at the top of the model menu, until setup is done.
 *
 * States move only on events: `armed` when onboarding ends, `shown` on the
 * first eligible `message.complete`, `dismissed` on ✕, `accepted` on the card's
 * button. Local setup completing ends it by eligibility: a machine with a
 * runtime and a staged model no longer qualifies, so every surface hides.
 *
 * The predecessor was a tip in the rotation: 5-10 min settle, a six-hour
 * cooldown shared with every tip, a quiet check, a 22 s linger, and a 7-day
 * reshow. It fired on rtxspark and vespyr with nobody watching and recorded
 * itself as shown, which hid it for a week.
 */

import { atom, computed } from 'nanostores'

import { getLocalCatalog, getLocalModelsStatus } from '@/hermes'
import { isOnboardingEnabled } from '@/lib/onboarding-enabled'
import { Codecs, persistentAtom } from '@/lib/persisted'
import { readKey } from '@/lib/storage'
import type { LocalCatalogModel, LocalModelsStatus } from '@/types/hermes'

import { hasSeenIntroReveal } from './intro-reveal'
import { $localModelsEnabled } from './local-models-flag'
import { $desktopOnboarding } from './onboarding'
import { $onboardingGate, type OnboardingPhase } from './onboarding-gate'
import { $connection } from './session'
import { $retiredTips } from './tips'

type LocalSetupOfferState = 'accepted' | 'armed' | 'dismissed' | 'shown' | 'unarmed'

interface OfferRecord {
  armedBy: null | string
  /** ISO time of the last transition, for `state()` readouts. */
  at: null | string
  /** Runtime id of the session whose finished turn showed the card (diagnostics only). */
  sessionId: null | string
  state: LocalSetupOfferState
}

const OFFER_STATES: readonly LocalSetupOfferState[] = ['accepted', 'armed', 'dismissed', 'shown', 'unarmed']
const EMPTY: OfferRecord = { armedBy: null, at: null, sessionId: null, state: 'unarmed' }

/** Persisted as a flat string record so `Codecs.stringRecord` does the parsing at the storage boundary. */
function toRecord(stored: Record<string, string>): OfferRecord {
  const state = OFFER_STATES.find(candidate => candidate === stored.state)

  return state
    ? { armedBy: stored.armedBy ?? null, at: stored.at ?? null, sessionId: stored.sessionId ?? null, state }
    : EMPTY
}

function toStored(record: OfferRecord): Record<string, string> {
  return Object.fromEntries(Object.entries(record).filter((entry): entry is [string, string] => entry[1] !== null))
}

const STORAGE_KEY = 'hermes.desktop.offers.local-setup.v1'

/** Desktop-global like the tip ledgers: the offer is about this machine, not a profile. */
const $stored = persistentAtom<Record<string, string>>(STORAGE_KEY, {}, Codecs.stringRecord)

export const $localSetupOffer = computed($stored, toRecord)

/** The stored record as another window may have left it. */
function readStoredRecord(): OfferRecord {
  const raw = readKey(STORAGE_KEY)

  return raw === null ? EMPTY : toRecord(Codecs.stringRecord.decode(raw))
}

function setRecord(record: OfferRecord): void {
  $stored.set(toStored(record))
}

// Another window moved the offer (dismissed it, say): adopt that, so this window
// neither keeps showing a card the user closed nor writes an older state over it.
window.addEventListener('storage', event => {
  if (event.key === STORAGE_KEY) {
    $stored.set(event.newValue === null ? {} : Codecs.stringRecord.decode(event.newValue))
  }
})

/** The recommended catalog row that fits, when this machine qualifies. */
interface LocalSetupFit {
  model: LocalCatalogModel
}

interface Eligibility {
  checkedAt: number
  fit: LocalSetupFit | null
  reason: string
  /** A failed read or a connection not up yet: shown in `state()` but never cached, so the next read retries. */
  transient?: boolean
}

/** Session cache: eligibility is a fact about the backend's machine, so a connection change drops it. */
export const $localSetupEligibility = atom<Eligibility | null>(null)

let eligibilityRead: Promise<Eligibility> | null = null
/** Bumped on every invalidation: a read that started before it must not land after it. */
let eligibilityGeneration = 0

function pickLocalSetupFit(
  connectionMode: null | string,
  status: LocalModelsStatus | null,
  catalog: readonly LocalCatalogModel[] | null
): Eligibility {
  const at = Date.now()

  if (connectionMode === null) {
    return { checkedAt: at, fit: null, reason: 'connection not established yet', transient: true }
  }

  if (connectionMode !== 'local') {
    return { checkedAt: at, fit: null, reason: `connection is ${connectionMode}, not local` }
  }

  if (!status || !catalog) {
    return { checkedAt: at, fit: null, reason: 'no local-models status or catalog' }
  }

  if (status.runtime_installed && status.models.length > 0) {
    return { checkedAt: at, fit: null, reason: 'local models already set up' }
  }

  const fitting = catalog.filter(model => model.fits)
  const model = fitting.find(candidate => candidate.recommended) ?? fitting[0]

  return model
    ? { checkedAt: at, fit: { model }, reason: `fits ${model.id}` }
    : { checkedAt: at, fit: null, reason: 'no catalog model fits this machine' }
}

/** One status + catalog read per connection; a transient miss is retried on the next read. */
export function readLocalSetupEligibility(): Promise<Eligibility> {
  const cached = $localSetupEligibility.get()

  return cached && !cached.transient ? Promise.resolve(cached) : refreshLocalSetupEligibility()
}

/** Read again, keeping the current answer on screen until the new one lands. */
export function refreshLocalSetupEligibility(): Promise<Eligibility> {
  const mode = $connection.get()?.mode ?? null

  // Settled without a request: the flag, a missing connection, or a remote
  // backend (whose hardware is not this computer's) all answer "no" locally.
  if (!$localModelsEnabled.get() || mode !== 'local') {
    const settled = $localModelsEnabled.get()
      ? pickLocalSetupFit(mode, null, null)
      : { checkedAt: Date.now(), fit: null, reason: 'local models are off in this build' }

    $localSetupEligibility.set(settled)

    return Promise.resolve(settled)
  }

  const generation = eligibilityGeneration

  eligibilityRead ??= Promise.all([getLocalModelsStatus(), getLocalCatalog()])
    .then(([status, catalog]) => pickLocalSetupFit(mode, status, catalog.models))
    .catch((error: Error) => ({
      checkedAt: Date.now(),
      fit: null,
      reason: `eligibility read failed: ${error.message}`,
      transient: true
    }))
    .then(result => {
      if (generation !== eligibilityGeneration) {
        return result
      }

      eligibilityRead = null

      // A failed re-read keeps a good answer on screen rather than hiding the offer on one bad fetch.
      if (!(result.transient && $localSetupEligibility.get()?.fit)) {
        $localSetupEligibility.set(result)
      }

      return result
    })

  return eligibilityRead
}

/** Drop the cached answer: the backend it described is gone (connection or profile changed; debug reset). */
function invalidateLocalSetupEligibility(): void {
  eligibilityGeneration += 1
  eligibilityRead = null
  $localSetupEligibility.set(null)
}

/** Which backend's machine the cached answer describes. Window-state republishes keep it. */
function backendIdentity(): string {
  const connection = $connection.get()

  return connection ? `${connection.mode ?? ''}|${connection.connectionId ?? ''}|${connection.baseUrl}` : ''
}

let lastBackend = backendIdentity()

$connection.listen(() => {
  const next = backendIdentity()

  if (next !== lastBackend) {
    lastBackend = next
    invalidateLocalSetupEligibility()
  }
})

const FINAL_STATES: readonly LocalSetupOfferState[] = ['accepted', 'dismissed']

function transition(state: LocalSetupOfferState, patch: Partial<OfferRecord>): void {
  // Another window may have closed the offer since this one last read it; a final state stays final.
  const stored = readStoredRecord()

  if (FINAL_STATES.includes(stored.state)) {
    $stored.set(toStored(stored))

    return
  }

  setRecord({ ...$localSetupOffer.get(), ...patch, at: new Date().toISOString(), state })
}

/** The guide will not run for this identity: film seen already, or "choose a provider later". */
function guideWillNotStart(): boolean {
  return hasSeenIntroReveal() || $desktopOnboarding.get().firstRunSkipped
}

/**
 * Guided onboarding ending (either way) arms the offer, and so does an install
 * where the guide never runs (flag off, or `idle` with the guide ruled out).
 * `cinematic`/`guided`/`handoff`, and an `idle` the guide may still leave, wait:
 * the card must not land on top of the guide.
 */
function armFromPhase(phase: OnboardingPhase): void {
  if ($localSetupOffer.get().state !== 'unarmed') {
    return
  }

  // Someone who closed the old local-setup tip already answered this offer.
  if ($retiredTips.get().includes('local-setup')) {
    transition('dismissed', { armedBy: 'retired-tip' })
  } else if (phase === 'done' || phase === 'skipped') {
    transition('armed', { armedBy: `onboarding:${phase}` })
  } else if (!isOnboardingEnabled()) {
    transition('armed', { armedBy: 'no-guided-onboarding' })
  } else if (phase === 'idle' && guideWillNotStart()) {
    transition('armed', { armedBy: 'guide-will-not-start' })
  }
}

$onboardingGate.subscribe(gate => armFromPhase(gate.phase))
$desktopOnboarding.listen(() => armFromPhase($onboardingGate.get().phase))

interface TurnCompleteSignal {
  /** Anything but a completed turn: errored, interrupted, cancelled. */
  failed: boolean
  sessionId: null | string
}

/**
 * `message.complete` for the session the user is looking at. The whole agent
 * loop has returned at this point (tool calls and interim messages come before
 * it), so this is the end of a task, not a step in one. Turns that did not
 * complete do not count; the caller only reports the session on screen.
 */
export function reportLocalSetupTurnComplete({ failed, sessionId }: TurnCompleteSignal): void {
  if (failed || !sessionId || $localSetupOffer.get().state !== 'armed') {
    return
  }

  void readLocalSetupEligibility().then(({ fit }) => {
    if (fit && $localSetupOffer.get().state === 'armed') {
      transition('shown', { sessionId })
    }
  })
}

export function dismissLocalSetupOffer(): void {
  transition('dismissed', {})
}

export function acceptLocalSetupOffer(): void {
  transition('accepted', {})
}

/** The card is live: shown, still eligible. The composer adds "and this session is idle". */
export const $localSetupCardLive = computed(
  [$localSetupOffer, $localSetupEligibility],
  (offer, eligibility) => offer.state === 'shown' && Boolean(eligibility?.fit)
)

/** The model-menu row stays until setup completes, whatever happened to the card. */
export const $localSetupRowFit = computed($localSetupEligibility, eligibility => eligibility?.fit ?? null)

/** Asks the pill of one composer (by scope target) to open its menu, where the row sits on top. */
export const $localSetupMenuRequest = atom<null | { seq: number; target: string }>(null)

export function requestLocalSetupMenu(target: string): void {
  $localSetupMenuRequest.set({ seq: ($localSetupMenuRequest.get()?.seq ?? 0) + 1, target })
}

// ── Debug handle ────────────────────────────────────────────────────────────
// Ships in packaged builds on purpose: rtxspark/vespyr run MSIX bundles we
// drive over CDP, and `state()` is how a run that showed nothing explains itself.

interface LocalSetupOfferDebug {
  force: () => Promise<Eligibility>
  reset: () => void
  state: () => {
    eligibility: Eligibility | null
    localModelsEnabled: boolean
    offer: OfferRecord
    onboardingPhase: OnboardingPhase
  }
}

declare global {
  interface Window {
    __hermesTips?: LocalSetupOfferDebug
  }
}

window.__hermesTips = {
  /** Skip onboarding and waiting: mark the card shown now, if the machine qualifies. */
  force: async () => {
    invalidateLocalSetupEligibility()
    const result = await readLocalSetupEligibility()

    if (result.fit) {
      transition('shown', { armedBy: 'debug:force', sessionId: 'debug' })
    }

    return result
  },
  reset: () => {
    $stored.set({})
    invalidateLocalSetupEligibility()
    armFromPhase($onboardingGate.get().phase)
  },
  state: () => ({
    eligibility: $localSetupEligibility.get(),
    localModelsEnabled: $localModelsEnabled.get(),
    offer: $localSetupOffer.get(),
    onboardingPhase: $onboardingGate.get().phase
  })
}
