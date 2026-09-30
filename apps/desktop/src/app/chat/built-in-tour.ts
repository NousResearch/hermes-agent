/**
 * The app's own tour — what `gui_tour` runs when it starts with no steps.
 *
 * The app owns the stops (stable `data-tour` handles) and their copy, so a
 * look around is one tool call. Before this, the model scanned `targets` and
 * wrote its own step list: two round trips before anything moved on screen,
 * built from raw DOM labels. A stop whose handle is not on screen right now (a
 * mode without that nav row, a collapsed pane) is dropped, never guessed at.
 */

import { getLocalCatalog, getLocalModelsStatus } from '@/hermes'
import { runtimeTranslations } from '@/i18n'
import { localSetupEligible } from '@/lib/tips/local-cta'
import { startTour, type TourResult } from '@/lib/tour'
import { $interfaceMode } from '@/store/interface-mode'
import { $localModelsEnabled } from '@/store/local-models-flag'
import { $connection } from '@/store/session'

export type BuiltInTourPreset = 'quick' | 'full'

type StopId = 'capabilities' | 'composer' | 'messaging' | 'model' | 'newSession' | 'rightPane' | 'sessions'

const SELECTORS: Record<StopId, string> = {
  capabilities: '[data-tour="sidebar-nav-capabilities"]',
  composer: '[data-tour="composer"]',
  messaging: '[data-tour="sidebar-nav-messaging"]',
  model: '[data-tour="model-pill"]',
  newSession: '[data-tour="sidebar-nav-new-session"]',
  rightPane: '[data-tour="right-pane-toggle"]',
  sessions: '[data-tour="sessions-sidebar"]'
}

/** Where their conversations live, where they ask for a job, how to start a
 *  fresh one. */
const ESSENTIALS: StopId[] = ['sessions', 'composer', 'newSession']

/** Full adds the model picker and what the interface mode puts on screen:
 *  Simple is for talking to Hermes, Advanced has the working pane. */
const SIMPLE_EXTRAS: StopId[] = ['capabilities', 'messaging']
const ADVANCED_EXTRAS: StopId[] = ['rightPane', 'capabilities']

const PRESET_STOPS: Record<BuiltInTourPreset, () => StopId[]> = {
  full: () => [...ESSENTIALS, 'model', ...($interfaceMode.get() === 'simple' ? SIMPLE_EXTRAS : ADVANCED_EXTRAS)],
  quick: () => ESSENTIALS
}

/** A slow fit answer drops the local line rather than holding the tour. */
const LOCAL_FIT_WAIT_MS = 2_000

/** Same rule as the tour collector: a keep-alive tab hidden with
 *  `data-pane-hidden` keeps its box, so the attribute decides, then size. */
function onScreen(selector: string): boolean {
  const node = document.querySelector(selector)

  if (!node || node.closest('[data-pane-hidden]')) {
    return false
  }

  const { height, width } = node.getBoundingClientRect()

  return width >= 4 && height >= 4
}

/** Whether the model stop says this computer can run a local model. It is the
 *  local-setup tip's answer (the backend's catalog `fits` check, on a local
 *  connection, with nothing set up yet), and only where the Local Models pane
 *  exists to point at. */
async function canRunLocalModel(): Promise<boolean> {
  const mode = $connection.get()?.mode ?? null

  if (!$localModelsEnabled.get() || mode !== 'local') {
    return false
  }

  const read = Promise.all([getLocalModelsStatus(), getLocalCatalog()]).then(
    ([status, catalog]) => localSetupEligible(mode, status, catalog.models),
    () => false
  )

  const late = new Promise<boolean>(resolve => setTimeout(() => resolve(false), LOCAL_FIT_WAIT_MS))

  return Promise.race([read, late])
}

/** Run the built-in tour. Never throws; failures come back like any tour
 *  action's, so the agent can say so in words. */
export async function runBuiltInTour(preset: BuiltInTourPreset): Promise<TourResult> {
  const stops = PRESET_STOPS[preset]().filter(id => onScreen(SELECTORS[id]))

  if (stops.length === 0) {
    return { error: 'None of the built-in tour targets are on screen.', success: false }
  }

  const localLine = stops.includes('model') && (await canRunLocalModel())
  const copy = runtimeTranslations().appTour

  return startTour(
    stops.map(id => ({
      selector: SELECTORS[id],
      text: id === 'model' && localLine ? `${copy.model.text} ${copy.modelLocal}` : copy[id].text,
      title: copy[id].title
    }))
  )
}
