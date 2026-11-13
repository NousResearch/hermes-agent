/** Name the stop the updater already recorded, instead of a bare "exit 1" dialog. */

import fs from 'node:fs'
import path from 'node:path'

/** The closed token set `hermes_cli.update_receipt.record_stop_reason` records. */
const STOP_CLASS_TOKEN = /^[a-z][a-z0-9_]{0,39}$/

const UPDATES_GUIDE = 'https://hermes-agent.nousresearch.com/docs/getting-started/updating'

export interface UpdateStop {
  stopClass: string
  outcome: string
}

export interface UpdateStopExplanation {
  /** Dialog headline: says what happened to the install, not that it "failed". */
  message: string
  /** Dialog body: the stop, why it is the whole story, and where the way out is. */
  detail: string
}

/**
 * Receipts live under the ROOT home: `hermes -p <name> update` files its receipt
 * in `<root>/logs/update_receipts/`, and the hand-off script resolves the same
 * root before it reads the receipt back (scripts/desktop-update/posix.sh,
 * `update_committed_after_exit`). A profile home passed here therefore still
 * reads the root's store.
 */
export function receiptDirectory(hermesHome: string): string {
  const home: string = path.resolve(hermesHome)
  const root: string = path.basename(path.dirname(home)) === 'profiles' ? path.dirname(path.dirname(home)) : home

  return path.join(root, 'logs', 'update_receipts')
}

/**
 * This run's own stop, read from the receipt the update wrote.
 *
 * `correlation_id` is the hand-off `run_id` (the launcher exports it as
 * `HERMES_UPDATE_CORRELATION_ID`), so a receipt left behind by an earlier run
 * cannot explain this one. A record still `running` keeps `finished_at` null and
 * never matches — the same rule the hand-off script applies before it trusts an
 * outcome. Anything missing or unexpected returns null and the caller keeps its
 * generic text: the dialog must never invent a stop it cannot prove.
 */
export function readStoppedRun(hermesHome: string, correlationId: string | null): UpdateStop | null {
  if (!correlationId) {
    return null // no run identity: a receipt on disk cannot be attributed to this result
  }

  let parsed: any

  try {
    parsed = JSON.parse(fs.readFileSync(path.join(receiptDirectory(hermesHome), 'latest.json'), 'utf8'))
  } catch {
    return null
  }

  if (!parsed || typeof parsed !== 'object' || parsed.correlation_id !== correlationId) {
    return null
  }

  if (typeof parsed.finished_at !== 'string' || parsed.finished_at === '') {
    return null // still running: its stop is not written yet
  }

  const outcome: string = typeof parsed.outcome === 'string' ? parsed.outcome : ''

  if (outcome !== 'failed' && outcome !== 'refused') {
    return null
  }

  const stopClass: string = typeof parsed.stop_class === 'string' ? parsed.stop_class : ''

  return STOP_CLASS_TOKEN.test(stopClass) ? { stopClass, outcome } : null
}

/**
 * What each recorded stop means for the user, where the receipt alone is enough
 * to say it. Kept to the stops whose remedy the CLI already owns: a token with
 * no entry keeps the generic failure dialog rather than a guessed explanation.
 *
 * `downgrade_refused` is the refusal `hermes_cli/update_downgrade_guard.py`
 * prints before anything moves (the receipt records the exit, not that text), so
 * the remedy here is the one the guide names: follow the checkout's channel.
 */
const STOP_EXPLANATIONS: Record<string, UpdateStopExplanation> = {
  downgrade_refused: {
    message: 'Hermes kept your install: the offered release is older than it',
    detail:
      'The updater refused to move this install back to the release it offered, because that release predates the ' +
      'gateway runtime your session stores already use. Downgrading is unsupported, and nothing was changed: you are ' +
      'still on the version you had.\n\n' +
      'To follow this checkout instead of the release channel, run `hermes update --set-channel main`, or pick a ' +
      `channel in About ▸ Updates.\n\n${UPDATES_GUIDE}`
  }
}

export function explainUpdateStop(stopClass: string | null): UpdateStopExplanation | null {
  return stopClass === null ? null : Object.prototype.hasOwnProperty.call(STOP_EXPLANATIONS, stopClass)
    ? STOP_EXPLANATIONS[stopClass]
    : null
}
