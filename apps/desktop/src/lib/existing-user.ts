/**
 * Is this someone who used Hermes before the redesign? Only they get the rename splash: a
 * brand-new install meets the redesign's words first and has nothing to translate.
 *
 * The answer comes from what is on the machine, not from a version history the app never kept:
 * a chat, a second agent or a routine is prior use. A new install has none of these when its
 * first session list settles (first-run setup is chatless), so that first settled look decides
 * `fresh`, and the caller records it so a chat started later never turns a new user into an
 * "existing" one. Until the look is authoritative the answer is `unknown` and nothing is decided.
 */

export type ExistingUserVerdict = 'existing' | 'fresh' | 'unknown'

export interface ExistingUserEvidence {
  /** The session list answered (loaded, no error). */
  sessionsSettled: boolean
  /** First-run setup is on screen: its outcome is not prior use, so wait for it. */
  setupRunning: boolean
  chats: number
  /** Every agent the machine lists, the default one included. */
  agents: number
  routines: number
}

export function existingUserVerdict(evidence: ExistingUserEvidence): ExistingUserVerdict {
  if (evidence.setupRunning || !evidence.sessionsSettled) {
    return 'unknown'
  }

  return evidence.chats > 0 || evidence.agents > 1 || evidence.routines > 0 ? 'existing' : 'fresh'
}
