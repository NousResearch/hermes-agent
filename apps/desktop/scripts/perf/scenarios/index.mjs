// Scenario registry. Add a scenario module here and it's automatically
// available to the runner, the default suite (tier 'ci'), and the baseline gate.

import coldStart from './cold-start.mjs'
import firstToken from './first-token.mjs'
import keystroke from './keystroke.mjs'
import profileSwitch from './profile-switch.mjs'
import sessionSwitch from './session-switch.mjs'
import stream from './stream.mjs'
import submit from './submit.mjs'
import transcript from './transcript.mjs'

export const SCENARIOS = {
  [stream.name]: stream,
  [keystroke.name]: keystroke,
  [transcript.name]: transcript,
  [coldStart.name]: coldStart,
  [firstToken.name]: firstToken,
  [submit.name]: submit,
  [sessionSwitch.name]: sessionSwitch,
  [profileSwitch.name]: profileSwitch
}

/** Scenarios safe to run with no LLM credits / no live backend — the default suite. */
export const CI_SCENARIOS = Object.values(SCENARIOS)
  .filter(s => s.tier === 'ci')
  .map(s => s.name)
