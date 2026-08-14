// Scenario registry. Add a scenario module here and it's automatically
// available to the runner, the default suite (tier 'ci'), and the baseline gate.

import coldStart from './cold-start.mjs'
import firstToken from './first-token.mjs'
import hitchClassify from './hitch-classify.mjs'
import idleCost from './idle-cost.mjs'
import keystroke from './keystroke.mjs'
import multitab from './multitab.mjs'
import profileSwitch from './profile-switch.mjs'
import renderChurn from './render-churn.mjs'
import rightPane from './right-pane.mjs'
import sessionLoad from './session-load.mjs'
import sessionSwitch from './session-switch.mjs'
import stream from './stream.mjs'
import streamHistory from './stream-history.mjs'
import submit from './submit.mjs'
import transcript from './transcript.mjs'

export const SCENARIOS = {
  [stream.name]: stream,
  [streamHistory.name]: streamHistory,
  [keystroke.name]: keystroke,
  [transcript.name]: transcript,
  [multitab.name]: multitab,
  [renderChurn.name]: renderChurn,
  [hitchClassify.name]: hitchClassify,
  [rightPane.name]: rightPane,
  [idleCost.name]: idleCost,
  [coldStart.name]: coldStart,
  [firstToken.name]: firstToken,
  [submit.name]: submit,
  [sessionLoad.name]: sessionLoad,
  [sessionSwitch.name]: sessionSwitch,
  [profileSwitch.name]: profileSwitch
}

/** Scenarios safe to run with no LLM credits / no live backend — the default suite. */
export const CI_SCENARIOS = Object.values(SCENARIOS)
  .filter(s => s.tier === 'ci')
  .map(s => s.name)
