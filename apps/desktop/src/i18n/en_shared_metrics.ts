import type { Translations } from './types'

export const enSharedMetrics: Translations['sharedMetrics'] = {
  consentTitle: 'Help improve Hermes?',
  consentBody:
    'Shared metrics contain only bounded counters. Never prompts, files, paths or error text. Collection is local. Sending them to Nous is a separate opt-in.',
  whatIsCollected: 'What is collected',
  collectedIntro: 'Only bounded counters:',
  collectedActivity:
    'Activity, session length, outcomes and error classes, including a fixed-list reason when a memory write or context compression is refused, fails or is skipped',
  collectedModels: 'Model routes and token totals',
  collectedNames: 'Built-in tool, command and catalog names',
  collectedMilestones: 'Bucketed setup counts',
  collectedReliability: 'Update and install results and timing (with a fixed-list reason and the stage when one fails, including a fresh install recorded on this machine and counted only once you opt in), crashes, startup and reply speed, messaging-platform health',
  collectedUsage:
    'How Hermes gets used: agent accuracy and efficiency (edit matches, loops, recoveries, tokens and tool calls per task, cache breaks), active time per surface and Desktop mode, which app areas, actions and settings are used, closed quickly or switched off, and provider setup outcomes',
  collectedMachine:
    'Coarse machine facts: RAM range, GPU type, Hermes version age and release channel, updates behind, whether a local model server is used',
  installId:
    'Sending uploads each daily package to the Nous telemetry service. Packages carry this profile’s install ID: a stable random UUID with no personal information, reset by deleting the shared-metrics directory.',
  consentWindow:
    'Only packages whose entire collection period falls inside a recorded consent window are ever sent. Apart from the fresh-install note (noted on this machine and counted only once you opt in), data from before you opt in, or from any gap while sending was off, stays on this machine. Sending can be turned off again at any time.',
  readDocs: 'Read the full details',
  share: 'Collect and send to Nous',
  local: 'Collect locally only',
  off: 'No thanks',
  changeLater: 'You can change this any time in Settings → Safety.',
  saveFailed: 'Couldn’t save your choice',
  collectLabel: 'Collect usage stats',
  collectDesc: 'Bounded counters kept on this device. Never prompts, files, paths or error text.',
  sendLabel: 'Send usage stats to Nous',
  sendDesc:
    'Upload each daily package to the Nous telemetry service. Only data from inside a consent window is sent. Needs collection on.',
  unavailable: 'Update the Hermes backend to change this setting.',
  stripBody: 'Bounded counters only, never prompts or files.',
  stripReaskBody: 'Asking once more: an earlier version could save “No thanks” before you saw this.',
  stripChoices: { share: 'Send to Nous', local: 'Local only', off: 'No thanks' },
  stripDetails: 'Details'
}
