import type { Translations } from './types'

export const enSharedMetrics: Translations['sharedMetrics'] = {
  consentTitle: 'Share usage stats?',
  dialogTitle: 'Usage stats',
  consentBody:
    'Hermes can count how you use it: session length, which models and tools run, and when something fails. It never records your messages, files, paths or error text.',
  whatIsCollected: 'What is counted',
  collectedActivity: 'Sessions: length, outcome, error type, active time per day',
  collectedModels: 'Models: which ones, token totals',
  collectedNames: 'Features: built-in tools, commands, app areas and settings used or turned off',
  collectedMilestones: 'Setup: which steps finished, provider connections, how many skills, plugins and jobs',
  collectedReliability: 'App health: crashes, startup and reply speed, updates, messaging connections',
  collectedUsage: 'Agent quality: missed edits, broken tool calls, stuck loops, cost per task',
  collectedMachine: 'Machine: OS, RAM range, GPU type, Hermes version, local model use',
  sending:
    'Stats stay on this computer unless you choose Share. Shared stats go to Nous once a day with a random ID for this profile. Apart from a one-time note that Hermes was installed, counted only once you opt in, stats from before you opted in are never sent. Change this anytime in Settings.',
  readDocs: 'Read the full details',
  share: 'Share with Nous',
  local: 'Keep on this computer',
  off: 'No thanks',
  saveFailed: 'Couldn’t save your choice',
  collectLabel: 'Collect usage stats',
  collectDesc: 'Counts only, kept on this computer. Never your messages, files, paths or error text.',
  sendLabel: 'Share usage stats with Nous',
  sendDesc:
    'Sends stats to Nous once a day with a random ID for this profile. Apart from the one-time install note, stats from before you opted in are never sent. Needs collection on.',
  unavailable: 'Update the Hermes backend to change this setting.',
  stripBody: 'Counts only. Never your messages or files.',
  stripReaskBody: 'Asking once more: an earlier version could save “No thanks” before you saw this.',
  stripChoices: { share: 'Share with Nous', local: 'Keep on this computer', off: 'No thanks' },
  stripDetails: 'Details'
}
