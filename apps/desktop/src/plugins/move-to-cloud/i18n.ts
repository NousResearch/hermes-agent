import type { PluginLocaleBundles, PluginTranslate } from '@hermes/plugin-sdk'

/** English is the only bundle: other locales fall back to it until translators add theirs. */
const en = {
  command: 'Move to Hermes Cloud',
  title: 'Move to Hermes Cloud',
  subtitle:
    'Copy this agent’s skills, memory, persona, schedules and recent conversations to a Hermes Cloud agent. Nothing on this computer changes.',
  from: 'From',
  thisComputer: 'This computer',
  to: 'To',
  signInTitle: 'Sign in to Hermes Cloud',
  signInHelp: 'Use the Nous account your cloud agents belong to.',
  signIn: 'Sign in',
  noAgentsTitle: 'No cloud agents yet',
  noAgentsHelp: 'Create one in the Nous Portal, then come back here.',
  openPortal: 'Open the Nous Portal',
  pickOrg: 'Choose a team',
  provisioning: 'Setting up…',
  down: 'Offline',
  chooseAgent: 'Choose the cloud agent to move into',
  scanning: 'Checking what will move…',
  whatMoves: 'What moves',
  skills: (n: number) => `${n} skill${n === 1 ? '' : 's'}`,
  memory: (notes: number, about: number) => `${notes} memory notes, ${about} about you`,
  persona: 'Persona',
  jobs: (n: number) => `${n} scheduled job${n === 1 ? '' : 's'}, paused until you resume them`,
  conversations: (n: number) => `${n} recent conversation${n === 1 ? '' : 's'}`,
  includeHistory: 'Include recent conversations',
  staysTitle: 'Stays on this computer',
  keysStay: 'API keys, bot tokens and sign-ins. You will enter the ones you need on the cloud agent.',
  tooLarge: (n: number) => `${n} item${n === 1 ? '' : 's'} too large to send`,
  localMcp: (names: string) => `MCP servers that run a program from this computer: ${names}`,
  redacted: (n: number) =>
    `${n} key${n === 1 ? '' : 's'} quoted in conversations or skills, replaced with a placeholder`,
  start: 'Move',
  building: 'Packing this agent…',
  reading: 'Packing this agent…',
  uploading: 'Sending to the cloud agent…',
  finishing: 'Checking what arrived…',
  applying: 'Merging into the cloud agent…',
  cancel: 'Cancel',
  resumeHint: 'You can close this. Starting again picks up where it stopped.',
  confirmTitle: 'Ready to merge',
  confirmHelp: 'The cloud agent keeps everything it already has. Where both have something, its own version wins.',
  newSkills: (n: number) => `${n} new skill${n === 1 ? '' : 's'}`,
  keptSkills: (n: number) => `${n} skill${n === 1 ? '' : 's'} the cloud agent already has (kept as they are)`,
  personaReplace: 'Persona: copied from this computer',
  personaKeep: 'Persona: the cloud agent keeps its own',
  merge: 'Merge',
  doneTitle: 'Moved',
  doneHelp: 'Your cloud agent now has what this one knows.',
  partialTitle: 'Moved, with problems',
  partialHelp: 'Some parts did not merge. Try again to finish them; the parts that worked are not repeated.',
  failedStep: (step: string, error: string) => `${step}: ${error}`,
  retry: 'Try again',
  keysTitle: 'Enter these keys on the cloud agent',
  keysHelp: 'Keys never leave this computer. Add each one in the cloud agent’s settings.',
  botWarning:
    'A bot token works in one place at a time. Add it to the cloud agent when you are ready to switch, and remove it here.',
  openAgent: 'Open the cloud agent',
  undo: 'Undo this move',
  undone: 'The move was undone. The cloud agent is as it was.',
  errorTitle: 'The move stopped',
  tryAgain: 'Try again'
}

export type MoveCopy = typeof en
export const MOVE_LOCALES: PluginLocaleBundles = { en }

/** Bind the message shape to the plugin translator so components keep typed access. */
export function bindCopy(t: PluginTranslate): MoveCopy {
  return Object.fromEntries(
    Object.entries(en).map(([key, value]) => [
      key,
      typeof value === 'function' ? (...args: unknown[]) => t(key, ...args) : t(key)
    ])
  ) as MoveCopy
}
