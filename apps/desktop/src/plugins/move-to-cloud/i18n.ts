import type { PluginLocaleBundles, PluginTranslate } from '@hermes/plugin-sdk'

/** English is the only bundle: other locales fall back to it until translators add theirs. */
const en = {
  command: 'Move to Hermes Cloud',
  backupCommand: 'Back up this agent',
  nav: 'Backup & Move',
  title: 'Backup & Move',
  backupOnlyTitle: 'Backup',
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
  confirmTitle: 'Ready to move',
  confirmHelp: 'The cloud agent keeps everything it already has. Where both have something, its own version wins.',
  modeMerge: 'Add to the cloud agent',
  modeReplace: 'Replace the cloud agent',
  replaceHelp: 'Your local copy replaces the cloud agent’s for each part you tick:',
  replaceMemory: 'Memory: the cloud agent’s notes about you and its work are overwritten',
  replaceSoul: 'Persona: the cloud agent’s persona is overwritten',
  replaceSkills: (n: number) =>
    n ? `Skills: ${n} skill${n === 1 ? '' : 's'} with the same name are swapped for yours` : 'Skills: yours are added',
  replaceCron: (n: number) =>
    `Scheduled jobs: ${n ? `its ${n} job${n === 1 ? ' is' : 's are'} removed and ` : ''}yours arrive paused`,
  replaceSessions: (n: number) =>
    n
      ? `Chat history: its ${n} conversation${n === 1 ? ' is' : 's are'} archived (kept, out of the list) and yours take their place`
      : 'Chat history: yours are added',
  replaceConfig: 'Settings: yours replace the cloud agent’s',
  replaceKeeps:
    'Always kept as Hermes Cloud set them: the Nous Portal sign-in, model and provider, tool and scheduler setup, and every key. Undo puts everything back.',
  replaceAcknowledge: 'I understand this overwrites the ticked parts on the cloud agent',
  replace: 'Replace',
  newAgentTitle: 'New cloud agent',
  newAgentHelp: 'Create one in the Nous Portal, then come back here to move into it.',
  backupTitle: 'Back up this agent',
  backupHelp:
    'One file with everything: every profile, settings, memory, skills, schedules, chat history and keys. Keep it private.',
  backup: 'Back up',
  backupRunning: 'Backing up…',
  backupSaved: 'Backup saved.',
  backupKept: (path: string) => `The backup is on this agent at ${path}.`,
  backupFailed: 'The backup did not finish',
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
  failedItems: (step: string, n: number) => `${step}: ${n} item${n === 1 ? '' : 's'} did not move`,
  retry: 'Try again',
  keysTitle: 'Enter these keys on the cloud agent',
  keysHelp: 'Keys never leave this computer. Add each one in the cloud agent’s settings.',
  botWarning:
    'A bot token works in one place at a time. Add it to the cloud agent when you are ready to switch, and remove it here.',
  openAgent: 'Open the cloud agent',
  undo: 'Undo this move',
  undone: 'The move was undone. The cloud agent is as it was.',
  undoProblemsTitle: 'Undo did not finish',
  undoProblemsHelp: 'These could not be put back. Try again to finish the undo.',
  retryUndo: 'Try undo again',
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
