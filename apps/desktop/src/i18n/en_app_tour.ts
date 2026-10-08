import type { Translations } from './types'

export const enAppTour: Translations['appTour'] = {
  sessions: { title: 'Your chats', text: 'Every conversation lives here. Search, pin or reopen any of them.' },
  composer: { title: 'Ask here', text: 'Say what you want done. Type @ to bring in a file.' },
  newSession: { title: 'Start fresh', text: 'A new session gets its own context. Use one per job.' },
  model: { title: 'Model picker', text: 'Chooses which model answers you.' },
  modelLocal: 'This computer can run one locally: Settings > Providers > Local Models.',
  capabilities: { title: 'Capabilities', text: 'Skills, tools and plugins Hermes can use. Add more here.' },
  messaging: { title: 'Messaging', text: 'Reach Hermes from Telegram, Slack, Discord and more.' },
  rightPane: { title: 'The working pane', text: 'Opens files, terminal, review and the in-app browser on the right.' }
}

export const enHandoffTour: Translations['handoffTour'] = {
  profileTitle: 'Your first task runs on the default profile',
  profileText:
    'This rail switches profiles. The one lit up now is default, where the task session lives. The other one is the setup profile, where the welcome chat lives.',
  sessionsTitle: 'Each profile keeps its own sessions',
  sessionsText:
    'This list belongs to the default profile. New session starts one on whichever profile is selected. Switch profiles on the rail and the list changes with it.',
  stayTitle: 'Hermes is one click away',
  stayText: 'Switch to the setup profile and open Welcome to Hermes whenever you want a hand. It stays there.',
  localTitle: 'This machine can run models locally',
  localText: (model: string) =>
    `${model} fits your hardware. It runs free, and chats never leave your computer. Pick it here, in the model menu, whenever you want.`
}
