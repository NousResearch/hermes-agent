import type { FirstRunCopy } from './types_first_run'

export const enFirstRun: FirstRunCopy = {
  guide: {
    title: 'Set up your agent',
    resume: 'Resume setup',
    upgradeRequired:
      'Hermes on this machine needs an update for sessionless setup. On that machine, run hermes update, restart Hermes, then retry. Your chats are unchanged.',
    intro: 'A few optional preferences, then your normal chat. Setup never creates a conversation.',
    appearance: 'Appearance',
    model: 'Choose a model',
    ready: 'Your model is ready.',
    providerNeeded: 'Connect a provider or use the free model before sending a message.',
    freeModel: 'Use free model',
    freeUnavailable: 'The free model is not available here. Connect a provider or choose one later.',
    providers: 'Connect a provider',
    finish: 'Finish setup',
    skip: 'Skip setup',
    back: 'Back',
    check: 'Check readiness',
    checking: 'Checking…',
    privacy: 'Sharing usage metrics is optional.'
  },
  beats: {
    name: 'Name',
    theme: 'Accent color',
    layout: 'Layout',
    connectors: 'Apps',
    plugins: 'Plugins',
    tour: 'Tour',
    build: 'First task'
  },
  namePlaceholder: 'Your name',
  useName: name => `Use “${name}”`,
  install: 'Install',
  installed: 'Installed',
  loading: 'Loading…',
  noPlugins: 'No plugins to show right now.',
  noneChosen: 'None',
  noAnswer: 'No answer',
  answerRefused: 'Answer not accepted',
  tours: {
    quick: 'Quick tour',
    full: 'Show me everything',
    skip: "Skip, let's build something"
  },
  skippedAll: 'Setup skipped',
  skip: 'Skip setup',
  skipTip: 'End the guided setup and open a new chat.',
  skipFailed: 'Could not skip setup',
  slashOffTitle: 'Slash commands are off during setup',
  slashOffBody: 'They work as usual once setup is done.',
  localModel: {
    title: 'This computer can run a model locally',
    text: (model, size) => `${model} (${size}) fits your hardware. It runs free, and your chats stay on this computer.`,
    setUp: 'Set it up',
    notNow: 'Not now'
  },
  intro: {
    greeting: "Hi, I'm your agent.",
    plan: "Let's set things up for you. Then we'll get something cool done."
  },
  rerun: {
    title: 'Run onboarding again',
    description: 'Open a new setup chat that walks through the first-run steps with your current model.',
    action: 'Run again',
    failed: 'Could not start onboarding'
  }
}
