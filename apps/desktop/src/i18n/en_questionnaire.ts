import type { Translations } from './types'

export const enQuestionnaire: Translations['questionnaire'] = {
  skipSetup: 'Skip setup',
  back: 'Back',
  trailLabel: 'Your answers',
  changeAnswer: 'Change this answer',
  retry: 'Try again',
  saveFailed: 'Could not mark setup as done',
  skipped: 'Skipped',
  otherLabel: 'Other answer',
  kinds: {
    accent: 'Accent color',
    layout: 'Layout',
    local: 'Local model',
    apps: 'Apps on this machine',
    connectors: 'Connectors'
  },
  name: {
    greeting: "Hi, I'm Hermes.",
    question: 'What should I call you?',
    placeholder: 'Your name'
  },
  accent: {
    title: 'Pick a color',
    custom: 'Custom color'
  },
  layout: {
    title: 'How do you want to work?',
    basic: 'Basic',
    basicDetail: 'For talking to Hermes.',
    elite: 'Elite',
    eliteDetail: 'For developers: terminal, files, diffs.'
  },
  local: {
    titleSpark: 'Your Spark can run Hermes offline',
    title: kind => `This ${kind} can run a model locally`,
    download: model => `Download ${model}`,
    downloadDetail: 'Downloads in the background once your first chat starts.',
    notNow: 'Not now',
    notNowDetail: 'Stays in the model menu as “Run locally”.',
    chip: 'Chip',
    gpu: 'GPU',
    memory: 'Memory',
    os: 'OS',
    memoryValue: gb => `${gb} GB`
  },
  apps: {
    title: 'Use Hermes inside these apps?',
    plugin: 'Plugin',
    pluginNotRunning: 'Plugin · app not running',
    approve: 'You approve each install in your first chat.',
    found: 'Found on this machine. Apps that are not installed are not listed.'
  },
  connectors: {
    title: 'Connect your apps?',
    checking: 'Checking which apps your account can connect…',
    offered: 'This is the list your account offers right now. You sign in to each one in your first chat.'
  },
  task: {
    title: 'What should we do first?',
    question: 'Pick one, or type your own.',
    blender: 'Build a scene in Blender',
    nvidia: 'Tune my GPU',
    broadcast: 'Set up my mic and camera',
    brief: 'Brief me on today',
    tidy: 'Tidy my Downloads'
  },
  tour: {
    title: 'Want the tour?',
    question: 'A quick look around after?',
    quick: 'Show me around',
    quickDetail: 'A few stops, under a minute.',
    none: "I'll figure it out",
    noneDetail: 'You can ask Hermes for a tour later.'
  },
  start: {
    title: name => (name ? `Ready, ${name}.` : 'Ready.'),
    sub: 'This is the first message Hermes gets. The ask shows first; the rest scrolls under it.',
    aboutMe: 'About me:',
    beforeYouStart: 'Before you start:',
    noTask: 'No task picked: Hermes asks what you want to do.',
    action: 'Start',
    waiting: 'Getting your free account ready…',
    failed: 'Could not start your first chat'
  },
  summary: {
    custom: 'Custom',
    local: model => `Local: ${model}`,
    localNo: 'Local: not now',
    noApps: 'No apps',
    noConnectors: 'No connectors',
    more: (label, count) => `${label} +${count}`,
    tour: 'Tour',
    noTour: 'No tour'
  },
  status: {
    settingUp: 'Setting up your free account',
    stillSettingUp: 'Still setting up your free account',
    unavailable: 'Free account unavailable',
    ready: 'Nous · free tier',
    downloadingEngine: 'Downloading the local engine',
    downloading: model => `Downloading ${model}`
  },
  settings: {
    title: 'Run setup again',
    description: 'Opens the first-run questions again. Your answers apply to the default profile.',
    action: 'Run setup',
    failed: 'Could not start setup'
  }
}
