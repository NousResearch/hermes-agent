export interface QuestionnaireTranslations {
  skipSetup: string
  back: string
  trailLabel: string
  changeAnswer: string
  retry: string
  saveFailed: string
  skipped: string
  otherLabel: string
  kinds: {
    accent: string
    layout: string
    local: string
    apps: string
    connectors: string
  }
  name: {
    greeting: string
    question: string
    /** The name field's placeholder when the OS gave no name to offer. */
    placeholder: string
  }
  accent: {
    title: string
    custom: string
  }
  layout: {
    title: string
    basic: string
    basicDetail: string
    elite: string
    eliteDetail: string
  }
  local: {
    titleSpark: string
    title: (kind: string) => string
    download: (model: string) => string
    downloadDetail: string
    notNow: string
    notNowDetail: string
    chip: string
    gpu: string
    memory: string
    os: string
    memoryValue: (gb: number) => string
  }
  apps: {
    title: string
    plugin: string
    pluginNotRunning: string
    approve: string
    found: string
  }
  connectors: {
    title: string
    checking: string
    offered: string
  }
  task: {
    title: string
    question: string
    blender: string
    nvidia: string
    broadcast: string
    brief: string
    tidy: string
  }
  tour: {
    title: string
    question: string
    quick: string
    quickDetail: string
    none: string
    noneDetail: string
  }
  start: {
    title: (name: null | string) => string
    sub: string
    aboutMe: string
    beforeYouStart: string
    noTask: string
    action: string
    waiting: string
    failed: string
  }
  summary: {
    custom: string
    local: (model: string) => string
    localNo: string
    noApps: string
    noConnectors: string
    more: (label: string, count: number) => string
    tour: string
    noTour: string
  }
  status: {
    settingUp: string
    stillSettingUp: string
    unavailable: string
    ready: string
    downloadingEngine: string
    downloading: (model: string) => string
  }
  settings: {
    title: string
    description: string
    action: string
    failed: string
  }
}
