import { enArtifacts } from './en_artifacts'
import { enAssistant } from './en_assistant'
import { enBoot } from './en_boot'
import { enCapabilities } from './en_capabilities'
import { enChat } from './en_chat'
import { enChrome } from './en_chrome'
import { enCommandCenter } from './en_command_center'
import { enCommon } from './en_common'
import { enConnectors } from './en_connectors'
import { enDiagnostics } from './en_diagnostics'
import { enSettings } from './en_settings'
import type { Translations } from './types'

export const en: Translations = {
  connectors: enConnectors.connectors,
  sessionImport: enConnectors.sessionImport,
  common: enCommon.common,
  fileMenu: enChrome.fileMenu,
  boot: enBoot.boot,
  notifications: enDiagnostics.notifications,
  remoteDisplayBanner: enBoot.remoteDisplayBanner,
  billingBlock: enCommon.billingBlock,
  sendDiagnostics: enDiagnostics.sendDiagnostics,
  titlebar: enChrome.titlebar,
  keybinds: enChrome.keybinds,
  findInPage: enChrome.findInPage,
  language: enSettings.language,
  settings: enSettings.settings,
  skills: enCapabilities.skills,
  starmap: enCapabilities.starmap,
  agents: enCapabilities.agents,
  commandCenter: enCommandCenter.commandCenter,
  messaging: enCommandCenter.messaging,
  webhooks: enCommandCenter.webhooks,
  profiles: enCommandCenter.profiles,
  cron: enCommandCenter.cron,
  artifacts: enArtifacts.artifacts,
  artifactCard: enArtifacts.artifactCard,
  artifactPreview: enArtifacts.artifactPreview,
  sidebar: enChrome.sidebar,
  composer: enChat.composer,
  statusStack: enChat.statusStack,
  updates: enBoot.updates,
  handoffTour: enBoot.handoffTour,
  guidedGreeting: enBoot.guidedGreeting,
  install: enBoot.install,
  onboarding: enBoot.onboarding,
  freeTier: enBoot.freeTier,
  modelPicker: enSettings.modelPicker,
  modelVisibility: enSettings.modelVisibility,
  shell: enChrome.shell,
  rightSidebar: enChrome.rightSidebar,
  preview: enArtifacts.preview,
  zones: enChrome.zones,
  contextMenu: enChrome.contextMenu,
  assistant: enAssistant.assistant,
  prompts: enChat.prompts,
  desktop: enChat.desktop,
  tips: enChat.tips,
  errors: enDiagnostics.errors,
  ui: enCommon.ui
}
