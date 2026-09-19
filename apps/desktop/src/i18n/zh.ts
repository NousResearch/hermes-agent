import { defineLocale } from './define-locale'
import { zhArtifacts } from './zh_artifacts'
import { zhAssistant } from './zh_assistant'
import { zhBoot } from './zh_boot'
import { zhCapabilities } from './zh_capabilities'
import { zhChat } from './zh_chat'
import { zhChrome } from './zh_chrome'
import { zhCommandCenter } from './zh_command_center'
import { zhCommon } from './zh_common'
import { zhConnectors } from './zh_connectors'
import { zhDiagnostics } from './zh_diagnostics'
import { zhSettings } from './zh_settings'

export const zh = defineLocale({
  connectors: zhConnectors.connectors,
  sessionImport: zhConnectors.sessionImport,
  common: zhCommon.common,
  fileMenu: zhChrome.fileMenu,
  boot: zhBoot.boot,
  notifications: zhDiagnostics.notifications,
  remoteDisplayBanner: zhBoot.remoteDisplayBanner,
  billingBlock: zhCommon.billingBlock,
  sendDiagnostics: zhDiagnostics.sendDiagnostics,
  titlebar: zhChrome.titlebar,
  keybinds: zhChrome.keybinds,
  findInPage: zhChrome.findInPage,
  language: zhSettings.language,
  settings: zhSettings.settings,
  skills: zhCapabilities.skills,
  starmap: zhCapabilities.starmap,
  agents: zhCapabilities.agents,
  commandCenter: zhCommandCenter.commandCenter,
  messaging: zhCommandCenter.messaging,
  webhooks: zhCommandCenter.webhooks,
  profiles: zhCommandCenter.profiles,
  cron: zhCommandCenter.cron,
  artifacts: zhArtifacts.artifacts,
  artifactCard: zhArtifacts.artifactCard,
  artifactPreview: zhArtifacts.artifactPreview,
  sidebar: zhChrome.sidebar,
  composer: zhChat.composer,
  statusStack: zhChat.statusStack,
  updates: zhBoot.updates,
  guidedGreeting: zhBoot.guidedGreeting,
  install: zhBoot.install,
  onboarding: zhBoot.onboarding,
  freeTier: zhBoot.freeTier,
  modelPicker: zhSettings.modelPicker,
  modelVisibility: zhSettings.modelVisibility,
  shell: zhChrome.shell,
  rightSidebar: zhChrome.rightSidebar,
  preview: zhArtifacts.preview,
  zones: zhChrome.zones,
  contextMenu: zhChrome.contextMenu,
  assistant: zhAssistant.assistant,
  prompts: zhChat.prompts,
  desktop: zhChat.desktop,
  tips: zhChat.tips,
  errors: zhDiagnostics.errors,
  ui: zhCommon.ui
})
