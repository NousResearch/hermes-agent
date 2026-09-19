import { defineLocale } from './define-locale'
import { jaArtifacts } from './ja_artifacts'
import { jaAssistant } from './ja_assistant'
import { jaBoot } from './ja_boot'
import { jaCapabilities } from './ja_capabilities'
import { jaChat } from './ja_chat'
import { jaChrome } from './ja_chrome'
import { jaCommandCenter } from './ja_command_center'
import { jaCommon } from './ja_common'
import { jaConnectors } from './ja_connectors'
import { jaDiagnostics } from './ja_diagnostics'
import { jaSettings } from './ja_settings'

export const ja = defineLocale({
  sessionImport: jaConnectors.sessionImport,
  common: jaCommon.common,
  fileMenu: jaChrome.fileMenu,
  boot: jaBoot.boot,
  notifications: jaDiagnostics.notifications,
  remoteDisplayBanner: jaBoot.remoteDisplayBanner,
  billingBlock: jaCommon.billingBlock,
  sendDiagnostics: jaDiagnostics.sendDiagnostics,
  titlebar: jaChrome.titlebar,
  language: jaSettings.language,
  settings: jaSettings.settings,
  skills: jaCapabilities.skills,
  starmap: jaCapabilities.starmap,
  agents: jaCapabilities.agents,
  commandCenter: jaCommandCenter.commandCenter,
  messaging: jaCommandCenter.messaging,
  profiles: jaCommandCenter.profiles,
  cron: jaCommandCenter.cron,
  artifacts: jaArtifacts.artifacts,
  artifactCard: jaArtifacts.artifactCard,
  artifactPreview: jaArtifacts.artifactPreview,
  sidebar: jaChrome.sidebar,
  composer: jaChat.composer,
  statusStack: jaChat.statusStack,
  updates: jaBoot.updates,
  guidedGreeting: jaBoot.guidedGreeting,
  install: jaBoot.install,
  onboarding: jaBoot.onboarding,
  modelPicker: jaSettings.modelPicker,
  modelVisibility: jaSettings.modelVisibility,
  shell: jaChrome.shell,
  rightSidebar: jaChrome.rightSidebar,
  preview: jaArtifacts.preview,
  zones: jaChrome.zones,
  contextMenu: jaChrome.contextMenu,
  assistant: jaAssistant.assistant,
  prompts: jaChat.prompts,
  desktop: jaChat.desktop,
  tips: jaChat.tips,
  errors: jaDiagnostics.errors,
  ui: jaCommon.ui
})
