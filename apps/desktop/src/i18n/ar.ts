import { arArtifacts } from './ar_artifacts'
import { arAssistant } from './ar_assistant'
import { arBoot } from './ar_boot'
import { arCapabilities } from './ar_capabilities'
import { arChat } from './ar_chat'
import { arChrome } from './ar_chrome'
import { arCommandCenter } from './ar_command_center'
import { arCommon } from './ar_common'
import { arConnectors } from './ar_connectors'
import { arDiagnostics } from './ar_diagnostics'
import { arSettings } from './ar_settings'
import { defineLocale } from './define-locale'

export const ar = defineLocale({
  quickCapture: {
    append: 'إضافة إلى المسودة',
    allowResend: 'السماح بإرسال آخر',
    handoffUnconfirmed: 'طُلب الإرسال إلى المحادثة، ولم يُؤكَّد التسليم. تحقّق من المحادثة قبل الإرسال مجددًا.',
    handoffRejected: 'رُفض الإرسال إلى المحادثة. لا يزال النص هنا.',

    save: 'حفظ الفكرة',
    saved: 'تم الحفظ على الجهاز',
    saving: 'جارٍ الحفظ…',
    browse: 'الأفكار المحفوظة',
    back: 'رجوع',
    empty: 'لا توجد أفكار محفوظة بعد.',
    open: 'إعادة إلى حقل الإدخال',
    placeholder: 'دوّن فكرة…',
    send: 'Enter للإرسال إلى',
    current: 'المحادثة الحالية',
    newSession: 'جلسة جديدة',
    offline: 'غير متصل — لا يزال بإمكانك الحفظ على الجهاز.',
    loadFailed: 'تعذر تحميل الأفكار. افتح اتصالاً وملفاً شخصياً في Hermes ثم أعد فتح الإدخال السريع.',
    saveFailed: 'تعذر الحفظ. النص ما زال هنا؛ حاول الحفظ مجدداً.',
    local: 'محفوظ على هذا الجهاز'
  },
  sharedMetrics: arCommon.sharedMetrics,
  externalOpenFailed: arChrome.externalOpenFailed,
  catalog: arCapabilities.catalog,
  sessionImport: arConnectors.sessionImport,
  sendDiagnostics: arDiagnostics.sendDiagnostics,
  common: arCommon.common,
  fileMenu: arChrome.fileMenu,
  boot: arBoot.boot,
  notifications: arDiagnostics.notifications,
  remoteDisplayBanner: arBoot.remoteDisplayBanner,
  titlebar: arChrome.titlebar,
  keybinds: arChrome.keybinds,
  language: arSettings.language,
  settings: arSettings.settings,
  skills: arCapabilities.skills,
  agents: arCapabilities.agents,
  commandCenter: arCommandCenter.commandCenter,
  messaging: arCommandCenter.messaging,
  profiles: arCommandCenter.profiles,
  modelAssignment: arSettings.modelAssignment,
  cron: arCommandCenter.cron,
  artifacts: arArtifacts.artifacts,
  artifactCard: arArtifacts.artifactCard,
  artifactPreview: arArtifacts.artifactPreview,
  sidebar: arChrome.sidebar,
  composer: arChat.composer,
  statusStack: arChat.statusStack,
  updates: arBoot.updates,
  guidedGreeting: arBoot.guidedGreeting,
  install: arBoot.install,
  onboarding: arBoot.onboarding,
  modelPicker: arSettings.modelPicker,
  modelVisibility: arSettings.modelVisibility,
  shell: arChrome.shell,
  rightSidebar: arChrome.rightSidebar,
  preview: arArtifacts.preview,
  interfaceMode: arSettings.interfaceMode,
  zones: arChrome.zones,
  contextMenu: arChrome.contextMenu,
  assistant: arAssistant.assistant,
  prompts: arChat.prompts,
  desktop: arChat.desktop,
  errors: arDiagnostics.errors,
  tips: arChat.tips,
  ui: arCommon.ui
})
