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
  sharedMetrics: arCommon.sharedMetrics,
  externalOpenFailed: arChrome.externalOpenFailed,
  sessionImport: arConnectors.sessionImport,
  sendDiagnostics: arDiagnostics.sendDiagnostics,
  common: arCommon.common,
  fileMenu: arChrome.fileMenu,
  boot: arBoot.boot,
  notifications: arDiagnostics.notifications,
  remoteDisplayBanner: arBoot.remoteDisplayBanner,
  butterbar: arBoot.butterbar,
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
  shell: {
    ...arChrome.shell,
    statusbar: {
      ...arChrome.shell.statusbar,
      toggleAccountUsage: 'استخدام الحساب',
      accountUsage: 'استخدام الحساب',
      accountUsageLeft: remaining => `متبقٍ ${remaining}%`,
      accountUsagePanel: {
        openUsageSettings: 'فتح إعدادات الاستخدام',
        plan: plan => `خطة ${plan}`,
        refresh: 'تحديث',
        remaining: remaining => `متبقٍ ${remaining}%`,
        resets: time => `يُعاد التعيين ${time}`,
        sessionLine: (input, output, total) => `${input} وارد · ${output} صادر · ${total} إجمالي`,
        stale: 'تعذر تحديث الاستخدام. يتم عرض آخر نتيجة ناجحة.',
        thisSession: 'هذه الجلسة',
        title: 'استخدام الحساب',
        unavailable: 'غير متاح',
        updated: time => `حُدّث ${time}`,
        updatedUnknown: 'وقت التحديث غير معروف',
        used: used => `مستخدم ${used}%`,
        apiKeyUsage: 'استخدام مفتاح API',
        bankedResets: count =>
          count === 1
            ? 'لديك إعادة تعيين واحدة محفوظة — استخدم /usage reset للتفعيل'
            : `لديك ${count} إعادة تعيين محفوظة — استخدم /usage reset للتفعيل`,
        creditsBalance: 'رصيد الاعتمادات',
        creditsUnlimited: 'غير محدود',
        extraUsage: 'استخدام إضافي',
        extraUsageValue: (used, limit) => `${used} / ${limit}`,
        ofLimitRemaining: (remaining, limit) => `متبقٍ ${remaining} من ${limit}`,
        resetIntervals: {
          daily: 'يُعاد التعيين يومياً',
          monthly: 'يُعاد التعيين شهرياً',
          weekly: 'يُعاد التعيين أسبوعياً'
        },
        usageThisMonth: value => `${value} هذا الشهر`,
        usageThisWeek: value => `${value} هذا الأسبوع`,
        usageToday: value => `${value} اليوم`,
        usageTotal: value => `${value} إجمالي`,
        windowLabels: {
          api_key_quota: 'حصة مفتاح API',
          current_session: 'الجلسة الحالية',
          current_week: 'الأسبوع الحالي',
          opus_week: 'أسبوع Opus',
          session: 'الجلسة',
          sonnet_week: 'أسبوع Sonnet',
          weekly: 'أسبوعي'
        }
      }
    }
  },
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
