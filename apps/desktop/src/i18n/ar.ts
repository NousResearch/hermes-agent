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
import { defineLocale, type TranslationOverrides } from './define-locale'

export const arOverrides = {
  sharedMetrics: arCommon.sharedMetrics,
  externalOpenFailed: arChrome.externalOpenFailed,
  sessionImport: arConnectors.sessionImport,
  codingWorkspace: {
    openFailed: 'تعذر فتح مجلد مساحة العمل',
    worktree: 'شجرة العمل',
    folder: 'مجلد',
    inUse: 'قيد الاستخدام',
    project: 'المشروع',
    workIn: 'العمل في',
    noProject: 'بدون مشروع',
    newWorktree: 'شجرة عمل جديدة',
    existingWorktree: 'شجرة عمل موجودة',
    currentCheckout: 'نسخة العمل الحالية',
    agentWorktree: 'شجرة عمل الوكيل',
    agentWorktreeNote: 'أنشأ الوكيل شجرة العمل هذه أثناء المحادثة.',
    newChatHere: 'محادثة جديدة في شجرة العمل هذه',
    projectFolder: 'مجلد المشروع',
    initializeGit: 'تهيئة مستودع Git',
    initializeGitDescription: 'ينشئ المستودع بإيداع أول فارغ. تبقى ملفاتك غير متتبعة.',
    initializeGitFailed: 'تعذر تهيئة مستودع Git',
    searchProjects: 'البحث في المشاريع',
    noProjects: 'لا توجد مشاريع مطابقة',
    projectsFailed: 'تعذر تحميل المشاريع',
    browse: 'استعراض…',
    base: 'من الفرع',
    dirty: 'تغييرات غير مثبتة',
    clean: 'دون تغييرات',
    detached: 'HEAD منفصل',
    dirtyNotCopied: 'لا تُنسخ التغييرات غير المثبتة إلى شجرة العمل الجديدة.',
    createOnSend: 'تُنشأ عند الإرسال الأول، وليس الآن.',
    shared: 'قد تشترك محادثات أخرى في نسخة العمل هذه. التغييرات غير معزولة.',
    preparing: 'جارٍ إعداد مساحة العمل',
    workInProject: 'العمل في مشروع…',
    useAsProject: 'استخدام كمشروع',
    newChat: 'محادثة جديدة في مساحة أخرى…',
    binding: 'تبقى هذه المحادثة في مساحة عملها الأصلية.',
    selectCheckout: 'اختر نسخة العمل',
    folderFailed: 'تعذر استخدام هذا المجلد كمشروع',
    unavailable: 'لا يمكن اختيار مساحة العمل حتى يتصل مالك المحادثة.',

    showControls: 'إظهار أدوات البرمجة',
    showControlsDescription:
      'اعرض المشروع ومكان العمل فوق حقل الإدخال للمحادثات الجديدة في هذا الملف الشخصي. لا تتغير المحادثات الحالية أو تفضيلات الشريط الجانبي.',
    saveFailed: 'تعذر حفظ إعداد أدوات البرمجة'
  },
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
  ui: arCommon.ui,
  handoffTour: {
    profileTitle: 'مهمتك الأولى تعمل على الملف الشخصي الافتراضي',
    profileText:
      'يبدّل هذا الشريط بين الملفات الشخصية. المضاء الآن هو الافتراضي، حيث توجد جلسة المهمة. والآخر هو ملف الإعداد، حيث توجد محادثة الترحيب.',
    sessionsTitle: 'لكل ملف شخصي جلساته الخاصة',
    sessionsText:
      'هذه القائمة تخص الملف الافتراضي. «جلسة جديدة» تبدأ جلسة على الملف المحدد. بدّل الملف من الشريط فتتغير القائمة معه.',
    stayTitle: 'Hermes على بُعد نقرة',
    stayText: 'انتقل إلى ملف الإعداد وافتح «مرحبًا بك في Hermes» متى احتجت إلى مساعدة. ستبقى هناك.',
    localTitle: 'يمكن لهذا الجهاز تشغيل النماذج محليًا',
    localText: (model: string) =>
      `${model} يناسب أجهزتك. يعمل مجانًا، ولا تغادر المحادثات جهازك. اختره من هنا، من قائمة النماذج، متى شئت.`
  },
  freeTier: {
    offer: {
      heading: 'واصل مع Hermes',
      body: 'أنت تستخدم الحصة المجانية. إذا واصلت استخدام Hermes فستبدأ بمواجهة حدود الاستخدام. سجّل الدخول بحساب Nous مجاني للحصول على حصة أكبر.',
      signIn: 'تسجيل الدخول',
      notNow: 'ليس الآن'
    }
  }
} satisfies TranslationOverrides

export const ar = defineLocale(arOverrides)
