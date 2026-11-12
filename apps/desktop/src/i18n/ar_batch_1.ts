// Arabic translations batch 1. Glossary: keep Hermes, Terminal(s),
// Webhooks, Tokens, MCP, API, GPU, RAM, SSH, JSON, GPT-Live and Nous in English;
// Skills = المهارات, Memory = الذاكرة. Deep-merged over the section overlays in ar.ts.
import { formatPercent } from '@/lib/usage-percent'

import type { TranslationOverrides } from './define-locale'

export const arBatch1 = {
  assistant: {
    approval: {
      commandDetails: 'تفاصيل الأمر',
      openSafetySettings: 'فتح إعدادات الأمان',
      reconnect: 'إعادة الاتصال',
      timedOutSystemLine:
        'انتهت مهلة الموافقة ولم يُنفَّذ الأمر. اطلب من Hermes المحاولة مرة أخرى، أو ارفع المهلة من الإعدادات ← الأمان ← مهلة الموافقة.'
    },
    catalogInstall: {
      alreadyInstalled: 'مثبّت مسبقًا، وتُرك كما هو',
      envVar: name => `متغير البيئة ${name}`,
      missingEnv: names => `اضبط ${names} لإكمال الإعداد`,
      notEnabled: 'مثبّت لكنه غير مفعّل',
      phase: {
        downloading: 'جارٍ التنزيل…',
        loading_tools: 'جارٍ تحميل أدواته…',
        python_packages: 'جارٍ تثبيت حزم Python…'
      },
      requiresHermes: range => `Hermes ${range}`,
      serverNotConnected: (server, reason) => `خادم MCP ‏${server} غير متصل${reason ? `: ${reason}` : ''}`
    },
    clarify: {
      noAnswer: 'لا توجد إجابة',
      notDelivered:
        'لم يصل هذا السؤال إلى التطبيق، فلا يمكن الإجابة عنه هنا. اضغط «إيقاف» لإنهاء الدور، ثم ردّ في المحادثة.',
      skipped: 'تم التخطي'
    },
    mcpSetup: {
      authorizeAction: 'منح الإذن',
      authorizeTitle: 'منح الإذن لخوادم MCP',
      authorized: server => `تم منح الإذن لـ ${server}`,
      enableAction: 'تفعيل',
      enableTitle: 'تفعيل خوادم MCP',
      enabled: server => `تم تفعيل ${server}`,
      envRequired: 'أدخل بيانات الاعتماد المطلوبة أولًا',
      failed: server => `فشل إعداد ${server}`,
      gatewayDisconnected: 'Hermes غير متصل الآن. أعد الاتصال ثم أرسلها مرة أخرى.',
      installAction: 'تثبيت',
      installTitle: 'إضافة خوادم MCP',
      installed: server => `تم تثبيت ${server}`,
      reloadFailed: 'تم حفظ الخادم، لكن فشلت إعادة تحميل أدوات MCP، وستُحمَّل في الجلسة القادمة',
      sendFailed: 'تعذّر إرسال رد إعداد MCP',
      toolCount: count =>
        count === 1 ? 'أداة واحدة' : count === 2 ? 'أداتان' : count <= 10 ? `${count} أدوات` : `${count} أداة`
    },
    thread: {
      errorChooseModel: 'اختر نموذجًا',
      errorCodes: {
        SESSION_OWNER_BUSY: {
          body: 'يوجد دور آخر يعمل هنا، ولم تنضم رسالتك إلى قائمة الانتظار. لم يُرسَل شيء. أعد المحاولة بعد انتهاء ذلك الدور.',
          title: 'Hermes مشغول في هذه المحادثة'
        },
        SESSION_OWNER_UNCONFIRMED: {
          body: 'لم تؤكد البوابة استلام رسالتك في الوقت المحدد. إن كانت في قائمة الانتظار فستُنفَّذ بعد انتهاء الدور السابق أو حلّه. تحقق من المحادثة قبل إرسالها مرة أخرى.',
          title: 'ربما تكون رسالتك في قائمة الانتظار'
        },
        stream_drop: {
          body: 'انقطع الاتصال قبل اكتمال الرد. أعد المحاولة لإرسالها مرة أخرى.',
          title: 'انقطع الرد'
        }
      },
      errorCompressConversation: 'ضغط المحادثة',
      errorCompressFailed: 'تعذّر ضغط المحادثة',
      errorGenericProvider: 'خدمة الذكاء الاصطناعي',
      errorLayerBodies: {
        auth: 'رفضت خدمة الذكاء الاصطناعي تسجيل دخولك. تحقق من بيانات الاعتماد لهذا المزوّد، ثم أرسل رسالتك مرة أخرى.',
        billing: 'لا يوجد رصيد في حسابك لدى هذا المزوّد. اشحن رصيدك أو بدّل المزوّد، ثم أرسل مرة أخرى.',
        disk: 'القرص ممتلئ، فلم يستطع Hermes حفظ هذه المحادثة. حرّر بعض المساحة ثم أعد المحاولة.',
        endpoint: 'لا يستطيع Hermes الوصول إلى خادم النموذج المخصص. تأكد أنه يعمل، ثم أرسل رسالتك مرة أخرى.',
        gateway:
          'واجه Hermes مشكلة داخلية في بدء هذا الرد. أرسل رسالتك مرة أخرى، وإن تكررت المشكلة فأرسل بيانات التشخيص.',
        generic: 'حدث خطأ أثناء رد Hermes. أعد المحاولة، أو انسخ التفاصيل إن تكرر الخطأ.',
        provider: 'تعذّر على خدمة الذكاء الاصطناعي إكمال هذا الطلب. أعد المحاولة بعد قليل أو بدّل المزوّد.',
        runtime:
          'واجه Hermes مشكلة داخلية في بدء هذا الرد. أرسل رسالتك مرة أخرى، وإن تكررت المشكلة فأرسل بيانات التشخيص.',
        streaming: 'انقطع الاتصال قبل اكتمال الرد. أعد المحاولة لإرسالها مرة أخرى.'
      },
      errorOpenHermesFolder: 'فتح مجلد Hermes',
      errorOpenHermesFolderFailed: 'تعذّر فتح مجلد Hermes',
      errorSignInFreeTier: 'سجّل الدخول بحساب Nous',
      errorToastTitle: 'لم يستطع Hermes إكمال الرد',
      errorUpdateApiKey: 'تحديث مفتاح API',
      expandMessage: 'توسيع الرسالة',
      loadingLocalModel: model => `جارٍ تحميل ${model} إلى الذاكرة`,
      processingPrompt: 'جارٍ معالجة الطلب',
      responseStopped: 'تم إيقاف الرد'
    },
    tool: {
      actions: {
        failedToOpen: 'تعذّر الفتح'
      }
    }
  },
  common: {
    expand: 'توسيع',
    formatJson: 'تنسيق JSON',
    tryHint: term => `جرّب «${term}»`
  },
  composer: {
    hiddenQueued: 'ملاحظة إعداد',
    queueResume: 'استئناف',
    queueResumeTip: 'أُوقف بزر الإيقاف — استأنف إرسال الأدوار المنتظرة',
    queuedPaused: count => `${count} في الانتظار — متوقف مؤقتًا`,
    speakReplies: 'قراءة الردود بصوت مسموع',
    stopSpeakingReplies: 'إيقاف قراءة الردود',
    voiceEngine: 'محرك المحادثة الصوتية',
    voiceEngineChained: 'تحويل الكلام إلى نص + صوت Hermes',
    voiceEngineChainedShort: 'تحويل الكلام إلى نص',
    voiceEngineChangeFailed: 'تعذّر تغيير محرك المحادثة الصوتية',
    voiceEngineLive: 'GPT-Live (محادثة مباشرة في الاتجاهين، ويحيل المهام إلى Hermes)',
    voiceEngineLiveNeedsKey: 'يحتاج مفتاح OpenAI API',
    voiceEngineLiveShort: 'GPT-Live',
    wakeWord: phrase => `كلمة التنبيه «${phrase}»`,
    wakeWordListening: phrase => `كلمة التنبيه: «${phrase}» — يستمع`,
    wakeWordOff: phrase => `كلمة التنبيه: «${phrase}» — متوقفة`,
    wakeWordPausedVoice: phrase => `كلمة التنبيه: «${phrase}» — متوقفة مؤقتًا أثناء المحادثة الصوتية`
  },
  errors: {
    boundaryDetails: 'التفاصيل',
    sendDiagnostics: 'إرسال بيانات التشخيص'
  },
  rightSidebar: {
    terminalCloseAll: 'إغلاق الكل',
    terminalCloseOthers: 'إغلاق البقية',
    terminalDropControl: paths => `تحتوي هذه الأسماء على رموز تحكم سينفّذها Terminal كمفاتيح: ${paths}`,
    terminalDropControlTitle: 'لم تُكتب في Terminal',
    terminalDropTooLarge: (names, limit) => `الحد الأقصى لكل ملف ${limit}. لم يُرفع: ${names}`,
    terminalDropTooLargeTitle: 'أكبر من الحد المسموح للرفع',
    terminalDropUploadFailed: name => `تعذّر رفع ${name}`,
    terminalNew: 'Terminal جديد',
    terminalsAria: 'Terminals'
  },
  shell: {
    approvalMode: {
      ariaLabel: mode => `وضع الموافقة: ${mode}`,
      manual: 'يدوي',
      manualDescription: 'يسأل قبل الإجراءات التي تحتاج موافقة',
      off: 'متوقف',
      offDescription: 'يعمل بدون طلبات موافقة',
      smart: 'ذكي',
      smartDescription: 'يقيّم الإجراءات تلقائيًا ويسأل عند الحاجة',
      title: 'وضع الموافقة'
    },
    modelOptions: {
      ultra: 'فائق',
      xhigh: 'عالٍ جدًا'
    },
    statusbar: {
      cacheHitRateTitle:
        'نسبة استخدام ذاكرة الطلبات المؤقتة في هذه الجلسة — الـ Tokens المخزّنة أرخص، فكلما ارتفعت النسبة قلّت التكلفة',
      connectionCloud: host => `السحابة: ${host}`,
      connectionCloudTooltip: host => `Hermes Cloud · ${host}`,
      connectionRemote: host => `عن بُعد: ${host}`,
      connectionRemoteTooltip: host => `عن بُعد · ${host}`,
      connectionSsh: host => `SSH: ${host}`,
      connectionSshTooltip: host => `SSH · ${host}`,
      contextUsagePanel: {
        categories: {
          conversation: 'المحادثة',
          mcp: 'MCP',
          memory: 'الذاكرة',
          rules: 'القواعد',
          skills: 'المهارات',
          subagent_definitions: 'تعريفات الوكلاء الفرعيين',
          system_prompt: 'تعليمات النظام',
          tool_definitions: 'تعريفات الأدوات'
        },
        empty: 'لا توجد بيانات سياق بعد',
        loading: 'جارٍ تحميل التفاصيل…',
        percentFull: percent => `ممتلئ \u2068${formatPercent(percent / 100, 'ar')}\u2069`,
        title: 'استخدام السياق',
        tokenSummary: (used, max) => `${used} / ${max} Tokens`
      },
      customizeTitle: 'إظهار في شريط الحالة',
      hideStatusbar: 'إخفاء شريط الحالة',
      openStarmap: 'فتح خريطة الذاكرة',
      openWebhooks: 'فتح Webhooks',
      planUsageTitle: (used, resetsIn) =>
        `استُخدم ${used} من باقتك${resetsIn ? ` · يتجدد بعد ${resetsIn}` : ''} · افتح الفوترة`,
      resetStatusbar: 'استعادة الافتراضي',
      starmap: 'خريطة الذاكرة',
      systemResources: {
        gpuMemory: 'ذاكرة GPU',
        gpuUtilization: 'استخدام GPU',
        loading: 'الموارد…',
        ram: 'الذاكرة RAM',
        title: 'موارد النظام',
        toggle: 'موارد النظام',
        unifiedNote: 'ذاكرة موحّدة — يتشارك GPU والنظام هذه الذاكرة.'
      },
      toggleApprovalMode: 'الموافقات',
      toggleBackendVersion: 'إصدار الخادم',
      toggleCacheHitRate: 'نسبة استخدام الذاكرة المؤقتة',
      toggleCommandCenter: 'مركز الأوامر',
      toggleContextUsage: 'مؤشر السياق',
      toggleFreeTier: 'الباقة المجانية',
      togglePlanUsage: 'استخدام الباقة',
      toggleRunningTimer: 'مؤقت الدور',
      toggleSessionTimer: 'مؤقت الجلسة',
      toggleTerminal: 'Terminal',
      toggleTokensPerSecond: 'Tokens في الثانية',
      toggleVersion: 'الإصدار والتحديثات',
      toggleWorkspace: 'مساحة العمل',
      tokensPerSecondTitle: 'Tokens الإخراج في الثانية، بمتوسط آخر 10 استدعاءات للنموذج',
      webhooks: 'Webhooks'
    }
  },
  sidebar: {
    dateDivider: {
      lastWeek: 'الأسبوع الماضي',
      thisMonth: 'في وقت سابق هذا الشهر',
      thisWeek: 'في وقت سابق هذا الأسبوع',
      today: 'في وقت سابق اليوم',
      yesterday: 'أمس'
    },
    messageCount: count =>
      count === 1 ? 'رسالة واحدة' : count === 2 ? 'رسالتان' : count <= 10 ? `${count} رسائل` : `${count} رسالة`,
    projects: {
      baseBranchNone: 'لا توجد فروع',
      baseBranchPlaceholder: 'ابحث في الفروع…',
      branchOff: () => ({ after: '', before: 'تفرّع من ' }),
      moveBusy: 'أوقف الدور الجاري، ثم انقل المحادثة',
      moveFailed: 'تعذّر نقل الجلسة',
      moveNoFolder: 'هذا المشروع ليس له مجلد للنقل إليه',
      moveNoProjects: 'لا توجد مشاريع أخرى',
      moveToProject: 'نقل إلى مشروع',
      movedTo: name => `تم النقل إلى ${name}`,
      staleBackend:
        'حدّث خادم Hermes لإنشاء المشاريع — الخادم أقدم من تطبيق سطح المكتب (الإعدادات ← التحديثات ← الخادم).'
    },
    row: {
      messageCount: count =>
        count === 1 ? 'رسالة واحدة' : count === 2 ? 'رسالتان' : count <= 10 ? `${count} رسائل` : `${count} رسالة`,
      todoProgress: 'المهام المنجزة'
    },
    statusDivider: {
      done: 'منتهية',
      working: 'قيد العمل'
    },
    storageCorrupt: {
      action: 'أغلق Hermes على هذا الملف الشخصي، ثم افحص الملف دون تعديله، أو استرجع نسخة احتياطية:',
      body: (profiles: string) =>
        `لا يستطيع Hermes قراءة سجل الجلسات كاملًا لـ ${profiles}. المحادثات الناقصة من هذه القائمة لم تُحذف، لكن الملف المحفوظة فيه تالف.`,
      guide: 'دليل الاسترداد',
      title: 'قاعدة بيانات الجلسات تالفة'
    },
    toolCallCount: count =>
      count === 1
        ? 'استدعاء أداة واحد'
        : count === 2
          ? 'استدعاءا أداة'
          : count <= 10
            ? `${count} استدعاءات أدوات`
            : `${count} استدعاء أداة`
  }
} satisfies TranslationOverrides
