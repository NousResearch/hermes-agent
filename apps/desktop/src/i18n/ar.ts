import { defineFieldCopy } from '@/app/settings/field-copy'

import type { Translations } from './types'

export const ar: Translations = {
  common: {
    apply: 'تطبيق',
    back: 'رجوع',
    save: 'حفظ',
    saving: 'جارٍ الحفظ…',
    cancel: 'إلغاء',
    change: 'تغيير',
    choose: 'اختيار',
    clear: 'مسح',
    close: 'إغلاق',
    collapse: 'طي',
    confirm: 'تأكيد',
    connect: 'اتصال',
    connecting: 'جارٍ الاتصال',
    continue: 'متابعة',
    copied: 'نُسخ',
    copy: 'نسخ',
    copyFailed: 'تعذر النسخ',
    delete: 'حذف',
    docs: 'التوثيق',
    done: 'تم',
    error: 'خطأ',
    expand: 'توسيع',
    failed: 'فشل',
    formatJson: 'تنسيق JSON',
    free: 'مجاني',
    loading: 'جارٍ التحميل…',
    notSet: 'غير مضبوط',
    refresh: 'تحديث',
    remove: 'إزالة',
    replace: 'استبدال',
    retry: 'إعادة المحاولة',
    run: 'تشغيل',
    send: 'إرسال',
    set: 'ضبط',
    skip: 'تخطي',
    update: 'تحديث',
    tryHint: term => `جرّب «${term}»`,
    on: 'مفعّل',
    off: 'معطّل'
  },

  intro: {
    wordmark: 'وكيل هرمس',
    body: 'أرسل خللًا أو فرعًا أو خطةً أو فكرةً أولية، وسأفحص المستودع وأحوّلها إلى خطوةٍ تنفيذية ملموسة.'
  },

  fileMenu: {
    revealFinder: 'إظهار في Finder',
    revealExplorer: 'إظهار في مستكشف الملفات',
    revealFileManager: 'فتح المجلد الحاوي',
    revealInSidebar: 'إظهار في شجرة الملفات',
    copyPath: 'نسخ المسار',
    copyRelativePath: 'نسخ المسار النسبي',
    rename: 'إعادة تسمية…',
    delete: 'حذف',
    renameTitle: 'إعادة تسمية',
    renameLabel: 'الاسم الجديد',
    deleteTitle: name => `حذف ${name}؟`,
    deleteBody: 'سيُنقَل إلى سلّة المهملات — يمكنك استرجاعه من هناك.',
    pathCopied: 'نُسخ المسار'
  },

  boot: {
    ready: 'هرمس لسطح المكتب جاهز',
    desktopBootFailedWithMessage: message => `فشل بدء سطح المكتب: ${message}`,
    steps: {
      connectingGateway: 'جارٍ الاتصال ببوابة سطح المكتب المباشرة',
      loadingSettings: 'جارٍ تحميل إعدادات هرمس',
      loadingSessions: 'جارٍ تحميل الجلسات الأخيرة',
      startingDesktopConnection: 'جارٍ بدء اتصال سطح المكتب',
      startingHermesDesktop: 'جارٍ بدء هرمس لسطح المكتب…'
    },
    errors: {
      backgroundExited: 'توقفت عملية هرمس الخلفية.',
      backgroundExitedDuringStartup: 'توقفت عملية هرمس الخلفية أثناء البدء.',
      backendStopped: 'توقفت الواجهة الخلفية',
      desktopBootFailed: 'فشل بدء سطح المكتب',
      gatewayConnectionLost: 'انقطع الاتصال بالبوابة',
      gatewaySignInRequired: 'يلزم تسجيل الدخول إلى البوابة',
      ipcBridgeUnavailable: 'جسر الاتصال الداخلي لسطح المكتب غير متاح.'
    },
    failure: {
      title: 'تعذر بدء هرمس',
      description: 'لم تبدأ البوابة الخلفية. جرّب إحدى خطوات الاسترداد أدناه. لن يحذف أي خيار محادثاتك أو إعداداتك.',
      remoteTitle: 'يلزم تسجيل الدخول إلى البوابة البعيدة',
      remoteDescription:
        'انتهت جلسة البوابة البعيدة. سجّل الدخول مجددًا لإعادة الاتصال. لن يحذف ذلك محادثاتك أو إعداداتك.',
      retry: 'إعادة المحاولة',
      repairInstall: 'إصلاح التثبيت',
      useLocalGateway: 'استخدام البوابة المحلية',
      gatewaySettings: 'إعدادات البوابة',
      back: 'رجوع',
      openLogs: 'فتح السجلات',
      repairHint: 'يعيد الإصلاح تشغيل المثبّت، وقد يستغرق بضع دقائق على جهاز جديد.',
      remoteSignInHint: signInLabel =>
        `يسجّل الخروج من جلسة المتصفّح البعيد المحفوظة، ثمّ يفتح ${signInLabel}. استخدم البوابة المحلية للانتقال إلى الواجهة الخلفية المضمّنة.`,
      signOutAndSignIn: 'تسجيل الخروج ثم الدخول',
      remoteFailureHint: 'تحقّق من رابط البوابة وتسجيل الدخول ضمن إعدادات البوابة، أو انتقل إلى البوابة المحلية.',
      hideRecentLogs: 'إخفاء السجلات الأخيرة',
      showRecentLogs: 'إظهار السجلات الأخيرة',
      signedInTitle: 'تم تسجيل الدخول',
      signedInMessage: 'جارٍ إعادة الاتصال بالبوابة البعيدة…',
      signInIncompleteTitle: 'لم يكتمل تسجيل الدخول',
      signInIncompleteMessage: 'أُغلقت نافذة الدخول قبل اكتمال المصادقة.',
      signInFailed: 'فشل تسجيل الدخول',
      signInToRemoteGateway: 'تسجيل الدخول إلى البوابة البعيدة',
      signInWithProvider: provider => `تسجيل الدخول عبر ${provider}`,
      identityProvider: 'مزوّد الهوية'
    }
  },

  notifications: {
    region: 'الإشعارات',
    hide: 'إخفاء',
    show: 'إظهار',
    more: count =>
      count === 1
        ? 'إشعار إضافي واحد'
        : count === 2
          ? 'إشعاران إضافيان'
          : count <= 10
            ? `${count} إشعارات إضافية`
            : `${count} إشعارًا إضافيًا`,
    clearAll: 'مسح الكل',
    dismiss: 'إغلاق الإشعار',
    details: 'التفاصيل',
    copyDetail: 'نسخ التفاصيل',
    copyDetailFailed: 'تعذر نسخ تفاصيل الإشعار',
    backendOutOfDateTitle: 'الواجهة الخلفية قديمة',
    backendOutOfDateMessage: 'واجهة هرمس الخلفية أقدم من إصدار سطح المكتب، وقد لا تعمل على نحو صحيح. حدّثها لتتوافقا.',
    installMethodUnsupportedTitle: 'طريقة تثبيت غير مدعومة',
    updateHermes: 'تحديث هرمس',
    updateReadyTitle: 'التحديث جاهز',
    updateReadyMessage: count =>
      count === 1
        ? 'تغيير جديد واحد متاح.'
        : count === 2
          ? 'تغييران جديدان متاحان.'
          : count <= 10
            ? `${count} تغييرات جديدة متاحة.`
            : `${count} تغييرًا جديدًا متاحًا.`,
    seeWhatsNew: 'عرض الجديد',
    errors: {
      elevenLabsNeedsKey: 'يتطلب تحويل الكلام إلى نص من ElevenLabs المفتاح ELEVENLABS_API_KEY.',
      elevenLabsRejectedKey: 'رفض ElevenLabs مفتاح الواجهة البرمجية (401).',
      methodNotAllowed:
        'رفضت واجهة سطح المكتب الخلفية الطلب (405 Method Not Allowed). جرّب إعادة تشغيل هرمس لسطح المكتب.',
      microphonePermission: 'رُفض إذن الميكروفون.',
      openaiRejectedApiKey: 'رفض OpenAI مفتاح الواجهة البرمجية.',
      openaiRejectedApiKeyWithStatus: status => `رفض OpenAI مفتاح الواجهة البرمجية (${status} invalid_api_key).`,
      openaiTtsNeedsKey: 'يتطلب تحويل النص إلى كلام من OpenAI المفتاح VOICE_TOOLS_OPENAI_KEY أو OPENAI_API_KEY.'
    },
    voice: {
      configureSpeechToText: 'اضبط تحويل الكلام إلى نص لاستخدام الوضع الصوتي.',
      couldNotStartSession: 'تعذر بدء الجلسة الصوتية',
      microphoneAccessDenied: 'رُفض الوصول إلى الميكروفون.',
      microphoneConstraintsUnsupported: 'لا يدعم هذا الجهاز قيود الميكروفون المطلوبة.',
      microphoneFailed: 'تعطل الميكروفون',
      microphoneInUse: 'الميكروفون مستخدم في تطبيق آخر.',
      microphonePermissionDenied: 'رُفض إذن الميكروفون.',
      microphoneStartFailed: 'تعذر بدء تسجيل الميكروفون.',
      microphoneUnsupported: 'لا تدعم بيئة التشغيل هذه تسجيل الميكروفون.',
      noMicrophone: 'لم يُعثر على ميكروفون.',
      noSpeechDetected: 'لم يُكتشف كلام',
      playbackFailed: 'فشل تشغيل الصوت',
      recordingFailed: 'فشل تسجيل الصوت',
      transcriptionFailed: 'فشل تفريغ الصوت',
      transcriptionUnavailable: 'تفريغ الصوت غير متاح بعد.',
      tryRecordingAgain: 'جرّب التسجيل مجددًا.',
      unavailable: 'الصوت غير متاح'
    },
    native: {
      approvalTitle: 'مطلوب موافقة',
      approveAction: 'موافقة',
      rejectAction: 'رفض',
      inputTitle: 'مطلوب إدخال',
      inputBody: 'ينتظر هرمس ردك.',
      turnDoneTitle: 'أكمل هرمس الدورة',
      turnDoneBody: 'الرد جاهز.',
      turnErrorTitle: 'فشلت الدورة',
      backgroundDoneTitle: 'اكتملت المهمة في الخلفية',
      backgroundFailedTitle: 'فشلت المهمة في الخلفية'
    }
  },

  remoteDisplayBanner: {
    message: reason =>
      `التصيير البرمجي مُفعّل — اكتُشف عرض بعيد (${reason}). تعطيل تسريع كرت الرسومات لمنع الوميض.`
  },

  titlebar: {
    hideSidebar: 'إخفاء الشريط الجانبي',
    showSidebar: 'إظهار الشريط الجانبي',
    search: 'بحث',
    searchTitle: 'البحث في الجلسات وطرق العرض والإجراءات',
    swapSidebarSides: 'تبديل جانبي الشريطين',
    swapSidebarSidesTitle: 'تبديل موضعي الجلسات ومتصفح الملفات',
    hideRightSidebar: 'إخفاء الشريط الجانبي الأيمن',
    showRightSidebar: 'إظهار الشريط الجانبي الأيمن',
    muteHaptics: 'كتم الاستجابة اللمسية',
    unmuteHaptics: 'تشغيل الاستجابة اللمسية',
    openSettings: 'فتح الإعدادات',
    openKeybinds: 'اختصارات لوحة المفاتيح',
    openStarmap: 'فتح مخطّط الذاكرة'
  },

  keybinds: {
    title: 'اختصارات لوحة المفاتيح',
    subtitle: open => `انقر اختصارًا لإعادة تعيينه · يفتح ${open} هذه اللوحة مجددًا.`,
    rebind: 'إعادة تعيين',
    reset: 'استعادة الافتراضي',
    resetAll: 'استعادة الكل',
    pressKey: 'اضغط مفتاحًا…',
    set: 'ضبط',
    conflictWith: label => `مُعيّن أيضًا إلى «${label}»`,
    categories: {
      composer: 'محرر الرسالة',
      profiles: 'الملفات الشخصية',
      session: 'الجلسة',
      navigation: 'التنقل',
      view: 'العرض'
    },
    actions: {
      'keybinds.openPanel': 'فتح اختصارات لوحة المفاتيح',
      'nav.commandPalette': 'فتح لوحة الأوامر',
      'nav.commandCenter': 'فتح مركز الأوامر',
      'nav.settings': 'فتح الإعدادات',
      'nav.profiles': 'فتح الملفات الشخصية',
      'nav.skills': 'فتح المهارات',
      'nav.messaging': 'فتح المراسلة',
      'nav.artifacts': 'فتح المخرجات',
      'nav.cron': 'فتح المهام المجدولة',
      'nav.agents': 'فتح الوكلاء',
      'session.new': 'جلسة جديدة',
      'session.newWindow': 'جلسة جديدة في نافذة',
      'session.next': 'الجلسة التالية',
      'session.prev': 'الجلسة السابقة',
      'session.slot.1': 'الانتقال إلى الجلسة الأخيرة 1',
      'session.slot.2': 'الانتقال إلى الجلسة الأخيرة 2',
      'session.slot.3': 'الانتقال إلى الجلسة الأخيرة 3',
      'session.slot.4': 'الانتقال إلى الجلسة الأخيرة 4',
      'session.slot.5': 'الانتقال إلى الجلسة الأخيرة 5',
      'session.slot.6': 'الانتقال إلى الجلسة الأخيرة 6',
      'session.slot.7': 'الانتقال إلى الجلسة الأخيرة 7',
      'session.slot.8': 'الانتقال إلى الجلسة الأخيرة 8',
      'session.slot.9': 'الانتقال إلى الجلسة الأخيرة 9',
      'session.focusSearch': 'البحث في الجلسات',
      'session.togglePin': 'تثبيت الجلسة الحالية أو إلغاء تثبيتها',
      'workspace.newWorktree': 'شجرة عمل جديدة',
      'composer.focus': 'التركيز على محرر الرسالة',
      'composer.modelPicker': 'فتح منتقي النموذج',
      'composer.voice': 'بدء محادثة صوتية أو إيقافها',
      'view.toggleSidebar': 'تبديل شريط الجلسات',
      'view.toggleRightSidebar': 'تبديل متصفح الملفات',
      'view.toggleReview': 'تبديل لوح المراجعة',
      'view.showFiles': 'إظهار متصفح الملفات',
      'view.showTerminal': 'إظهار الطرفية',
      'view.newTerminal': 'طرفية جديدة',
      'view.nextTerminal': 'الطرفية التالية',
      'view.prevTerminal': 'الطرفية السابقة',
      'view.closeTerminal': 'إغلاق الطرفية',
      'view.terminalSelection': 'إرسال تحديد الطرفية إلى محرر الرسالة',
      'view.closePreviewTab': 'إغلاق تبويب المعاينة',
      'view.flipPanes': 'تبديل جانبي الشريطين',
      'appearance.toggleMode': 'تبديل الوضع الفاتح والداكن',
      'profile.default': 'الانتقال إلى الملف الشخصي الافتراضي',
      'profile.switch.1': 'الانتقال إلى الملف الشخصي 1',
      'profile.switch.2': 'الانتقال إلى الملف الشخصي 2',
      'profile.switch.3': 'الانتقال إلى الملف الشخصي 3',
      'profile.switch.4': 'الانتقال إلى الملف الشخصي 4',
      'profile.switch.5': 'الانتقال إلى الملف الشخصي 5',
      'profile.switch.6': 'الانتقال إلى الملف الشخصي 6',
      'profile.switch.7': 'الانتقال إلى الملف الشخصي 7',
      'profile.switch.8': 'الانتقال إلى الملف الشخصي 8',
      'profile.switch.9': 'الانتقال إلى الملف الشخصي 9',
      'profile.switch.10': 'الانتقال إلى الملف الشخصي 10',
      'profile.switch.11': 'الانتقال إلى الملف الشخصي 11',
      'profile.switch.12': 'الانتقال إلى الملف الشخصي 12',
      'profile.switch.13': 'الانتقال إلى الملف الشخصي 13',
      'profile.switch.14': 'الانتقال إلى الملف الشخصي 14',
      'profile.switch.15': 'الانتقال إلى الملف الشخصي 15',
      'profile.switch.16': 'الانتقال إلى الملف الشخصي 16',
      'profile.switch.17': 'الانتقال إلى الملف الشخصي 17',
      'profile.switch.18': 'الانتقال إلى الملف الشخصي 18',
      'profile.next': 'الملف الشخصي التالي',
      'profile.prev': 'الملف الشخصي السابق',
      'profile.toggleAll': 'تبديل عرض جميع الملفات الشخصية',
      'profile.create': 'إنشاء ملف شخصي',
      'composer.send': 'إرسال الرسالة',
      'composer.newline': 'إدراج سطر جديد',
      'composer.steer': 'توجيه الدورة الجارية',
      'composer.sendQueued': 'إرسال الدورة التالية في قائمة الانتظار',
      'composer.mention': 'الإشارة إلى ملفات ومجلدات وروابط',
      'composer.slash': 'لوحة الأوامر المائلة',
      'composer.help': 'المساعدة السريعة',
      'composer.history': 'التنقل في النافذة المنبثقة أو السجل',
      'composer.cancel': 'إغلاق النافذة المنبثقة وإلغاء التشغيل'
    }
  },

  language: {
    label: 'اللغة',
    description: 'اختر لغة واجهة سطح المكتب.',
    saving: 'جارٍ حفظ اللغة…',
    saveError: 'فشل تحديث اللغة',
    switchTo: 'تغيير اللغة',
    searchPlaceholder: 'البحث في اللغات…',
    noResults: 'لم يُعثر على لغات'
  },

  settings: {
    closeSettings: 'إغلاق الإعدادات',
    exportConfig: 'تصدير الإعدادات',
    importConfig: 'استيراد الإعدادات',
    resetToDefaults: 'استعادة القيم الافتراضية',
    resetConfirm: 'هل تريد استعادة جميع إعدادات هرمس الافتراضية؟',
    exportFailed: 'فشل التصدير',
    resetFailed: 'فشلت الاستعادة',
    nav: {
      providers: 'المزوّدون',
      providerAccounts: 'الحسابات',
      providerApiKeys: 'مفاتيح الواجهة البرمجية',
      gateway: 'البوابة',
      apiKeys: 'الأدوات والمفاتيح',
      keysTools: 'الأدوات',
      keysSettings: 'الإعدادات',
      mcp: 'MCP',
      archivedChats: 'المحادثات المؤرشفة',
      about: 'حول',
      notifications: 'الإشعارات'
    },
    notifications: {
      title: 'الإشعارات',
      intro:
        'إشعارات نظام سطح المكتب منفصلة عن التنبيهات داخل التطبيق. هذه الإعدادات محلية على الجهاز؛ يحتفظ كل حاسوب بإعداداته الخاصة.',
      enableAll: 'تفعيل الإشعارات',
      enableAllDesc: 'المفتاح الرئيسي. أوقفه لكتم جميع الإشعارات أدناه.',
      focusedHint: 'لا تظهر تنبيهات الاكتمال إلا عندما يكون هرمس في الخلفية.',
      kinds: {
        approval: {
          label: 'مطلوب موافقة',
          description: 'ينتظر أمر موافقتك أو رفضك.'
        },
        input: {
          label: 'مطلوب إدخال',
          description: 'طرح هرمس سؤالًا أو يحتاج كلمة مرور أو سرًا.'
        },
        turnDone: {
          label: 'الرد جاهز',
          description: 'اكتملت دورة بينما كان هرمس في الخلفية.'
        },
        turnError: {
          label: 'فشلت الدورة',
          description: 'انتهت دورة بخطأ.'
        },
        backgroundDone: {
          label: 'اكتملت المهمة في الخلفية',
          description: 'اكتمل أمر طرفية يعمل في الخلفية.'
        }
      },
      test: 'إرسال إشعار تجريبي',
      testTitle: 'هرمس',
      testBody: 'تعمل الإشعارات.',
      testSent: 'أُرسل الاختبار. إذا لم يظهر شيء، فتحقق من أذونات إشعارات النظام ووضع التركيز أو عدم الإزعاج.',
      testUnsupported: 'لا يدعم هذا النظام إشعارات سطح المكتب.',
      completionSoundTitle: 'صوت الاكتمال',
      completionSoundDesc: 'يُشغّل عند اكتمال دورة للوكيل. اختر صوتًا معدًا مسبقًا وعاينه هنا.',
      completionSoundPreview: 'معاينة'
    },
    sections: {
      model: 'النموذج',
      chat: 'المحادثة',
      appearance: 'المظهر',
      workspace: 'مساحة العمل',
      safety: 'الأمان',
      memory: 'الذاكرة والسياق',
      voice: 'الصوت',
      advanced: 'متقدم'
    },
    searchPlaceholder: {
      about: 'حول هرمس لسطح المكتب',
      config: 'البحث في الإعدادات...',
      gateway: 'اتصال البوابة...',
      keys: 'البحث في مفاتيح الواجهة البرمجية...',
      mcp: 'البحث في خوادم بروتوكول سياق النموذج...',
      sessions: 'البحث في الجلسات المؤرشفة...'
    },
    modeOptions: {
      light: { label: 'فاتح', description: 'أسطح مكتبية ساطعة' },
      dark: { label: 'داكن', description: 'مساحة عمل منخفضة الوهج' },
      system: { label: 'النظام', description: 'اتباع مظهر النظام' }
    },
    appearance: {
      title: 'المظهر',
      uiScaleTitle: 'مقياس الواجهة',
      uiScaleDesc: (percent: number) =>
        `يضبط حجم النصّ وعناصر التحكّم في التطبيق كلّه. يعمل أيضًا Cmd/Ctrl مع + و- و0. الحالي: ${percent}%.`,
      translucencyTitle: 'شفافية النافذة',
      translucencyDesc: 'شاهد سطح مكتبك عبر النافذة كاملة. لنظامي ماك وويندوز فقط.',
      intro:
        'هذه تفضيلات عرض خاصة بسطح المكتب. يتحكم الوضع في السطوع، وتتحكم السمة في ألوان التمييز وأسلوب سطح المحادثة.',
      colorMode: 'وضع الألوان',
      colorModeDesc: 'اختر وضعًا ثابتًا أو دع هرمس يتبع إعداد النظام.',
      toolViewTitle: 'عرض استدعاءات الأدوات',
      toolViewDesc: 'يخفي العرض المبسّط بيانات الأدوات الخام، ويعرض الوضع التقني المدخلات والمخرجات كاملة.',
      embedsTitle: 'التضمينات السطرية',
      embedsDesc:
        'تُحمّل المعاينات الغنية من مواقع طرف ثالث (يوتيوب، إكس، …). «اسأل» يعرض عنصرًا نائبًا حتى تسمح بكلٍّ منها؛ «دائمًا» يحمّلها تلقائيًّا؛ «إيقاف» يبقي الروابط نصًّا.',
      embedsAsk: 'اسأل',
      embedsAlways: 'دائمًا',
      embedsOff: 'إيقاف',
      embedsReset: (count: number) =>
        count === 1
          ? 'إعادة ضبط خدمة مسموحة واحدة'
          : count === 2
            ? 'إعادة ضبط خدمتين مسموحتين'
            : count >= 3 && count <= 10
              ? `إعادة ضبط ${count} خدمات مسموحة`
              : `إعادة ضبط ${count} خدمة مسموحة`,
      product: 'مبسّط',
      productDesc: 'نشاط أدوات واضح للبشر مع ملخصات موجزة.',
      technical: 'تقني',
      technicalDesc: 'يتضمن معاملات الأدوات ونتائجها الخام والتفاصيل منخفضة المستوى.',
      themeTitle: 'السمة',
      themeDesc: 'لوحات ألوان سطح المكتب فقط، ويُطبّق الوضع المحدد فوقها.',
      themeProfileNote: profile => `حُفظت للملف الشخصي «${profile}»؛ يحتفظ كل ملف بسمته الخاصة.`,
      installTitle: 'التثبيت من VS Code',
      installDesc:
        'الصق معرّف إضافة من المتجر، مثل dracula-theme.theme-dracula، لتحويل سمة ألوانها إلى لوحة لسطح المكتب.',
      installPlaceholder: 'publisher.extension',
      installButton: 'تثبيت',
      installing: 'جارٍ التثبيت…',
      installError: 'تعذر تثبيت هذه السمة.',
      installed: name => `ثُبّتت «${name}».`,
      removeTheme: 'إزالة السمة',
      importedBadge: 'مستوردة',
      themeNames: {
        nous: 'نوس',
        midnight: 'منتصف الليل',
        ember: 'جمرة',
        mono: 'أحادي',
        cyberpunk: 'سايبربانك',
        slate: 'إردواز'
      },
      themeDescriptions: {
        nous: 'حياديات زجاجية بلمسات نوس الزرقاء',
        midnight: 'أزرق بنفسجي عميق بلمسات باردة',
        ember: 'قرمزي دافئ وبرونزي — أجواء المِصهر',
        mono: 'تدرّج رمادي نقي — بساطة وتركيز',
        cyberpunk: 'أخضر نيون على أسود — طرفية ماتريكس',
        slate: 'أزرق إردوازي هادئ — سمة مطوّر مركّزة'
      },
      pet: {
        title: 'حيوان أليف',
        intro:
          'تبنَّ تميمةً متحركةً من معرض الحيوانات تطفو فوق التطبيق وتتفاعل مع ما يفعله هرمس — تجري أثناء تنفيذ الأدوات، وتحتفل عند النجاح، وتتجهّم عند الأخطاء.',
        restartHint:
          'تحتاج الحيوانات إلى إعادة تشغيل سريعة — التطبيق العامل بدأ قبل إضافة هذه الميزة. أغلق هرمس وأعد فتحه، ثم عُد إلى هنا.',
        on: 'تشغيل',
        off: 'إيقاف',
        scaleTitle: 'الحجم',
        scaleDesc: 'غيّر حجم التميمة الطافية. يُطبَّق في كل مكان فورًا.',
        roamTitle: 'تجوال',
        roamDesc: 'دع الحيوان يتجوّل في النافذة وحده أثناء الخمول.',
        chooseTitle: 'اختر حيوانًا',
        chooseDesc: 'اختيار واحدٍ يثبّته (إن لزم) ويجعله نشطًا.',
        searchPlaceholder: 'ابحث عن حيوان…',
        unreachable: 'تعذّر الوصول إلى معرض الحيوانات. تحقّق من اتصالك وأعد فتح هذه الصفحة.',
        noMatch: query => `لا حيوان يطابق "${query}".`,
        installedTag: 'مثبّت',
        generatedTag: 'مُولَّد',
        countCapped: (cap, total) => `عرض ${cap} من ${total} — اكتب لتضييق النتائج.`,
        count: n => {
          if (n === 1) return 'حيوان واحد.'
          if (n === 2) return 'حيوانان.'
          if (n >= 3 && n <= 10) return `${n} حيوانات.`

          return `${n} حيوانًا.`
        },
        uninstall: name => `إلغاء تثبيت ${name}`,
        delete: name => `حذف ${name}`,
        deleteTitle: name => `حذف ${name}؟`,
        deleteBody: 'يحذف هذا الحيوان نهائيًّا — لا يمكن إعادة تثبيته.',
        deleteConfirm: 'حذف',
        rename: name => `إعادة تسمية ${name}`,
        renameTitle: 'إعادة تسمية الحيوان',
        renamePlaceholder: 'سمِّ حيوانك',
        renameSave: 'حفظ',
        exportPet: name => `تصدير ${name}`,
        adoptFailed: slug => `تعذّر تبنّي ${slug}`,
        uninstallFailed: slug => `تعذّر إلغاء تثبيت ${slug}`,
        renameFailed: slug => `تعذّر إعادة تسمية ${slug}`,
        exportFailed: slug => `تعذّر تصدير ${slug}`,
        noneAvailable: 'لا حيوانات متاحة للتشغيل الآن.',
        turnOnFailed: 'تعذّر تشغيل الحيوان.',
        turnOffFailed: 'تعذّر إيقاف الحيوان.'
      }
    },
    fieldLabels: defineFieldCopy({
      model: 'النموذج الافتراضي',
      modelContextLength: 'نافذة السياق',
      fallbackProviders: 'النماذج الاحتياطية',
      toolsets: 'مجموعات الأدوات المفعّلة',
      timezone: 'المنطقة الزمنية',
      display: {
        personality: 'الشخصية',
        showReasoning: 'إظهار الاستدلال',
        skin: 'سمة سطر الأوامر',
        resumeDisplay: 'عرض سجل الجلسات المستأنفة',
        busyInputMode: 'سلوك الإدخال أثناء عمل الوكيل'
      },
      agent: {
        maxTurns: 'الحد الأقصى لخطوات الوكيل',
        imageInputMode: 'مرفقات الصور',
        apiMaxRetries: 'محاولات الواجهة البرمجية',
        serviceTier: 'فئة الخدمة',
        toolUseEnforcement: 'فرض استخدام الأدوات'
      },
      terminal: {
        cwd: 'مجلد العمل',
        backend: 'واجهة التنفيذ الخلفية',
        modalMode: 'وضع صندوق Modal الرملي',
        timeout: 'مهلة الأمر',
        persistentShell: 'صدفة مستمرة',
        envPassthrough: 'تمرير متغيرات البيئة',
        dockerImage: 'صورة Docker',
        singularityImage: 'صورة Singularity',
        modalImage: 'صورة Modal',
        daytonaImage: 'صورة Daytona'
      },
      fileReadMaxChars: 'حد قراءة الملفات',
      toolOutput: {
        maxBytes: 'حد مخرجات الطرفية',
        maxLines: 'حد صفحات الملفات',
        maxLineLength: 'حد طول السطر'
      },
      codeExecution: { mode: 'وضع تنفيذ الشيفرة' },
      approvals: {
        mode: 'وضع الموافقات',
        timeout: 'مهلة الموافقة',
        mcpReloadConfirm: 'تأكيد إعادة تحميل بروتوكول سياق النموذج'
      },
      commandAllowlist: 'قائمة الأوامر المسموح بها',
      security: { redactSecrets: 'حجب الأسرار', allowPrivateUrls: 'السماح بالروابط الخاصة' },
      browser: {
        allowPrivateUrls: 'روابط المتصفح الخاصة',
        autoLocalForPrivateUrls: 'استخدام المتصفح المحلي للروابط الخاصة'
      },
      checkpoints: { enabled: 'نقاط تحقق الملفات', maxSnapshots: 'حد نقاط التحقق' },
      voice: {
        recordKey: 'اختصار الصوت',
        maxRecordingSeconds: 'أقصى مدة للتسجيل',
        autoTts: 'قراءة الردود'
      },
      stt: {
        enabled: 'تحويل الكلام إلى نص',
        provider: 'مزوّد تحويل الكلام إلى نص',
        local: { model: 'نموذج التفريغ المحلي', language: 'لغة التفريغ' },
        openai: { model: 'نموذج OpenAI للتفريغ' },
        groq: { model: 'نموذج Groq للتفريغ' },
        mistral: { model: 'نموذج Mistral للتفريغ' },
        elevenlabs: {
          modelId: 'نموذج ElevenLabs للتفريغ',
          languageCode: 'لغة ElevenLabs',
          tagAudioEvents: 'وسم الأحداث الصوتية',
          diarize: 'تمييز المتحدثين'
        }
      },
      tts: {
        provider: 'مزوّد تحويل النص إلى كلام',
        edge: { voice: 'صوت Edge' },
        openai: { model: 'نموذج OpenAI الصوتي', voice: 'صوت OpenAI' },
        elevenlabs: { voiceId: 'صوت ElevenLabs', modelId: 'نموذج ElevenLabs' },
        xai: { voiceId: 'صوت xAI (Grok)', language: 'لغة xAI' },
        minimax: { model: 'نموذج MiniMax الصوتي', voiceId: 'صوت MiniMax' },
        mistral: { model: 'نموذج Mistral الصوتي', voiceId: 'صوت Mistral' },
        gemini: { model: 'نموذج Gemini الصوتي', voice: 'صوت Gemini' },
        neutts: { model: 'نموذج NeuTTS', device: 'جهاز NeuTTS' },
        kittentts: { model: 'نموذج KittenTTS', voice: 'صوت KittenTTS' },
        piper: { voice: 'صوت Piper' }
      },
      memory: {
        memoryEnabled: 'الذاكرة المستمرة',
        userProfileEnabled: 'ملف المستخدم',
        memoryCharLimit: 'ميزانية الذاكرة',
        userCharLimit: 'ميزانية الملف',
        provider: 'مزوّد الذاكرة'
      },
      context: { engine: 'محرك السياق' },
      compression: {
        enabled: 'الضغط التلقائي',
        threshold: 'عتبة الضغط',
        targetRatio: 'نسبة الضغط المستهدفة',
        protectLastN: 'حماية الرسائل الأخيرة'
      },
      delegation: {
        model: 'نموذج الوكيل الفرعي',
        provider: 'مزوّد الوكيل الفرعي',
        maxIterations: 'حد دورات الوكيل الفرعي',
        maxConcurrentChildren: 'الوكلاء الفرعيون المتزامنون',
        childTimeoutSeconds: 'مهلة الوكيل الفرعي',
        reasoningEffort: 'جهد استدلال الوكيل الفرعي'
      },
      updates: { nonInteractiveLocalChanges: 'التغييرات المحلية أثناء التحديث داخل التطبيق' },
      // Gateway-schema-only fields (no local English copy): without these the
      // labels fall back to prettified keys and stay English.
      dashboard: { theme: 'سمة لوحة التحكم' },
      humanDelay: { mode: 'محاكاة مهلة الكتابة' },
      logging: { level: 'مستوى السجل' }
    }),
    enumValueLabels: {
      // Generic enum values surfaced by config selects. Brand, model, and
      // protocol identifiers are intentionally omitted (they fall back to the
      // raw value).
      auto: 'تلقائي',
      native: 'أصلي',
      text: 'نصي',
      manual: 'يدوي',
      smart: 'ذكي',
      off: 'إيقاف',
      on: 'تشغيل',
      all: 'الكل',
      new: 'الجديد فقط',
      first: 'الأولى فقط',
      project: 'المشروع',
      strict: 'صارم',
      compressor: 'الضاغط',
      default: 'افتراضي',
      custom: 'مخصص',
      minimal: 'أدنى',
      low: 'منخفض',
      medium: 'متوسط',
      high: 'عالٍ',
      xhigh: 'أقصى',
      builtin: 'مدمج',
      honcho: 'هونتشو',
      local: 'محلي',
      stash: 'حفظ جانبي',
      discard: 'تجاهل',
      tiny: 'صغير جدًا',
      base: 'أساسي',
      small: 'صغير',
      large: 'كبير',
      // Built-in personalities (display.personality).
      helpful: 'مفيد',
      concise: 'موجز',
      technical: 'تقني',
      creative: 'مبدع',
      teacher: 'معلّم',
      kawaii: 'كاواي',
      catgirl: 'قطة',
      pirate: 'قرصان',
      shakespeare: 'شكسبير',
      surfer: 'راكب أمواج',
      noir: 'نوار',
      philosopher: 'فيلسوف',
      hype: 'حماسي'
    },
    fieldDescriptions: defineFieldCopy({
      model: 'يُستخدم للمحادثات الجديدة ما لم تختر نموذجًا آخر من محرر الرسالة.',
      modelContextLength: 'اتركه صفرًا لاستخدام نافذة السياق المكتشفة للنموذج المحدد.',
      fallbackProviders: 'إدخالات احتياطية بصيغة provider:model تُجرّب عند فشل النموذج الافتراضي.',
      display: {
        personality: 'أسلوب المساعد الافتراضي للجلسات الجديدة.',
        showReasoning: 'يعرض محتوى الاستدلال عندما توفره الواجهة الخلفية.',
        skin: 'سمة سطر الأوامر المرئية.',
        resumeDisplay: 'كيفية عرض السجل عند استئناف جلسة.',
        busyInputMode: 'سلوك حقل الإدخال أثناء انشغال الوكيل.'
      },
      timezone: 'تُستخدم عندما يحتاج هرمس إلى سياق الوقت المحلي. اتركها فارغة لاستخدام منطقة النظام.',
      agent: {
        imageInputMode: 'يتحكم في كيفية إرسال مرفقات الصور إلى النموذج.',
        maxTurns: 'أقصى عدد لدورات استدعاء الأدوات قبل أن يوقف هرمس التشغيل.',
        serviceTier: 'فئة خدمة الواجهة البرمجية (OpenAI/Anthropic).'
      },
      terminal: {
        cwd: 'مجلد المشروع الافتراضي لعمليات الأدوات والطرفية.',
        backend: 'بيئة تنفيذ أوامر الطرفية.',
        modalMode: 'وضع صندوق Modal الرملي.',
        persistentShell: 'يحافظ على حالة الصدفة بين الأوامر عندما تدعمها الواجهة الخلفية.',
        envPassthrough: 'متغيرات البيئة التي تُمرر إلى تنفيذ الأدوات.',
        dockerImage: 'صورة الحاوية المستخدمة عندما تكون واجهة التنفيذ الخلفية Docker.',
        singularityImage: 'الصورة المستخدمة عندما تكون واجهة التنفيذ الخلفية Singularity.',
        modalImage: 'الصورة المستخدمة عندما تكون واجهة التنفيذ الخلفية Modal.',
        daytonaImage: 'الصورة المستخدمة عندما تكون واجهة التنفيذ الخلفية Daytona.'
      },
      codeExecution: { mode: 'مدى تقييد تنفيذ الشيفرة بالمشروع الحالي.' },
      fileReadMaxChars: 'أقصى عدد من المحارف التي يقرأها هرمس في عملية قراءة ملف واحدة.',
      approvals: {
        mode: 'كيفية تعامل هرمس مع الأوامر التي تتطلب موافقة صريحة.',
        timeout: 'مدة انتظار مطالبة الموافقة قبل انتهاء مهلتها.'
      },
      security: { redactSecrets: 'يحجب الأسرار المكتشفة من المحتوى المرئي للنموذج قدر الإمكان.' },
      checkpoints: { enabled: 'ينشئ لقطات قابلة للاستعادة قبل تعديل الملفات.' },
      memory: {
        memoryEnabled: 'يحفظ ذكريات مستمرة تفيد الجلسات اللاحقة.',
        userProfileEnabled: 'يحافظ على ملف موجز لتفضيلات المستخدم.',
        provider: 'إضافة مزوّد الذاكرة.'
      },
      context: { engine: 'استراتيجية إدارة المحادثات الطويلة عند الاقتراب من حد السياق.' },
      compression: { enabled: 'يلخّص السياق الأقدم عندما تكبر المحادثة.' },
      voice: { autoTts: 'يقرأ ردود المساعد تلقائيًا.' },
      tts: {
        xai: {
          voiceId: 'معرّف صوت xAI (مثل eve) أو معرّف صوت مخصص.',
          language: 'رمز لغة النطق، مثل en.'
        },
        neutts: { device: 'جهاز الاستنتاج المحلي لـ NeuTTS.' }
      },
      stt: {
        enabled: 'يفعّل تفريغ الكلام محليًا أو عبر مزوّد.',
        elevenlabs: {
          modelId: 'نموذج Scribe من ElevenLabs للتفريغ.',
          languageCode: 'رمز لغة ISO-639-3 اختياري. اتركه فارغًا ليكتشف ElevenLabs اللغة تلقائيًا.'
        }
      },
      updates: {
        nonInteractiveLocalChanges:
          'عند تحديث هرمس من داخل التطبيق بلا مطالبة طرفية، احتفظ بتعديلات المصدر المحلية مؤقتًا أو تجاهلها. تسأل تحديثات الطرفية دائمًا.'
      },
      dashboard: { theme: 'سمة لوحة تحكم الويب.' },
      humanDelay: { mode: 'وضع محاكاة مهلة الكتابة البشرية.' },
      logging: { level: 'مستوى التفصيل في agent.log.' },
      delegation: { reasoningEffort: 'جهد الاستدلال للوكلاء الفرعيين المفوّضين.' }
    }),
    about: {
      heading: 'هرمس لسطح المكتب',
      version: value => `الإصدار ${value}`,
      versionUnavailable: 'الإصدار غير متاح',
      updates: 'التحديثات',
      checkNow: 'التحقق الآن',
      checking: 'جارٍ التحقق…',
      seeWhatsNew: 'عرض الجديد',
      updateNow: 'حدّث الآن',
      releaseNotes: 'ملاحظات الإصدار',
      onLatest: 'لديك أحدث إصدار.',
      installing: 'يجري تثبيت تحديث حاليًا.',
      cantUpdate: 'لا يستطيع هذا الإصدار تحديث نفسه من داخل التطبيق.',
      cantReach: 'تعذر الوصول إلى خادم التحديث.',
      tapCheck: 'اضغط «التحقق الآن» للبحث عن تحديثات.',
      updateReady: count =>
        count === 1
          ? 'تحديث جديد جاهز، ويتضمن تغييرًا واحدًا.'
          : count === 2
            ? 'تحديث جديد جاهز، ويتضمن تغييرين.'
            : count <= 10
              ? `تحديث جديد جاهز، ويتضمن ${count} تغييرات.`
              : `تحديث جديد جاهز، ويتضمن ${count} تغييرًا.`,
      lastChecked: age => `آخر تحقق ${age}`,
      justNowSuffix: ' · الآن',
      automaticUpdates: 'التحديثات التلقائية',
      automaticUpdatesDesc: 'يتحقق هرمس من التحديثات تلقائيًا في الخلفية وينبهك عند جاهزيتها.',
      branchCommit: (branch, commit) => `الفرع ${branch} · الإيداع ${commit}`,
      never: 'أبدًا',
      justNow: 'الآن',
      minAgo: count =>
        count === 1 ? 'قبل دقيقة' : count === 2 ? 'قبل دقيقتين' : count <= 10 ? `قبل ${count} دقائق` : `قبل ${count} دقيقة`,
      hoursAgo: count =>
        count === 1 ? 'قبل ساعة' : count === 2 ? 'قبل ساعتين' : count <= 10 ? `قبل ${count} ساعات` : `قبل ${count} ساعة`,
      daysAgo: count =>
        count === 1 ? 'قبل يوم' : count === 2 ? 'قبل يومين' : count <= 10 ? `قبل ${count} أيام` : `قبل ${count} يومًا`
    },
    uninstall: {
      dangerZone: 'منطقة الخطر',
      checking: 'جارٍ فحص ما هو مثبّت…',
      confirmTitle: 'تأكيد إلغاء التثبيت',
      confirmBody: consequence => `سيزيل هذا ${consequence}. لا يمكن التراجع عن هذه الخطوة.`,
      appPathLabel: 'التطبيق:',
      uninstalling: 'جارٍ إلغاء التثبيت…',
      confirmYes: 'نعم، ألغِ التثبيت',
      cancel: 'إلغاء',
      sectionTitle: 'إلغاء تثبيت هرمس',
      sectionBody: 'اختر مقدار ما يُزال. سيُغلق التطبيق لإتمام العملية، ويمكنك إعادة فتح المثبّت في أي وقت للعودة.',
      couldNotStart: 'تعذر بدء إلغاء التثبيت.',
      modes: {
        gui: {
          title: 'إلغاء تثبيت واجهة المحادثة فقط',
          description: 'يزيل تطبيق سطح المكتب هذا. يبقى وكيل هرمس وإعداداتك ومحادثاتك كلها.',
          consequence: 'واجهة المحادثة لسطح المكتب (هذا التطبيق وبياناته)'
        },
        lite: {
          title: 'إلغاء الواجهة والوكيل مع الاحتفاظ ببياناتي',
          description: 'يزيل التطبيق ووكيل هرمس، مع الاحتفاظ بالإعدادات والمحادثات والأسرار لإعادة تثبيت مستقبلية.',
          consequence: 'واجهة المحادثة ووكيل هرمس (تُحفظ الإعدادات والمحادثات والأسرار)'
        },
        full: {
          title: 'إلغاء تثبيت كل شيء',
          description:
            'يزيل التطبيق والوكيل وكل بيانات المستخدم — الإعدادات والمحادثات والمهام المجدولة والأسرار والسجلات.',
          consequence: 'كل شيء — واجهة المحادثة ووكيل هرمس وكل إعداداتك ومحادثاتك وأسرارك وسجلاتك'
        }
      }
    },
    config: {
      none: 'لا شيء',
      noneParen: '(لا شيء)',
      notSet: 'غير مضبوط',
      commaSeparated: 'قيم مفصولة بفواصل',
      loading: 'جارٍ تحميل إعدادات هرمس...',
      emptyTitle: 'لا إعدادات قابلة للتعديل',
      emptyDesc: 'لا يحتوي هذا القسم على إعدادات قابلة للتغيير.',
      failedLoad: 'فشل تحميل الإعدادات',
      autosaveFailed: 'فشل الحفظ التلقائي',
      imported: 'استُوردت الإعدادات',
      invalidJson: 'بيانات JSON غير صالحة'
    },
    credentials: {
      pasteKey: 'لصق المفتاح',
      pasteLabelKey: label => `لصق مفتاح ${label}`,
      optional: 'اختياري',
      enterValueFirst: 'أدخل قيمة أولًا.',
      couldNotSave: 'تعذر حفظ بيانات الاعتماد.',
      remove: 'إزالة',
      getKey: 'الحصول على مفتاح',
      saving: 'جارٍ الحفظ'
    },
    envActions: {
      actionsFor: label => `إجراءات ${label}`,
      credentialActions: 'إجراءات بيانات الاعتماد',
      docs: 'التوثيق',
      hideValue: 'إخفاء القيمة',
      revealValue: 'إظهار القيمة',
      replace: 'استبدال',
      set: 'ضبط',
      clear: 'مسح'
    },
    gateway: {
      loading: 'جارٍ تحميل إعدادات البوابة...',
      unavailableTitle: 'إعدادات البوابة غير متاحة',
      unavailableDesc: 'لا يتيح جسر اتصال سطح المكتب إعدادات البوابة.',
      title: 'اتصال البوابة',
      envOverride: 'تجاوز من البيئة',
      intro:
        'يبدأ هرمس لسطح المكتب بوابته المحلية تلقائيًا. استخدم بوابة بعيدة للتحكم في واجهة هرمس خلفية تعمل على جهاز آخر أو خلف وكيل موثوق. اختر ملفًا شخصيًا لمنحه مضيفًا بعيدًا مستقلًا.',
      appliesTo: 'ينطبق على',
      allProfiles: 'جميع الملفات الشخصية',
      defaultConnection: 'الاتصال الافتراضي لكل ملف شخصي لا يملك تجاوزًا مستقلًا.',
      profileConnection: profile =>
        `اتصال يُستخدم فقط عندما يكون «${profile}» الملف النشط. اختر «محلي» ليرث الإعداد الافتراضي.`,
      envOverrideTitle: 'تتحكم متغيرات البيئة في جلسة سطح المكتب هذه.',
      envOverrideDesc: 'أزل HERMES_DESKTOP_REMOTE_URL وHERMES_DESKTOP_REMOTE_TOKEN لاستخدام الإعداد المحفوظ أدناه.',
      modeTitle: 'وضع الاتصال',
      localTitle: 'بوابة محلية',
      localDesc: 'تشغّل واجهة هرمس خلفية خاصة على الجهاز المحلي. هذا هو الافتراضي ويعمل دون اتصال.',
      remoteTitle: 'بوابة بعيدة',
      remoteDesc: 'تصل غلاف سطح المكتب بواجهة هرمس خلفية بعيدة.',
      remoteAuthHint:
        'تستخدم البوابات المستضافة OAuth أو اسم مستخدم وكلمة مرور، وقد تستخدم البوابات الذاتية رمز جلسة.',
      cloudTitle: 'سحابة هرمس',
      cloudDesc: 'سجّل الدخول مرة واحدة إلى سحابة هرمس واختر من الوكلاء في حسابك — دون لصق أي رابط.',
      cloudSignInTitle: 'سحابة هرمس',
      cloudSignIn: 'تسجيل الدخول إلى سحابة هرمس',
      cloudSignedIn: 'مُسجَّل الدخول إلى سحابة هرمس',
      cloudNeedsSignIn: 'سجّل الدخول إلى سحابة هرمس لاكتشاف الوكلاء في حسابك.',
      cloudSignedInDesc: 'أنت مُسجَّل الدخول. اختر وكيلًا أدناه؛ تتجدد الجلسة تلقائيًا.',
      cloudAgentsTitle: 'وكلاؤك',
      cloudOrgPickerTitle: 'اختر مؤسسة',
      cloudOrgSelect: 'اختيار',
      cloudOrgChange: 'تغيير المؤسسة',
      cloudOrgRole: role => `الدور: ${role}`,
      cloudLoadingAgents: 'جارٍ تحميل وكلائك…',
      cloudNoAgents: {
        before: 'لا وكلاء في هذا الحساب. أنشئ واحدًا في ',
        linkText: 'بوابة Nous',
        after: '، ثم حدّث.'
      },
      cloudRefresh: 'تحديث',
      cloudConnect: 'اتصال',
      cloudConnecting: 'جارٍ الاتصال…',
      cloudDiscoverFailed: 'تعذّر تحميل وكلاء سحابة هرمس',
      cloudConnectFailed: 'تعذّر الاتصال بذلك الوكيل',
      cloudSignInFailed: 'فشل تسجيل الدخول إلى سحابة هرمس',
      cloudSignedOutTitle: 'تم تسجيل الخروج من سحابة هرمس',
      cloudSignedOutMessage: 'مُسحت جلسة سحابة هرمس.',
      cloudConnectedTitle: 'متصل',
      cloudConnectedPill: 'متصل',
      cloudConnectedTo: name => `متصل بـ ${name}.`,
      cloudAgentProvisioning: 'جارٍ التهيئة…',
      cloudStatusLabel: status => `الحالة: ${status}`,
      remoteUrlTitle: 'الرابط البعيد',
      remoteUrlDesc: 'الرابط الأساسي للوحة التحكم البعيدة. تُدعم بادئات المسارات، مثل /hermes.',
      probing: 'جارٍ التحقق من طريقة مصادقة البوابة…',
      probeError: 'تعذر الوصول إلى هذه البوابة. تحقق من الرابط؛ ستظهر طريقة المصادقة بعد استجابتها.',
      signedIn: 'تم تسجيل الدخول',
      signIn: 'تسجيل الدخول',
      signOut: 'تسجيل الخروج',
      signInWith: provider => `تسجيل الدخول عبر ${provider}`,
      authTitle: 'المصادقة',
      authSignedInPassword: 'تستخدم هذه البوابة اسم مستخدم وكلمة مرور. أنت مسجل الدخول، وتُجدّد الجلسة تلقائيًا.',
      authSignedInOauth: 'تستخدم هذه البوابة OAuth. أنت مسجل الدخول، وتُجدّد الجلسة تلقائيًا.',
      authNeedsPassword: 'تستخدم هذه البوابة اسم مستخدم وكلمة مرور. سجّل الدخول لتفويض تطبيق سطح المكتب.',
      authNeedsOauth: provider => `تستخدم هذه البوابة OAuth. سجّل الدخول عبر ${provider} لتفويض تطبيق سطح المكتب.`,
      tokenTitle: 'رمز الجلسة',
      tokenDesc: 'رمز جلسة لوحة التحكم المستخدم للوصول عبر REST وWebSocket. اتركه فارغًا للإبقاء على الرمز المحفوظ.',
      existingToken: value => `الرمز الحالي ${value}`,
      savedToken: 'محفوظ',
      pasteSessionToken: 'لصق رمز الجلسة',
      testRemote: 'اختبار الاتصال البعيد',
      saveForRestart: 'الحفظ لإعادة التشغيل التالية',
      saveAndReconnect: 'الحفظ وإعادة الاتصال',
      diagnostics: 'التشخيص',
      diagnosticsDesc: 'يكشف desktop.log في مدير الملفات، وهو مفيد عند تعذر بدء البوابة.',
      openLogs: 'فتح السجلات',
      incompleteTitle: 'إعداد البوابة البعيدة غير مكتمل',
      incompleteSignIn: 'أدخل رابطًا بعيدًا وسجّل الدخول قبل التبديل إلى الوضع البعيد.',
      incompleteToken: 'أدخل رابطًا بعيدًا ورمز جلسة قبل التبديل إلى الوضع البعيد.',
      incompleteSignInTest: 'أدخل رابطًا بعيدًا وسجّل الدخول قبل الاختبار.',
      incompleteTokenTest: 'أدخل رابطًا بعيدًا ورمز جلسة قبل الاختبار.',
      enterUrlFirst: 'أدخل رابطًا بعيدًا أولًا.',
      restartingTitle: 'جارٍ إعادة تشغيل اتصال البوابة',
      savedTitle: 'حُفظت إعدادات البوابة',
      restartingMessage: 'سيعيد هرمس لسطح المكتب الاتصال بالإعدادات المحفوظة.',
      savedMessage: 'حُفظت لإعادة التشغيل التالية.',
      connectedTo: (baseUrl, version) => `متصل بـ ${baseUrl}${version ? ` · هرمس ${version}` : ''}`,
      reachableTitle: 'يمكن الوصول إلى البوابة البعيدة',
      signedOutTitle: 'تم تسجيل الخروج',
      signedOutMessage: 'مُسحت جلسة البوابة البعيدة.',
      failedLoad: 'فشل تحميل إعدادات البوابة',
      signInFailed: 'فشل تسجيل الدخول',
      signOutFailed: 'فشل تسجيل الخروج',
      testFailed: 'فشل اختبار البوابة البعيدة',
      applyFailed: 'تعذر تطبيق إعدادات البوابة',
      saveFailed: 'تعذر حفظ إعدادات البوابة'
    },
    keys: {
      loading: 'جارٍ تحميل مفاتيح الواجهة البرمجية وبيانات الاعتماد...',
      failedLoad: 'فشل تحميل مفاتيح الواجهة البرمجية',
      empty: 'لا يوجد شيء مضبوط في هذه الفئة بعد.'
    },
    mcp: {
      loading: 'جارٍ تحميل خوادم بروتوكول سياق النموذج...',
      failedLoad: 'فشل تحميل إعداد بروتوكول سياق النموذج',
      nameRequiredTitle: 'الاسم مطلوب',
      nameRequiredMessage: 'أعط هذا الخادم مفتاح إعداد.',
      objectRequired: 'يجب أن يكون إعداد الخادم كائن JSON',
      invalidJson: 'بيانات JSON للخادم غير صالحة',
      saveFailed: 'فشل الحفظ',
      removeFailed: 'فشلت الإزالة',
      gatewayUnavailableTitle: 'البوابة غير متاحة',
      gatewayUnavailableMessage: 'أعد اتصال البوابة قبل إعادة تحميل الأدوات.',
      reloadedTitle: 'أُعيد تحميل أدوات بروتوكول سياق النموذج',
      reloadedMessage: 'تُطبق مخططات الأدوات الجديدة على الدورات الجديدة.',
      reloadFailed: 'فشلت إعادة تحميل بروتوكول سياق النموذج',
      savedTitle: 'حُفظ الخادم',
      savedMessage: name => `يُطبق ${name} بعد إعادة تحميل بروتوكول سياق النموذج.`,
      newServer: 'خادم جديد',
      reload: 'إعادة تحميل الأدوات',
      reloading: 'جارٍ إعادة التحميل...',
      emptyTitle: 'لا توجد خوادم',
      emptyDesc: 'أضف خادم stdio أو HTTP لإتاحة الأدوات.',
      disabled: 'معطّل',
      editServer: 'تعديل الخادم',
      name: 'الاسم',
      serverJson: 'بيانات JSON للخادم',
      remove: 'إزالة',
      saveServer: 'حفظ الخادم',
      test: 'اختبار الاتصال',
      testing: 'جارٍ الاختبار...',
      testOk: count => `متصل — ${count} أداة متاحة`,
      testFailed: 'فشل الاتصال',
      enableServer: name => `تفعيل ${name}`,
      disableServer: name => `تعطيل ${name}`,
      serverEnabled: name => `فُعّل ${name} — يسري على الجلسات الجديدة.`,
      serverDisabled: name => `عُطّل ${name} — يسري على الجلسات الجديدة.`,
      toggleFailed: name => `فشل تبديل ${name}`,
      tabServers: 'الخوادم',
      tabCatalog: 'الكتالوج',
      catalogLoading: 'جارٍ تحميل كتالوج MCP...',
      catalogLoadFailed: 'فشل تحميل كتالوج MCP',
      catalogEmpty: 'لا توجد مدخلات في الكتالوج.',
      catalogInstalled: 'مثبت',
      catalogEnabled: 'مفعل',
      catalogNeedsInstall: 'يحتاج بناء',
      catalogInstall: 'تثبيت',
      catalogInstalling: 'جارٍ التثبيت...',
      catalogInstallStarted: name => `جارٍ تثبيت ${name}... يسري على الجلسات الجديدة عند الاكتمال.`,
      catalogInstallFailed: name => `فشل تثبيت ${name}`,
      catalogEnvPrompt: name => `يتطلب ${name} بيانات اعتماد`,
      catalogEnvRequired: 'املأ القيم المطلوبة قبل التثبيت.',
      capabilitySummary: (tools, prompts, resources) =>
        `${[`${tools} أداة`, ...(prompts ? [`${prompts} موجّه`] : []), ...(resources ? [`${resources} مورد`] : [])].join('، ')} مفعّلة`,
      statusConnecting: 'جارٍ الاتصال…',
      statusNeedsAuth: 'يحتاج مصادقة',
      statusError: 'خطأ',
      statusOff: 'إيقاف',
      allServers: 'جميع الخوادم',
      authenticatedTitle: 'تمت المصادقة',
      authenticatedMessage: (server, count) => `${server}: ${count} أداة`,
      waitingForBrowser: 'في انتظار المتصفح…',
      authenticate: 'مصادقة',
      unsavedConnect: 'غير محفوظ — احفظ mcp.json للاتصال.',
      enableTool: tool => `تفعيل ${tool}`,
      disableTool: tool => `تعطيل ${tool}`,
      noOutput: 'لا مخرجات بعد.'
    },
    model: {
      loading: 'جارٍ تحميل إعداد النموذج...',
      appliesDesc: 'ينطبق على الجلسات الجديدة. استخدم منتقي النموذج في محرر الرسالة لتبديل نموذج المحادثة النشطة.',
      provider: 'المزوّد',
      model: 'النموذج',
      applying: 'جارٍ التطبيق...',
      defaultsLabel: 'الافتراضيات',
      reasoning: 'الاستدلال',
      reasoningOff: 'إيقاف',
      defaultsFailed: 'فشل حفظ افتراضيات النموذج',
      auxiliaryTitle: 'النماذج المساعدة',
      resetAllToMain: 'إعادة الكل إلى النموذج الرئيسي',
      auxiliaryDesc: 'تعمل المهام المساعدة على النموذج الرئيسي افتراضيًا. عيّن نموذجًا مستقلًا لأي مهمة لتجاوز ذلك.',
      setToMain: 'استخدام الرئيسي',
      change: 'تغيير',
      autoUseMain: 'تلقائي · استخدام النموذج الرئيسي',
      providerDefault: '(افتراضي المزوّد)',
      fallbackAdd: 'إضافة احتياطيّ',
      fallbackEmpty: 'لا نماذج احتياطيّة — يُستعمل النموذج الافتراضيّ ما لم يفشل.',
      notInCatalog: 'ليس في قائمة نماذج هذا المزوّد — قد تلجأ الطلبات إلى نموذج احتياطيّ.',
      tasks: {
        vision: { label: 'الرؤية', hint: 'تحليل الصور' },
        web_extract: { label: 'استخراج الويب', hint: 'تلخيص الصفحات' },
        compression: { label: 'الضغط', hint: 'ضغط السياق' },
        skills_hub: { label: 'مركز المهارات', hint: 'البحث عن المهارات' },
        approval: { label: 'الموافقة', hint: 'الموافقة التلقائية الذكية' },
        mcp: { label: 'MCP', hint: 'توجيه أدوات MCP' },
        title_generation: { label: 'إنشاء العناوين', hint: 'عناوين الجلسات' },
        curator: { label: 'القيّم', hint: 'مراجعة استخدام المهارات' }
      }
    },
    providers: {
      connectAccount: 'ربط حساب',
      haveApiKey: 'لديك مفتاح واجهة برمجية بدلًا من ذلك؟',
      intro: 'سجّل الدخول باشتراك دون نسخ مفتاح. يدير هرمس تسجيل الدخول عبر المتصفح من داخل التطبيق.',
      connected: 'متصل',
      collapse: 'طي',
      connectAnother: 'ربط مزوّد آخر',
      otherProviders: 'مزوّدون آخرون',
      disconnect: 'قطع الاتصال',
      disconnectInTerminal: 'قطع الاتصال (يشغّل أمر الإزالة في الطرفية)',
      removeConfirm: provider => `إزالة ${provider}؟`,
      removeExternalGeneric: provider => `يُدار ${provider} عبر أداة سطر أوامره؛ أزله من هناك.`,
      removeKeyManaged: provider => `أُعد ${provider} من مفتاح واجهة برمجية. أزله من مفاتيح الواجهة البرمجية.`,
      removeTerminalConfirm: (provider, command) =>
        `قطع اتصال ${provider}؟ سيشغّل هذا الأمر "${command}" في الطرفية لمسح بيانات الاعتماد.`,
      removeTerminalRunning: provider => `جارٍ تشغيل قطع اتصال ${provider} في الطرفية…`,
      removedTitle: 'أُزيل الحساب',
      removedMessage: provider => `أُزيل ${provider}.`,
      failedRemove: provider => `تعذرت إزالة ${provider}`,
      noProviderKeys: 'لا تتوفر مفاتيح واجهة برمجية للمزوّدين.',
      searchKeys: 'ابحث عن مزوّد…',
      noKeysMatch: 'لا مزوّد يطابق بحثك.',
      loading: 'جارٍ تحميل المزوّدين...'
    },
    sessions: {
      loading: 'جارٍ تحميل الجلسات المؤرشفة…',
      archivedTitle: 'الجلسات المؤرشفة',
      archivedIntro:
        'تُخفى المحادثات المؤرشفة من الشريط الجانبي مع الاحتفاظ برسائلها. انقر مع Ctrl أو ⌘ على محادثة في الشريط لأرشفتها.',
      emptyArchivedTitle: 'لا جلسات مؤرشفة',
      emptyArchivedDesc: 'أرشف محادثة لإخفائها هنا.',
      unarchive: 'إلغاء الأرشفة',
      deletePermanently: 'حذف نهائي',
      messages: count =>
        count === 1 ? 'رسالة واحدة' : count === 2 ? 'رسالتان' : count <= 10 ? `${count} رسائل` : `${count} رسالة`,
      restored: 'استُعيدت',
      deleteConfirm: title => `حذف «${title}» نهائيًا؟ لا يمكن التراجع عن ذلك.`,
      defaultDirTitle: 'مجلد المشروع الافتراضي',
      defaultDirDesc: 'تبدأ الجلسات الجديدة في هذا المجلد ما لم تختر غيره. اتركه فارغًا لاستخدام مجلد المنزل.',
      defaultDirUpdated: 'حُدّث مجلد المشروع الافتراضي؛ ابدأ محادثة جديدة (Ctrl/⌘+N) لتطبيقه',
      defaultsTo: label => `الافتراضي هو ${label}.`,
      change: 'تغيير',
      choose: 'اختيار',
      clear: 'مسح',
      notSet: 'غير مضبوط',
      failedLoad: 'تعذر تحميل الجلسات المؤرشفة',
      unarchiveFailed: 'فشل إلغاء الأرشفة',
      deleteFailed: 'فشل الحذف',
      updateDirFailed: 'تعذر تحديث المجلد الافتراضي',
      clearDirFailed: 'تعذر مسح المجلد الافتراضي'
    },
    toolsets: {
      loadingConfig: 'جارٍ تحميل الإعداد',
      savedTitle: 'حُفظت بيانات الاعتماد',
      savedMessage: key => `حُدّث ${key}.`,
      removedTitle: 'أُزيلت بيانات الاعتماد',
      removedMessage: key => `أُزيل ${key}.`,
      failedSave: key => `فشل حفظ ${key}`,
      failedRemove: key => `فشلت إزالة ${key}`,
      failedReveal: key => `فشل إظهار ${key}`,
      removeConfirm: key => `إزالة ${key} من .env؟`,
      set: 'مضبوط',
      notSet: 'غير مضبوط',
      selectedTitle: 'حُدد المزوّد',
      selectedMessage: provider => `أصبح ${provider} المزوّد النشط.`,
      failedSelect: provider => `فشل تحديد ${provider}`,
      failedLoad: 'فشل تحميل إعداد الأدوات',
      noProviderOptions: 'لا تتطلب مجموعة الأدوات هذه خيارات مزوّد؛ فعّلها وستعمل مع إعدادك الحالي.',
      noProviders: 'لا يتوفر مزوّدون لمجموعة الأدوات هذه حاليًا.',
      ready: 'جاهز',
      nousIncluded: 'مضمّن مع اشتراك Nous؛ سجّل الدخول إلى بوابة Nous لتفعيله.',
      noApiKeyRequired: 'لا يلزم مفتاح واجهة برمجية.',
      postSetupHint: step =>
        `تتطلب هذه الواجهة تثبيتًا لمرة واحدة (${step}). تعمل على هذا الجهاز وقد تستغرق بضع دقائق.`,
      postSetupRun: 'تشغيل الإعداد',
      postSetupRunning: 'جارٍ التثبيت…',
      postSetupStarting: 'جارٍ البدء…',
      postSetupCompleteTitle: 'اكتمل الإعداد',
      postSetupCompleteMessage: step => `ثُبّت ${step}.`,
      postSetupErrorTitle: 'اكتمل الإعداد مع أخطاء',
      postSetupErrorMessage: step => `راجع سجل ${step}.`,
      postSetupFailed: step => `فشل تشغيل إعداد ${step}`,
      loadingModels: 'جارٍ تحميل كتالوج النماذج...',
      modelSectionTitle: 'النموذج',
      modelCount: count => `${count} نموذج`,
      modelInUse: 'قيد الاستخدام',
      modelDefault: 'افتراضي',
      modelInactiveHint: 'اختر هذه الواجهة أولًا لتغيير نموذجها.',
      modelSelectedTitle: 'اختير النموذج',
      modelSelectedMessage: model => `يسري ${model} على الجلسات الجديدة.`,
      failedSelectModel: model => `فشل اختيار ${model}`
    }
  },

  skills: {
    tabSkills: 'المهارات',
    tabToolsets: 'مجموعات الأدوات',
    tabMcp: 'MCP',
    tabHub: 'تصفح المركز',
    all: 'الكل',
    searchSkills: 'البحث في المهارات...',
    searchToolsets: 'البحث في مجموعات الأدوات...',
    refresh: 'تحديث المهارات',
    refreshing: 'جارٍ تحديث المهارات',
    loading: 'جارٍ تحميل القدرات...',
    noSkillsTitle: 'لم يُعثر على مهارات',
    noSkillsDesc: 'جرّب بحثًا أوسع أو فئة مختلفة.',
    noToolsetsTitle: 'لم يُعثر على مجموعات أدوات',
    noToolsetsDesc: 'جرّب عبارة بحث أوسع.',
    noDescription: 'لا يوجد وصف.',
    configured: 'مضبوطة',
    needsKeys: 'تحتاج مفاتيح',
    toolsetsEnabled: (enabled, total) => `${enabled}/${total} من مجموعات الأدوات مفعّلة`,
    configureToolset: label => `ضبط ${label}`,
    toggleToolset: label => `تبديل مجموعة أدوات ${label}`,
    skillsLoadFailed: 'فشل تحميل المهارات',
    toolsetsRefreshFailed: 'فشل تحديث مجموعات الأدوات',
    skillEnabled: 'فُعّلت المهارة',
    skillDisabled: 'عُطّلت المهارة',
    toolsetEnabled: 'فُعّلت مجموعة الأدوات',
    toolsetDisabled: 'عُطّلت مجموعة الأدوات',
    appliesToNewSessions: name => `يُطبق ${name} على الجلسات الجديدة.`,
    failedToUpdate: name => `فشل تحديث ${name}`,
    sortMostUsed: 'الأكثر استخدامًا',
    sortAlpha: 'أ–ي',
    sortMostUsedDesc: '↓ الأكثر استخدامًا',
    sortLeastUsedAsc: '↑ الأقل استخدامًا',
    enableAll: 'تفعيل الكل',
    disableAll: 'تعطيل الكل',
    disableUnused: 'تعطيل غير المستخدمة',
    bulkUpdated: count =>
      count === 1
        ? 'حُدّث عنصر واحد للجلسات الجديدة.'
        : count === 2
          ? 'حُدّث عنصران للجلسات الجديدة.'
          : count <= 10
            ? `حُدّثت ${count} عناصر للجلسات الجديدة.`
            : `حُدّث ${count} عنصرًا للجلسات الجديدة.`,
    bulkNoChange: 'لا شيء لتغييره.',
    usageCount: count => `استُخدمت ${count}×`,
    provenance: {
      agent: 'مُتعلَّمة',
      bundled: 'مدمجة',
      hub: 'المركز'
    },
    emptyNoneFound: noun => `لم يُعثر على ${noun}`,
    emptyNothingMatches: query => `لا شيء يطابق «${query}».`,
    emptyNoneAvailable: noun => `لا ${noun} متاحة بعد.`,
    changesApplyNewSessions: 'تُطبق التغييرات على الجلسات الجديدة.',
    skillUpdated: 'حُدّثت المهارة',
    edit: 'تحرير',
    archive: 'أرشفة',
    skillArchivedTitle: 'أُرشفت المهارة',
    skillArchivedMessage: 'قابلة للاستعادة عبر hermes curator restore.',
    hub: {
      searchPlaceholder: 'ابحث في مركز المهارات (الرسمي، GitHub، المجتمع)...',
      search: 'بحث',
      searching: 'جارٍ البحث...',
      connectingHubs: 'جارٍ الاتصال بمراكز المهارات...',
      connectedHubs: 'المراكز المتصلة:',
      featured: 'مهارات مميزة',
      landingHint: 'ابحث في المركز لتصفح المهارات القابلة للتثبيت من الفهرس الرسمي وGitHub ومصادر المجتمع.',
      noResults: 'لا توجد مهارات مطابقة في المركز.',
      resultCount: (count, ms) => `${count} نتيجة${ms !== null ? ` خلال ${ms} م.ث` : ''}`,
      timedOut: sources => `انتهت المهلة: ${sources}`,
      installed: 'مثبتة',
      install: 'تثبيت',
      installing: 'جارٍ التثبيت...',
      uninstall: 'إزالة',
      uninstalling: 'جارٍ الإزالة…',
      updateAll: 'تحديث المثبتة',
      updating: 'جارٍ التحديث...',
      preview: 'معاينة',
      scan: 'فحص',
      scanning: 'جارٍ الفحص...',
      close: 'إغلاق',
      files: 'الملفات',
      noReadme: 'لا توجد معاينة SKILL.md لهذه المهارة.',
      trust: {
        builtin: 'مدمجة',
        trusted: 'موثوقة',
        community: 'مجتمعية'
      },
      verdictSafe: 'آمنة',
      verdictCaution: 'تحذير',
      verdictDangerous: 'خطرة',
      policyAllow: 'التثبيت مسموح',
      policyAsk: 'راجع قبل التثبيت',
      policyBlock: 'التثبيت محظور بالسياسة',
      findings: count => `${count} ملاحظة`,
      noFindings: 'لا توجد ملاحظات أمنية.',
      installStarted: name => `جارٍ تثبيت ${name}...`,
      uninstallStarted: name => `جارٍ إزالة ${name}...`,
      updateStarted: 'جارٍ تحديث المهارات المثبتة...',
      actionFailed: 'فشل إجراء المهارة',
      actionLog: 'سجل الإجراءات',
      loadFailed: 'فشل تحميل مركز المهارات',
      previewFailed: 'فشلت معاينة المهارة',
      scanFailed: 'فشل الفحص الأمني',
      searchFailed: 'فشل البحث في المركز'
    },
    categoryLabels: {
      'apple': 'آبل',
      'autonomous-ai-agents': 'وكلاء مستقلون',
      'creative': 'إبداع',
      'data-science': 'علم البيانات',
      'email': 'البريد',
      'general': 'عام',
      'github': 'GitHub',
      'media': 'وسائط',
      'mlops': 'عمليات تعلم آلي',
      'note-taking': 'تدوين',
      'productivity': 'إنتاجية',
      'research': 'بحث',
      'smart-home': 'منزل ذكي',
      'social-media': 'تواصل اجتماعي',
      'software-development': 'تطوير برمجيات'
    }
  },

  starmap: {
    title: 'مخطّط الذاكرة',
    subtitle: (nodes, clusters) => `${nodes} مهارة عبر ${clusters} فئة`,
    close: 'إغلاق مخطّط الذاكرة',
    refresh: 'تحديث',
    memory: 'الذاكرة',
    filterAll: 'الكل',
    filterUsed: 'المستعملة',
    filterLearned: 'المتعلَّمة',
    viewGraph: 'المخطّط',
    loadFailed: 'تعذّر تحميل مخطّط الذاكرة',
    loading: 'جارٍ التحميل…',
    emptyTitle: 'لا شيء مُتعلَّم بعد',
    emptyDesc: 'كلّما بنى هرمس مهاراتٍ وذكرياتٍ لعملك، تظهر هنا.',
    share: 'مشاركة المخطّط',
    shareHint: 'انسخ الرمز لمشاركة هذا المخطّط، أو الصق رمزًا لتحميله. يتضمّن التخطيط فقط، لا نصّ ذاكرتك أو مهاراتك.',
    shareTitle: 'استيراد / تصدير المخطّط',
    sharePlaceholder: 'الصق رمز مخطّط…',
    copy: 'نسخ رمز المخطّط',
    copied: 'نُسخ!',
    importMap: 'استيراد مخطّط',
    importBtn: 'تحميل',
    importEmpty: 'الصق رمز مخطّط لتحميله.',
    importSuccess: nodes => `حُمِّل مخطّطٌ به ${nodes} ${nodes === 1 ? 'عقدة' : 'عقدة'}.`,
    importedBadge: 'مخطّط مستورَد',
    resetToMine: 'العودة إلى مخطّطي'
  },

  agents: {
    close: 'إغلاق الوكلاء',
    title: 'شجرة التفويض',
    subtitle: 'نشاط الوكلاء الفرعيين المباشر للدورة الحالية.',
    emptyTitle: 'لا وكلاء فرعيين نشطون',
    emptyDesc: 'عندما تفوّض دورة عملًا، يظهر تقدم الوكلاء الفرعيين هنا.',
    running: 'يعمل',
    failed: 'فشل',
    done: 'تم',
    streaming: 'يبث',
    files: 'الملفات',
    moreFiles: count =>
      count === 1
        ? '+ملف واحد آخر'
        : count === 2
          ? '+ملفان آخران'
          : count <= 10
            ? `+${count} ملفات أخرى`
            : `+${count} ملفًا آخر`,
    delegation: index => `التفويض ${index}`,
    workers: count =>
      count === 1 ? 'منفّذ واحد' : count === 2 ? 'منفّذان' : count <= 10 ? `${count} منفّذين` : `${count} منفّذًا`,
    workersActive: count =>
      count === 1 ? 'واحد نشط' : count === 2 ? 'اثنان نشطان' : count <= 10 ? `${count} نشطين` : `${count} نشطًا`,
    agentsCount: count =>
      count === 1 ? 'وكيل واحد' : count === 2 ? 'وكيلان' : count <= 10 ? `${count} وكلاء` : `${count} وكيلًا`,
    activeCount: count =>
      count === 1 ? 'واحد نشط' : count === 2 ? 'اثنان نشطان' : count <= 10 ? `${count} نشطين` : `${count} نشطًا`,
    failedCount: count => (count === 1 ? 'فشل واحد' : count === 2 ? 'فشل اثنان' : `فشل ${count}`),
    toolsCount: count =>
      count === 1 ? 'أداة واحدة' : count === 2 ? 'أداتان' : count <= 10 ? `${count} أدوات` : `${count} أداة`,
    filesCount: count =>
      count === 1 ? 'ملف واحد' : count === 2 ? 'ملفان' : count <= 10 ? `${count} ملفات` : `${count} ملفًا`,
    updatedAgo: age => `حُدّث ${age}`,
    ageNow: 'الآن',
    ageSeconds: seconds => `قبل ${seconds} ث`,
    ageMinutes: minutes => `قبل ${minutes} د`,
    ageHours: hours => `قبل ${hours} س`,
    durationSeconds: seconds => `${seconds} ث`,
    durationMinutes: (minutes, seconds) => `${minutes} د ${seconds} ث`,
    tokens: value => `${value} وحدة`
  },

  commandCenter: {
    close: 'إغلاق مركز الأوامر',
    paletteTitle: 'لوحة الأوامر',
    back: 'رجوع',
    searchPlaceholder: 'البحث في الجلسات وطرق العرض والإجراءات',
    goTo: 'انتقال إلى',
    goToSession: 'انتقال إلى الجلسة',
    branches: 'الفروع',
    startInBranch: branch => `محادثة جديدة في ${branch}`,
    commandCenter: 'مركز الأوامر',
    appearance: 'المظهر',
    settings: 'الإعدادات',
    changeTheme: 'تغيير السمة...',
    pets: {
      title: 'الحيوانات',
      placeholder: 'ابحث عن حيوان…',
      loading: 'جارٍ تحميل معرض الحيوانات…',
      error: 'تعذّر الوصول إلى معرض الحيوانات.',
      staleBackend: 'أعد تشغيل هرمس لاستخدام الحيوانات — الخلفية أقدم من هذه الميزة.',
      empty: 'لا حيوانات مطابقة.',
      turnOff: 'إيقاف',
      turnOn: 'تشغيل',
      installed: 'مثبّت',
      generatedTag: 'مُولَّد',
      adoptFailed: 'تعذّر تبنّي ذلك الحيوان.',
      toggleFailed: 'تعذّر تبديل حالة الحيوان.',
      noneAvailable: 'لا حيوانات متاحة — اختر واحدًا أدناه لتثبيته.'
    },
    changeColorMode: 'تغيير وضع الألوان...',
    generatePet: {
      title: 'توليد حيوان',
      placeholder: 'صِف حيوانًا لتوليده…',
      promptHint: 'اكتب وصفًا، ثم اضغط Enter لرسم أربعة أشكال.',
      readyHint: 'اضغط Enter لرسم أربعة أشكال من وصفك.',
      generate: 'توليد',
      generating: 'جارٍ التوليد…',
      retry: 'إعادة المحاولة',
      hatch: 'فقس',
      spawning: 'جارٍ الإنشاء…',
      hatching: 'جارٍ فقس حيوانك…',
      hatchingSub: 'جارٍ بثّ الحياة فيه…',
      hatched: 'لقد فقس!',
      hatchRow: (_state, done, total) => `جارٍ رسم الإطار ${done} من ${total}…`,
      hatchComposing: 'جارٍ تجميع أجزائه…',
      hatchSaving: 'أوشكنا على الانتهاء…',
      namePlaceholder: 'سمِّ حيوانك',
      staleBackend: 'حدّث هرمس لتوليد الحيوانات.',
      backgroundHint: 'يمكنك إغلاق هذا — سينبّهك هرمس عند الانتهاء.',
      slowProviderHint: 'قد يستغرق هذا عدة دقائق',
      remix: 'إعادة مزج',
      remixConfirmTitle: 'إعادة مزج هذا الشكل؟',
      remixConfirmBody:
        'يولّد هذا مجموعة جديدة من المسودّات منطلقًا من هذا الشكل. قد يستغرق عدة دقائق.',
      genericError: 'فشل التوليد — أعد المحاولة أو اختر اقتراحًا.',
      referenceImageTooLarge: 'الصورة المرجعية كبيرة جدًّا. استخدم واحدةً أصغر من ١٦ ميغابايت.',
      referenceImageInvalid: 'تعذّر قراءة تلك الصورة المرجعية. جرّب صيغة PNG أو JPG أو WebP أو GIF.',
      adopt: 'تبنٍّ',
      startOver: 'البدء من جديد'
    },
    installTheme: {
      title: 'تثبيت سمة...',
      pageTitle: 'تثبيت السمة',
      placeholder: 'البحث في متجر VS Code...',
      loading: 'جارٍ البحث في المتجر...',
      error: 'تعذر الوصول إلى المتجر.',
      empty: 'لا توجد سمات مطابقة.',
      install: 'تثبيت',
      installing: 'جارٍ التثبيت...',
      installed: 'مثبّتة',
      installs: count => `عمليات التثبيت: ${count}`
    },
    settingsFields: 'حقول الإعدادات',
    mcpServers: 'خوادم بروتوكول سياق النموذج',
    archivedChats: 'المحادثات المؤرشفة',
    sections: { maintenance: 'الصيانة', sessions: 'الجلسات', system: 'النظام', usage: 'الاستخدام' },
    sectionDescriptions: {
      maintenance: 'التشخيص والنسخ الاحتياطي وبيانات الذاكرة',
      sessions: 'البحث في الجلسات وإدارتها',
      system: 'الحالة والسجلات وإجراءات النظام',
      usage: 'نشاط الوحدات والتكلفة والمهارات عبر الزمن'
    },
    nav: {
      newChat: { title: 'جلسة جديدة', detail: 'بدء جلسة جديدة' },
      settings: { title: 'الإعدادات', detail: 'ضبط هرمس لسطح المكتب' },
      skills: { title: 'المهارات والأدوات', detail: 'تفعيل المهارات ومجموعات الأدوات والمزوّدين' },
      messaging: { title: 'المراسلة', detail: 'إعداد تيليجرام وسلاك وديسكورد وغيرها' },
      artifacts: { title: 'المخرجات', detail: 'تصفح المخرجات المنشأة' }
    },
    sectionEntries: {
      sessions: { title: 'لوحة الجلسات', detail: 'البحث في الجلسات وتثبيتها وإدارتها' },
      system: { title: 'لوحة النظام', detail: 'حالة البوابة والسجلات وإعادة التشغيل والتحديث' },
      usage: { title: 'لوحة الاستخدام', detail: 'نشاط الوحدات والتكلفة والمهارات' }
    },
    providerNavigate: 'التنقل',
    providerSessions: 'الجلسات',
    refresh: 'تحديث',
    refreshing: 'جارٍ التحديث...',
    noResults: 'لم يُعثر على نتائج مطابقة.',
    pinSession: 'تثبيت الجلسة',
    unpinSession: 'إلغاء تثبيت الجلسة',
    exportSession: 'تصدير الجلسة',
    deleteSession: 'حذف الجلسة',
    noSessions: 'لا توجد جلسات بعد.',
    gatewayRunning: 'بوابة المراسلة تعمل',
    gatewayStopped: 'بوابة المراسلة متوقفة',
    hermesActiveSessions: (version, count) => `هرمس ${version} · الجلسات النشطة ${count}`,
    restartGateway: 'إعادة تشغيل البوابة',
    gatewayRestartFailed: 'فشلت إعادة تشغيل البوابة.',
    updateHermes: 'تحديث هرمس',
    actionRunning: 'يعمل',
    actionDone: 'تم',
    actionFailed: 'فشل',
    actionStartedWaiting: 'بدأ الإجراء، وفي انتظار الحالة...',
    loadingStatus: 'جارٍ تحميل الحالة...',
    recentLogs: 'السجلات الأخيرة',
    noLogs: 'لم تُحمّل سجلات بعد.',
    days: count => `${count} ي`,
    statSessions: 'الجلسات',
    statApiCalls: 'استدعاءات الواجهة البرمجية',
    statTokens: 'الوحدات الداخلة والخارجة',
    statCost: 'التكلفة المقدرة',
    actualCost: cost => `الفعلية ${cost}`,
    loadingUsage: 'جارٍ تحميل الاستخدام...',
    noUsage: period =>
      period === 1
        ? 'لا استخدام خلال اليوم الأخير.'
        : period === 2
          ? 'لا استخدام خلال اليومين الأخيرين.'
          : period <= 10
            ? `لا استخدام خلال آخر ${period} أيام.`
            : `لا استخدام خلال آخر ${period} يومًا.`,
    retry: 'إعادة المحاولة',
    dailyTokens: 'الوحدات اليومية',
    dayUsageTooltip: (day, input, output) => `اليوم ${day} · إدخال ${input} · إخراج ${output}`,
    input: 'إدخال',
    output: 'إخراج',
    noDailyActivity: 'لا نشاط يومي.',
    topModels: 'أكثر النماذج استخدامًا',
    noModelUsage: 'لا يوجد استخدام للنماذج بعد.',
    topSkills: 'أكثر المهارات استخدامًا',
    noSkillActivity: 'لا يوجد نشاط للمهارات بعد.',
    actions: count => `الإجراءات: ${count}`,
    logFile: 'ملف السجل',
    logLevel: 'المستوى',
    logSearchPlaceholder: 'تصفية أسطر السجل...',
    maintenance: {
      runOps: 'التشخيص',
      doctor: 'تشغيل الفاحص',
      doctorDesc: 'فحص صحة التثبيت والإعداد والمزودات',
      securityAudit: 'التدقيق الأمني',
      securityAuditDesc: 'فحص الإعداد والمهارات بحثًا عن إعدادات خطرة',
      backup: 'إنشاء نسخة احتياطية',
      backupDesc: 'ضغط الإعداد والذكريات والمهارات والجلسات',
      debugShare: 'مشاركة التصحيح',
      debugShareDesc: 'رفع تقرير مموه مع السجلات والحصول على روابط قابلة للمشاركة (تُحذف تلقائيًا خلال ٦ ساعات)',
      debugShareRunning: 'جارٍ رفع تقرير التصحيح...',
      debugShareLinks: 'روابط المشاركة',
      debugShareFailed: 'فشلت مشاركة التصحيح',
      copyLink: 'نسخ الرابط',
      linkCopied: 'نُسخ الرابط',
      curator: 'منسق المهارات',
      curatorDesc: 'مراجعة خلفية تؤرشف المهارات الراكدة التي أنشأها الوكيل',
      curatorPaused: 'موقوف مؤقتًا',
      curatorActive: 'نشط',
      curatorDisabled: 'معطل',
      curatorLastRun: when => `آخر تشغيل ${when}`,
      curatorNeverRan: 'لم يعمل قط',
      pause: 'إيقاف مؤقت',
      resume: 'استئناف',
      runNow: 'تشغيل الآن',
      memoryData: 'بيانات الذاكرة',
      memoryDataDesc: 'ملفات الذاكرة المدمجة المحقونة في كل جلسة',
      memoryProvider: name => `المزود النشط: ${name}`,
      builtinMemory: 'مدمج',
      memoryFile: 'ذاكرة الوكيل (MEMORY.md)',
      userFile: 'ملف المستخدم (USER.md)',
      bytes: size => size,
      empty: 'فارغ',
      resetMemory: 'تصفير الذاكرة',
      resetUser: 'تصفير الملف الشخصي',
      resetAll: 'تصفير الاثنين',
      resetConfirm: target => `حذف ${target}؟ لا يمكن التراجع عن هذا.`,
      resetDone: files => `حُذف ${files}.`,
      resetFailed: 'فشل تصفير الذاكرة',
      actionStarted: name => `بدأ ${name} — متابعة السجل...`,
      actionFailed: name => `فشل بدء ${name}`,
      running: 'قيد التشغيل...',
      viewLog: 'سجل الإجراءات'
    }
  },

  messaging: {
    search: 'البحث في المراسلة...',
    loading: 'جارٍ تحميل منصات المراسلة...',
    loadFailed: 'فشل تحميل منصات المراسلة',
    states: {
      connected: 'متصل',
      connecting: 'جارٍ الاتصال',
      disabled: 'معطّل',
      fatal: 'خطأ',
      gateway_stopped: 'بوابة المراسلة متوقفة',
      not_configured: 'يحتاج إعدادًا',
      pending_restart: 'تستلزم إعادة التشغيل',
      retrying: 'جارٍ إعادة المحاولة',
      startup_failed: 'فشل البدء'
    },
    unknown: 'غير معروف',
    hintPendingRestart: 'أعد تشغيل البوابة من شريط الحالة لتطبيق هذا التغيير.',
    hintGatewayStopped: 'شغّل البوابة من شريط الحالة للاتصال.',
    credentialsSet: 'بيانات الاعتماد مضبوطة',
    needsSetup: 'يحتاج إعدادًا',
    gatewayStopped: 'بوابة المراسلة متوقفة',
    getCredentials: 'الحصول على بيانات الاعتماد',
    openSetupGuide: 'فتح دليل الإعداد',
    required: 'مطلوب',
    recommended: 'موصى به',
    advanced: count => `متقدم (${count})`,
    noTokenNeeded: 'لا تحتاج هذه المنصة إلى رمز هنا. استخدم دليل الإعداد أعلاه ثم فعّلها أدناه.',
    enabled: 'مفعّل',
    disabled: 'معطّل',
    unsavedChanges: 'تغييرات غير محفوظة',
    saving: 'جارٍ الحفظ...',
    saveChanges: 'حفظ التغييرات',
    saved: 'حُفظ',
    replaceValue: 'استبدال القيمة الحالية',
    openDocs: 'فتح التوثيق',
    clearField: key => `مسح ${key}`,
    enableAria: name => `تفعيل ${name}`,
    disableAria: name => `تعطيل ${name}`,
    platformEnabled: name => `فُعّل ${name}`,
    platformDisabled: name => `عُطّل ${name}`,
    restartToApply: 'أعد تشغيل البوابة ليصبح هذا التغيير نافذًا.',
    setupSaved: name => `حُفظ إعداد ${name}`,
    restartToReconnect: 'أعد تشغيل البوابة للاتصال ببيانات الاعتماد الجديدة.',
    keyCleared: key => `مُسح ${key}`,
    setupUpdated: name => `حُدّث إعداد ${name}.`,
    failedUpdate: name => `فشل تحديث ${name}`,
    failedSave: name => `فشل حفظ ${name}`,
    failedClear: key => `فشل مسح ${key}`,
    fieldCopy: {
      TELEGRAM_BOT_TOKEN: {
        label: 'رمز البوت',
        help: 'أنشئ بوتًا عبر @BotFather ثم الصق الرمز الذي يمنحك إياه.',
        placeholder: 'الصق رمز بوت تيليجرام'
      },
      TELEGRAM_ALLOWED_USERS: {
        label: 'معرّفات مستخدمي تيليجرام المسموح لهم',
        help: 'موصى به. معرّفات رقمية مفصولة بفواصل من @userinfobot. من دونها يستطيع أي شخص مراسلة البوت.'
      },
      TELEGRAM_PROXY: { label: 'رابط الوكيل', help: 'يلزم فقط على الشبكات التي تحجب تيليجرام.' },
      DISCORD_BOT_TOKEN: {
        label: 'رمز البوت',
        help: 'أنشئ تطبيقًا في بوابة مطوري ديسكورد، وأضف بوتًا، ثم الصق رمزه.'
      },
      DISCORD_ALLOWED_USERS: {
        label: 'معرّفات مستخدمي ديسكورد المسموح لهم',
        help: 'موصى به. معرّفات مستخدمي ديسكورد مفصولة بفواصل.'
      },
      DISCORD_REPLY_TO_MODE: { label: 'أسلوب الرد', help: 'first أو all أو off.' },
      DISCORD_ALLOW_ALL_USERS: {
        label: 'السماح لجميع مستخدمي ديسكورد',
        help: 'للتطوير فقط. عند تفعيله يستطيع أي شخص مراسلة البوت دون قائمة سماح.'
      },
      DISCORD_HOME_CHANNEL: {
        label: 'معرّف القناة الرئيسية',
        help: 'القناة التي يرسل إليها البوت الرسائل الاستباقية، مثل مخرجات المهام المجدولة والتذكيرات.'
      },
      DISCORD_HOME_CHANNEL_NAME: {
        label: 'اسم القناة الرئيسية',
        help: 'الاسم المعروض للقناة الرئيسية في السجلات والحالة.'
      },
      BLUEBUBBLES_ALLOW_ALL_USERS: {
        label: 'السماح لجميع مستخدمي iMessage',
        help: 'عند تفعيله تُتجاوز قائمة سماح BlueBubbles.'
      },
      MATTERMOST_ALLOW_ALL_USERS: { label: 'السماح لجميع مستخدمي Mattermost' },
      MATTERMOST_HOME_CHANNEL: { label: 'القناة الرئيسية' },
      QQ_ALLOW_ALL_USERS: { label: 'السماح لجميع مستخدمي QQ' },
      QQBOT_HOME_CHANNEL: { label: 'قناة QQ الرئيسية', help: 'القناة أو المجموعة الافتراضية لتسليم المهام المجدولة.' },
      QQBOT_HOME_CHANNEL_NAME: { label: 'اسم قناة QQ الرئيسية' },
      SLACK_BOT_TOKEN: {
        label: 'رمز بوت سلاك',
        help: 'استخدم رمز البوت من OAuth & Permissions بعد تثبيت تطبيق سلاك.',
        placeholder: 'الصق رمز بوت سلاك'
      },
      SLACK_APP_TOKEN: {
        label: 'رمز تطبيق سلاك',
        help: 'استخدم رمز مستوى التطبيق المطلوب لوضع Socket.',
        placeholder: 'الصق رمز تطبيق سلاك'
      },
      SLACK_ALLOWED_USERS: {
        label: 'معرّفات مستخدمي سلاك المسموح لهم',
        help: 'موصى به. معرّفات مستخدمي سلاك مفصولة بفواصل.'
      },
      MATTERMOST_URL: { label: 'رابط الخادم', placeholder: 'https://mattermost.example.com' },
      MATTERMOST_TOKEN: { label: 'رمز البوت' },
      MATTERMOST_ALLOWED_USERS: {
        label: 'معرّفات المستخدمين المسموح لهم',
        help: 'موصى به. معرّفات مستخدمي Mattermost مفصولة بفواصل.'
      },
      MATRIX_HOMESERVER: { label: 'رابط الخادم الرئيسي (homeserver)', placeholder: 'https://matrix.org' },
      MATRIX_ACCESS_TOKEN: { label: 'رمز الوصول' },
      MATRIX_USER_ID: { label: 'معرّف مستخدم البوت', placeholder: '@hermes:example.org' },
      MATRIX_ALLOWED_USERS: {
        label: 'معرّفات مستخدمي Matrix المسموح لهم',
        help: 'موصى به. معرّفات مفصولة بفواصل بصيغة @user:server.'
      },
      SIGNAL_HTTP_URL: {
        label: 'رابط جسر Signal',
        placeholder: 'http://127.0.0.1:8080',
        help: 'رابط جسر signal-cli REST يعمل حاليًا.'
      },
      SIGNAL_ACCOUNT: { label: 'رقم الهاتف', help: 'الرقم المسجل في جسر signal-cli.' },
      SIGNAL_ALLOWED_USERS: {
        label: 'مستخدمو Signal المسموح لهم',
        help: 'موصى به. معرّفات Signal مفصولة بفواصل.'
      },
      WHATSAPP_ENABLED: {
        label: 'تفعيل جسر واتساب',
        help: 'يضبطه المفتاح أدناه تلقائيًا. لا تغيّره إلا إذا كنت تعرف أنك تحتاج إليه.'
      },
      WHATSAPP_MODE: { label: 'وضع الجسر' },
      WHATSAPP_ALLOWED_USERS: {
        label: 'مستخدمو واتساب المسموح لهم',
        help: 'موصى به. أرقام الهواتف أو معرّفات واتساب مفصولة بفواصل.'
      },
      HASS_URL: {
        label: 'رابط Home Assistant',
        help: 'الرابط الأساسي لخادم Home Assistant، مثل https://homeassistant.local:8123.'
      },
      HASS_TOKEN: {
        label: 'رمز وصول Home Assistant',
        help: 'رمز وصول طويل الأمد من Home Assistant (الملف الشخصي → الأمان).'
      },
      EMAIL_ADDRESS: { label: 'عنوان البريد الإلكتروني', help: 'العنوان الذي يرسل هرمس منه ويستقبل عليه.' },
      EMAIL_PASSWORD: { label: 'كلمة مرور البريد', help: 'كلمة مرور الحساب أو كلمة مرور تطبيق مخصصة.' },
      EMAIL_IMAP_HOST: { label: 'خادم IMAP', help: 'مضيف خادم IMAP، مثل imap.gmail.com.' },
      EMAIL_SMTP_HOST: { label: 'خادم SMTP', help: 'مضيف خادم SMTP، مثل smtp.gmail.com.' },
      TWILIO_ACCOUNT_SID: {
        label: 'معرّف حساب Twilio (Account SID)',
        help: 'من لوحة تحكم Twilio.'
      },
      TWILIO_AUTH_TOKEN: {
        label: 'رمز مصادقة Twilio (Auth Token)',
        help: 'من لوحة تحكم Twilio.'
      },
      DINGTALK_CLIENT_ID: { label: 'معرّف العميل (Client ID)', help: 'معرّف عميل DingTalk (مفتاح التطبيق).' },
      DINGTALK_CLIENT_SECRET: { label: 'سر العميل (Client Secret)', help: 'سر عميل DingTalk (سر التطبيق).' },
      FEISHU_APP_ID: { label: 'معرّف التطبيق (App ID)', help: 'معرّف تطبيق Feishu / Lark.' },
      FEISHU_APP_SECRET: { label: 'سر التطبيق (App secret)', help: 'سر تطبيق Feishu / Lark.' },
      FEISHU_ENCRYPT_KEY: { label: 'مفتاح التشفير', help: 'مفتاح تشفير أحداث Feishu / Lark.' },
      FEISHU_VERIFICATION_TOKEN: { label: 'رمز التحقق', help: 'رمز تحقق Feishu / Lark.' },
      WECOM_BOT_ID: { label: 'معرّف بوت WeCom', help: 'مفتاح الويب هوك لروبوت المجموعة في WeCom.' },
      WECOM_SECRET: { label: 'سر WeCom', help: 'سر روبوت المجموعة في WeCom.' },
      WECOM_CALLBACK_CORP_ID: { label: 'معرّف مؤسسة WeCom', help: 'معرّف المؤسسة في WeCom.' },
      WECOM_CALLBACK_CORP_SECRET: { label: 'سر مؤسسة WeCom', help: 'سر تطبيق WeCom الخاص بالمؤسسة.' },
      WECOM_CALLBACK_AGENT_ID: { label: 'معرّف وكيل WeCom', help: 'معرّف الوكيل لتطبيق WeCom الذاتي البناء.' },
      WECOM_CALLBACK_TOKEN: { label: 'رمز WeCom', help: 'رمز التحقق من الاستدعاء الراجع في WeCom.' },
      WECOM_CALLBACK_ENCODING_AES_KEY: {
        label: 'مفتاح AES لـ WeCom',
        help: 'مفتاح ترميز AES للاستدعاء الراجع في WeCom.'
      },
      WEIXIN_ACCOUNT_ID: { label: 'معرّف الحساب', help: 'معرّف الحساب الرسمي في WeChat.' },
      WEIXIN_TOKEN: { label: 'رمز الاستدعاء الراجع', help: 'رمز الاستدعاء الراجع في WeChat.' },
      WEIXIN_BASE_URL: { label: 'الرابط الأساسي', help: 'الرابط الأساسي لمنصة WeChat.' },
      BLUEBUBBLES_SERVER_URL: {
        label: 'رابط خادم BlueBubbles',
        help: 'رابط خادم BlueBubbles لتكامل iMessage، مثل http://192.168.1.10:1234.'
      },
      BLUEBUBBLES_PASSWORD: {
        label: 'كلمة مرور خادم BlueBubbles',
        help: 'من BlueBubbles Server → Settings → API.'
      },
      BLUEBUBBLES_ALLOWED_USERS: {
        label: 'عناوين iMessage المسموح لها',
        help: 'موصى به. عناوين iMessage (بريد إلكتروني أو هاتف) مفصولة بفواصل.'
      },
      QQ_APP_ID: { label: 'معرّف تطبيق QQ', help: 'معرّف التطبيق من منصة QQ المفتوحة (q.qq.com).' },
      QQ_CLIENT_SECRET: { label: 'سر عميل QQ', help: 'سر العميل من منصة QQ المفتوحة.' },
      QQ_ALLOWED_USERS: {
        label: 'معرّفات مستخدمي QQ المسموح لهم',
        help: 'موصى به. معرّفات مستخدمي QQ مفصولة بفواصل.'
      },
      API_SERVER_ENABLED: {
        label: 'تفعيل خادم الواجهة البرمجية',
        help: 'يفعّل الواجهة البرمجية المتوافقة مع OpenAI ليتصل بها مثل Open WebUI و LobeChat.'
      },
      API_SERVER_KEY: {
        label: 'مفتاح مصادقة الواجهة البرمجية',
        help: 'رمز Bearer لمصادقة الخادم. مطلوب متى كان الخادم مفعّلًا، ويرفض الخادم البدء من دونه.'
      },
      API_SERVER_PORT: { label: 'منفذ الخادم', help: 'منفذ خادم الواجهة البرمجية (الافتراضي 8642).' },
      API_SERVER_HOST: {
        label: 'مضيف الخادم',
        help: 'عنوان الربط (الافتراضي 127.0.0.1). يبقى API_SERVER_KEY مطلوبًا حتى على الربط المحلي.'
      },
      API_SERVER_MODEL_NAME: {
        label: 'اسم النموذج المعلن',
        help: 'الاسم المعروض على المسار /v1/models. الافتراضي اسم الملف الشخصي.'
      },
      WEBHOOK_ENABLED: {
        label: 'تفعيل الويب هوك',
        help: 'يفعّل محوّل الويب هوك لاستقبال الأحداث من GitHub و GitLab وغيرهما.'
      },
      WEBHOOK_PORT: { label: 'منفذ الويب هوك', help: 'منفذ خادم HTTP للويب هوك (الافتراضي 8644).' },
      WEBHOOK_SECRET: { label: 'سر الويب هوك', help: 'سر HMAC عام للتحقق من توقيعات الويب هوك.' }
    },
    platformIntro: {
      telegram:
        'في تيليجرام، راسل @BotFather ونفّذ الأمر /newbot وانسخ الرمز الذي يعطيك إياه، ثم خذ معرّفك الرقمي من @userinfobot.',
      discord:
        'افتح بوابة مطوري ديسكورد، وأنشئ تطبيقًا، وأضف إليه بوتًا، ثم انسخ رمزه. وادعُ البوت إلى خادمك بالنطاقات الصحيحة.',
      slack: 'أنشئ تطبيق سلاك، وفعّل وضع Socket، وثبّته في مساحة العمل، ثم انسخ رمز البوت ورمز مستوى التطبيق.',
      mattermost: 'على خادم Mattermost لديك، أنشئ حساب بوت أو رمز وصول شخصيًا، ثم الصق رابط الخادم والرمز هنا.',
      matrix: 'سجّل الدخول إلى خادمك الرئيسي بحساب البوت، ثم انسخ رمز الوصول ومعرّف المستخدم ورابط الخادم الرئيسي.',
      signal: 'شغّل جسر signal-cli REST في مكان يمكن الوصول إليه، ثم وجّه هرمس إلى الرابط ورقم الهاتف المسجل.',
      whatsapp: 'شغّل جسر واتساب المرفق مع هرمس، وامسح رمز QR عند أول تشغيل، ثم فعّل المنصة.',
      bluebubbles:
        'شغّل خادم BlueBubbles على جهاز Mac يعمل عليه iMessage، واكشف واجهته البرمجية، ثم وجّه هرمس إلى الرابط مع كلمة مرور الخادم.',
      homeassistant: 'في Home Assistant، افتح ملفك الشخصي وأنشئ رمز وصول طويل الأمد، ثم الصقه هنا مع رابط HA لديك.',
      email:
        'استخدم صندوق بريد مخصصًا. لحسابات Gmail/Workspace، أنشئ كلمة مرور تطبيق واستخدم imap.gmail.com / smtp.gmail.com.',
      sms: 'احصل على Account SID و Auth Token من لوحة تحكم Twilio، إضافة إلى رقم هاتف يمكنه إرسال الرسائل النصية.',
      dingtalk: 'أنشئ تطبيق DingTalk في لوحة المطورين، ثم انسخ Client ID (مفتاح التطبيق) و Client Secret هنا.',
      feishu: 'أنشئ تطبيق Feishu / Lark، وفعّل قدرة البوت فيه، وانسخ App ID و App secret ومفاتيح تشفير الأحداث.',
      wecom:
        'أضف روبوت مجموعة في WeCom وانسخ مفتاح الويب هوك الخاص به في WECOM_BOT_ID. للإرسال فقط — استخدم خيار WeCom (تطبيق) للتواصل ثنائي الاتجاه.',
      wecom_callback:
        'جهّز تطبيق WeCom ذاتي البناء، واكشف رابط الاستدعاء الراجع الخاص به، وأدخل معرّف المؤسسة والسر ومعرّف الوكيل ومفتاح AES.',
      weixin:
        'سجّل الدخول إلى منصة الحسابات الرسمية في WeChat، وانسخ AppID و Token، ووجّه رابط الاستدعاء الراجع للرسائل إلى هرمس.',
      qqbot: 'سجّل تطبيقًا على منصة QQ المفتوحة (q.qq.com) وانسخ App ID و Client Secret.',
      yuanbao: 'اربط هرمس بخدمة Yuanbao (元宝) من Tencent.',
      api_server:
        'اكشف هرمس كواجهة برمجية متوافقة مع OpenAI. اضبط مفتاح مصادقة، ثم وجّه Open WebUI أو LobeChat وغيرهما إلى المضيف والمنفذ.',
      webhook:
        'شغّل خادم HTTP تستطيع الأدوات الأخرى (GitHub و GitLab والتطبيقات المخصصة) إرسال POST إليه. استخدم السر للتحقق من التوقيعات.'
    }
  },

  profiles: {
    close: 'إغلاق الملفات الشخصية',
    nameHint: 'أحرف إنجليزية صغيرة وأرقام وشرطات وشرطات سفلية. يجب أن يبدأ بحرف أو رقم.',
    title: 'الملفات الشخصية',
    count: count =>
      count === 1
        ? 'ملف شخصي واحد'
        : count === 2
          ? 'ملفان شخصيان'
          : count <= 10
            ? `${count} ملفات شخصية`
            : `${count} ملفًا شخصيًا`,
    search: 'البحث في الملفات الشخصية...',
    loading: 'جارٍ تحميل الملفات الشخصية...',
    newProfile: 'ملف شخصي جديد',
    allProfiles: 'جميع الملفات الشخصية',
    showAllProfiles: 'إظهار جميع الملفات الشخصية',
    switchToProfile: name => `الانتقال إلى ${name}`,
    manageProfiles: 'إدارة الملفات الشخصية...',
    actionsFor: name => `إجراءات ${name}`,
    color: 'اللون...',
    colorFor: name => `لون ${name}`,
    setColor: color => `ضبط اللون ${color}`,
    autoColor: 'تلقائي',
    noProfiles: 'لا توجد ملفات شخصية بعد.',
    selectPrompt: 'اختر ملفًا شخصيًا لعرض تفاصيله.',
    refresh: 'تحديث الملفات الشخصية',
    refreshing: 'جارٍ تحديث الملفات الشخصية',
    default: 'افتراضي',
    skills: count =>
      count === 1 ? 'مهارة واحدة' : count === 2 ? 'مهارتان' : count <= 10 ? `${count} مهارات` : `${count} مهارة`,
    env: 'البيئة',
    defaultBadge: 'افتراضي',
    rename: 'إعادة التسمية',
    renameMenu: 'إعادة تسمية…',
    editSoul: 'تحرير SOUL.md…',
    copySetup: 'نسخ أمر الإعداد',
    copying: 'جارٍ النسخ...',
    modelLabel: 'النموذج',
    skillsLabel: 'المهارات',
    notSet: 'غير مضبوط',
    soulDesc: 'موجّه النظام وتعليمات الشخصية المضمّنة في هذا الملف.',
    soulOptional: 'اختياري',
    soulPlaceholder: mode => `موجّه النظام أو شخصية هذا الملف.\nاتركه فارغًا ليبقى ${mode}.`,
    soulPlaceholderCloned: 'مستنسخًا من الافتراضي',
    soulPlaceholderEmpty: 'فارغًا',
    unsavedChanges: 'تغييرات غير محفوظة',
    loadingSoul: 'جارٍ تحميل SOUL.md...',
    emptySoul: 'ملف SOUL.md فارغ؛ ابدأ كتابة الشخصية...',
    saving: 'جارٍ الحفظ...',
    saveSoul: 'حفظ SOUL.md',
    deleteTitle: 'حذف الملف الشخصي؟',
    deleteDescPrefix: 'سيؤدي ذلك إلى حذف ',
    deleteDescMid: ' وإزالة مجلد ',
    deleteDescSuffix: '. لا يمكن التراجع عن ذلك.',
    deleting: 'جارٍ الحذف...',
    createDesc: 'الملفات الشخصية بيئات هرمس مستقلة، ولكل منها إعداداتها ومهاراتها وملف SOUL.md.',
    nameLabel: 'الاسم',
    cloneFromDefault: 'نسخ الملف الافتراضي',
    cloneFromDefaultDesc: 'ينسخ الإعدادات والمهارات وSOUL.md من ملفك الشخصي الافتراضي.',
    cloneFrom: 'النسخ من',
    cloneFromNone: 'بلا (فارغ)',
    cloneFromDesc: 'ينسخ الإعدادات والمهارات وSOUL.md من الملف الشخصي المصدر المحدد.',
    invalidName: hint => `الاسم غير صالح. ${hint}`,
    nameRequired: 'الاسم مطلوب.',
    creating: 'جارٍ الإنشاء...',
    createAction: 'إنشاء ملف شخصي',
    renameTitle: 'إعادة تسمية الملف الشخصي',
    renameDescPrefix: 'تحدّث إعادة التسمية مجلد الملف وأي سكربتات تغليف في ',
    renameDescSuffix: '.',
    newNameLabel: 'الاسم الجديد',
    renaming: 'جارٍ إعادة التسمية...',
    created: 'أُنشئ الملف الشخصي',
    renamed: 'أُعيدت تسمية الملف الشخصي',
    deleted: 'حُذف الملف الشخصي',
    setupCopied: 'نُسخ أمر الإعداد',
    soulSaved: 'حُفظ SOUL.md',
    failedLoad: 'فشل تحميل الملفات الشخصية',
    failedDelete: 'فشل حذف الملف الشخصي',
    failedCopy: 'فشل نسخ أمر الإعداد',
    failedLoadSoul: 'فشل تحميل SOUL.md',
    failedSaveSoul: 'فشل حفظ SOUL.md',
    failedCreate: 'فشل إنشاء الملف الشخصي',
    failedRename: 'فشلت إعادة تسمية الملف الشخصي'
  },

  cron: {
    close: 'إغلاق المهام المجدولة',
    title: 'المهام المجدولة',
    count: count => (count === 1 ? 'مهمة واحدة' : count === 2 ? 'مهمتان' : count <= 10 ? `${count} مهام` : `${count} مهمة`),
    search: 'البحث في المهام المجدولة...',
    loading: 'جارٍ تحميل المهام المجدولة...',
    states: {
      enabled: 'مفعّلة',
      scheduled: 'مجدولة',
      running: 'تعمل',
      paused: 'متوقفة مؤقتًا',
      disabled: 'معطّلة',
      error: 'خطأ',
      completed: 'مكتملة'
    },
    deliveryLabels: {
      local: 'سطح المكتب هذا',
      telegram: 'تيليجرام',
      discord: 'ديسكورد',
      slack: 'سلاك',
      email: 'البريد الإلكتروني'
    },
    scheduleLabels: {
      daily: 'يوميًا',
      weekdays: 'أيام العمل',
      weekly: 'أسبوعيًا',
      monthly: 'شهريًا',
      hourly: 'كل ساعة',
      'every-15-minutes': 'كل 15 دقيقة',
      custom: 'مخصص'
    },
    scheduleHints: {
      daily: 'كل يوم الساعة 9:00 صباحًا',
      weekdays: 'من الاثنين إلى الجمعة الساعة 9:00 صباحًا',
      weekly: 'كل اثنين الساعة 9:00 صباحًا',
      monthly: 'اليوم الأول من كل شهر الساعة 9:00 صباحًا',
      hourly: 'عند بداية كل ساعة',
      'every-15-minutes': 'كل 15 دقيقة',
      custom: 'صيغة Cron أو عبارة إنجليزية طبيعية'
    },
    days: {
      '0': 'الأحد',
      '1': 'الاثنين',
      '2': 'الثلاثاء',
      '3': 'الأربعاء',
      '4': 'الخميس',
      '5': 'الجمعة',
      '6': 'السبت',
      '7': 'الأحد'
    },
    dayFallback: value => `اليوم رقم ${value}`,
    everyDayAt: time => `كل يوم عند ${time}`,
    weekdaysAt: time => `أيام العمل عند ${time}`,
    everyDayOfWeekAt: (day, time) => `كل ${day} عند ${time}`,
    monthlyOnDayAt: (dayOfMonth, time) => `شهريًا في اليوم ${dayOfMonth} عند ${time}`,
    topOfHour: 'عند بداية كل ساعة',
    everyHourAt: minute => `كل ساعة عند :${minute}`,
    newCron: 'مهمة مجدولة جديدة',
    emptyDescNew: 'جدول موجّهًا ليعمل وفق تعبير Cron. سيشغله هرمس ويسلّم النتيجة إلى الوجهة التي تختارها.',
    emptyDescSearch: 'جرّب عبارة بحث أوسع.',
    emptyTitleNew: 'لا توجد مهام مجدولة بعد',
    emptyTitleSearch: 'لا نتائج مطابقة',
    last: 'السابق:',
    next: 'التالي:',
    noRuns: 'لا عمليات تشغيل بعد',
    manage: 'إدارة',
    showRuns: 'إظهار عمليات التشغيل',
    hideRuns: 'إخفاء عمليات التشغيل',
    runHistory: 'سجل التشغيل',
    actionsFor: title => `إجراءات ${title}`,
    actionsTitle: 'إجراءات المهمة المجدولة',
    resume: 'استئناف المهمة',
    pause: 'إيقاف المهمة مؤقتًا',
    resumeTitle: 'استئناف',
    pauseTitle: 'إيقاف مؤقت',
    triggerNow: 'تشغيل الآن',
    edit: 'تعديل المهمة',
    deleteTitle: 'حذف المهمة المجدولة؟',
    deleteDescPrefix: 'سيؤدي ذلك إلى إزالة ',
    deleteDescSuffix: ' نهائيًا وإيقاف تشغيلها فورًا.',
    deleting: 'جارٍ الحذف...',
    resumed: 'استؤنفت المهمة المجدولة',
    paused: 'أُوقفت المهمة مؤقتًا',
    triggered: 'شُغّلت المهمة المجدولة',
    deleted: 'حُذفت المهمة المجدولة',
    created: 'أُنشئت المهمة المجدولة',
    updated: 'حُدّثت المهمة المجدولة',
    failedLoad: 'فشل تحميل المهام المجدولة',
    failedUpdate: 'فشل تحديث المهمة المجدولة',
    failedTrigger: 'فشل تشغيل المهمة المجدولة',
    failedDelete: 'فشل حذف المهمة المجدولة',
    failedSave: 'فشل حفظ المهمة المجدولة',
    editTitle: 'تعديل المهمة المجدولة',
    createTitle: 'مهمة مجدولة جديدة',
    editDesc: 'حدّث الجدول أو الموجّه أو وجهة التسليم. تُطبق التغييرات في التشغيل التالي.',
    createDesc: 'جدول موجّهًا ليعمل تلقائيًا. استخدم صيغة Cron أو عبارة إنجليزية طبيعية مثل "every 15 minutes".',
    nameLabel: 'الاسم',
    namePlaceholder: 'الموجز الصباحي',
    promptLabel: 'الموجّه',
    promptPlaceholder: 'لخّص محادثات سلاك غير المقروءة وأرسل أهم 5 منها بالبريد...',
    frequencyLabel: 'التكرار',
    deliverLabel: 'التسليم إلى',
    customScheduleLabel: 'جدول مخصص',
    customPlaceholder: '0 9 * * * أو weekdays at 9am',
    customHint: 'تعبير Cron أو عبارة إنجليزية مثل "every hour" أو "weekdays at 9am".',
    optional: 'اختياري',
    promptRequired: 'الموجّه مطلوب.',
    promptScheduleRequired: 'الموجّه والجدول مطلوبان.',
    scheduleRequired: 'الجدول مطلوب.',
    scriptOnlyEditHint: 'مهمّة برمجيّة فقط (بلا موجّه ذكاء اصطناعيّ). معرّف المهمّة:',
    saveChanges: 'حفظ التغييرات',
    createAction: 'إنشاء المهمة'
  },

  artifacts: {
    search: 'البحث في المخرجات...',
    refresh: 'تحديث المخرجات',
    refreshing: 'جارٍ تحديث المخرجات',
    indexing: 'جارٍ فهرسة مخرجات الجلسات الأخيرة',
    tabAll: 'الكل',
    tabImages: 'الصور',
    tabFiles: 'الملفات',
    tabLinks: 'الروابط',
    noArtifactsTitle: 'لم يُعثر على مخرجات',
    noArtifactsDesc: 'ستظهر الصور المنشأة ومخرجات الملفات هنا حين تنتجها الجلسات.',
    failedLoad: 'فشل تحميل المخرجات',
    openFailed: 'فشل الفتح',
    itemsImage: 'صور',
    itemsLink: 'روابط',
    itemsFile: 'ملفات',
    itemsGeneric: 'عناصر',
    zero: '0',
    rangeOf: (start, end, total) => `${start}-${end} من ${total}`,
    goToPage: (itemLabel, page) => `الانتقال إلى صفحة ${itemLabel} رقم ${page}`,
    colTitleLink: 'عنوان الرابط',
    colTitleFile: 'الاسم',
    colTitleDefault: 'العنوان أو الاسم',
    colLocationLink: 'الرابط',
    colLocationFile: 'المسار',
    colLocationDefault: 'الموقع',
    colSession: 'الجلسة',
    kindImage: 'صورة',
    kindFile: 'ملف',
    kindLink: 'رابط',
    chat: 'المحادثة',
    copyUrl: 'نسخ الرابط',
    copyPath: 'نسخ المسار'
  },

  sidebar: {
    nav: {
      'new-session': 'جلسة جديدة',
      skills: 'المهارات والأدوات',
      messaging: 'المراسلة',
      artifacts: 'المخرجات'
    },
    searchAria: 'البحث في الجلسات',
    searchPlaceholder: 'البحث في الجلسات…',
    clearSearch: 'مسح البحث',
    noMatch: query => `لا توجد جلسات تطابق «${query}».`,
    results: 'النتائج',
    pinned: 'المثبتة',
    sessions: 'الجلسات',
    cronJobs: 'المهام المجدولة',
    groupAriaGrouped: 'إظهار الجلسات في قائمة واحدة',
    groupAriaUngrouped: 'تجميع الجلسات حسب مساحة العمل',
    showProjects: 'إظهار المشاريع',
    showSessions: 'إظهار الجلسات',
    groupTitleGrouped: 'إلغاء تجميع الجلسات',
    groupTitleUngrouped: 'التجميع حسب مساحة العمل',
    allPinned: 'كل ما هنا مثبت. ألغ تثبيت محادثة لتظهر ضمن المحادثات الأخيرة.',
    shiftClickHint: 'انقر مع Shift لتثبيت المحادثة',
    noWorkspace: 'بلا مساحة عمل',
    noProject: 'لا يوجد مشروع',
    projectEmpty: 'لا توجد جلسات بعد',
    noSessions: 'لا توجد جلسات بعد',
    projects: {
      sectionLabel: 'المشاريع',
      newButton: 'مشروع جديد',
      createTitle: 'مشروع جديد',
      createDesc: 'سمِّ مساحة عمل وأضف مجلدًا أو أكثر.',
      renameTitle: 'إعادة تسمية المشروع',
      addFolderTitle: 'إضافة مجلد',
      namePlaceholder: 'مثال: Skunkworks',
      foldersLabel: 'المجلدات',
      ideaLabel: 'الفكرة',
      ideaPlaceholder: 'فيمَ يدور هذا المشروع؟ (يُحفظ في IDEA.md)',
      ideaGenerate: 'توليد فكرة',
      ideaGenerating: 'جارٍ التوليد…',
      ideaShuffle: 'خلط القوالب',
      noFolders: 'لم تُضَف مجلدات بعد.',
      addFolder: 'إضافة مجلد',
      primaryBadge: 'أساسي',
      removeFolder: 'إزالة',
      create: 'إنشاء',
      menu: 'إجراءات المشروع',
      menuRename: 'إعادة تسمية',
      menuAppearance: 'المظهر',
      noColor: 'بلا لون',
      menuAddFolder: 'إضافة مجلد',
      menuSetActive: 'تعيين كنشط',
      menuDelete: 'حذف',
      reveal: 'إظهار في المجلد',
      copyPath: 'نسخ المسار',
      removeFromSidebar: 'إخفاء من الشريط الجانبي',
      createFailed: 'تعذّر إنشاء المشروع',
      staleBackend:
        'حدّث تطبيق هرمس العامل لإنشاء المشاريع — تطبيقك العامل أقدم من تطبيق سطح المكتب هذا (الإعدادات ← التحديثات ← التطبيق العامل).',
      deleteConfirm: 'يزيل هذا المشروع المحفوظ من هرمس. تبقى الملفات ومستودعات git وأشجار العمل دون مساس.',
      startWork: 'شجرة عمل جديدة',
      newWorktreeTitle: 'شجرة عمل جديدة',
      newWorktreeDesc: 'سمِّ الفرع لهذه الشجرة.',
      branchPlaceholder: 'مثال: my-feature',
      startWorkFailed: 'تعذّر إنشاء شجرة العمل',
      convertBranch: 'تحويل فرع…',
      convertBranchTitle: 'تحويل فرع',
      convertBranchDesc: 'افتح الفروع المسحوبة، أو أنشئ شجرة عمل لفرع حرّ.',
      convertBranchPlaceholder: 'ابحث في الفروع…',
      convertBranchInstead: 'تحويل فرع موجود',
      branchOpenExisting: 'فتح',
      branchSwitchHome: 'تبديل الرئيسي',
      branchCreateWorktree: 'شجرة عمل جديدة',
      branchesLoading: 'جارٍ تحميل الفروع…',
      noBranches: 'لم تُعثَر على فروع',
      removeWorktree: 'إزالة شجرة العمل',
      removeWorktreeFailed: 'تعذّر إزالة شجرة العمل (تغييرات غير مودَعة؟)',
      removeWorktreeConfirm:
        'أزِلها من git (يحذف مجلد شجرة العمل؛ يبقى الفرع)، أو اكتفِ بإخفاء المسار من الشريط الجانبي وترك شجرة العمل على القرص.',
      removeWorktreeDirty:
        'لهذه الشجرة تغييرات غير مودَعة. أزِلها قسرًا (يتجاهل تلك التغييرات)، أو اكتفِ بإخفاء المسار وإبقائها على القرص.',
      forceRemove: 'إزالة قسرية',
      enter: label => `فتح ${label}`,
      reorder: label => `إعادة ترتيب ${label}`,
      toggle: label => `تبديل جلسات ${label}`,
      back: 'كل المشاريع'
    },
    newSessionIn: label => `جلسة جديدة في ${label}`,
    showMoreIn: (count, label) =>
      count === 1
        ? `إظهار جلسة أخرى في ${label}`
        : count === 2
          ? `إظهار جلستين أخريين في ${label}`
          : count <= 10
            ? `إظهار ${count} جلسات أخرى في ${label}`
            : `إظهار ${count} جلسة أخرى في ${label}`,
    loading: 'جارٍ التحميل…',
    loadMore: 'تحميل المزيد',
    loadCount: step =>
      step === 1
        ? 'تحميل جلسة أخرى'
        : step === 2
          ? 'تحميل جلستين أخريين'
          : step <= 10
            ? `تحميل ${step} جلسات أخرى`
            : `تحميل ${step} جلسة أخرى`,
    row: {
      pin: 'تثبيت',
      unpin: 'إلغاء التثبيت',
      copyId: 'نسخ المعرّف',
      export: 'تصدير',
      branchFrom: 'تفريع',
      rename: 'إعادة التسمية',
      archive: 'أرشفة',
      newWindow: 'نافذة جديدة',
      copyIdFailed: 'تعذر نسخ معرّف الجلسة',
      actionsFor: title => `إجراءات ${title}`,
      sessionActions: 'إجراءات الجلسة',
      sessionRunning: 'الجلسة تعمل',
      needsInput: 'تحتاج إدخالك',
      waitingForAnswer: 'في انتظار إجابتك',
      handoffOrigin: platform => `محوّلة من ${platform}`,
      renamed: 'أُعيدت التسمية',
      renameFailed: 'فشلت إعادة التسمية',
      renameTitle: 'إعادة تسمية الجلسة',
      renameDesc: 'أعط هذه المحادثة عنوانًا يسهل تذكره. اتركه فارغًا لمسحه.',
      untitledPlaceholder: 'جلسة بلا عنوان',
      ageNow: 'الآن',
      ageDay: 'ي',
      ageHour: 'س',
      ageMin: 'د'
    }
  },

  composer: {
    message: 'رسالة',
    wakingProfile: profile => `جارٍ تنشيط ${profile}…`,
    placeholderStarting: 'جارٍ بدء هرمس...',
    placeholderReconnecting: 'جارٍ إعادة الاتصال بهرمس…',
    placeholderFollowUp: 'أرسل متابعة',
    newSessionPlaceholders: [
      'ماذا سنبني؟',
      'أعط هرمس مهمة',
      'بماذا تفكر؟',
      'صف ما تحتاج إليه',
      'ما الذي ينبغي أن ننجزه؟',
      'اسأل عن أي شيء',
      'ابدأ بهدف'
    ],
    followUpPlaceholders: [
      'أرسل متابعة',
      'أضف سياقًا آخر',
      'حسّن الطلب',
      'ما التالي؟',
      'واصل العمل',
      'تعمّق أكثر',
      'عدّل أو تابع'
    ],
    startVoice: 'بدء محادثة صوتية',
    queueMessage: 'إضافة الرسالة إلى قائمة الانتظار',
    steer: 'توجيه التشغيل الحالي',
    stop: 'إيقاف',
    send: 'إرسال',
    speaking: 'يتحدث',
    transcribing: 'يفرّغ الصوت',
    thinking: 'يفكر',
    muted: 'مكتوم',
    listening: 'يستمع',
    muteMic: 'كتم الميكروفون',
    unmuteMic: 'تشغيل الميكروفون',
    stopListening: 'إيقاف الاستماع والإرسال',
    stopShort: 'إيقاف',
    endConversation: 'إنهاء المحادثة الصوتية',
    endShort: 'إنهاء',
    stopDictation: 'إيقاف الإملاء',
    transcribingDictation: 'جارٍ تفريغ الإملاء',
    voiceDictation: 'إملاء صوتي',
    speakReplies: 'قراءة الردود بصوت عالٍ',
    stopSpeakingReplies: 'إيقاف قراءة الردود بصوت عالٍ',
    lookupLoading: 'جارٍ البحث…',
    lookupNoMatches: 'لا نتائج مطابقة.',
    lookupTry: 'جرّب',
    lookupOr: 'أو',
    commonCommands: 'الأوامر الشائعة',
    hotkeys: 'الاختصارات',
    helpFooter: 'يفتح اللوحة الكاملة · ومفتاح الحذف ⌫ يغلقها',
    commandDescs: {
      '/help': 'القائمة الكاملة للأوامر والاختصارات',
      '/clear': 'بدء جلسة جديدة',
      '/resume': 'استئناف جلسة سابقة',
      '/details': 'التحكم في مستوى تفاصيل السجل',
      '/copy': 'نسخ التحديد أو آخر رسالة للمساعد',
      '/quit': 'الخروج من هرمس'
    },
    hotkeyDescs: {
      'composer.mention': 'الإشارة إلى الملفات والمجلدات والروابط ومستودع Git',
      'composer.slash': 'لوحة الأوامر المائلة',
      'composer.help': 'هذه المساعدة السريعة، احذفها لإغلاقها',
      'composer.sendNewline': 'إرسال · Shift+Enter لسطر جديد',
      'composer.sendQueued': 'إرسال الدورة التالية في الانتظار',
      'keybinds.openPanel': 'جميع اختصارات لوحة المفاتيح',
      'composer.cancel': 'إغلاق النافذة وإلغاء التشغيل',
      'composer.history': 'التنقل في النافذة أو السجل'
    },
    attachUrlTitle: 'إرفاق رابط',
    attachUrlDesc: 'سيجلب هرمس الصفحة ويدرجها في سياق هذه الدورة.',
    urlPlaceholder: 'https://example.com/post',
    urlHintPre: 'أدرج الرابط كاملًا، مثل ',
    attach: 'إرفاق',
    queued: count => `${count} في الانتظار`,
    queueStuckTitle: 'الرسالة المنتظرة لم تُرسَل',
    queueStuckBody: 'تعذّر إرسال دورة منتظرة مرارًا. لا تزال في قائمة الانتظار — حاول إرسالها مجددًا.',
    attachmentOnly: 'دورة مرفقات فقط',
    emptyTurn: 'دورة فارغة',
    attachments: count =>
      count === 1 ? 'مرفق واحد' : count === 2 ? 'مرفقان' : count <= 10 ? `${count} مرفقات` : `${count} مرفقًا`,
    editingInComposer: 'جارٍ التعديل في محرر الرسالة',
    editingQueuedInComposer: 'جارٍ تعديل دورة منتظرة في محرر الرسالة',
    queueEdit: 'تعديل',
    queueSendNext: 'التالية',
    queueSend: 'إرسال',
    queueDelete: 'حذف',
    previewUnavailable: 'المعاينة غير متاحة',
    previewLabel: label => `معاينة ${label}`,
    couldNotPreview: label => `تعذرت معاينة ${label}`,
    removeAttachment: label => `إزالة ${label}`,
    dictating: 'يملي',
    preparingAudio: 'جارٍ تجهيز الصوت',
    speakingResponse: 'جارٍ نطق الرد',
    readingAloud: 'جارٍ القراءة بصوت عالٍ',
    themeSuggestions: 'اقتراحات سمات سطح المكتب',
    noMatchingThemes: 'لا توجد سمات مطابقة.',
    themeTryPre: 'جرّب ',
    themeTryPost: '.',
    attachLabel: 'إرفاق',
    files: 'ملفات…',
    folder: 'مجلد…',
    images: 'صور…',
    pasteImage: 'لصق صورة',
    url: 'رابط…',
    promptSnippets: 'مقتطفات موجّهات…',
    tipPre: 'تلميح: اكتب ',
    tipPost: ' للإشارة إلى الملفات داخل النص.',
    snippetsTitle: 'مقتطفات الموجّهات',
    snippetsDesc: 'اختر موجّهًا أوليًا لإدراجه في محرر الرسالة.',
    dropFiles: 'أفلت الملفات لإرفاقها',
    dropSession: 'أفلت لربط هذه المحادثة',
    snippets: {
      codeReview: {
        label: 'مراجعة الشيفرة',
        description: 'راجع التغيير الحالي بحثًا عن التراجعات والحالات الطرفية المفقودة والاختبارات الناقصة.',
        text: 'راجع هذا بحثًا عن الأخطاء والتراجعات والاختبارات الناقصة.'
      },
      implementationPlan: {
        label: 'خطة التنفيذ',
        description: 'ضع نهجًا قبل تعديل الشيفرة ليبقى التغيير مركزًا.',
        text: 'ضع خطة تنفيذ موجزة قبل تغيير الشيفرة.'
      },
      explainThis: {
        label: 'اشرح هذا',
        description: 'اشرح كيفية عمل الشيفرة المحددة وأشر إلى الملفات الأساسية.',
        text: 'اشرح كيفية عمل هذا وأشر إلى الملفات الأساسية.'
      }
    }
  },

  statusStack: {
    agents: 'الوكلاء',
    background: count => `${count} في الخلفية`,
    subagents: count =>
      count === 1
        ? 'وكيل فرعي واحد'
        : count === 2
          ? 'وكيلان فرعيان'
          : count <= 10
            ? `${count} وكلاء فرعيون`
            : `${count} وكيلًا فرعيًا`,
    todos: (done, total) => `المهام ${done}/${total}`,
    running: 'قيد التشغيل',
    stop: 'إيقاف',
    dismiss: 'إخفاء',
    exit: code => `الخروج ${code}`,
    coding: {
      title: 'شجرة العمل',
      noBranch: 'لا فرع',
      detached: 'منفصل',
      clean: 'نظيفة',
      changed: count => `${count} مُعدَّل`,
      ahead: count => `${count} متقدّم`,
      behind: count => `${count} متأخّر`,
      review: 'مراجعة',
      close: 'إغلاق',
      openChanges: 'فتح التغييرات',
      openFile: 'فتح الملف',
      stage: 'تجهيز',
      unstage: 'إلغاء التجهيز',
      stageAll: 'تجهيز الكل',
      viewAsTree: 'عرض كشجرة',
      viewAsList: 'عرض كقائمة',
      revert: 'تراجع',
      revertAll: 'التراجع عن الكل',
      revertConfirm: 'هل تتجاهل التغييرات على هذا الملف وتعيده إلى الحالة المودَعة؟ لا يمكن التراجع عن هذا.',
      revertAllConfirm: 'هل تتجاهل كل التغييرات وتعيد الملفات إلى الحالة المودَعة؟ لا يمكن التراجع عن هذا.',
      staged: 'مُجهَّز',
      noChanges: 'لا تغييرات',
      notRepo: 'ليس مستودع git',
      noDiff: 'لا فروق لعرضها',
      scopeUncommitted: 'غير مودَع',
      scopeBranch: 'الفرع',
      scopeLastTurn: 'الخطوة الأخيرة',
      commit: 'إيداع',
      commitAndPush: 'إيداع ودفع',
      commitPlaceholder: 'الرسالة (⌘↵ للإيداع)',
      generateCommitMessage: 'توليد رسالة الإيداع',
      stopGenerating: 'إيقاف التوليد',
      createPr: 'إنشاء طلب سحب',
      openPr: 'فتح طلب السحب',
      ghMissing: 'ثبّت واجهة GitHub السطرية (gh) وسجّل الدخول لفتح طلبات السحب',
      agentShip: 'اطلب من هرمس فتح طلب سحب',
      agentShipPrompt: 'راجع التغييرات الحالية، وأودِعها برسالة إيداع تقليدية واضحة، وادفع الفرع، وافتح طلب سحب.',
      newBranch: 'فرع جديد',
      branchOffFrom: base => `فرع جديد من ${base}`,
      switchTo: branch => `التبديل إلى ${branch}`,
      switchFailed: branch => `تعذّر التبديل إلى ${branch}`,
      worktrees: 'أشجار العمل'
    }
  },

  updates: {
    stages: {
      idle: 'جارٍ الاستعداد…',
      prepare: 'جارٍ الاستعداد…',
      fetch: 'جارٍ التنزيل…',
      pull: 'أوشكنا على الانتهاء…',
      pydeps: 'جارٍ الإنهاء…',
      update: 'جارٍ تحديث هرمس…',
      rebuild: 'جارٍ إعادة بناء تطبيق سطح المكتب…',
      restart: 'جارٍ إعادة تشغيل هرمس…',
      done: 'اكتمل التحديث',
      manual: 'حدّث من الطرفية',
      guiSkew: 'حدّث تطبيق سطح المكتب',
      error: 'توقف التحديث'
    },
    checking: 'جارٍ البحث عن تحديثات…',
    checkFailedTitle: 'تعذر التحقق من التحديثات',
    tryAgain: 'إعادة المحاولة',
    notAvailableTitle: 'لا يتوفر تحديث',
    unsupportedMessage: 'لا يستطيع هذا الإصدار من هرمس تحديث نفسه من داخل التطبيق.',
    connectionRetry: 'تحقق من اتصالك وأعد المحاولة.',
    latestBody: 'لديك أحدث إصدار.',
    latestBodyBackend: 'تعمل الواجهة الخلفية بأحدث إصدار.',
    allSetTitle: 'كل شيء جاهز',
    availableTitle: 'يتوفر تحديث جديد',
    availableBody: 'إصدار جديد من هرمس جاهز للتثبيت.',
    availableTitleBackend: 'يتوفر تحديث للواجهة الخلفية',
    availableBodyBackend: 'إصدار أحدث من واجهة هرمس الخلفية المتصلة جاهز للتثبيت.',
    availableBodyNoChangelog: 'يتوفر إصدار أحدث، لكن ملاحظات الإصدار غير متاحة لهذا النوع من التثبيت.',
    updateNow: 'التحديث الآن',
    maybeLater: 'لاحقًا',
    moreChanges: count =>
      count === 1
        ? '+ يتضمن تغييرًا واحدًا آخر.'
        : count === 2
          ? '+ يتضمن تغييرين آخرين.'
          : count <= 10
            ? `+ يتضمن ${count} تغييرات أخرى.`
            : `+ يتضمن ${count} تغييرًا آخر.`,
    manualTitle: 'التحديث من الطرفية',
    manualBody: 'ثبّتَّ هرمس من سطر الأوامر، ولذلك تجري تحديثاته هناك أيضًا. الصق هذا في الطرفية:',
    manualPickedUp: 'سيستخدم هرمس الإصدار الجديد في المرة التالية التي تشغله فيها.',
    guiSkewTitle: 'حدّث تطبيق سطح المكتب',
    guiSkewBody:
      'حُدّثت الخلفية، لكنّ حزمة تطبيق سطح المكتب هذه لم تتغيّر. حدّث تطبيق هرمس لسطح المكتب أو أعد تثبيته (ملف AppImage / ‎.deb / ‎.rpm) ليتطابقا.',
    copy: 'نسخ',
    copied: 'نُسخ',
    done: 'تم',
    applyingBody: 'ستتولى أداة تحديث هرمس العملية في نافذتها، ثم تعيد فتح هرمس عند الانتهاء.',
    applyingBodyBackend:
      'تطبّق الواجهة الخلفية البعيدة التحديث وستُعاد تشغيلها. يعيد هرمس الاتصال تلقائيًا عند عودتها.',
    applyingClose: 'سيُغلق هرمس لتطبيق التحديث.',
    errorTitle: 'لم يكتمل التحديث',
    errorBody: 'لم تُفقد أي بيانات. يمكنك إعادة المحاولة الآن.',
    notNow: 'ليس الآن',
    applyStatus: {
      preparing: 'جارٍ تحديث الواجهة الخلفية…',
      pulling: 'جارٍ تحديث الواجهة الخلفية…',
      restarting: 'جارٍ إعادة تشغيل الواجهة الخلفية لتحميل التحديث…',
      notAvailable: 'لا يتوفر تحديث لهذه الواجهة الخلفية.',
      failed: 'فشل تحديث الواجهة الخلفية.',
      noReturn: 'لم تعد الواجهة الخلفية إلى العمل. ربما لم يكتمل التحديث؛ تحقق من مضيفها.'
    }
  },

  install: {
    stageStates: {
      pending: 'قيد الانتظار',
      running: 'جارٍ التثبيت',
      succeeded: 'تم',
      skipped: 'تم التخطي',
      failed: 'فشل'
    },
    oneTimeTitle: 'يحتاج هرمس إلى تثبيت لمرة واحدة',
    unsupportedDesc: platform =>
      `التثبيت التلقائي عند أول تشغيل غير متاح على ${platform} بعد. افتح الطرفية وشغّل الأمر أدناه ثم أعد تشغيل التطبيق. ستتجاوز عمليات التشغيل التالية هذه الخطوة.`,
    installCommand: 'أمر التثبيت',
    copyCommand: 'نسخ الأمر',
    viewDocs: 'عرض توثيق التثبيت',
    installTo: 'سيُثبّت في',
    retryAfterRun: 'شغّلته؛ أعد المحاولة',
    failedTitle: 'فشل التثبيت',
    settingUpTitle: 'جارٍ إعداد وكيل هرمس',
    finishingTitle: 'جارٍ الإنهاء',
    failedDesc:
      'فشلت إحدى خطوات التثبيت. قد يحدث هذا على ويندوز إذا كان مثيل آخر من سطر أوامر هرمس أو سطح المكتب يعمل. أوقف أي مثيلات عاملة ثم أعد المحاولة. راجع التفاصيل أدناه أو سجل سطح المكتب للنص الكامل.',
    activeDesc:
      'هذا إعداد لمرة واحدة. ينزّل مثبّت هرمس التبعيات ويضبط جهازك. ستتجاوز عمليات التشغيل التالية هذه الخطوة.',
    progress: (completed, total) =>
      total === 1
        ? `اكتملت ${completed} من خطوة واحدة`
        : total === 2
          ? `اكتملت ${completed} من خطوتين`
          : total <= 10
            ? `اكتملت ${completed} من ${total} خطوات`
            : `اكتملت ${completed} من ${total} خطوة`,
    currentStage: stage => ` — الآن: ${stage}`,
    fetchingManifest: 'جارٍ جلب بيان المثبّت...',
    error: 'خطأ',
    hideOutput: 'إخفاء مخرجات المثبّت',
    showOutput: 'إظهار مخرجات المثبّت',
    lines: count =>
      count === 1 ? 'سطر واحد' : count === 2 ? 'سطران' : count <= 10 ? `${count} أسطر` : `${count} سطرًا`,
    noOutput: 'لا توجد مخرجات بعد.',
    cancelling: 'جارٍ الإلغاء...',
    cancelInstall: 'إلغاء التثبيت',
    transcriptSaved: 'حُفظ النص الكامل في',
    copiedOutput: 'نُسخ!',
    copyOutput: 'نسخ المخرجات',
    reloadRetry: 'إعادة التحميل والمحاولة'
  },

  onboarding: {
    headerTitle: 'لنجهّز لك وكيل هرمس',
    headerDesc: 'صل مزوّد نموذج لبدء المحادثة. لا تتطلب معظم الخيارات سوى نقرة واحدة.',
    preparingInstall: 'ينهي هرمس التثبيت. يستغرق ذلك عادة أقل من دقيقة عند التشغيل الأول.',
    starting: 'جارٍ بدء هرمس…',
    lookingUpProviders: 'جارٍ البحث عن المزوّدين...',
    collapse: 'طي',
    otherProviders: 'مزوّدون آخرون',
    haveApiKey: 'لدي مفتاح واجهة برمجية',
    chooseLater: 'سأختار مزوّدًا لاحقًا',
    recommended: 'موصى به',
    connected: 'متصل',
    featuredPitch: 'اشتراك واحد وأكثر من 300 نموذج متقدم؛ الطريقة الموصى بها لتشغيل هرمس',
    openRouterPitch: 'مفتاح واحد ومئات النماذج؛ خيار افتراضي جيد',
    apiKeyOptions: {
      fireworks: {
        short: 'واجهة نماذج مباشرة',
        description: 'وصول مباشر إلى النماذج المستضافة لدى Fireworks AI.'
      },
      openrouter: {
        short: 'مفتاح واحد ونماذج كثيرة',
        description: 'يستضيف مئات النماذج خلف مفتاح واحد. خيار افتراضي جيد للتثبيتات الجديدة.'
      },
      openai: { short: 'نماذج فئة GPT', description: 'وصول مباشر إلى نماذج OpenAI.' },
      gemini: { short: 'نماذج Gemini', description: 'وصول مباشر إلى نماذج Google Gemini.' },
      xai: { short: 'نماذج Grok', description: 'وصول مباشر إلى نماذج xAI Grok.' },
      local: {
        short: 'مستضاف ذاتيًا',
        description: 'وجّه هرمس إلى نقطة نهاية محلية أو مستضافة ذاتيًا متوافقة مع OpenAI، مثل vLLM وllama.cpp وOllama.'
      }
    },
    backToSignIn: 'العودة إلى تسجيل الدخول',
    getKey: 'الحصول على مفتاح',
    replaceCurrent: 'استبدال القيمة الحالية',
    pasteApiKey: 'لصق مفتاح الواجهة البرمجية',
    localApiKeyPlaceholder: 'مفتاح الواجهة البرمجية (اختياري — فقط إن كانت نقطتك تتطلبه)',
    couldNotSave: 'تعذر حفظ بيانات الاعتماد.',
    connecting: 'جارٍ الاتصال',
    update: 'تحديث',
    flowSubtitles: {
      pkce: 'يفتح المتصفح لتسجيل الدخول ثم يتابع هنا',
      device_code: 'يفتح صفحة تحقق في المتصفح، ويتصل هرمس تلقائيًا',
      loopback: 'يفتح المتصفح لتسجيل الدخول، ويتصل هرمس تلقائيًا',
      external: 'سجّل الدخول مرة واحدة في الطرفية ثم عد إلى المحادثة'
    },
    startingSignIn: provider => `جارٍ بدء تسجيل الدخول إلى ${provider}...`,
    verifyingCode: provider => `جارٍ التحقق من رمزك لدى ${provider}...`,
    connectedProvider: provider => `تم ربط ${provider}`,
    connectedPicking: provider => `تم ربط ${provider}. جارٍ اختيار نموذج افتراضي...`,
    signInFailed: 'فشل تسجيل الدخول. أعد المحاولة.',
    pickDifferentProvider: 'اختيار مزوّد آخر',
    signInWith: provider => `تسجيل الدخول عبر ${provider}`,
    openedBrowser: provider => `فتحنا ${provider} في المتصفح.`,
    authorizeThere: 'فوّض هرمس هناك.',
    copyAuthCode: 'انسخ رمز التفويض والصقه أدناه.',
    pasteAuthCode: 'لصق رمز التفويض',
    reopenAuthPage: 'إعادة فتح صفحة التفويض',
    autoBrowser: provider => `فتحنا ${provider} في المتصفح. فوّض هرمس هناك وسيجري الاتصال تلقائيًا دون نسخ أو لصق.`,
    reopenSignInPage: 'إعادة فتح صفحة تسجيل الدخول',
    waitingAuthorize: 'في انتظار التفويض...',
    externalPending: provider =>
      `يسجّل ${provider} الدخول عبر أداة سطر الأوامر الخاصة به. شغّل هذا الأمر في الطرفية ثم عد واختر «سجّلت الدخول»:`,
    signedIn: 'سجّلت الدخول',
    deviceCodeOpened: provider => `فتحنا ${provider} في المتصفح. أدخل هذا الرمز هناك:`,
    reopenVerification: 'إعادة فتح صفحة التحقق',
    copy: 'نسخ',
    defaultModel: 'النموذج الافتراضي',
    freeTier: 'الفئة المجانية',
    pro: 'احترافي',
    free: 'مجاني',
    price: (input, output) => `${input} إدخال / ${output} إخراج لكل مليون وحدة`,
    change: 'تغيير',
    startChatting: 'بدء',
    docs: provider => `توثيق ${provider}`
  },

  modelPicker: {
    title: 'تبديل النموذج',
    current: 'الحالي:',
    unknown: '(غير معروف)',
    search: 'تصفية المزوّدين والنماذج...',
    noModels: 'لم يُعثر على نماذج.',
    addProvider: 'إضافة مزوّد',
    loadFailed: 'تعذر تحميل النماذج',
    noAuthenticatedProviders: 'لا يوجد مزوّدون موثّقون.',
    pro: 'احترافي',
    proNeedsSubscription: 'تحتاج النماذج الاحترافية إلى اشتراك Nous مدفوع.',
    free: 'مجاني',
    freeTier: 'الفئة المجانية',
    priceTitle: 'سعر الإدخال والإخراج لكل مليون وحدة'
  },

  modelVisibility: {
    title: 'النماذج',
    search: 'البحث في النماذج',
    noAuthenticatedProviders: 'لا يوجد مزوّدون موثّقون.',
    addProvider: 'إضافة مزوّد…'
  },

  shell: {
    windowControls: 'عناصر تحكم النافذة',
    paneControls: 'عناصر تحكم الألواح',
    appControls: 'عناصر تحكم التطبيق',
    modelMenu: {
      search: 'البحث في النماذج',
      noModels: 'لم يُعثر على نماذج',
      editModels: 'تعديل النماذج…',
      refreshModels: 'تحديث النماذج',
      fast: 'سريع',
      medium: 'متوسط'
    },
    modelOptions: {
      noOptions: 'لا خيارات لهذا النموذج',
      options: 'الخيارات',
      thinking: 'التفكير',
      fast: 'سريع',
      effort: 'الجهد',
      minimal: 'أدنى',
      low: 'منخفض',
      medium: 'متوسط',
      high: 'مرتفع',
      xhigh: 'عالٍ جدًا',
      max: 'أقصى',
      ultra: 'خارق',
      updateFailed: 'فشل تحديث خيار النموذج',
      fastFailed: 'فشل تحديث الوضع السريع'
    },
    gatewayMenu: {
      gateway: 'البوابة',
      connected: 'متصل',
      connecting: 'جارٍ الاتصال',
      offline: 'غير متصل',
      inferenceReady: 'الاستنتاج جاهز',
      inferenceNotReady: 'الاستنتاج غير جاهز',
      checkingInference: 'جارٍ التحقق من الاستنتاج',
      disconnected: 'انقطع الاتصال',
      openSystem: 'فتح لوحة النظام',
      connection: label => `الاتصال: ${label}`,
      recentActivity: 'النشاط الأخير',
      viewAllLogs: 'عرض جميع السجلات ←',
      messagingPlatforms: 'منصات المراسلة'
    },
    statusbar: {
      unknown: 'غير معروف',
      restart: 'إعادة التشغيل',
      update: 'تحديث',
      updateInProgress: 'التحديث جارٍ',
      commitsBehind: (count, branch) =>
        count === 1
          ? `متأخر عن ${branch} بإيداع واحد`
          : count === 2
            ? `متأخر عن ${branch} بإيداعين`
            : count <= 10
              ? `متأخر عن ${branch} بـ ${count} إيداعات`
              : `متأخر عن ${branch} بـ ${count} إيداعًا`,
      tokensShort: value => `${value} وحدة`,
      reasoningShort: { none: 'إيقاف', minimal: 'أدنى', low: 'منخفض', medium: 'متوسط', high: 'عالٍ', xhigh: 'فائق', max: 'أقصى', ultra: 'خارق' },
      desktopVersion: version => `هرمس لسطح المكتب ${version}`,
      backendVersion: version => `الواجهة الخلفية ${version}`,
      clientLabel: version => `العميل ${version}`,
      backendLabel: version => `الواجهة الخلفية ${version}`,
      commit: sha => `الإيداع ${sha}`,
      branch: branch => `الفرع ${branch}`,
      closeCommandCenter: 'إغلاق مركز الأوامر',
      openCommandCenter: 'فتح مركز الأوامر',
      showTerminal: 'إظهار الطرفية',
      hideTerminal: 'إخفاء الطرفية',
      gateway: 'البوابة',
      gatewayReady: 'جاهزة',
      gatewayNeedsSetup: 'تحتاج إعدادًا',
      gatewayChecking: 'جارٍ التحقق',
      gatewayConnecting: 'جارٍ الاتصال',
      gatewayOffline: 'غير متصلة',
      gatewayRestarting: 'إعادة التشغيل…',
      gatewayTitle: 'حالة بوابة استنتاج هرمس',
      agents: 'الوكلاء',
      closeAgents: 'إغلاق الوكلاء',
      openAgents: 'فتح الوكلاء',
      subagents: count =>
        count === 1
          ? 'وكيل فرعي واحد'
          : count === 2
            ? 'وكيلان فرعيان'
            : count <= 10
              ? `${count} وكلاء فرعيين`
              : `${count} وكيلًا فرعيًا`,
      failed: count => (count === 1 ? 'فشل واحد' : count === 2 ? 'فشل اثنان' : `فشل ${count}`),
      running: count => (count === 1 ? 'يعمل واحد' : count === 2 ? 'يعمل اثنان' : `يعمل ${count}`),
      cron: 'المهام المجدولة',
      openCron: 'فتح المهام المجدولة',
      starmap: 'مخطّط الذاكرة',
      openStarmap: 'فتح مخطّط الذاكرة',
      turnRunning: 'تعمل',
      currentTurnElapsed: 'المدة المنقضية للدورة الحالية',
      contextUsage: 'استخدام السياق',
      contextUsagePanel: {
        categories: {
          conversation: 'المحادثة',
          mcp: 'MCP',
          memory: 'الذاكرة',
          rules: 'القواعد',
          skills: 'المهارات',
          subagent_definitions: 'تعريفات الوكلاء الفرعيّين',
          system_prompt: 'موجّه النظام',
          tool_definitions: 'تعريفات الأدوات'
        },
        empty: 'لا بيانات سياق بعد',
        loading: 'جارٍ تحميل التفصيل…',
        percentFull: percent => `ممتلئ ${percent}٪`,
        title: 'استخدام السياق',
        tokenSummary: (used, max) => `${used} / ${max} رمز`
      },
      openContextUsage: 'فتح تفصيل استخدام السياق',
      session: 'الجلسة',
      runtimeSessionElapsed: 'المدة المنقضية لجلسة التشغيل',
      yoloOn:
        'وضع YOLO مفعّل؛ ستتم الموافقة تلقائيًا على الأوامر الخطرة. انقر لتعطيله، أو انقر مع Shift لتبديله عامًا.',
      yoloOff: 'وضع YOLO معطّل؛ انقر للموافقة تلقائيًا على الأوامر الخطرة، أو انقر مع Shift لتبديله عامًا.',
      modelNone: 'لا شيء',
      noModel: 'لا نموذج',
      switchModel: 'تبديل النموذج',
      openModelPicker: 'فتح منتقي النموذج',
      modelTitle: (provider, model) => `النموذج · ${provider}: ${model}`,
      providerModelTitle: (provider, model) => `${provider} · ${model}`
    }
  },

  rightSidebar: {
    aria: 'الشريط الجانبي الأيمن',
    panelsAria: 'لوحات الشريط الجانبي الأيمن',
    files: 'نظام الملفات',
    terminal: 'الطرفية',
    noFolderSelected: 'لم يُحدد مجلد',
    changeCwdTitle: 'تغيير مجلد العمل',
    remotePickerTitle: 'اختيار مجلد بعيد',
    remotePickerDescription: 'تصفح المجلدات على الواجهة الخلفية المتصلة.',
    remotePickerSelect: 'اختيار المجلد',
    folderTip: cwd => `${cwd} — انقر لتغيير المجلد`,
    openFolder: 'فتح مجلد',
    refreshTree: 'تحديث الشجرة',
    collapseAll: 'طي جميع المجلدات',
    previewUnavailable: 'المعاينة غير متاحة',
    couldNotPreview: path => `تعذرت معاينة ${path}`,
    noProjectTitle: 'لا يوجد مشروع',
    noProjectBody: 'اضبط مجلد عمل من شريط الحالة لتصفح الملفات.',
    noProjectOpen: 'لا مشروع مفتوح',
    noDiffs: 'لا فروق',
    unreadableTitle: 'غير قابل للقراءة',
    unreadableBody: error => `تعذرت قراءة هذا المجلد (${error}).`,
    emptyTitle: 'فارغ',
    emptyBody: 'هذا المجلد فارغ.',
    treeErrorTitle: 'خطأ في الشجرة',
    treeErrorBody: 'واجهت شجرة الملفات خطأ أثناء عرض هذا المجلد.',
    tryAgain: 'إعادة المحاولة',
    loadingTree: 'جارٍ تحميل شجرة الملفات',
    loadingFiles: 'جارٍ تحميل الملفات',
    terminalHide: 'إخفاء الطرفية',
    terminalsAria: 'الطرفيات',
    terminalNew: 'طرفية جديدة',
    terminalCloseOthers: 'إغلاق الأخرى',
    terminalCloseAll: 'إغلاق الكل',
    addToChat: 'إضافة إلى المحادثة'
  },

  preview: {
    tab: 'المعاينة',
    closeTab: label => `إغلاق ${label}`,
    closeOthers: 'إغلاق الأخرى',
    closeToRight: 'إغلاق ما على اليمين',
    closeAll: 'إغلاق الكل',
    closePane: 'إغلاق لوحة المعاينة',
    loading: 'جارٍ تحميل المعاينة',
    unavailable: 'المعاينة غير متاحة',
    opening: 'جارٍ الفتح...',
    hide: 'إخفاء',
    openPreview: 'فتح المعاينة',
    openInBrowser: 'فتح في المتصفّح',
    linkHint: 'النقر مع ⌘/Ctrl لفتح لوحة المعاينة',
    sourceLineTitle: 'انقر للتحديد · انقر مع Shift للتوسيع · اسحب إلى محرر الرسالة',
    source: 'المصدر',
    renderedPreview: 'المعاينة',
    diff: 'الفروق',
    unknownSize: 'حجم غير معروف',
    binaryTitle: 'يبدو أن هذا ملف ثنائي',
    binaryBody: label => `قد تُظهر معاينة ${label} نصًا غير مقروء.`,
    largeTitle: 'هذا الملف كبير',
    largeBody: (label, size) => `حجم ${label} هو ${size}. سيعرض هرمس أول 512 كيلوبايت فقط.`,
    previewAnyway: 'المعاينة على أي حال',
    truncated: 'يُعرض أول 512 كيلوبايت.',
    noInlineTitle: 'لا تتوفر معاينة مضمنة',
    noInlineBody: mimeType => `يمكن إرفاق ${mimeType || 'هذا النوع من الملفات'} في السياق.`,
    edit: 'تعديل',
    editing: 'جارٍ التعديل',
    unsavedChanges: 'تغييرات غير محفوظة',
    saveFailed: message => `تعذّر الحفظ: ${message}`,
    diskChangedTitle: 'تغيّر الملف على القرص',
    diskChangedBody: 'تغيّر هذا الملف منذ أن فتحته. هل تكتب فوقه بنسختك، أم تتجاهل تعديلاتك وتعيد التحميل؟',
    overwrite: 'الكتابة فوقه',
    discardReload: 'تجاهل وإعادة التحميل',
    console: {
      deselect: 'إلغاء تحديد الإدخال',
      select: 'تحديد الإدخال',
      copyFailed: 'تعذر نسخ مخرجات وحدة التحكم',
      copyEntry: 'نسخ هذا الإدخال',
      sendEntry: 'إرسال هذا الإدخال إلى المحادثة',
      messages: count =>
        count === 1
          ? 'رسالة واحدة في وحدة التحكم'
          : count === 2
            ? 'رسالتان في وحدة التحكم'
            : count <= 10
              ? `${count} رسائل في وحدة التحكم`
              : `${count} رسالة في وحدة التحكم`,
      resize: 'تغيير حجم وحدة تحكم المعاينة',
      title: 'وحدة تحكم المعاينة',
      selected: count =>
        count === 1
          ? 'إدخال واحد محدد'
          : count === 2
            ? 'إدخالان محددان'
            : count <= 10
              ? `${count} إدخالات محددة`
              : `${count} إدخالًا محددًا`,
      sendToChat: 'إرسال إلى المحادثة',
      copySelected: 'نسخ المحدد إلى الحافظة',
      copyAll: 'نسخ الكل إلى الحافظة',
      copy: 'نسخ',
      clear: 'مسح',
      empty: 'لا توجد رسائل في وحدة التحكم بعد.',
      promptHeader: 'وحدة تحكم المعاينة:',
      sentTitle: 'أُرسل إلى المحادثة',
      sentMessage: count =>
        count === 1
          ? 'أُضيف إدخال سجل واحد إلى محرر الرسالة'
          : count === 2
            ? 'أُضيف إدخالا سجل إلى محرر الرسالة'
            : count <= 10
              ? `أُضيفت ${count} إدخالات سجل إلى محرر الرسالة`
              : `أُضيف ${count} إدخال سجل إلى محرر الرسالة`
    },
    web: {
      appFailedToBoot: 'فشل بدء تطبيق المعاينة',
      serverNotFound: 'لم يُعثر على الخادم',
      failedToLoad: 'فشل تحميل المعاينة',
      tryAgain: 'إعادة المحاولة',
      restarting: 'جارٍ إعادة تشغيل هرمس...',
      askRestart: 'اطلب من هرمس إعادة تشغيل الخادم',
      lookingRestart: taskId => `يبحث هرمس عن خادم معاينة لإعادة تشغيله (${taskId})`,
      restartingTitle: 'جارٍ إعادة تشغيل خادم المعاينة',
      restartingMessage: 'يعمل هرمس في الخلفية. راقب وحدة تحكم المعاينة لمتابعة التقدم.',
      startRestartFailed: message => `تعذر بدء إعادة تشغيل الخادم: ${message}`,
      restartFailed: 'فشلت إعادة تشغيل الخادم',
      hideConsole: 'إخفاء وحدة تحكم المعاينة',
      showConsole: 'إظهار وحدة تحكم المعاينة',
      hideDevTools: 'إخفاء أدوات مطور المعاينة',
      openDevTools: 'فتح أدوات مطور المعاينة',
      finishedRestarting: message => `أنهى هرمس إعادة تشغيل خادم المعاينة${message ? `: ${message}` : ''}`,
      failedRestarting: message => `فشلت إعادة تشغيل الخادم: ${message}`,
      unknownError: 'خطأ غير معروف',
      restartedTitle: 'أُعيد تشغيل خادم المعاينة',
      reloadingNow: 'جارٍ إعادة تحميل المعاينة الآن.',
      restartFailedTitle: 'فشلت إعادة تشغيل المعاينة',
      restartFailedMessage: 'تعذر على هرمس إعادة تشغيل الخادم.',
      stillWorking: 'لا يزال هرمس يعمل، لكن لم تصل نتيجة إعادة التشغيل بعد. ربما يعمل أمر الخادم في المقدمة لا في الخلفية.',
      workspaceReloading: 'تغيرت مساحة العمل؛ جارٍ إعادة تحميل المعاينة',
      fileChanged: url => `تغير ملف؛ جارٍ إعادة تحميل المعاينة: ${url}`,
      filesChanged: (count, url) =>
        count === 1
          ? `تغيير واحد في الملفات؛ جارٍ إعادة تحميل المعاينة: ${url}`
          : count === 2
            ? `تغييران في الملفات؛ جارٍ إعادة تحميل المعاينة: ${url}`
            : count <= 10
              ? `${count} تغييرات في الملفات؛ جارٍ إعادة تحميل المعاينة: ${url}`
              : `${count} تغييرًا في الملفات؛ جارٍ إعادة تحميل المعاينة: ${url}`,
      watchFailed: message => `تعذرت مراقبة ملف المعاينة: ${message}`,
      moduleMimeDescription:
        'تُقدّم سكربتات الوحدات بنوع MIME غير صحيح. يعني ذلك غالبًا أن خادم ملفات ثابتًا يقدّم تطبيق Vite أو React بدل خادم تطوير المشروع.',
      loadFailedConsole: (code, message) => `فشل التحميل${code ? ` (${code})` : ''}: ${message}`,
      unreachableDescription: 'تعذر الوصول إلى صفحة المعاينة.',
      openTarget: url => `فتح ${url}`,
      fallbackTitle: 'المعاينة'
    }
  },

  assistant: {
    thread: {
      loadingSession: 'جارٍ تحميل الجلسة',
      showEarlier: 'عرض الرسائل الأقدم',
      expandMessage: 'توسيع الرسالة',
      scrollToBottom: 'التمرير إلى الأسفل',
      loadingResponse: 'يحمّل هرمس ردًا',
      resumeWhenBackgroundDone: count =>
        count === 1
          ? 'سيُستأنف عند انتهاء المهمّة الخلفيّة'
          : count === 2
            ? 'سيُستأنف عند انتهاء المهمّتين الخلفيّتين'
            : count <= 10
              ? `سيُستأنف عند انتهاء ${count} مهامّ خلفيّة`
              : `سيُستأنف عند انتهاء ${count} مهمّةً خلفيّة`,
      thinking: 'يفكر',
      today: time => `اليوم، ${time}`,
      yesterday: time => `أمس، ${time}`,
      copy: 'نسخ',
      refresh: 'تحديث',
      moreActions: 'إجراءات أخرى',
      branchNewChat: 'تفريع إلى محادثة جديدة',
      dismissError: 'تجاهل الخطأ',
      readAloudFailed: 'فشلت القراءة بصوت عالٍ',
      preparingAudio: 'جارٍ تجهيز الصوت...',
      stopReading: 'إيقاف القراءة',
      readAloud: 'قراءة بصوت عالٍ',
      editMessage: 'تعديل الرسالة',
      stop: 'إيقاف',
      restorePrevious: 'استعادة نقطة التحقق السابقة',
      restoreCheckpoint: 'استعادة نقطة التحقق',
      restoreFromHere: 'استعادة نقطة التحقق — إعادة التشغيل من هذه المطالبة',
      restoreTitle: 'الاستعادة إلى نقطة التحقق هذه؟',
      restoreBody: 'يُزال كل ما بعد هذه المطالبة من المحادثة، وتُشغَّل المطالبة مجددًا من هنا.',
      restoreConfirm: 'استعادة وإعادة تشغيل',
      restoreNext: 'استعادة نقطة التحقق التالية',
      goForward: 'تقدم',
      sendEdited: 'إرسال الرسالة المعدلة',
      attachingFile: 'جارٍ إرفاق الملف…'
    },
    approval: {
      gatewayDisconnected: 'بوابة هرمس غير متصلة',
      sendFailed: 'تعذر إرسال رد الموافقة',
      run: 'تشغيل',
      command: 'الأمر',
      moreOptions: 'خيارات موافقة أخرى',
      allowSession: 'السماح في هذه الجلسة',
      alwaysAllowMenu: 'السماح دائمًا…',
      jumpToApproval: 'مطلوب موافقة',
      reject: 'رفض',
      alwaysTitle: 'السماح بهذا الأمر دائمًا؟',
      alwaysDescription: pattern =>
        `يضيف ذلك النمط «${pattern}» إلى قائمة السماح الدائمة (~/.hermes/config.yaml). لن يطلب هرمس موافقتك مجددًا على أوامر كهذه في هذه الجلسة أو الجلسات اللاحقة.`,
      alwaysAllow: 'السماح دائمًا'
    },
    clarify: {
      notReady: 'طلب الاستيضاح غير جاهز بعد',
      gatewayDisconnected: 'بوابة هرمس غير متصلة',
      sendFailed: 'تعذر إرسال رد الاستيضاح',
      loadingQuestion: 'جارٍ تحميل السؤال…',
      other: 'إجابة أخرى (اكتبها بنفسك)',
      placeholder: 'اكتب إجابتك…',
      skip: 'تخطي',
      skipped: 'مُتخطّى',
      continueLabel: 'متابعة'
    },
    tool: {
      code: 'الشيفرة',
      copyCode: 'نسخ الشيفرة',
      renderingImage: 'جارٍ عرض الصورة',
      copyOutput: 'نسخ المخرجات',
      copyCommand: 'نسخ الأمر',
      copyContent: 'نسخ المحتوى',
      copyUrl: 'نسخ الرابط',
      copyResults: 'نسخ النتائج',
      copyQuery: 'نسخ الاستعلام',
      copyFile: 'نسخ الملف',
      copyPath: 'نسخ المسار',
      outputAlt: 'مخرجات الأداة',
      rawResponse: 'الاستجابة الخام',
      copyActivity: 'نسخ النشاط',
      recoveredOne: 'نجح بعد محاولة فاشلة واحدة',
      recoveredMany: count =>
        count === 2
          ? 'نجح بعد محاولتين فاشلتين'
          : count <= 10
            ? `نجح بعد ${count} محاولات فاشلة`
            : `نجح بعد ${count} محاولة فاشلة`,
      failedOne: 'فشلت خطوة واحدة',
      failedMany: count =>
        count === 2 ? 'فشلت خطوتان' : count <= 10 ? `فشلت ${count} خطوات` : `فشلت ${count} خطوة`,
      statusRunning: 'يعمل',
      statusError: 'خطأ',
      statusRecovered: 'نجح',
      statusDone: 'تم',
      actions: {
        read: 'قرأ',
        reading: 'يقرأ',
        opened: 'فتح',
        opening: 'يفتح',
        failedToOpen: 'تعذّر الفتح',
        searched: 'بحث',
        searching: 'يبحث',
        ran: 'شغّل',
        running: 'يشغّل',
        ranCode: 'شغّل الشيفرة',
        runningCode: 'يكتب الشيفرة'
      },
      prefixes: {
        browser: 'المتصفّح',
        web: 'الويب'
      },
      titleTemplates: {
        actionCommand: (action, command) => `${action} ${command}`,
        actionQuoted: (action, value) => `${action} «${value}»`,
        actionTarget: (action, target) => `${action} ${target}`,
        prefixedDone: (prefix, action) => `${prefix} ${action}`,
        runningPrefixedTool: (prefix, action) => `جارٍ تشغيل ${prefix} ${action}`,
        runningTool: action => `جارٍ تشغيل ${action}`
      },
      titles: {
        browser_click: { done: 'نقر عنصر الصفحة', pending: 'ينقر عنصر الصفحة', pendingAction: 'ينقر' },
        browser_fill: { done: 'ملأ حقل النموذج', pending: 'يملأ حقل النموذج', pendingAction: 'يملأ' },
        browser_navigate: { done: 'فتح الصفحة', pending: 'يفتح الصفحة', pendingAction: 'يفتح' },
        browser_snapshot: {
          done: 'التقط لقطة الصفحة',
          pending: 'يلتقط لقطة الصفحة',
          pendingAction: 'يلتقط'
        },
        browser_take_screenshot: {
          done: 'التقط لقطة الشاشة',
          pending: 'يلتقط لقطة الشاشة',
          pendingAction: 'يلتقط'
        },
        browser_type: { done: 'كتب على الصفحة', pending: 'يكتب على الصفحة', pendingAction: 'يكتب' },
        clarify: { done: 'طرح سؤالًا', pending: 'يطرح سؤالًا', pendingAction: 'يسأل' },
        cronjob: { done: 'مهمة مجدولة', pending: 'يجدول مهمة', pendingAction: 'يجدول' },
        edit_file: { done: 'عدّل الملف', pending: 'يعدّل الملف', pendingAction: 'يعدّل' },
        execute_code: { done: 'شغّل الشيفرة', pending: 'يكتب الشيفرة', pendingAction: 'يكتب الشيفرة' },
        image_generate: { done: 'ولّد صورة', pending: 'يولّد صورة', pendingAction: 'يولّد' },
        list_files: { done: 'سرد الملفات', pending: 'يسرد الملفات', pendingAction: 'يسرد' },
        patch: { done: 'رقّع الملف', pending: 'يرقّع الملف', pendingAction: 'يرقّع' },
        read_file: { done: 'قرأ الملف', pending: 'يقرأ الملف', pendingAction: 'يقرأ' },
        search_files: { done: 'بحث في الملفات', pending: 'يبحث في الملفات', pendingAction: 'يبحث' },
        session_search_recall: {
          done: 'بحث في سجل الجلسة',
          pending: 'يبحث في سجل الجلسة',
          pendingAction: 'يبحث'
        },
        terminal: { done: 'شغّل الأمر', pending: 'يشغّل الأمر', pendingAction: 'يشغّل' },
        todo: { done: 'حدّث المهام', pending: 'يحدّث المهام', pendingAction: 'يحدّث' },
        vision_analyze: { done: 'حلّل الصورة', pending: 'يحلّل الصورة', pendingAction: 'يحلّل' },
        web_extract: { done: 'قرأ صفحة الويب', pending: 'يقرأ صفحة الويب', pendingAction: 'يقرأ' },
        web_search: { done: 'بحث في الويب', pending: 'يبحث في الويب', pendingAction: 'يبحث' },
        write_file: { done: 'عدّل الملف', pending: 'يعدّل الملف', pendingAction: 'يعدّل' }
      }
    }
  },

  prompts: {
    gatewayDisconnected: 'بوابة هرمس غير متصلة',
    sudoSendFailed: 'تعذر إرسال كلمة مرور sudo',
    secretSendFailed: 'تعذر إرسال السر',
    sudoTitle: 'كلمة مرور المسؤول',
    sudoDesc: 'يحتاج هرمس إلى كلمة مرور sudo لتشغيل أمر ذي صلاحيات مرتفعة. لا تُرسل إلا إلى وكيلك المحلي.',
    sudoPlaceholder: 'كلمة مرور sudo',
    secretTitle: 'يلزم سر',
    secretDesc: 'يحتاج هرمس إلى بيانات اعتماد للمتابعة.',
    secretPlaceholder: 'قيمة السر'
  },

  desktop: {
    audioReadFailed: 'تعذرت قراءة الصوت المسجل',
    sessionUnavailable: 'الجلسة غير متاحة',
    createSessionFailed: 'تعذر إنشاء جلسة جديدة',
    promptFailed: 'فشل الموجّه',
    providerCredentialRequired: 'أضف بيانات اعتماد لمزوّد قبل إرسال رسالتك الأولى.',
    emptySlashCommand: 'أمر مائل فارغ',
    desktopCommands: 'أوامر سطح المكتب',
    skillCommandsAvailable: count =>
      count === 1
        ? 'يتوفر أمر مهارة واحد.'
        : count === 2
          ? 'يتوفر أمرا مهارات.'
          : count <= 10
            ? `تتوفر ${count} أوامر مهارات.`
            : `يتوفر ${count} أمر مهارة.`,
    warningLine: message => `تحذير: ${message}`,
    yoloArmed: 'فُعّل وضع YOLO لهذه المحادثة',
    yoloOff: 'وضع YOLO معطّل',
    yoloSystem: active => `وضع YOLO ${active ? 'مفعّل' : 'معطّل'} لهذه الجلسة`,
    yoloTitle: 'YOLO',
    yoloToggleFailed: 'تعذر تبديل وضع YOLO',
    profileStatus: current =>
      `الملف الشخصي: ${current}. استخدم /profile <name> أو منتقي «جلسة جديدة» لبدء محادثة في ملف آخر.`,
    unknownProfile: 'ملف شخصي غير معروف',
    noProfileNamed: (target, available) => `لا يوجد ملف شخصي باسم «${target}». المتاح: ${available}`,
    newChatsProfile: name => `ستستخدم المحادثات الجديدة الملف الشخصي ${name}.`,
    setProfileFailed: 'فشل ضبط الملف الشخصي',
    sttDisabled: 'تحويل الكلام إلى نص معطّل في الإعدادات.',
    stopFailed: 'فشل الإيقاف',
    regenerateFailed: 'فشلت إعادة الإنشاء',
    editFailed: 'فشل التعديل',
    resumeFailed: 'فشل الاستئناف',
    resumeStrandedTitle: 'تعذّر تحميل هذه الجلسة',
    resumeStrandedBody:
      'فشل الاتصال بهذه الجلسة وتوقّفت إعادة المحاولة التلقائية. تأكّد أن البوابة تعمل، ثم أعد المحاولة.',
    resumeRetry: 'إعادة المحاولة',
    nothingToBranch: 'لا يوجد ما يمكن تفريعه',
    branchNeedsChat: 'ابدأ محادثة أو استأنفها قبل التفريع.',
    sessionBusy: 'الجلسة مشغولة',
    branchStopCurrent: 'أوقف الدورة الحالية قبل تفريع هذه المحادثة.',
    branchNoText: 'لا تحتوي هذه الرسالة على نص للتفريع منه.',
    branchTitle: n => `مسودّة: فرع رقم ${n}`,
    branchFailed: 'فشل التفريع',
    deleteFailed: 'فشل الحذف',
    archived: 'أُرشفت',
    archiveFailed: 'فشلت الأرشفة',
    cwdChangeFailed: 'فشل تغيير مجلد العمل',
    cwdStagedTitle: 'جُهّز تغيير مجلد العمل',
    cwdStagedMessage: 'أعد تشغيل واجهة سطح المكتب الخلفية لتطبيق تغيير المجلد على هذه الجلسة النشطة.',
    modelSwitchFailed: 'فشل تبديل النموذج',
    sessionExported: 'صُدّرت الجلسة',
    sessionExportFailed: 'تعذر تصدير الجلسة',
    imageSaved: 'حُفظت الصورة',
    downloadStarted: 'بدأ التنزيل',
    restartToUseSaveImage: 'أعد تشغيل هرمس لسطح المكتب لاستخدام حفظ الصورة.',
    restartToSaveImages: 'أعد تشغيل هرمس لسطح المكتب لحفظ الصور',
    imageDownloadFailed: 'فشل تنزيل الصورة',
    openImage: 'فتح الصورة',
    downloadImage: 'تنزيل الصورة',
    savingImage: 'جارٍ حفظ الصورة',
    imagePreviewFailed: 'فشلت معاينة الصورة',
    imageAttach: 'إرفاق الصورة',
    imageWriteFailed: 'فشلت كتابة الصورة على القرص.',
    imageAttachFailed: 'فشل إرفاق الصورة',
    attachImages: 'إرفاق صور',
    clipboard: 'الحافظة',
    noClipboardImage: 'لم يُعثر على صورة في الحافظة',
    clipboardPasteFailed: 'فشل اللصق من الحافظة',
    dropFiles: 'إفلات الملفات',
    handoff: {
      pickPlatform: 'اختيار وجهة',
      success: platform => `حُوّلت إلى ${platform}. يمكنك الاستئناف هنا في أي وقت.`,
      systemNote: platform => `↻ حُوّلت إلى ${platform}؛ يمكنك الاستئناف هنا في أي وقت.`,
      failed: error => `فشل التحويل: ${error}`,
      timedOut: 'انتهت مهلة انتظار البوابة. هل الأمر `hermes gateway` يعمل؟'
    }
  },

  errors: {
    genericFailure: 'حدث خطأ ما',
    boundaryTitle: 'تعطل جزء من الواجهة',
    boundaryDesc: 'واجه العرض خطأ غير متوقع. محادثاتك وإعداداتك آمنة.',
    reloadWindow: 'إعادة تحميل النافذة',
    openLogs: 'فتح السجلات'
  },

  ui: {
    search: { clear: 'مسح البحث' },
    pagination: {
      label: 'ترقيم الصفحات',
      previous: 'السابق',
      previousAria: 'الانتقال إلى الصفحة السابقة',
      next: 'التالي',
      nextAria: 'الانتقال إلى الصفحة التالية'
    },
    sidebar: {
      title: 'الشريط الجانبي',
      description: 'يعرض الشريط الجانبي على الأجهزة المحمولة.',
      toggle: 'تبديل الشريط الجانبي'
    }
  }
}
