import { defineFieldCopy } from '@/app/settings/field-copy'

import type { Translations } from './types'

export const ar: Translations = {
  common: {
    actions: 'إجراءات',
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
    reset: 'إعادة الضبط',
    retry: 'إعادة المحاولة',
    run: 'تشغيل',
    send: 'إرسال',
    set: 'ضبط',
    skip: 'تخطي',
    update: 'تحديث',
    zoomIn: 'تكبير',
    zoomOut: 'تصغير',
    openFullView: 'فتح العرض الكامل',
    openDiagram: 'فتح المخطّط',
    holdModifierToZoom: 'اضغط على مفتاح الأوامر للتكبير',
    resizePane: id => `تغيير حجم ${id}`,
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
      backendTimeout: seconds =>
        `لم يستجب هرمس خلال ${Number.isFinite(Number(seconds)) ? Number(seconds).toLocaleString('ar-EG') : seconds} ثانية. تحقق من الاتصال ثم أعد المحاولة.`,
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
    message: reason => `التصيير البرمجي مُفعّل — اكتُشف عرض بعيد (${reason}). تعطيل تسريع كرت الرسومات لمنع الوميض.`
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
    openStarmap: 'فتح مخطّط الذاكرة',
    layoutEditor: 'محرر التخطيط',
    layoutEditorTitle: 'محرر التخطيط — النقر مع ⌘ يعيد التخطيط كما كان'
  },

  keybinds: {
    title: 'اختصارات لوحة المفاتيح',
    subtitle: open => `انقر اختصارًا لإعادة تعيينه · يفتح ${open} هذه اللوحة مجددًا.`,
    search: 'البحث في الاختصارات',
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
      'session.newTab': 'تبويب جلسة جديد',
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
      'view.closeTab': 'إغلاق التبويب',
      'view.reopenTab': 'إعادة فتح التبويب المغلق',
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
      keybinds: 'اختصارات لوحة المفاتيح',
      keysTools: 'الأدوات',
      keysSettings: 'الإعدادات',
      mcp: 'بروتوكول سياق النموذج',
      archivedChats: 'المحادثات المؤرشفة',
      about: 'حول',
      billing: 'الفوترة',
      notifications: 'الإشعارات',
      plugins: 'الإضافات'
    },
    plugins: {
      title: 'إضافات سطح المكتب',
      blurb:
        'امتدادات واجهة تُحمَّل في هذا التطبيق — مضمّنة مع البناء، أو موضوعة في مجلد desktop-plugins (بما فيها ما يكتبه هرمس). التعطيل يُفرغ الإضافة فورًا ويبقى ساريًا بعد إعادة التشغيل.',
      count: n =>
        n === 1
          ? 'إضافة واحدة مثبّتة'
          : n === 2
            ? 'إضافتان مثبّتتان'
            : n >= 3 && n <= 10
              ? `${n} إضافات مثبّتة`
              : `${n} إضافة مثبّتة`,
      openFolder: 'فتح مجلد الإضافات',
      rescan: 'إعادة الفحص',
      reveal: 'إظهار في مدير الملفات',
      enable: 'تفعيل',
      disable: 'تعطيل',
      failed: 'فشلت',
      empty: 'لا توجد إضافات سطح مكتب مثبّتة بعد.',
      kinds: { bundled: 'مضمّنة', disk: 'على القرص', runtime: 'وقت التشغيل' }
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
      completionSoundPreview: 'معاينة',
      completionSoundNames: {
        1: 'راحة بنغمتين',
        2: 'رنين زجاجي',
        3: 'ماريمبا هادئة',
        4: 'رسالة بثلاث نغمات',
        5: 'وشوشة هوائية',
        6: 'عنقود اكتشاف',
        7: 'الأنظمة جاهزة',
        8: 'طرفية آي بي إم',
        9: 'زقزقة مودم',
        10: 'أجراس الريح',
        11: 'وعاء رنان',
        12: 'تصاعد القيثارة',
        13: 'نبضة سونار',
        14: 'صندوق موسيقي'
      }
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
        `يضبط حجم النصّ وعناصر التحكّم في التطبيق كلّه. ويمكن استخدام اختصارات التكبير والتصغير وإعادة الضبط. الحالي: ${percent}%.`,
      translucencyTitle: 'شفافية النافذة',
      translucencyDesc: 'شاهد سطح مكتبك عبر النافذة كاملة. لنظامي ماك وويندوز فقط.',
      backdropTitle: 'خلفية المحادثة',
      backdropDesc: 'صورة التمثال الباهتة خلف المحادثة.',
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
      marketplaceHeader: 'من متجر فيجوال ستوديو كود',
      searchThemesPlaceholder: 'ابحث في سماتك أو متجر فيجوال ستوديو كود…',
      noInstalledThemes: query => `لا توجد سمة مثبّتة تطابق «${query}».`,
      themeNames: {
        nous: 'Nous',
        midnight: 'منتصف الليل',
        ember: 'جمرة',
        mono: 'أحادي',
        cyberpunk: 'سايبربانك',
        slate: 'إردواز'
      },
      themeDescriptions: {
        nous: 'حياديات زجاجية بلمسات Nous الزرقاء',
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
          if (n === 1) {
            return 'حيوان واحد.'
          }

          if (n === 2) {
            return 'حيوانان.'
          }

          if (n >= 3 && n <= 10) {
            return `${n} حيوانات.`
          }

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
        modalMode: 'وضع صندوق مودال الرملي',
        timeout: 'مهلة الأمر',
        persistentShell: 'صدفة مستمرة',
        envPassthrough: 'تمرير متغيرات البيئة',
        dockerImage: 'صورة دوكر',
        singularityImage: 'صورة سنجيولاريتي',
        modalImage: 'صورة مودال',
        daytonaImage: 'صورة دايتونا'
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
        echoTranscripts: 'إظهار النصوص المفرّغة',
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
        edge: { voice: 'صوت Microsoft Edge' },
        openai: { model: 'نموذج OpenAI الصوتي', voice: 'صوت OpenAI' },
        elevenlabs: { voiceId: 'صوت ElevenLabs', modelId: 'نموذج ElevenLabs' },
        xai: { voiceId: 'صوت xAI Grok', language: 'لغة xAI' },
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
      honcho: 'Honcho',
      local: 'محلي',
      edge: 'Microsoft Edge',
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
      fallbackProviders: 'إدخالات احتياطية بصيغة مزوّد:نموذج تُجرّب عند فشل النموذج الافتراضي.',
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
        serviceTier: 'فئة خدمة الواجهة البرمجية لـ OpenAI وAnthropic.'
      },
      terminal: {
        cwd: 'مجلد المشروع الافتراضي لعمليات الأدوات والطرفية.',
        backend: 'بيئة تنفيذ أوامر الطرفية.',
        modalMode: 'وضع صندوق مودال الرملي.',
        persistentShell: 'يحافظ على حالة الصدفة بين الأوامر عندما تدعمها الواجهة الخلفية.',
        envPassthrough: 'متغيرات البيئة التي تُمرر إلى تنفيذ الأدوات.',
        dockerImage: 'صورة الحاوية المستخدمة عندما تكون واجهة التنفيذ الخلفية دوكر.',
        singularityImage: 'الصورة المستخدمة عندما تكون واجهة التنفيذ الخلفية سنجيولاريتي.',
        modalImage: 'الصورة المستخدمة عندما تكون واجهة التنفيذ الخلفية مودال.',
        daytonaImage: 'الصورة المستخدمة عندما تكون واجهة التنفيذ الخلفية دايتونا.'
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
        provider: 'المزوّد المستخدم لتحويل النص إلى كلام.',
        xai: {
          voiceId: 'معرّف صوت xAI، مثل eve، أو معرّف صوت مخصص.',
          language: 'رمز لغة النطق، مثل en.'
        },
        neutts: { device: 'جهاز الاستنتاج المحلي لـ NeuTTS.' }
      },
      stt: {
        enabled: 'يفعّل تفريغ الكلام محليًا أو عبر مزوّد.',
        echoTranscripts: 'يعيد نشر النص الخام للرسالة الصوتية في المحادثة.',
        provider: 'المزوّد المستخدم لتحويل الكلام إلى نص.',
        elevenlabs: {
          modelId: 'نموذج سكرايب من ElevenLabs للتفريغ.',
          languageCode: 'رمز لغة اختياري وفق معيار آيزو. اتركه فارغًا لتكتشف ElevenLabs اللغة تلقائيًا.'
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
      unknown: 'غير معروف',
      never: 'أبدًا',
      justNow: 'الآن',
      minAgo: count =>
        count === 1
          ? 'قبل دقيقة'
          : count === 2
            ? 'قبل دقيقتين'
            : count <= 10
              ? `قبل ${count} دقائق`
              : `قبل ${count} دقيقة`,
      hoursAgo: count =>
        count === 1
          ? 'قبل ساعة'
          : count === 2
            ? 'قبل ساعتين'
            : count <= 10
              ? `قبل ${count} ساعات`
              : `قبل ${count} ساعة`,
      daysAgo: count =>
        count === 1 ? 'قبل يوم' : count === 2 ? 'قبل يومين' : count <= 10 ? `قبل ${count} أيام` : `قبل ${count} يومًا`,
      sourceInstallOnly: path => `${path} ليس نسخة جيت؛ لا يعمل التحديث الذاتي لسطح المكتب إلا من تثبيت مصدري.`
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
    memoryProvider: {
      leaveBlankToKeep: 'اتركه فارغًا للاحتفاظ بالقيمة الحالية',
      set: 'مضبوط',
      failedLoad: 'تعذر تحميل إعدادات مزوّد الذاكرة',
      savedTitle: provider => `حُفظت إعدادات ${provider}`,
      savedMessage: 'حُدّث إعداد مزوّد الذاكرة.',
      failedSave: provider => `تعذر حفظ إعدادات ${provider}`,
      loading: 'جارٍ تحميل إعدادات مزوّد الذاكرة...',
      settings: provider => `إعدادات ${provider}`,
      fieldSet: field => `${field}: مضبوط`,
      fieldNotSet: field => `${field}: غير مضبوط`,
      save: 'حفظ',
      connectFailed: 'تعذر بدء الاتصال',
      couldNotStart: 'تعذر بدء الاتصال.',
      timedOut: 'انتهت المهلة. أعد المحاولة.',
      connectionFailed: 'فشل الاتصال.',
      connectViaOauth: 'الاتصال عبر التفويض',
      reconnect: 'إعادة الاتصال',
      connect: 'اتصال',
      apiKeySet: 'مفتاح الواجهة مضبوط',
      oauthSet: 'التفويض مضبوط',
      waitingForConsent: 'في انتظار الموافقة في المتصفح...',
      cancel: 'إلغاء',
      providerNames: { hindsight: 'Hindsight' },
      fieldLabels: {
        mode: 'الوضع',
        api_key: 'مفتاح الواجهة البرمجية',
        api_url: 'رابط الواجهة البرمجية',
        bank_id: 'معرّف مخزن الذاكرة',
        recall_budget: 'ميزانية الاسترجاع'
      },
      fieldDescriptions: {
        mode: 'كيفية اتصال هرمس بخدمة Hindsight.',
        api_key: 'يُستخدم للمصادقة مع واجهة Hindsight البرمجية.'
      },
      fieldPlaceholders: { api_key: 'أدخل مفتاح واجهة Hindsight البرمجية' },
      optionLabels: {
        cloud: 'سحابي',
        local_external: 'محلي خارجي',
        low: 'منخفضة',
        mid: 'متوسطة',
        high: 'مرتفعة'
      },
      optionDescriptions: {
        cloud: 'واجهة Hindsight السحابية؛ خفيفة ولا تحتاج إلا إلى مفتاح واجهة.',
        local_external: 'الاتصال بنسخة Hindsight قائمة.'
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
        'تستخدم البوابات المستضافة أو أوث أو اسم مستخدم وكلمة مرور، وقد تستخدم البوابات الذاتية رمز جلسة.',
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
      authSignedInOauth: 'تستخدم هذه البوابة أو أوث. أنت مسجل الدخول، وتُجدّد الجلسة تلقائيًا.',
      authNeedsPassword: 'تستخدم هذه البوابة اسم مستخدم وكلمة مرور. سجّل الدخول لتفويض تطبيق سطح المكتب.',
      authNeedsOauth: provider => `تستخدم هذه البوابة أو أوث. سجّل الدخول عبر ${provider} لتفويض تطبيق سطح المكتب.`,
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
      empty: 'لا يوجد شيء مضبوط في هذه الفئة بعد.',
      fieldLabels: {
        AGENT_BROWSER_ENGINE: 'محرك المتصفح المحلي',
        BRAVE_SEARCH_API_KEY: 'مفتاح بحث بريف',
        BROWSER_USE_API_KEY: 'مفتاح Browser Use',
        BROWSERBASE_API_KEY: 'مفتاح Browserbase',
        BROWSERBASE_PROJECT_ID: 'معرّف مشروع Browserbase',
        BRV_API_KEY: 'مفتاح ByteRover',
        CAMOFOX_API_KEY: 'مفتاح Camoufox',
        CAMOFOX_URL: 'رابط Camoufox',
        ELEVENLABS_API_KEY: 'مفتاح ElevenLabs',
        EXA_API_KEY: 'مفتاح Exa',
        FAL_KEY: 'مفتاح FAL',
        FIRECRAWL_API_KEY: 'مفتاح Firecrawl',
        FIRECRAWL_API_URL: 'رابط واجهة Firecrawl',
        FIRECRAWL_BROWSER_TTL: 'مدة جلسة متصفح Firecrawl',
        FIRECRAWL_GATEWAY_URL: 'رابط بوابة Firecrawl',
        GITHUB_TOKEN: 'رمز GitHub',
        HERMES_LANGFUSE_BASE_URL: 'رابط خادم Langfuse',
        HERMES_LANGFUSE_PUBLIC_KEY: 'المفتاح العام لـ Langfuse',
        HERMES_LANGFUSE_SECRET_KEY: 'المفتاح السري لـ Langfuse',
        HINDSIGHT_API_KEY: 'مفتاح Hindsight',
        HINDSIGHT_API_URL: 'رابط واجهة Hindsight',
        HONCHO_API_KEY: 'مفتاح Honcho',
        HONCHO_BASE_URL: 'الرابط الأساسي لـ Honcho',
        KREA_API_KEY: 'مفتاح Krea',
        MEM0_API_KEY: 'مفتاح Mem0',
        MISTRAL_API_KEY: 'مفتاح Mistral',
        OPENVIKING_API_KEY: 'مفتاح OpenViking',
        OPENVIKING_ENDPOINT: 'نقطة OpenViking',
        PARALLEL_API_KEY: 'مفتاح Parallel',
        RETAINDB_API_KEY: 'مفتاح ريتين دي بي',
        RETAINDB_BASE_URL: 'الرابط الأساسي لريتين دي بي',
        SEARXNG_URL: 'رابط SearXNG',
        SUPERMEMORY_API_KEY: 'مفتاح Supermemory',
        TAVILY_API_KEY: 'مفتاح Tavily',
        TOOL_GATEWAY_DOMAIN: 'نطاق بوابة الأدوات',
        TOOL_GATEWAY_SCHEME: 'بروتوكول بوابة الأدوات',
        TOOL_GATEWAY_USER_TOKEN: 'رمز مستخدم بوابة الأدوات',
        VOICE_TOOLS_OPENAI_KEY: 'مفتاح OpenAI للصوت',
        GATEWAY_ALLOW_ALL_USERS: 'السماح لجميع مستخدمي البوابة',
        GATEWAY_PROXY_KEY: 'مفتاح وكيل البوابة',
        GATEWAY_PROXY_URL: 'رابط وكيل البوابة',
        HERMES_EPHEMERAL_SYSTEM_PROMPT: 'توجيه النظام المؤقت',
        HERMES_PREFILL_MESSAGES_FILE: 'ملف رسائل التمهيد',
        HERMES_SIMPLEX_TEXT_BATCH_DELAY: 'مهلة تجميع نصوص سيمبلكس',
        RAFT_PROFILE: 'ملف رافت الشخصي',
        SMS_ALLOWED_USERS: 'مستخدمو الرسائل النصية المسموح لهم',
        SMS_HOME_CHANNEL: 'قناة الرسائل النصية الرئيسية',
        SUDO_PASSWORD: 'كلمة مرور الصلاحيات المرتفعة',
        WECOM_ALLOWED_USERS: 'مستخدمو وي كوم المسموح لهم',
        WECOM_HOME_CHANNEL: 'قناة وي كوم الرئيسية',
        WECOM_WEBSOCKET_URL: 'رابط اتصال وي كوم الفوري'
      },
      fieldDescriptions: {
        AGENT_BROWSER_ENGINE:
          'محرك التصفح في الوضع المحلي: تلقائي لاستخدام كروم افتراضيًا، أو لايت باندا الأسرع من دون لقطات، أو كروم صراحة.',
        BRAVE_SEARCH_API_KEY: 'رمز اشتراك واجهة بحث بريف؛ تتيح الفئة المجانية ألفي استعلام شهريًا.',
        BROWSER_USE_API_KEY: 'مفتاح Browser Use للمتصفح السحابي؛ وهو اختياري لأن المتصفح المحلي يعمل بدونه.',
        BROWSERBASE_API_KEY: 'مفتاح Browserbase للمتصفح السحابي؛ وهو اختياري لأن المتصفح المحلي يعمل بدونه.',
        BROWSERBASE_PROJECT_ID: 'معرّف مشروع Browserbase؛ لا يلزم إلا عند استخدام المتصفح السحابي.',
        BRV_API_KEY: 'مفتاح ByteRover للمزامنة السحابية الاختيارية؛ التخزين محلي أولًا افتراضيًا.',
        CAMOFOX_API_KEY: 'رمز اختياري يُرسل إلى خادم Camoufox البعيد الذي يتطلب مصادقة.',
        CAMOFOX_URL: 'رابط خادم Camoufox للتصفح المحلي المقاوم للكشف.',
        ELEVENLABS_API_KEY: 'مفتاح ElevenLabs للأصوات المتميزة وتحويل الكلام إلى نص.',
        EXA_API_KEY: 'مفتاح Exa للبحث الأصلي بالذكاء الاصطناعي واستخراج المحتوى.',
        FAL_KEY: 'مفتاح FAL لتوليد الصور والفيديو.',
        FIRECRAWL_API_KEY: 'مفتاح Firecrawl للبحث في الويب واستخراج الصفحات.',
        FIRECRAWL_API_URL: 'رابط واجهة Firecrawl للنسخ المستضافة ذاتيًا.',
        FIRECRAWL_BROWSER_TTL: 'مدة بقاء جلسة متصفح Firecrawl بالثواني؛ الافتراضي ثلاثمئة.',
        FIRECRAWL_GATEWAY_URL: 'رابط بديل دقيق لبوابة Firecrawl، مخصص لمشتركي Nous.',
        GITHUB_TOKEN: 'رمز GitHub لمركز المهارات، يرفع حدود الاستدعاء ويفعّل نشر المهارات.',
        HERMES_LANGFUSE_BASE_URL: 'رابط خادم Langfuse.',
        HERMES_LANGFUSE_PUBLIC_KEY: 'المفتاح العام لمشروع Langfuse.',
        HERMES_LANGFUSE_SECRET_KEY: 'المفتاح السري لمشروع Langfuse.',
        HINDSIGHT_API_KEY: 'مفتاح Hindsight للذاكرة المستمرة الواعية بالعلاقات.',
        HINDSIGHT_API_URL: 'الرابط الأساسي لواجهة Hindsight.',
        HONCHO_API_KEY: 'مفتاح Honcho للذاكرة المستمرة الأصلية بالذكاء الاصطناعي.',
        HONCHO_BASE_URL: 'الرابط الأساسي لنسخة Honcho المستضافة ذاتيًا، ولا تحتاج إلى مفتاح.',
        KREA_API_KEY: 'مفتاح Krea لتوليد الصور بنماذج Krea 2.',
        MEM0_API_KEY: 'مفتاح منصة Mem0 للذاكرة الدلالية المستمرة.',
        MISTRAL_API_KEY: 'مفتاح Mistral لتحويل النص إلى كلام والكلام إلى نص.',
        OPENVIKING_API_KEY: 'مفتاح OpenViking؛ اتركه فارغًا في وضع التطوير المحلي.',
        OPENVIKING_ENDPOINT: 'رابط خادم OpenViking.',
        PARALLEL_API_KEY: 'مفتاح Parallel للبحث في الويب واستخراج المحتوى.',
        RETAINDB_API_KEY: 'مفتاح ريتين دي بي للذاكرة المستمرة.',
        RETAINDB_BASE_URL: 'الرابط الأساسي لنسخة ريتين دي بي المستضافة ذاتيًا.',
        SEARXNG_URL: 'رابط نسخة SearXNG للبحث المجاني المستضاف ذاتيًا.',
        SUPERMEMORY_API_KEY: 'مفتاح Supermemory للذاكرة المستمرة ضمن المحادثة.',
        TAVILY_API_KEY: 'مفتاح Tavily للبحث في الويب واستخراج المحتوى.',
        TOOL_GATEWAY_DOMAIN: 'لاحقة النطاق المشتركة لبوابة أدوات مشتركي Nous، وتُشتق منها عناوين المزوّدين.',
        TOOL_GATEWAY_SCHEME: 'بروتوكول رابط بوابة الأدوات؛ الآمن افتراضيًا، ويمكن تغييره للاختبار المحلي.',
        TOOL_GATEWAY_USER_TOKEN: 'رمز وصول صريح لمشترك Nous عند طلبات بوابة الأدوات؛ وهو اختياري.',
        VOICE_TOOLS_OPENAI_KEY: 'مفتاح OpenAI لتفريغ الصوت وتحويل النص إلى كلام.',
        GATEWAY_ALLOW_ALL_USERS: 'يسمح لجميع المستخدمين بالتفاعل مع بوتات المراسلة؛ وهو معطّل افتراضيًا.',
        GATEWAY_PROXY_KEY: 'رمز مصادقة مع خادم هرمس البعيد في وضع الوكيل.',
        GATEWAY_PROXY_URL: 'رابط خادم هرمس بعيد تُحوّل إليه الرسائل في وضع الوكيل.',
        HERMES_EPHEMERAL_SYSTEM_PROMPT: 'توجيه نظام مؤقت يُحقن وقت الاستدعاء ولا يُحفظ في الجلسات.',
        HERMES_PREFILL_MESSAGES_FILE: 'مسار ملف بيانات يحوي رسائل تمهيد مؤقتة للأمثلة المسبقة.',
        HERMES_SIMPLEX_TEXT_BATCH_DELAY: 'مدة الهدوء بالثواني لجمع الرسائل النصية المتتابعة في حدث واحد.',
        RAFT_PROFILE: 'معرّف ملف وكيل رافت؛ يفعّل المحوّل تلقائيًا عند ضبطه.',
        SMS_ALLOWED_USERS: 'أرقام الهواتف المسموح لها بمحادثة البوت، مفصولة بفواصل.',
        SMS_HOME_CHANNEL: 'رقم الهاتف الافتراضي لتسليم المهام المجدولة والإشعارات.',
        SUDO_PASSWORD: 'كلمة مرور أوامر الطرفية التي تتطلب صلاحيات الجذر.',
        WECOM_ALLOWED_USERS: 'معرّفات مستخدمي وي كوم المسموح لهم بمحادثة البوت، مفصولة بفواصل.',
        WECOM_HOME_CHANNEL: 'معرّف المحادثة الافتراضي لتسليم المهام المجدولة والإشعارات.',
        WECOM_WEBSOCKET_URL: 'رابط الاتصال الفوري لروبوت وي كوم الذكي.'
      }
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
      catalogLoading: 'جارٍ تحميل كتالوج بروتوكول سياق النموذج...',
      catalogLoadFailed: 'فشل تحميل كتالوج بروتوكول سياق النموذج',
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
      noOutput: 'لا مخرجات بعد.',
      authOauth: 'أو أوث',
      authApiKey: 'مفتاح الواجهة البرمجية'
    },
    model: {
      loading: 'جارٍ تحميل إعداد النموذج...',
      appliesDesc: 'ينطبق على الجلسات الجديدة. استخدم منتقي النموذج في محرر الرسالة لتبديل نموذج المحادثة النشطة.',
      provider: 'المزوّد',
      genericProvider: 'مزوّد',
      model: 'النموذج',
      applying: 'جارٍ التطبيق...',
      pasteProviderKey: key => `الصق ${key}`,
      activatingProvider: 'جارٍ التفعيل...',
      activateProvider: 'تفعيل',
      setUpProvider: provider => `إعداد ${provider}`,
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
      moaTitle: 'مزيج الوكلاء',
      moaDesc: 'اضبط إعدادات مسبقة مسماة تظهر كنماذج ضمن مزود مزيج الوكلاء. النموذج الجامع هو النموذج المنفذ.',
      moaPreset: 'الإعداد المسبق',
      moaSetDefault: 'تعيين افتراضيا',
      moaDelete: 'حذف',
      moaNewPreset: 'إعداد مسبق جديد',
      moaAddPreset: 'إضافة إعداد مسبق',
      moaDefault: 'الافتراضي',
      moaAggregator: 'النموذج الجامع',
      moaReference: index => `النموذج المرجعي ${index}`,
      moaRemove: 'إزالة',
      moaAddReference: 'إضافة نموذج مرجعي',
      moaPresets: 'إعدادات مزيج الوكلاء',
      moaPrefix: 'مزيج',
      otherProviders: 'مزودين آخرين',
      staleAuxWarning: (count, names) => ({
        before:
          count === 1
            ? `لا تزال مهمة مساعدة واحدة (${names}) تعمل على `
            : count === 2
              ? `لا تزال مهمتان مساعدتان (${names}) تعملان على `
              : count <= 10
                ? `لا تزال ${count} مهام مساعدة (${names}) تعمل على `
                : `لا تزال ${count} مهمة مساعدة (${names}) تعمل على `,
        after: '، لا على نموذجك الرئيسي.'
      }),
      tasks: {
        vision: { label: 'الرؤية', hint: 'تحليل الصور' },
        web_extract: { label: 'استخراج الويب', hint: 'تلخيص الصفحات' },
        compression: { label: 'الضغط', hint: 'ضغط السياق' },
        skills_hub: { label: 'مركز المهارات', hint: 'البحث عن المهارات' },
        approval: { label: 'الموافقة', hint: 'الموافقة التلقائية الذكية' },
        mcp: { label: 'بروتوكول سياق النموذج', hint: 'توجيه أدوات بروتوكول سياق النموذج' },
        title_generation: { label: 'إنشاء العناوين', hint: 'عناوين الجلسات' },
        curator: { label: 'القيّم', hint: 'مراجعة استخدام المهارات' }
      }
    },
    computerUse: {
      linuxNote: 'يتحكم في سطح المكتب عبر منظومة الإتاحة لإكس دون مطالبة أذونات.',
      windowsNote: 'قد يعرض التشغيل الأول مطالبة الحماية في ويندوز لعامل إتاحة برنامج التحكم؛ اسمح له.',
      granted: 'ممنوح',
      notGranted: 'غير ممنوح',
      unknown: 'غير معروف',
      readStatusFailed: 'تعذر قراءة حالة التحكم بالحاسوب',
      requestFailed: 'تعذر طلب الأذونات',
      approveTitle: 'الموافقة في إعدادات النظام',
      approveMessage: 'سيعرض ماك حوار أذونات منسوبا إلى كوا درايفر. وافق عليه ثم عد إلى هنا.',
      checking: 'جارٍ فحص حالة التحكم بالحاسوب…',
      unsupported: platform => `التحكم بالحاسوب غير مدعوم على هذه المنصة (${platform}).`,
      installDriver: 'ثبّت مشغل كوا درايفر أدناه للتحكم في هذا الجهاز.',
      grantAfterInstall: ' ثم امنحه إذني الإتاحة وتسجيل الشاشة هنا.',
      grantIdentity:
        'ترتبط الأذونات بهوية كوا درايفر نفسه (com.trycua.driver)، لا بهرمس، لذلك يُنسب الحوار إلى العملية التي تتحكم في ماك.',
      recheck: 'إعادة الفحص',
      accessibility: 'الإتاحة',
      accessibilityHint: 'يتيح لكوا درايفر النقر والكتابة وقراءة شجرة الإتاحة.',
      screenRecording: 'تسجيل الشاشة',
      screenRecordingHint: 'يتيح لكوا درايفر التقاط صور لنوافذ التطبيقات.',
      driverHealth: 'سلامة المشغل',
      ready: 'جاهز',
      notReady: 'غير جاهز',
      readyDescription: 'التحكم بالحاسوب جاهز. اطلب من الوكيل التقاط تطبيق والتفاعل معه.',
      waitingApproval: 'في انتظار الموافقة…',
      grantPermissions: 'منح الأذونات'
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
      localEndpoint: {
        title: 'نقطة نهاية محلية أو مخصصة',
        description:
          'وجّه هرمس إلى أي نقطة نهاية متوافقة مع واجهة أوبن أي آي، مثل زيفرا وفي إل إل إم ولاما دوت سي بلس بلس وأولاما وغيرها.'
      },
      loading: 'جارٍ تحميل المزوّدين...',
      providerNames: {
        OpenRouter: 'OpenRouter',
        Anthropic: 'Anthropic',
        xAI: 'xAI',
        DeepSeek: 'DeepSeek',
        MiniMax: 'MiniMax',
        'MiniMax (China)': 'MiniMax (China)',
        'OpenCode Zen': 'OpenCode Zen',
        'OpenCode Go': 'OpenCode Go',
        'NVIDIA NIM': 'NVIDIA NIM',
        'Ollama Cloud': 'Ollama Cloud',
        'LM Studio': 'LM Studio',
        'Xiaomi MiMo': 'Xiaomi MiMo',
        'Arcee AI': 'Arcee AI',
        'GMI Cloud': 'GMI Cloud',
        'Azure Foundry': 'Azure Foundry',
        'Alibaba Cloud (Coding Plan)': 'Alibaba Cloud (Coding Plan)',
        'Fireworks AI': 'Fireworks AI',
        'GitHub Copilot': 'GitHub Copilot',
        'Google AI Studio': 'Google AI Studio',
        HuggingFace: 'HuggingFace',
        'Kilo Code': 'Kilo Code',
        'Kimi / Kimi Coding Plan': 'Kimi / Kimi Coding Plan',
        'Kimi / Moonshot (China)': 'Kimi / Moonshot (China)',
        NovitaAI: 'NovitaAI',
        'OpenAI API': 'OpenAI API',
        'Qwen Cloud': 'Qwen Cloud',
        'StepFun Step Plan': 'StepFun Step Plan',
        'Tencent TokenHub': 'Tencent TokenHub',
        'Z.AI (GLM)': 'Z.AI (GLM)'
      },
      providerDescriptions: {
        OpenRouter: 'مئات النماذج المتقدمة خلف مفتاح واحد.',
        Anthropic: 'وصول مباشر إلى نماذج Claude.',
        xAI: 'وصول مباشر إلى نماذج Grok.',
        DeepSeek: 'وصول مباشر إلى نماذج DeepSeek.',
        MiniMax: 'نقطة MiniMax الدولية ونماذجها.',
        'MiniMax (China)': 'نقطة MiniMax داخل الصين.',
        'OpenCode Zen': 'وصول مدفوع بحسب الاستخدام إلى نماذج برمجة منتقاة.',
        'OpenCode Go': 'اشتراك شهري لنماذج البرمجة المفتوحة.',
        'NVIDIA NIM': 'خدمة NVIDIA NIM أو نقطة محلية خاصة بك.',
        'Ollama Cloud': 'نماذج مفتوحة مستضافة في Ollama Cloud.',
        'LM Studio': 'خادم محلي متوافق مع واجهة OpenAI.',
        'Xiaomi MiMo': 'نماذج Xiaomi MiMo.',
        'Arcee AI': 'نماذج صغيرة ومتوسطة تستضيفها Arcee AI.',
        'GMI Cloud': 'حوسبة رسومية وخدمة نماذج عبر GMI Cloud.',
        'Azure Foundry': 'نقاط Azure Foundry المخصصة والمتوافقة مع مزوّدي النماذج.',
        'Alibaba Cloud (Coding Plan)': 'خطة البرمجة السحابية من Alibaba.',
        'Fireworks AI': 'وصول مباشر إلى النماذج المستضافة لدى Fireworks.',
        'GitHub Copilot': 'وصول إلى نماذج GitHub Copilot.',
        'Google AI Studio': 'وصول مباشر إلى نماذج Gemini من استوديو Google.',
        HuggingFace: 'أكثر من عشرين نموذجًا مفتوحًا عبر مزوّدي الاستدلال.',
        'Kilo Code': 'وصول إلى نماذج Kilo Code.',
        'Kimi / Kimi Coding Plan': 'نماذج Kimi ونقاط البرمجة من Moonshot.',
        'Kimi / Moonshot (China)': 'نقطة Moonshot داخل الصين.',
        NovitaAI: 'وصول مباشر إلى نماذج NovitaAI.',
        'OpenAI API': 'وصول مباشر إلى نماذج OpenAI.',
        'Qwen Cloud': 'نماذج Qwen ومزوّدون متعددون عبر Alibaba Cloud.',
        'StepFun Step Plan': 'نماذج البرمجة ضمن خطة StepFun.',
        'Tencent TokenHub': 'نماذج Tencent عبر مركز الرموز.',
        'Z.AI (GLM)': 'نماذج GLM ونقاط Z.AI المستضافة.'
      },
      advancedFields: {
        baseUrl: 'الرابط الأساسي البديل',
        baseUrlDescription: provider => `رابط أساسي بديل لخدمة ${provider}.`,
        region: 'المنطقة',
        regionDescription: provider => `المنطقة المستخدمة عند الاتصال بخدمة ${provider}.`,
        profile: 'ملف المصادقة',
        profileDescription: provider => `ملف المصادقة المستخدم للاتصال بخدمة ${provider}.`,
        credentialsPath: 'مسار بيانات الاعتماد',
        credentialsPathDescription: provider => `مسار بيانات الاعتماد المستخدمة للاتصال بخدمة ${provider}.`,
        alternateKey: 'مفتاح بديل',
        alternateKeyDescription: provider => `بيانات اعتماد إضافية أو بديلة لخدمة ${provider}.`
      }
    },
    sessions: {
      loading: 'جارٍ تحميل الجلسات المؤرشفة…',
      archivedTitle: 'الجلسات المؤرشفة',
      archivedIntro:
        'تُخفى المحادثات المؤرشفة من الشريط الجانبي مع الاحتفاظ برسائلها. انقر مع مفتاح التحكم أو الأوامر على محادثة في الشريط لأرشفتها.',
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
      defaultDirUpdated: 'حُدّث مجلد المشروع الافتراضي؛ ابدأ محادثة جديدة لتطبيقه',
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
      failedSelectModel: model => `فشل اختيار ${model}`,
      providerNames: {
        'Nous Subscription': 'Nous Subscription',
        'Firecrawl Self-Hosted': 'Firecrawl Self-Hosted',
        'Brave Search (Free)': 'Brave Search (Free)',
        'DuckDuckGo (ddgs)': 'DuckDuckGo (ddgs)',
        Exa: 'Exa',
        Firecrawl: 'Firecrawl',
        Parallel: 'Parallel',
        SearXNG: 'SearXNG',
        Tavily: 'Tavily',
        'xAI Web Search (Grok)': 'xAI Web Search (Grok)',
        'Local Browser': 'Local Browser',
        'Nous Subscription (Browser Use cloud)': 'Nous Subscription (Browser Use cloud)',
        Camofox: 'Camofox',
        'Browser Use': 'Browser Use',
        Browserbase: 'Browserbase',
        'FAL.ai': 'FAL.ai',
        Krea: 'Krea',
        'Nous Portal (image)': 'Nous Portal (image)',
        OpenAI: 'OpenAI',
        'OpenAI (Codex auth)': 'OpenAI (Codex auth)',
        'OpenRouter (image)': 'OpenRouter (image)',
        'xAI Grok Imagine (image)': 'xAI Grok Imagine (image)',
        FAL: 'FAL',
        'xAI Grok Imagine': 'xAI Grok Imagine',
        'xAI Grok OAuth (SuperGrok / Premium+)': 'xAI Grok OAuth (SuperGrok / Premium+)',
        'xAI API key': 'xAI API key',
        'Microsoft Edge TTS': 'Microsoft Edge TTS',
        'OpenAI TTS': 'OpenAI TTS',
        'xAI TTS': 'xAI TTS',
        ElevenLabs: 'ElevenLabs',
        'Mistral (Voxtral TTS)': 'Mistral (Voxtral TTS)',
        'Google Gemini TTS': 'Google Gemini TTS',
        KittenTTS: 'KittenTTS',
        Piper: 'Piper',
        'Home Assistant': 'Home Assistant',
        'Spotify Web API': 'Spotify Web API',
        'cua-driver (background)': 'cua-driver (background)'
      },
      providerTags: {
        'Managed Firecrawl billed to your subscription': 'Firecrawl مُدار وتُحتسب تكلفته على اشتراكك.',
        'Run your own Firecrawl instance (Docker)': 'شغّل نسخة Firecrawl خاصة بك عبر دوكر.',
        'Free-tier API key — 2k queries/mo, search only.':
          'مفتاح من الفئة المجانية يتيح ألفي استعلام شهريًا، للبحث فقط.',
        'Search via the ddgs Python package — no API key (pair with any extract provider)':
          'بحث عبر حزمة ddgs لبايثون بلا مفتاح، ويمكن إقرانه بأي مزوّد استخراج.',
        'Semantic + neural web search with content extraction.': 'بحث دلالي وعصبي في الويب مع استخراج المحتوى.',
        'Full search + extract; supports direct API and Nous tool-gateway routing.':
          'بحث واستخراج كاملان، مع دعم الواجهة المباشرة والتوجيه عبر بوابة أدوات Nous.',
        'Objective-tuned search + parallel page extraction.': 'بحث مضبوط بحسب الهدف مع استخراج متوازٍ للصفحات.',
        'Free, privacy-respecting metasearch. Point SEARXNG_URL at your instance.':
          'بحث تجميعي مجاني يحترم الخصوصية؛ وجّه رابط SearXNG إلى نسختك.',
        'Search + extract in one provider.': 'بحث واستخراج في مزوّد واحد.',
        "Agentic web search via Grok's web_search tool — uses xAI Grok OAuth or XAI_API_KEY.":
          'بحث وكيلي في الويب عبر أداة Grok، باستخدام مصادقة Grok أو مفتاح xAI.',
        'Headless Chromium, no API key needed': 'Chromium بلا واجهة، ولا يحتاج إلى مفتاح واجهة برمجية.',
        'Managed Browser Use billed to your subscription': 'Browser Use مُدار وتُحتسب تكلفته على اشتراكك.',
        'Anti-detection browser (Firefox/Camoufox)': 'متصفح مقاوم للكشف مبني على Firefox وCamoufox.',
        'Cloud browser with remote execution': 'متصفح سحابي مع تنفيذ عن بُعد.',
        'Cloud browser with stealth and proxies': 'متصفح سحابي مع التخفي والخوادم الوكيلة.',
        'Managed FAL image generation billed to your subscription':
          'توليد صور مُدار عبر FAL وتُحتسب تكلفته على اشتراكك.',
        'Pick from flux-2-klein, flux-2-pro, gpt-image, nano-banana, etc. — text-to-image & image editing':
          'اختر من FLUX 2 Klein وFLUX 2 Pro وGPT Image وNano Banana وغيرها، لتحويل النص إلى صورة وتحرير الصور.',
        'Krea 2 foundation model — Medium ($0.03), Large ($0.06), Medium Turbo ($0.015). Style transfer, moodboards, reference-guided generation. Direct key or managed Nous Subscription gateway.':
          'نموذج Krea 2 الأساسي بثلاث فئات؛ يدعم نقل الأسلوب ولوحات المزاج والتوليد الموجّه بالمراجع، عبر مفتاح مباشر أو بوابة اشتراك Nous.',
        'Reference-grounded image generation via Nous Portal (OpenRouter-backed)':
          'توليد صور مستند إلى المراجع عبر Nous Portal المدعومة بـ OpenRouter.',
        'gpt-image-2 at low/medium/high quality tiers — text-to-image & image editing':
          'نموذج GPT Image 2 بفئات جودة منخفضة ومتوسطة وعالية، لتحويل النص إلى صورة وتحرير الصور.',
        'gpt-image-2 via ChatGPT/Codex OAuth — no API key required; supports text and image inputs':
          'نموذج GPT Image 2 عبر مصادقة ChatGPT أو Codex، بلا مفتاح، ويدعم مدخلات النص والصورة.',
        'Gemini Flash Image & more via OpenRouter; uses OPENROUTER_API_KEY':
          'Gemini Flash Image ونماذج أخرى عبر OpenRouter، باستخدام مفتاحه البرمجي.',
        'grok-imagine-image - text-to-image & image editing; uses xAI Grok OAuth or XAI_API_KEY. xAI Imagine storage is enabled so generated media gets a reusable public URL without an automatic expiry. xAI may bill for stored files and public URL hosting. Disable this with `image_gen.xai.storage.enabled: false` or set `expires_after` to change the retention.':
          'Grok Imagine للصور يحوّل النص إلى صورة ويحرر الصور عبر مصادقة Grok أو مفتاح xAI. التخزين مفعّل لمنح الوسائط رابطًا عامًا قابلًا لإعادة الاستخدام، وقد تترتب عليه تكلفة؛ ويمكن تعطيله أو تغيير مدة الاحتفاظ من الإعدادات.',
        'Managed FAL video generation billed to your subscription':
          'توليد فيديو مُدار عبر FAL وتُحتسب تكلفته على اشتراكك.',
        'LTX, Pixverse, Veo 3.1, Seedance 2.0, Kling 4K, Happy Horse — text-to-video & image-to-video':
          'نماذج LTX وPixVerse وVeo 3.1 وSeedance 2.0 وKling 4K وHappy Horse، لتحويل النص أو الصورة إلى فيديو.',
        'grok-imagine-video for text/reference; grok-imagine-video-1.5 for image-to-video; edit/extend: pass the stored public HTTPS MP4 (`video` / `public_url` from a prior Imagine result); uses xAI Grok OAuth or XAI_API_KEY. xAI Imagine storage is enabled so generated media gets a reusable public URL without an automatic expiry. xAI may bill for stored files and public URL hosting. Disable this with `video_gen.xai.storage.enabled: false` or set `expires_after` to change the retention.':
          'Grok Imagine للفيديو يدعم النص والمراجع وتحويل الصورة إلى فيديو وتحرير الفيديو أو تمديده، عبر مصادقة Grok أو مفتاح xAI. التخزين مفعّل لمنح الوسائط رابطًا عامًا قابلًا لإعادة الاستخدام، وقد تترتب عليه تكلفة؛ ويمكن تعطيله أو تغيير مدة الاحتفاظ من الإعدادات.',
        'Browser login at accounts.x.ai — no API key required':
          'تسجيل دخول بالمتصفح إلى حساب xAI، ولا يحتاج إلى مفتاح.',
        'Direct xAI API billing via XAI_API_KEY': 'فوترة مباشرة لواجهة xAI عبر مفتاحها البرمجي.',
        'Good quality, no API key needed': 'جودة جيدة، ولا يحتاج إلى مفتاح واجهة برمجية.',
        'Managed OpenAI TTS billed to your subscription':
          'تحويل نص إلى كلام مُدار عبر OpenAI وتُحتسب تكلفته على اشتراكك.',
        'High quality voices': 'أصوات عالية الجودة.',
        'Grok voices — uses xAI Grok OAuth or XAI_API_KEY': 'أصوات Grok عبر مصادقة Grok أو مفتاح xAI.',
        'Most natural voices': 'الأصوات الأكثر طبيعية.',
        'Multilingual, native Opus': 'متعدد اللغات مع دعم أصلي لتنسيق أوبس.',
        '30 prebuilt voices, controllable via prompts': 'ثلاثون صوتًا جاهزًا يمكن التحكم فيها بالتوجيهات.',
        'Lightweight local ONNX TTS (~25MB), no API key':
          'تحويل محلي خفيف للنص إلى كلام، بحجم يقارب خمسة وعشرين ميجابايت، وبلا مفتاح.',
        'Local neural TTS, 44 languages (voices ~20-90MB)':
          'تحويل عصبي محلي للنص إلى كلام يدعم أربعًا وأربعين لغة، بأصوات بين عشرين وتسعين ميجابايت تقريبًا.',
        'REST API integration': 'تكامل عبر واجهة برمجية بأسلوب ريست.',
        'PKCE OAuth — opens the setup wizard': 'مصادقة آمنة تفتح معالج الإعداد.',
        'Background computer-use via cua-driver — does NOT steal your cursor or focus. Works with any model.':
          'تحكم بالحاسوب في الخلفية عبر مشغّل كوا درايفر، من دون الاستحواذ على المؤشر أو التركيز، ويعمل مع أي نموذج.'
      },
      providerBadgeTerms: {
        subscription: 'اشتراك',
        free: 'مجاني',
        paid: 'مدفوع',
        preview: 'معاينة',
        local: 'محلي',
        'self-hosted': 'مستضاف ذاتيًا',
        'no key': 'بلا مفتاح',
        'search only': 'للبحث فقط',
        'optional gateway': 'بوابة اختيارية',
        '★ recommended': '★ موصى به'
      },
      envVarPrompts: {
        FIRECRAWL_API_URL: 'رابط نسخة Firecrawl الخاصة بك',
        BRAVE_SEARCH_API_KEY: 'مفتاح واجهة بحث بريف من الفئة المجانية',
        EXA_API_KEY: 'مفتاح واجهة Exa البرمجية',
        FIRECRAWL_API_KEY: 'مفتاح واجهة Firecrawl البرمجية',
        PARALLEL_API_KEY: 'مفتاح واجهة Parallel البرمجية',
        SEARXNG_URL: 'رابط نسخة SearXNG',
        TAVILY_API_KEY: 'مفتاح واجهة Tavily البرمجية',
        CAMOFOX_URL: 'رابط خادم Camoufox',
        BROWSER_USE_API_KEY: 'مفتاح واجهة Browser Use البرمجية',
        BROWSERBASE_API_KEY: 'مفتاح واجهة Browserbase البرمجية',
        BROWSERBASE_PROJECT_ID: 'معرّف مشروع Browserbase',
        FAL_KEY: 'مفتاح واجهة FAL البرمجية',
        KREA_API_KEY: 'مفتاح واجهة Krea البرمجية',
        OPENAI_API_KEY: 'مفتاح واجهة OpenAI البرمجية',
        OPENROUTER_API_KEY: 'مفتاح واجهة OpenRouter البرمجية',
        XAI_API_KEY: 'مفتاح واجهة xAI البرمجية',
        VOICE_TOOLS_OPENAI_KEY: 'مفتاح واجهة OpenAI البرمجية',
        ELEVENLABS_API_KEY: 'مفتاح واجهة ElevenLabs البرمجية',
        MISTRAL_API_KEY: 'مفتاح واجهة Mistral البرمجية',
        GEMINI_API_KEY: 'مفتاح واجهة Gemini البرمجية',
        HASS_TOKEN: 'رمز وصول طويل الأمد لمساعد المنزل',
        HASS_URL: 'رابط مساعد المنزل'
      },
      modelDetails: {
        '<1s': 'أقل من ثانية',
        '~6s': 'نحو ست ثوان',
        '~2s': 'نحو ثانيتين',
        '~8s': 'نحو ثماني ثوان',
        '~15s': 'نحو خمس عشرة ثانية',
        '~20s': 'نحو عشرين ثانية',
        '~5s': 'نحو خمس ثوان',
        '~12s': 'نحو اثنتي عشرة ثانية',
        '~15-25s': 'نحو خمس عشرة إلى خمس وعشرين ثانية',
        '~25-60s': 'نحو خمس وعشرين إلى ستين ثانية',
        '~8-15s': 'نحو ثماني إلى خمس عشرة ثانية',
        '~40s': 'نحو أربعين ثانية',
        '~2min': 'نحو دقيقتين',
        '~5-10s': 'نحو خمس إلى عشر ثوان',
        '~10-20s': 'نحو عشر إلى عشرين ثانية',
        '~30-60s': 'نحو ثلاثين إلى ستين ثانية',
        '~30-90s': 'نحو ثلاثين إلى تسعين ثانية',
        '~60-120s': 'نحو ستين إلى مئة وعشرين ثانية',
        '~120-300s': 'نحو مئة وعشرين إلى ثلاثمئة ثانية',
        '~60-240s': 'نحو ستين إلى مئتين وأربعين ثانية',
        'Fast, crisp text': 'سريع مع نص واضح.',
        'Studio photorealism': 'واقعية فوتوغرافية بمستوى الاستوديو.',
        'Bilingual EN/CN, 6B': 'ثنائي اللغة للإنجليزية والصينية، بستة مليارات معلمة.',
        'Gemini 3 Pro, reasoning depth, text rendering': 'Gemini 3 Pro، بعمق استدلال وإخراج جيد للنص.',
        'Prompt adherence': 'التزام قوي بالتوجيه.',
        'SOTA text rendering + CJK, world-aware photorealism':
          'إخراج متقدم للنص ولغات شرق آسيا، مع واقعية فوتوغرافية واعية بالعالم.',
        'Best typography': 'أفضل معالجة للخطوط.',
        'Design, brand systems, production-ready': 'مناسب للتصميم وأنظمة العلامات وجاهز للإنتاج.',
        'LLM-based, complex text': 'مبني على نموذج لغوي وقادر على النصوص المعقدة.',
        'Illustration, anime, painting, expressive/artistic styles':
          'ممتاز للرسوم والأنمي والرسم والأساليب التعبيرية والفنية.',
        'Photorealism, raw textured looks (motion blur, grain, film)':
          'واقعية فوتوغرافية ومظاهر خام ذات نسيج، مثل ضبابية الحركة والحبيبات والفيلم.',
        'Illustration, anime, painting, expressive styles. Faster + cheaper.':
          'للرسوم والأنمي والرسم والأساليب التعبيرية، وهو أسرع وأقل تكلفة.',
        'Photorealism, raw textured looks (motion blur, grain), expressive styles.':
          'واقعية فوتوغرافية ومظاهر خام ذات نسيج وأساليب تعبيرية.',
        'Fastest Krea 2 — medium quality at lower latency / cost.':
          'أسرع فئات Krea 2، بجودة متوسطة وزمن انتظار وتكلفة أقل.',
        'Highest fidelity; best prompt adherence; slower on OpenRouter':
          'أعلى دقة وأفضل التزام بالتوجيه، لكنه أبطأ على OpenRouter.',
        'Fast, reliable fallback with good layout adherence': 'بديل سريع وموثوق مع التزام جيد بالتخطيط.',
        'Fast iteration, lowest cost': 'تكرار سريع بأقل تكلفة.',
        'Balanced — default': 'متوازن، وهو الافتراضي.',
        'Highest fidelity, strongest prompt adherence': 'أعلى دقة وأقوى التزام بالتوجيه.',
        'Fast, high-quality': 'سريع وعالي الجودة.',
        'Higher fidelity / detail; slower than the standard model.': 'دقة وتفاصيل أعلى، لكنه أبطأ من النموذج القياسي.',
        '22B model with native audio generation. Affordable.':
          'نموذج باثنتين وعشرين مليار معلمة مع توليد صوت أصلي وتكلفة معقولة.',
        'Affordable. Negative prompts. 1-15s durations.':
          'تكلفة معقولة، ويدعم التوجيهات السلبية ومددًا من ثانية إلى خمس عشرة ثانية.',
        'Google DeepMind. Cinematic, native audio, strong prompt adherence.':
          'من Google DeepMind؛ سينمائي بصوت أصلي والتزام قوي بالتوجيه.',
        'ByteDance. Cinematic, synchronized audio + lip-sync, 4-15s.':
          'من ByteDance؛ سينمائي بصوت متزامن ومزامنة للشفاه ومدد من أربع إلى خمس عشرة ثانية.',
        '4K output, native audio (Chinese/English), 3-15s.':
          'إخراج فائق الدقة وصوت أصلي بالصينية والإنجليزية، ومدد من ثلاث إلى خمس عشرة ثانية.',
        'Alibaba. New model, sparse public docs — conservative defaults.':
          'من Alibaba؛ نموذج جديد قليل التوثيق العام مع إعدادات افتراضية محافظة.',
        'Text-to-video; legacy image-to-video fallback.': 'تحويل النص إلى فيديو، مع بديل قديم لتحويل الصورة إلى فيديو.',
        'Latest xAI image-to-video model.': 'أحدث نموذج من xAI لتحويل الصورة إلى فيديو.',
        '$0.006/MP': '٠٫٠٠٦ دولار لكل ميجابكسل',
        '$0.03/MP': '٠٫٠٣ دولار لكل ميجابكسل',
        '$0.005/MP': '٠٫٠٠٥ دولار لكل ميجابكسل',
        '$0.15/image (1K)': '٠٫١٥ دولار للصورة بدقة ألف',
        '$0.034/image': '٠٫٠٣٤ دولار للصورة',
        '$0.04–0.06/image': 'من ٠٫٠٤ إلى ٠٫٠٦ دولار للصورة',
        '$0.03-0.09/image': 'من ٠٫٠٣ إلى ٠٫٠٩ دولار للصورة',
        '$0.25/image': '٠٫٢٥ دولار للصورة',
        '$0.02/MP': '٠٫٠٢ دولار لكل ميجابكسل',
        '$0.030 (text) / $0.035 (style refs)': '٠٫٠٣ دولار للنص، و٠٫٠٣٥ لمراجع الأسلوب',
        '$0.060 (text) / $0.065 (style refs)': '٠٫٠٦ دولار للنص، و٠٫٠٦٥ لمراجع الأسلوب',
        '$0.030 (text) / $0.035 (style refs) / $0.040 (moodboards)':
          '٠٫٠٣ دولار للنص، و٠٫٠٣٥ لمراجع الأسلوب، و٠٫٠٤ للوحات المزاج',
        '$0.060 (text) / $0.065 (style refs) / $0.070 (moodboards)':
          '٠٫٠٦ دولار للنص، و٠٫٠٦٥ لمراجع الأسلوب، و٠٫٠٧ للوحات المزاج',
        '$0.015 (text) / $0.0175 (style refs)': '٠٫٠١٥ دولار للنص، و٠٫٠١٧٥ لمراجع الأسلوب',
        varies: 'متغير',
        cheap: 'اقتصادي',
        premium: 'متميز',
        'see https://docs.x.ai/developers/models/grok-imagine-video': 'راجع صفحة نموذج Grok Imagine للفيديو.',
        'see https://docs.x.ai/developers/pricing': 'راجع صفحة تسعير xAI.'
      }
    }
  },

  skills: {
    tabSkills: 'المهارات',
    tabToolsets: 'مجموعات الأدوات',
    tabMcp: 'بروتوكول سياق النموذج',
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
    toolCount: count =>
      count === 1 ? 'أداة واحدة' : count === 2 ? 'أداتان' : count <= 10 ? `${count} أدوات` : `${count} أداة`,
    toolsetNames: {
      web: 'بحث الويب واستخراجه',
      browser: 'أتمتة المتصفح',
      terminal: 'الطرفية والعمليات',
      file: 'عمليات الملفات',
      code_execution: 'تنفيذ الشيفرة',
      vision: 'الرؤية وتحليل الصور',
      video: 'تحليل الفيديو',
      image_gen: 'توليد الصور',
      video_gen: 'توليد الفيديو',
      x_search: 'البحث في إكس',
      tts: 'تحويل النص إلى كلام',
      skills: 'المهارات',
      todo: 'تخطيط المهام',
      memory: 'الذاكرة',
      context_engine: 'محرك السياق',
      session_search: 'البحث في الجلسات',
      clarify: 'أسئلة التوضيح',
      delegation: 'تفويض المهام',
      cronjob: 'المهام المجدولة',
      homeassistant: 'مساعد المنزل',
      spotify: 'Spotify',
      discord: 'ديسكورد للقراءة والمشاركة',
      discord_admin: 'إدارة خادم ديسكورد',
      yuanbao: 'يوانباو',
      computer_use: 'التحكم بالحاسوب'
    },
    toolsetDescriptions: {
      web: 'البحث في الويب واستخراج محتوى الصفحات',
      browser: 'التنقل والنقر والكتابة والتمرير',
      terminal: 'تشغيل أوامر الطرفية وإدارة العمليات',
      file: 'قراءة الملفات وكتابتها وتعديلها والبحث فيها',
      code_execution: 'تنفيذ الشيفرة',
      vision: 'تحليل الصور',
      video: 'تحليل الفيديو باستخدام نموذج يدعمه',
      image_gen: 'توليد الصور',
      video_gen: 'توليد الفيديو من نص أو صورة أو مرجع',
      x_search: 'البحث في منصة إكس',
      tts: 'تحويل النص إلى كلام',
      skills: 'عرض المهارات وإدارتها',
      todo: 'إنشاء خطط المهام ومتابعتها',
      memory: 'ذاكرة مستمرة بين الجلسات',
      context_engine: 'أدوات التشغيل التي يوفرها محرك السياق النشط',
      session_search: 'البحث في المحادثات السابقة',
      clarify: 'طرح أسئلة توضيحية',
      delegation: 'تفويض المهام إلى وكلاء فرعيين',
      cronjob: 'إنشاء المهام المجدولة وعرضها وتحديثها وتشغيلها',
      homeassistant: 'التحكم في أجهزة المنزل الذكي',
      spotify: 'التشغيل والبحث وقوائم التشغيل والمكتبة',
      discord: 'جلب الرسائل والبحث عن الأعضاء وإنشاء سلاسل',
      discord_admin: 'عرض القنوات والأدوار وتثبيت الرسائل وإسناد الأدوار',
      yuanbao: 'معلومات المجموعات واستعلامات الأعضاء والرسائل الخاصة',
      computer_use: 'التحكم في سطح المكتب عبر مشغّل كوا درايفر'
    },
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
    archiveSkillTitle: name => `أرشفة ${name}؟`,
    archiveSkillDescription: 'ستُؤرشف المهارة، ويمكن استعادتها بالأمر `hermes curator restore`.',
    archiveFailed: 'فشلت الأرشفة',
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
      searchFailed: 'فشل البحث في المركز',
      sourceLabels: {
        official: 'الرسمي (Nous)',
        'hermes-index': 'فهرس هرمس',
        'skills-sh': 'سكيلز دوت إس إتش',
        'well-known': 'المصادر المعروفة',
        url: 'رابط مباشر',
        github: 'GitHub',
        clawhub: 'كلاو هب',
        'claude-marketplace': 'سوق Claude',
        lobehub: 'لوب هب',
        'browse-sh': 'براوز دوت إس إتش'
      }
    },
    categoryLabels: {
      apple: 'آبل',
      'autonomous-ai-agents': 'وكلاء مستقلون',
      creative: 'إبداع',
      'data-science': 'علم البيانات',
      email: 'البريد',
      general: 'عام',
      github: 'GitHub',
      media: 'وسائط',
      mlops: 'عمليات تعلم آلي',
      'note-taking': 'تدوين',
      productivity: 'إنتاجية',
      research: 'بحث',
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
    resetToMine: 'العودة إلى مخطّطي',
    skill: 'مهارة',
    profileMemory: 'ذاكرة الملف الشخصي',
    learned: 'متعلَّمة',
    pinned: 'مثبّتة',
    unknown: 'غير معروف',
    coreAge: 'المركز أقدم · الأطراف أحدث',
    playTimeline: 'تشغيل خط الزمن',
    pauseTimeline: 'إيقاف خط الزمن مؤقتا',
    timelineScrubber: 'شريط خط الزمن',
    editNode: kind => (kind === 'skill' ? 'تحرير المهارة…' : 'تحرير الذاكرة…'),
    archiveSkill: 'أرشفة المهارة',
    deleteMemory: 'حذف الذاكرة',
    editTitle: label => `تحرير ${label}`,
    deleteMemoryTitle: label => `حذف ${label}؟`,
    deleteMemoryDescription: 'ستُحذف هذه الذاكرة نهائيا.'
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
    commands: 'الأوامر',
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
      remixConfirmBody: 'يولّد هذا مجموعة جديدة من المسودّات منطلقًا من هذا الشكل. قد يستغرق عدة دقائق.',
      genericError: 'فشل التوليد — أعد المحاولة أو اختر اقتراحًا.',
      referenceImageTooLarge: 'الصورة المرجعية كبيرة جدًّا. استخدم واحدةً أصغر من ١٦ ميغابايت.',
      referenceImageInvalid: 'تعذّر قراءة تلك الصورة المرجعية. جرّب صيغة PNG أو JPG أو WebP أو GIF.',
      reference: 'صورة مرجعية',
      addReference: 'إضافة صورة مرجعية',
      removeReference: 'إزالة الصورة المرجعية',
      unavailableTitle: 'أضف مزود صور للبدء بالتوليد',
      unavailableDesc: 'يتطلب إنشاء حيوان مخصص مزودا يستطيع الاستناد إلى صورة مرجعية.',
      setupImageGeneration: 'إعداد توليد الصور',
      grabKeyFrom: 'احصل على مفتاح من',
      nousPortal: 'Nous Portal',
      openRouter: 'OpenRouter',
      openAi: 'OpenAI',
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
    platformNames: {
      telegram: 'تيليجرام',
      discord: 'ديسكورد',
      slack: 'سلاك',
      mattermost: 'ماترموست',
      matrix: 'ماتريكس',
      signal: 'سيجنال',
      whatsapp: 'واتساب',
      bluebubbles: 'بلو بابلز',
      homeassistant: 'مساعد المنزل',
      email: 'البريد الإلكتروني',
      sms: 'الرسائل النصية عبر تويليو',
      dingtalk: 'دينج توك',
      feishu: 'فيشو ولارك',
      google_chat: 'محادثات Google',
      wecom: 'وي كوم لبوت المجموعة',
      wecom_callback: 'وي كوم للتطبيق',
      weixin: 'وي شين ووي تشات الشخصي',
      qqbot: 'بوت كيو كيو',
      teams: 'مايكروسوفت تيمز',
      yuanbao: 'يوانباو',
      api_server: 'خادم الواجهة البرمجية',
      webhook: 'خطاطيف الويب',
      irc: 'آي آر سي',
      line: 'لاين',
      msgraph_webhook: 'خطاف مايكروسوفت جراف',
      ntfy: 'إن تي إف واي',
      photon: 'فوتون',
      raft: 'رافت',
      relay: 'ريلاي',
      simplex: 'سيمبلكس',
      whatsapp_cloud: 'واتساب السحابي'
    },
    platformDescriptions: {
      telegram: 'شغّل هرمس من الرسائل الخاصة والمجموعات والموضوعات في تيليجرام.',
      discord: 'اربط هرمس بالرسائل الخاصة والقنوات والمواضيع في ديسكورد.',
      slack: 'استخدم هرمس من سلاك عبر وضع المقبس، وحدد أعضاء سلاك المسموح للبوت بالرد عليهم.',
      mattermost: 'اربط هرمس بقنوات ماترموست ورسائله المباشرة.',
      matrix: 'استخدم هرمس في غرف ماتريكس ورسائله المباشرة.',
      signal: 'اتصل عبر جسر سيجنال ذي واجهة نقل الحالة التمثيلية.',
      whatsapp: 'استخدم هرمس عبر جسر واتساب المرفق مع مصادقة برمز الاستجابة السريعة.',
      bluebubbles: 'استخدم هرمس عبر رسائل آبل من خلال خادم بلو بابلز.',
      homeassistant: 'تحكم في منزلك الذكي من هرمس عبر مساعد المنزل.',
      email: 'تحدث إلى هرمس عبر صندوق بريد يدعم بروتوكولي الاستقبال والإرسال.',
      sms: 'أرسل الرسائل النصية واستقبلها عبر تويليو.',
      dingtalk: 'اربط هرمس بمجموعات دينج توك.',
      feishu: 'استخدم هرمس داخل فيشو أو لارك.',
      google_chat: 'اربط هرمس بمحادثات Google عبر النشر والاشتراك السحابي.',
      wecom: 'بوت مجموعة في وي كوم للإرسال فقط عبر خطاف ويب.',
      wecom_callback: 'تكامل ثنائي الاتجاه مع وي كوم عبر تطبيق رد الاتصال.',
      weixin: 'اربط حساب وي تشات شخصيا عبر واجهة بوت آي لينك من تنسنت.',
      qqbot: 'اربط هرمس ببوت كيو كيو من منصته المفتوحة.',
      teams: 'اربط هرمس بقنوات مايكروسوفت تيمز ومحادثاته.',
      yuanbao: 'اربط هرمس بخدمة يوانباو من تنسنت.',
      api_server: 'اعرض هرمس كواجهة برمجية متوافقة مع OpenAI لأدوات مثل Open WebUI.',
      webhook: 'استقبل الأحداث من GitHub وجيت لاب ومصادر خطاطيف الويب الأخرى.',
      irc: 'اربط هرمس بقنوات آي آر سي ورسائلها الخاصة.',
      line: 'استخدم هرمس في محادثات لاين ومجموعاتها.',
      msgraph_webhook: 'استقبل أحداث مايكروسوفت جراف عبر خطاف ويب.',
      ntfy: 'تبادل الرسائل مع هرمس عبر موضوعات إن تي إف واي.',
      photon: 'اربط هرمس بمنصة فوتون سبيكترم.',
      raft: 'اربط هرمس بمنصة رافت.',
      relay: 'اربط هرمس بمنصة ريلاي.',
      simplex: 'استخدم هرمس عبر جهات اتصال ومجموعات سيمبلكس.',
      whatsapp_cloud: 'اربط هرمس بواجهة واتساب السحابية.'
    },
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
      TELEGRAM_ALLOW_ALL_USERS: {
        label: 'السماح لجميع مستخدمي تيليجرام',
        help: 'للتطوير فقط. عند تفعيله يستطيع أي مستخدم في تيليجرام تشغيل البوت.',
        placeholder: 'فعّل أو عطّل'
      },
      TELEGRAM_HOME_CHANNEL: {
        label: 'معرّف المحادثة الرئيسية',
        help: 'المحادثة الافتراضية لتسليم مخرجات المهام المجدولة والإشعارات.'
      },
      TELEGRAM_HOME_CHANNEL_NAME: {
        label: 'اسم المحادثة الرئيسية',
        help: 'الاسم المعروض للمحادثة الرئيسية في السجلات والحالة.',
        placeholder: 'اسم المحادثة الرئيسية'
      },
      DISCORD_BOT_TOKEN: {
        label: 'رمز البوت',
        help: 'أنشئ تطبيقًا في بوابة مطوري ديسكورد، وأضف بوتًا، ثم الصق رمزه.'
      },
      DISCORD_ALLOWED_USERS: {
        label: 'معرّفات مستخدمي ديسكورد المسموح لهم',
        help: 'موصى به. معرّفات مستخدمي ديسكورد مفصولة بفواصل.'
      },
      DISCORD_REPLY_TO_MODE: { label: 'أسلوب الرد', help: 'الرسالة الأولى أو كل الرسائل أو الإيقاف.' },
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
        label: 'السماح لجميع مستخدمي آي مسج',
        help: 'عند تفعيله تُتجاوز قائمة سماح بلو بابلز.'
      },
      MATTERMOST_ALLOW_ALL_USERS: { label: 'السماح لجميع مستخدمي ماترموست' },
      MATTERMOST_HOME_CHANNEL: { label: 'القناة الرئيسية' },
      QQ_ALLOW_ALL_USERS: { label: 'السماح لجميع مستخدمي كيو كيو' },
      QQBOT_HOME_CHANNEL: {
        label: 'قناة كيو كيو الرئيسية',
        help: 'القناة أو المجموعة الافتراضية لتسليم المهام المجدولة.'
      },
      QQBOT_HOME_CHANNEL_NAME: { label: 'اسم قناة كيو كيو الرئيسية' },
      SLACK_BOT_TOKEN: {
        label: 'رمز بوت سلاك',
        help: 'استخدم رمز البوت من صفحة أو أوث والأذونات بعد تثبيت تطبيق سلاك.',
        placeholder: 'الصق رمز بوت سلاك'
      },
      SLACK_APP_TOKEN: {
        label: 'رمز تطبيق سلاك',
        help: 'استخدم رمز مستوى التطبيق المطلوب لوضع المقبس.',
        placeholder: 'الصق رمز تطبيق سلاك'
      },
      SLACK_ALLOWED_USERS: {
        label: 'معرّفات مستخدمي سلاك المسموح لهم',
        help: 'موصى به. معرّفات مستخدمي سلاك مفصولة بفواصل.'
      },
      SLACK_ALLOW_ALL_USERS: {
        label: 'السماح لجميع مستخدمي سلاك',
        help: 'للتطوير فقط. عند تفعيله يستطيع أي مستخدم في سلاك تشغيل البوت.'
      },
      SLACK_HOME_CHANNEL: {
        label: 'معرّف قناة سلاك الرئيسية',
        help: 'القناة الافتراضية لتسليم مخرجات المهام المجدولة والإشعارات.'
      },
      SLACK_HOME_CHANNEL_NAME: {
        label: 'اسم قناة سلاك الرئيسية',
        help: 'الاسم المعروض لقناة سلاك الرئيسية.'
      },
      MATTERMOST_URL: {
        label: 'رابط الخادم',
        help: 'رابط خادم ماترموست، مثل الرابط الذي يبدأ ببروتوكول الاتصال الآمن.',
        placeholder: 'https://mattermost.example.com'
      },
      MATTERMOST_TOKEN: {
        label: 'رمز البوت',
        help: 'رمز بوت ماترموست أو رمز وصول شخصي.'
      },
      MATTERMOST_ALLOWED_USERS: {
        label: 'معرّفات المستخدمين المسموح لهم',
        help: 'موصى به. معرّفات مستخدمي ماترموست مفصولة بفواصل.'
      },
      MATTERMOST_ALLOWED_CHANNELS: {
        label: 'معرّفات القنوات المسموح بها',
        help: 'عند ضبطها لا يستجيب البوت إلا في هذه القنوات.'
      },
      MATTERMOST_FREE_RESPONSE_CHANNELS: {
        label: 'قنوات الاستجابة الحرة',
        help: 'قنوات ماترموست التي يستجيب فيها البوت دون الحاجة إلى الإشارة إليه.'
      },
      MATTERMOST_REPLY_MODE: {
        label: 'أسلوب الرد',
        help: 'اختر الرد داخل سلسلة أو الرد المسطح.'
      },
      MATTERMOST_REQUIRE_MENTION: {
        label: 'اشتراط الإشارة في القنوات',
        help: 'عند تفعيله لا يستجيب البوت في القنوات إلا عند الإشارة إليه.'
      },
      MATRIX_HOMESERVER: {
        label: 'رابط الخادم الرئيسي',
        help: 'رابط الخادم الرئيسي لشبكة ماتريكس.',
        placeholder: 'https://matrix.org'
      },
      MATRIX_ACCESS_TOKEN: {
        label: 'رمز الوصول',
        help: 'رمز وصول ماتريكس، وهو مفضّل على تسجيل الدخول بكلمة المرور.'
      },
      MATRIX_USER_ID: {
        label: 'معرّف مستخدم البوت',
        help: 'معرّف مستخدم ماتريكس الكامل للبوت.',
        placeholder: '@hermes:example.org'
      },
      MATRIX_ALLOWED_USERS: {
        label: 'معرّفات مستخدمي ماتريكس المسموح لهم',
        help: 'موصى به. معرّفات مفصولة بفواصل بصيغة @user:server.'
      },
      MATRIX_ALLOW_ALL_USERS: {
        label: 'السماح لجميع مستخدمي ماتريكس',
        help: 'للتطوير فقط. عند تفعيله يستطيع أي مستخدم تشغيل البوت.'
      },
      MATRIX_AUTO_THREAD: {
        label: 'إنشاء سلاسل تلقائيًا في الغرف',
        help: 'ينشئ سلسلة تلقائيًا لرسائل غرف ماتريكس.'
      },
      MATRIX_DEVICE_ID: {
        label: 'معرّف جهاز ماتريكس',
        help: 'معرّف ثابت يحافظ على مفاتيح التشفير بين مرات التشغيل.'
      },
      MATRIX_DM_AUTO_THREAD: {
        label: 'إنشاء سلاسل تلقائيًا في الرسائل الخاصة',
        help: 'ينشئ سلسلة تلقائيًا لرسائل ماتريكس الخاصة.'
      },
      MATRIX_FREE_RESPONSE_ROOMS: {
        label: 'غرف الاستجابة الحرة',
        help: 'غرف ماتريكس التي يستجيب فيها البوت دون الحاجة إلى الإشارة إليه.'
      },
      MATRIX_HOME_CHANNEL: {
        label: 'معرّف الغرفة الرئيسية',
        help: 'الغرفة الافتراضية لتسليم مخرجات المهام المجدولة والإشعارات.'
      },
      MATRIX_HOME_CHANNEL_NAME: {
        label: 'اسم الغرفة الرئيسية',
        help: 'الاسم المعروض لغرفة ماتريكس الرئيسية.'
      },
      MATRIX_PASSWORD: {
        label: 'كلمة مرور ماتريكس',
        help: 'بديل لتسجيل الدخول برمز الوصول.'
      },
      MATRIX_RECOVERY_KEY: {
        label: 'مفتاح استعادة ماتريكس',
        help: 'مفتاح استعادة التحقق بعد تبديل مفاتيح الجهاز.'
      },
      MATRIX_REQUIRE_MENTION: {
        label: 'اشتراط الإشارة في الغرف',
        help: 'عند تفعيله لا يستجيب البوت في الغرف إلا عند الإشارة إليه.'
      },
      SIGNAL_HTTP_URL: {
        label: 'رابط جسر سيجنال',
        placeholder: 'http://127.0.0.1:8080',
        help: 'رابط جسر سيجنال البرمجي الذي يعمل حاليًا.'
      },
      SIGNAL_ACCOUNT: { label: 'رقم الهاتف', help: 'الرقم المسجل في جسر سيجنال.' },
      SIGNAL_ALLOWED_USERS: {
        label: 'مستخدمو سيجنال المسموح لهم',
        help: 'موصى به. معرّفات سيجنال مفصولة بفواصل.'
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
      WHATSAPP_ALLOW_ALL_USERS: {
        label: 'السماح لجميع مستخدمي واتساب',
        help: 'للتطوير فقط. عند تفعيله يستطيع أي مستخدم تشغيل البوت.'
      },
      WHATSAPP_DM_POLICY: {
        label: 'سياسة الرسائل الخاصة',
        help: 'تحدد كيفية السماح برسائل واتساب الخاصة.'
      },
      WHATSAPP_HOME_CHANNEL: {
        label: 'معرّف المحادثة الرئيسية',
        help: 'المحادثة الافتراضية لتسليم مخرجات المهام المجدولة والإشعارات.'
      },
      WHATSAPP_HOME_CHANNEL_NAME: {
        label: 'اسم المحادثة الرئيسية',
        help: 'الاسم المعروض لمحادثة واتساب الرئيسية.'
      },
      HASS_URL: {
        label: 'رابط مساعد المنزل',
        help: 'الرابط الأساسي لخادم مساعد المنزل، مثل https://homeassistant.local:8123.'
      },
      HASS_TOKEN: {
        label: 'رمز وصول مساعد المنزل',
        help: 'رمز وصول طويل الأمد من مساعد المنزل، من الملف الشخصي ثم الأمان.'
      },
      EMAIL_ADDRESS: { label: 'عنوان البريد الإلكتروني', help: 'العنوان الذي يرسل هرمس منه ويستقبل عليه.' },
      EMAIL_PASSWORD: { label: 'كلمة مرور البريد', help: 'كلمة مرور الحساب أو كلمة مرور تطبيق مخصصة.' },
      EMAIL_IMAP_HOST: { label: 'خادم استقبال البريد', help: 'مضيف خادم استقبال البريد، مثل imap.gmail.com.' },
      EMAIL_SMTP_HOST: { label: 'خادم إرسال البريد', help: 'مضيف خادم إرسال البريد، مثل smtp.gmail.com.' },
      EMAIL_ALLOWED_USERS: {
        label: 'عناوين البريد المسموح لها',
        help: 'عناوين البريد الإلكتروني المسموح لها بمراسلة البوت، مفصولة بفواصل.'
      },
      EMAIL_HOME_ADDRESS: {
        label: 'عنوان البريد الرئيسي',
        help: 'العنوان الافتراضي لتسليم مخرجات المهام المجدولة والإشعارات.'
      },
      EMAIL_SMTP_PORT: {
        label: 'منفذ خادم الإرسال',
        help: 'منفذ خادم إرسال البريد، وقيمته الافتراضية ٥٨٧.'
      },
      TWILIO_ACCOUNT_SID: {
        label: 'معرّف حساب تويليو',
        help: 'من لوحة تحكم تويليو.'
      },
      TWILIO_AUTH_TOKEN: {
        label: 'رمز مصادقة تويليو',
        help: 'من لوحة تحكم تويليو.'
      },
      TWILIO_PHONE_NUMBER: {
        label: 'رقم هاتف تويليو',
        help: 'رقم قادر على إرسال الرسائل النصية بصيغة دولية.'
      },
      DINGTALK_CLIENT_ID: { label: 'معرّف العميل', help: 'معرّف عميل دينج توك، وهو مفتاح التطبيق.' },
      DINGTALK_CLIENT_SECRET: { label: 'سر العميل', help: 'سر عميل دينج توك، وهو سر التطبيق.' },
      DINGTALK_ALLOWED_USERS: {
        label: 'المستخدمون المسموح لهم',
        help: 'معرّفات الموظفين أو المرسلين المسموح لهم بمراسلة البوت، مفصولة بفواصل.'
      },
      DINGTALK_HOME_CHANNEL: {
        label: 'معرّف المحادثة الرئيسية',
        help: 'المحادثة الافتراضية لتسليم مخرجات المهام المجدولة والإشعارات.'
      },
      DINGTALK_HOME_CHANNEL_NAME: {
        label: 'اسم المحادثة الرئيسية',
        help: 'الاسم المعروض لمحادثة دينج توك الرئيسية.'
      },
      DINGTALK_WEBHOOK_URL: {
        label: 'رابط خطاف روبوت دينج توك',
        help: 'رابط اختياري لروبوت ثابت يُستخدم في التسليم بين المنصات والمهام المجدولة.'
      },
      FEISHU_APP_ID: { label: 'معرّف التطبيق', help: 'معرّف تطبيق فيشو أو لارك.' },
      FEISHU_APP_SECRET: { label: 'سر التطبيق', help: 'سر تطبيق فيشو أو لارك.' },
      FEISHU_ENCRYPT_KEY: { label: 'مفتاح التشفير', help: 'مفتاح تشفير أحداث فيشو أو لارك.' },
      FEISHU_VERIFICATION_TOKEN: { label: 'رمز التحقق', help: 'رمز تحقق فيشو أو لارك.' },
      FEISHU_ALLOWED_USERS: {
        label: 'مستخدمو فيشو المسموح لهم',
        help: 'معرّفات مستخدمي فيشو المسموح لهم بمراسلة البوت، مفصولة بفواصل.'
      },
      FEISHU_ALLOW_ALL_USERS: {
        label: 'السماح لجميع مستخدمي فيشو',
        help: 'للتطوير فقط. عند تفعيله يستطيع أي مستخدم تشغيل البوت.'
      },
      FEISHU_DOMAIN: {
        label: 'النطاق',
        help: 'اختر فيشو للصين أو لارك للنطاق الدولي.'
      },
      FEISHU_HOME_CHANNEL: {
        label: 'معرّف المحادثة الرئيسية',
        help: 'المحادثة الافتراضية لتسليم مخرجات المهام المجدولة والإشعارات.'
      },
      FEISHU_HOME_CHANNEL_NAME: {
        label: 'اسم المحادثة الرئيسية',
        help: 'الاسم المعروض لمحادثة فيشو الرئيسية.'
      },
      GOOGLE_CHAT_ALLOWED_USERS: {
        label: 'عناوين البريد المسموح لها',
        help: 'عناوين بريد المستخدمين المسموح لهم بالتفاعل مع البوت، مفصولة بفواصل.'
      },
      GOOGLE_CHAT_HOME_CHANNEL: {
        label: 'معرّف المساحة الرئيسية',
        help: 'المساحة الافتراضية لتسليم مخرجات المهام المجدولة والإشعارات.'
      },
      GOOGLE_CHAT_PROJECT_ID: {
        label: 'معرّف مشروع Google السحابي',
        help: 'المشروع الذي يستضيف موضوع النشر والاشتراك لأحداث المحادثة.'
      },
      GOOGLE_CHAT_SERVICE_ACCOUNT_JSON: {
        label: 'مسار مفتاح حساب الخدمة',
        help: 'مسار ملف مفتاح حساب الخدمة أو محتواه. اتركه فارغًا لاستخدام بيانات الاعتماد الافتراضية.'
      },
      GOOGLE_CHAT_SUBSCRIPTION_NAME: {
        label: 'اسم اشتراك النشر والاشتراك',
        help: 'المسار الكامل للاشتراك الذي يستقبل أحداث محادثات Google.'
      },
      WECOM_BOT_ID: { label: 'معرّف بوت وي كوم', help: 'مفتاح خطاف الويب لروبوت المجموعة في وي كوم.' },
      WECOM_SECRET: { label: 'سر وي كوم', help: 'سر روبوت المجموعة في وي كوم.' },
      WECOM_CALLBACK_CORP_ID: { label: 'معرّف مؤسسة وي كوم', help: 'معرّف المؤسسة في وي كوم.' },
      WECOM_CALLBACK_CORP_SECRET: { label: 'سر مؤسسة وي كوم', help: 'سر تطبيق وي كوم الخاص بالمؤسسة.' },
      WECOM_CALLBACK_AGENT_ID: {
        label: 'معرّف وكيل وي كوم',
        help: 'معرّف الوكيل لتطبيق وي كوم الذاتي البناء.'
      },
      WECOM_CALLBACK_TOKEN: { label: 'رمز وي كوم', help: 'رمز التحقق من الاستدعاء الراجع في وي كوم.' },
      WECOM_CALLBACK_ENCODING_AES_KEY: {
        label: 'مفتاح تشفير وي كوم',
        help: 'مفتاح تشفير الاستدعاء الراجع في وي كوم.'
      },
      WEIXIN_ACCOUNT_ID: { label: 'معرّف الحساب', help: 'معرّف الحساب الرسمي في وي تشات.' },
      WEIXIN_TOKEN: { label: 'رمز الاستدعاء الراجع', help: 'رمز الاستدعاء الراجع في وي تشات.' },
      WEIXIN_BASE_URL: { label: 'الرابط الأساسي', help: 'الرابط الأساسي لمنصة وي تشات.' },
      BLUEBUBBLES_SERVER_URL: {
        label: 'رابط خادم بلو بابلز',
        help: 'رابط خادم بلو بابلز لتكامل آي مسج، مثل http://192.168.1.10:1234.'
      },
      BLUEBUBBLES_PASSWORD: {
        label: 'كلمة مرور خادم بلو بابلز',
        help: 'من إعدادات واجهة خادم بلو بابلز البرمجية.'
      },
      BLUEBUBBLES_ALLOWED_USERS: {
        label: 'عناوين آي مسج المسموح لها',
        help: 'موصى به. عناوين آي مسج، من بريد إلكتروني أو هاتف، مفصولة بفواصل.'
      },
      QQ_APP_ID: { label: 'معرّف تطبيق كيو كيو', help: 'معرّف التطبيق من منصة كيو كيو المفتوحة.' },
      QQ_CLIENT_SECRET: { label: 'سر عميل كيو كيو', help: 'سر العميل من منصة كيو كيو المفتوحة.' },
      QQ_ALLOWED_USERS: {
        label: 'معرّفات مستخدمي كيو كيو المسموح لهم',
        help: 'موصى به. معرّفات مستخدمي كيو كيو مفصولة بفواصل.'
      },
      QQ_GROUP_ALLOWED_USERS: {
        label: 'معرّفات مجموعات كيو كيو المسموح لها',
        help: 'معرّفات المجموعات المسموح لها بالتفاعل مع البوت، مفصولة بفواصل.'
      },
      QQ_SANDBOX: {
        label: 'وضع كيو كيو التجريبي',
        help: 'يفعّل وضع الاختبار أثناء التطوير.'
      },
      IRC_ALLOWED_USERS: {
        label: 'الأسماء المستعارة المسموح لها',
        help: 'الأسماء المستعارة المسموح لها بمراسلة البوت، مفصولة بفواصل.'
      },
      IRC_ALLOW_ALL_USERS: {
        label: 'السماح لجميع مستخدمي آي آر سي',
        help: 'للتطوير فقط. عند تفعيله يستطيع أي شخص في القناة مراسلة البوت.'
      },
      IRC_CHANNEL: {
        label: 'قناة آي آر سي',
        help: 'القناة التي ينضم إليها البوت.'
      },
      IRC_HOME_CHANNEL: {
        label: 'القناة الرئيسية',
        help: 'القناة الافتراضية لتسليم مخرجات المهام المجدولة والإشعارات.'
      },
      IRC_NICKNAME: {
        label: 'الاسم المستعار للبوت',
        help: 'الاسم الذي يظهر به البوت في آي آر سي.'
      },
      IRC_NICKSERV_PASSWORD: {
        label: 'كلمة مرور خدمة الأسماء',
        help: 'كلمة مرور تعريف الاسم المستعار عند الحاجة.'
      },
      IRC_PORT: {
        label: 'منفذ آي آر سي',
        help: 'منفذ الخادم، ويُستخدم المنفذ الآمن افتراضيًا.'
      },
      IRC_SERVER: {
        label: 'خادم آي آر سي',
        help: 'اسم مضيف خادم آي آر سي.'
      },
      IRC_SERVER_PASSWORD: {
        label: 'كلمة مرور الخادم',
        help: 'كلمة مرور خادم آي آر سي عند الحاجة.'
      },
      IRC_USE_TLS: {
        label: 'استخدام الاتصال الآمن',
        help: 'يشفّر الاتصال بخادم آي آر سي.'
      },
      LINE_ALLOWED_GROUPS: {
        label: 'معرّفات مجموعات لاين المسموح لها',
        help: 'المجموعات التي يستجيب فيها البوت، مفصولة بفواصل.'
      },
      LINE_ALLOWED_ROOMS: {
        label: 'معرّفات غرف لاين المسموح لها',
        help: 'الغرف التي يستجيب فيها البوت، مفصولة بفواصل.'
      },
      LINE_ALLOWED_USERS: {
        label: 'معرّفات مستخدمي لاين المسموح لهم',
        help: 'المستخدمون المسموح لهم بمراسلة البوت مباشرة، مفصولون بفواصل.'
      },
      LINE_ALLOW_ALL_USERS: {
        label: 'السماح لجميع مستخدمي لاين',
        help: 'للتطوير فقط. يعطّل قائمة السماح.'
      },
      LINE_CHANNEL_ACCESS_TOKEN: {
        label: 'رمز وصول قناة لاين',
        help: 'رمز الوصول طويل الأمد من لوحة مطوري لاين.'
      },
      LINE_CHANNEL_SECRET: {
        label: 'سر قناة لاين',
        help: 'يُستخدم للتحقق من توقيع خطاف الويب.'
      },
      LINE_HOME_CHANNEL: {
        label: 'معرّف القناة الرئيسية',
        help: 'المستخدم أو المجموعة أو الغرفة الافتراضية لتسليم الإشعارات والمهام المجدولة.'
      },
      LINE_HOST: {
        label: 'مضيف خطاف الويب',
        help: 'عنوان ربط خادم خطاف الويب.'
      },
      LINE_PORT: {
        label: 'منفذ خطاف الويب',
        help: 'منفذ استماع خادم خطاف الويب.'
      },
      LINE_PUBLIC_URL: {
        label: 'الرابط العام الآمن',
        help: 'الرابط العام المستخدم لإرسال الصور والصوت والفيديو إلى لاين.'
      },
      LINE_SLOW_RESPONSE_THRESHOLD: {
        label: 'عتبة الاستجابة البطيئة',
        help: 'عدد الثواني قبل إظهار زر انتظار الاستجابة الطويلة.'
      },
      NTFY_ALLOWED_USERS: {
        label: 'الموضوعات المسموح لها',
        help: 'أسماء الموضوعات المسموح لها بمراسلة البوت، مفصولة بفواصل.'
      },
      NTFY_ALLOW_ALL_USERS: {
        label: 'السماح لجميع الموضوعات',
        help: 'للتطوير فقط. يعطّل قائمة السماح.'
      },
      NTFY_HOME_CHANNEL: {
        label: 'الموضوع الرئيسي',
        help: 'الموضوع الافتراضي لتسليم الإشعارات والمهام المجدولة.'
      },
      NTFY_HOME_CHANNEL_NAME: {
        label: 'اسم الموضوع الرئيسي',
        help: 'الاسم البشري المعروض للموضوع الرئيسي.'
      },
      NTFY_MARKDOWN: {
        label: 'تنسيق الردود',
        help: 'يرسل الردود بتنسيق النص المنسق عند تفعيله.'
      },
      NTFY_PUBLISH_TOPIC: {
        label: 'موضوع نشر الردود',
        help: 'الموضوع الذي تُنشر إليه ردود البوت.'
      },
      NTFY_SERVER_URL: {
        label: 'رابط خادم إن تي إف واي',
        help: 'الرابط الأساسي للخادم.'
      },
      NTFY_TOKEN: {
        label: 'رمز مصادقة إن تي إف واي',
        help: 'رمز المصادقة أو بيانات المستخدم وكلمة المرور.'
      },
      NTFY_TOPIC: {
        label: 'موضوع الاشتراك',
        help: 'اسم الموضوع الذي يشترك فيه البوت.'
      },
      PHOTON_ALLOWED_USERS: {
        label: 'المستخدمون المسموح لهم',
        help: 'أرقام الهواتف الدولية المسموح لها بمراسلة البوت، مفصولة بفواصل.'
      },
      PHOTON_ALLOW_ALL_USERS: {
        label: 'السماح لجميع المرسلين',
        help: 'للتطوير فقط. يعطّل قائمة السماح.'
      },
      PHOTON_DASHBOARD_HOST: {
        label: 'مضيف لوحة فوتون',
        help: 'عنوان الواجهة البرمجية للوحة فوتون.'
      },
      PHOTON_HOME_CHANNEL: {
        label: 'وجهة فوتون الرئيسية',
        help: 'الوجهة الافتراضية لتسليم الإشعارات والمهام المجدولة.'
      },
      PHOTON_HOME_CHANNEL_NAME: {
        label: 'اسم الوجهة الرئيسية',
        help: 'الاسم البشري المعروض للوجهة الرئيسية.'
      },
      PHOTON_MARKDOWN: {
        label: 'عرض الردود كنص منسق',
        help: 'يرسل ردود الوكيل بتنسيق منسق حيث تدعمه المنصة.'
      },
      PHOTON_MENTION_PATTERNS: {
        label: 'أنماط الإشارة في المجموعات',
        help: 'أنماط الكلمات التي توقظ البوت داخل المحادثات الجماعية.'
      },
      PHOTON_NODE_BIN: {
        label: 'مسار مشغّل نود',
        help: 'مسار الملف التنفيذي لنود.'
      },
      PHOTON_PROJECT_ID: {
        label: 'معرّف مشروع فوتون',
        help: 'معرّف مشروع سبيكترم المرتبط بالتكامل.'
      },
      PHOTON_PROJECT_SECRET: {
        label: 'سر مشروع فوتون',
        help: 'السر المقترن بمعرّف مشروع سبيكترم.'
      },
      PHOTON_REACTIONS: {
        label: 'تفعيل تفاعلات الحالة',
        help: 'يستخدم التفاعلات لإظهار حالة المعالجة وتمرير تفاعلات المستخدم إلى الوكيل.'
      },
      PHOTON_REQUIRE_MENTION: {
        label: 'اشتراط الإشارة في المجموعات',
        help: 'يتجاهل رسائل المجموعات التي لا تطابق كلمة إيقاظ.'
      },
      PHOTON_SIDECAR_AUTOSTART: {
        label: 'تشغيل الخدمة المرافقة تلقائيًا',
        help: 'يشغّل خدمة نود المرافقة عند الاتصال.'
      },
      PHOTON_SIDECAR_PORT: {
        label: 'منفذ الخدمة المرافقة',
        help: 'منفذ التحكم المحلي بالخدمة المرافقة.'
      },
      PHOTON_SPECTRUM_HOST: {
        label: 'مضيف فوتون سبيكترم',
        help: 'عنوان الواجهة البرمجية لفوتون سبيكترم.'
      },
      PHOTON_TELEMETRY: {
        label: 'تفعيل القياسات',
        help: 'يرسل قياسات حزمة سبيكترم من الخدمة المرافقة.'
      },
      SIMPLEX_ALLOWED_USERS: {
        label: 'جهات الاتصال المسموح لها',
        help: 'معرّفات جهات اتصال سيمبلكس المسموح لها بمراسلة البوت.'
      },
      SIMPLEX_ALLOW_ALL_USERS: {
        label: 'السماح لجميع جهات الاتصال',
        help: 'للتطوير فقط. يعطّل قائمة السماح.'
      },
      SIMPLEX_AUTO_ACCEPT: {
        label: 'قبول طلبات الاتصال تلقائيًا',
        help: 'يقبل طلبات الاتصال الواردة تلقائيًا.'
      },
      SIMPLEX_GROUP_ALLOWED: {
        label: 'المجموعات المسموح بها',
        help: 'معرّفات مجموعات سيمبلكس التي يشارك فيها البوت.'
      },
      SIMPLEX_HOME_CHANNEL: {
        label: 'جهة الاتصال أو المجموعة الرئيسية',
        help: 'الوجهة الافتراضية لتسليم الإشعارات والمهام المجدولة.'
      },
      SIMPLEX_HOME_CHANNEL_NAME: {
        label: 'اسم الوجهة الرئيسية',
        help: 'الاسم البشري المعروض للوجهة الرئيسية.'
      },
      SIMPLEX_WS_URL: {
        label: 'رابط خدمة سيمبلكس',
        help: 'رابط مقبس الويب لخدمة محادثات سيمبلكس.'
      },
      TEAMS_ALLOWED_USERS: {
        label: 'مستخدمو تيمز المسموح لهم',
        help: 'معرّفات مستخدمي تيمز أو أسماؤهم الرئيسية، مفصولة بفواصل.'
      },
      TEAMS_ALLOW_ALL_USERS: {
        label: 'السماح لجميع مستخدمي تيمز',
        help: 'للتطوير فقط. عند تفعيله يستطيع أي مستخدم تشغيل البوت.'
      },
      TEAMS_CLIENT_ID: {
        label: 'معرّف تطبيق تيمز',
        help: 'معرّف تطبيق مايكروسوفت المستخدم في إطار البوت.'
      },
      TEAMS_CLIENT_SECRET: {
        label: 'سر تطبيق تيمز',
        help: 'سر تطبيق مايكروسوفت المستخدم في إطار البوت.'
      },
      TEAMS_HOME_CHANNEL: {
        label: 'القناة الرئيسية',
        help: 'المحادثة أو القناة الافتراضية لتسليم الإشعارات والمهام المجدولة.'
      },
      TEAMS_HOME_CHANNEL_NAME: {
        label: 'اسم القناة الرئيسية',
        help: 'الاسم المعروض لقناة تيمز الرئيسية.'
      },
      TEAMS_PORT: {
        label: 'منفذ خطاف الويب',
        help: 'منفذ استماع إطار البوت.'
      },
      TEAMS_TENANT_ID: {
        label: 'معرّف مستأجر مايكروسوفت',
        help: 'معرّف المستأجر الذي يستضيف تطبيق البوت.'
      },
      API_SERVER_ENABLED: {
        label: 'تفعيل خادم الواجهة البرمجية',
        help: 'يفعّل الواجهة البرمجية المتوافقة مع OpenAI لتتصل بها أدوات مثل Open WebUI ولوب تشات.'
      },
      API_SERVER_KEY: {
        label: 'مفتاح مصادقة الواجهة البرمجية',
        help: 'رمز حامل لمصادقة الخادم. مطلوب متى كان الخادم مفعّلًا، ويرفض الخادم البدء من دونه.'
      },
      API_SERVER_PORT: { label: 'منفذ الخادم', help: 'منفذ خادم الواجهة البرمجية (الافتراضي 8642).' },
      API_SERVER_HOST: {
        label: 'مضيف الخادم',
        help: 'عنوان الربط المحلي. يبقى مفتاح مصادقة الخادم مطلوبًا حتى على الربط المحلي.'
      },
      API_SERVER_MODEL_NAME: {
        label: 'اسم النموذج المعلن',
        help: 'الاسم المعروض في مسار النماذج. الافتراضي اسم الملف الشخصي.'
      },
      WEBHOOK_ENABLED: {
        label: 'تفعيل الويب هوك',
        help: 'يفعّل محوّل خطاف الويب لاستقبال الأحداث من GitHub وجيت لاب وغيرهما.'
      },
      WEBHOOK_PORT: { label: 'منفذ الويب هوك', help: 'منفذ خادم الويب للخطاف، وقيمته الافتراضية ٨٦٤٤.' },
      WEBHOOK_SECRET: { label: 'سر الويب هوك', help: 'سر عام للتحقق من توقيعات خطاف الويب.' }
    },
    platformIntro: {
      telegram:
        'في تيليجرام، راسل @BotFather ونفّذ الأمر /newbot وانسخ الرمز الذي يعطيك إياه، ثم خذ معرّفك الرقمي من @userinfobot.',
      discord:
        'افتح بوابة مطوري ديسكورد، وأنشئ تطبيقًا، وأضف إليه بوتًا، ثم انسخ رمزه. وادعُ البوت إلى خادمك بالنطاقات الصحيحة.',
      slack: 'أنشئ تطبيق سلاك، وفعّل وضع المقبس، وثبّته في مساحة العمل، ثم انسخ رمز البوت ورمز مستوى التطبيق.',
      mattermost: 'على خادم ماترموست لديك، أنشئ حساب بوت أو رمز وصول شخصيًا، ثم الصق رابط الخادم والرمز هنا.',
      matrix: 'سجّل الدخول إلى خادمك الرئيسي بحساب البوت، ثم انسخ رمز الوصول ومعرّف المستخدم ورابط الخادم الرئيسي.',
      google_chat: 'اربط هرمس بمحادثات Google عبر خدمة النشر والاشتراك السحابية.',
      signal: 'شغّل جسر سيجنال البرمجي في مكان يمكن الوصول إليه، ثم وجّه هرمس إلى الرابط ورقم الهاتف المسجل.',
      whatsapp: 'شغّل جسر واتساب المرفق مع هرمس، وامسح رمز الاستجابة السريعة عند أول تشغيل، ثم فعّل المنصة.',
      bluebubbles:
        'شغّل خادم بلو بابلز على جهاز ماك يعمل عليه آي مسج، واكشف واجهته البرمجية، ثم وجّه هرمس إلى الرابط مع كلمة مرور الخادم.',
      homeassistant: 'في مساعد المنزل، افتح ملفك الشخصي وأنشئ رمز وصول طويل الأمد، ثم الصقه هنا مع رابط الخادم.',
      email:
        'استخدم صندوق بريد مخصصًا. لحسابات جيميل أو مساحة العمل، أنشئ كلمة مرور تطبيق واستخدم خادمي الاستقبال والإرسال المناسبين.',
      sms: 'احصل على معرّف الحساب ورمز المصادقة من لوحة تحكم تويليو، إضافة إلى رقم يرسل الرسائل النصية.',
      dingtalk: 'أنشئ تطبيق دينج توك في لوحة المطورين، ثم انسخ معرّف العميل وسره هنا.',
      feishu: 'أنشئ تطبيق فيشو أو لارك، وفعّل قدرة البوت فيه، وانسخ معرّف التطبيق وسره ومفاتيح تشفير الأحداث.',
      wecom:
        'أضف روبوت مجموعة في وي كوم وانسخ مفتاح خطاف الويب الخاص به. هذا الخيار للإرسال فقط؛ استخدم خيار تطبيق وي كوم للتواصل ثنائي الاتجاه.',
      wecom_callback:
        'جهّز تطبيق وي كوم ذاتي البناء، واكشف رابط الاستدعاء الراجع الخاص به، وأدخل معرّف المؤسسة والسر ومعرّف الوكيل ومفتاح التشفير.',
      weixin:
        'سجّل الدخول إلى منصة الحسابات الرسمية في وي تشات، وانسخ معرّف التطبيق والرمز، ووجّه رابط الاستدعاء الراجع للرسائل إلى هرمس.',
      qqbot: 'سجّل تطبيقًا على منصة كيو كيو المفتوحة وانسخ معرّف التطبيق وسر العميل.',
      yuanbao: 'اربط هرمس بخدمة يوانباو من Tencent.',
      api_server:
        'اكشف هرمس كواجهة برمجية متوافقة مع OpenAI. اضبط مفتاح مصادقة، ثم وجّه Open WebUI أو لوب تشات وغيرهما إلى المضيف والمنفذ.',
      webhook:
        'شغّل خادم ويب تستطيع الأدوات الأخرى، مثل GitHub وجيت لاب والتطبيقات المخصصة، إرسال الطلبات إليه. استخدم السر للتحقق من التوقيعات.'
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
    namePlaceholder: 'my-profile',
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
    count: count =>
      count === 1 ? 'مهمة واحدة' : count === 2 ? 'مهمتان' : count <= 10 ? `${count} مهام` : `${count} مهمة`,
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
      custom: 'تعبير جدولة أو عبارة طبيعية'
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
    emptyDescNew: 'جدول موجّهًا ليعمل وفق تعبير زمني. سيشغله هرمس ويسلّم النتيجة إلى الوجهة التي تختارها.',
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
    createDesc: 'جدول موجّهًا ليعمل تلقائيًا. استخدم تعبيرًا زمنيًا أو عبارة طبيعية مثل «كل خمس عشرة دقيقة».',
    nameLabel: 'الاسم',
    namePlaceholder: 'الموجز الصباحي',
    promptLabel: 'الموجّه',
    promptPlaceholder: 'لخّص محادثات سلاك غير المقروءة وأرسل أهم 5 منها بالبريد...',
    frequencyLabel: 'التكرار',
    deliverLabel: 'التسليم إلى',
    customScheduleLabel: 'جدول مخصص',
    customPlaceholder: '0 9 * * * أو weekdays at 9am',
    customHint: 'تعبير زمني أو عبارة طبيعية مثل «كل ساعة» أو «أيام العمل عند التاسعة صباحًا».',
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
    shiftClickHint: 'انقر مع مفتاح التبديل لتثبيت المحادثة',
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
      branchOff: () => ({ after: '', before: 'إنشاء فرع من ' }),
      baseBranchPlaceholder: 'البحث في الفروع…',
      baseBranchNone: 'لم يُعثر على فروع',
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
      hideTabBar: 'إخفاء شريط التبويبات',
      openInNewTab: 'فتح في تبويب جديد',
      openInSplit: 'فتح في تقسيم',
      copyIdFailed: 'تعذر نسخ معرّف الجلسة',
      actionsFor: title => `إجراءات ${title}`,
      sessionActions: 'إجراءات الجلسة',
      sessionRunning: 'الجلسة تعمل',
      needsInput: 'تحتاج إدخالك',
      waitingForAnswer: 'في انتظار إجابتك',
      finishedUnread: 'انتهت — غير مقروءة',
      backgroundRunning: 'مهمة تعمل في الخلفية',
      handoffOrigin: platform => `محوّلة من ${platform}`,
      ownedByProfile: profile => `الملف الشخصي: ${profile}`,
      renamed: 'أُعيدت التسمية',
      renameFailed: 'فشلت إعادة التسمية',
      renameTitle: 'إعادة تسمية الجلسة',
      renameDesc: 'أعط هذه المحادثة عنوانًا يسهل تذكره. اتركه فارغًا لمسحه.',
      untitledPlaceholder: 'جلسة بلا عنوان',
      untitledChat: id => `محادثة ${id}`,
      ageNow: 'الآن',
      ageDay: 'ي',
      ageHour: 'س',
      ageMin: 'د'
    }
  },

  composer: {
    message: 'رسالة',
    addContext: 'إضافة سياق',
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
      'composer.sendNewline': 'إرسال · مفتاح التبديل مع الإدخال لسطر جديد',
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
    backgroundProcess: 'عملية في الخلفية',
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
    fireworksPitch: 'واجهة نماذج مباشرة لنماذج متقدمة تستضيفها فايرووركس',
    openRouterPitch: 'مفتاح واحد ومئات النماذج؛ خيار افتراضي جيد',
    openRouterName: 'OpenRouter',
    providerNames: {
      nous: 'Nous Portal',
      'openai-codex': 'OpenAI Codex',
      'minimax-oauth': 'MiniMax',
      'qwen-oauth': 'Qwen Code',
      'xai-oauth': 'xAI Grok',
      anthropic: 'Anthropic',
      'claude-code': 'Claude Code'
    },
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
        description: 'وجّه هرمس إلى نقطة نهاية محلية أو مستضافة ذاتيًا متوافقة مع OpenAI.'
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
    docs: provider => `توثيق ${provider}`,
    runtime: {
      readyTitle: 'هرمس جاهز',
      connected: provider => `تم ربط ${provider}.`,
      gatewayToolsTitle: 'فُعّلت بوابة الأدوات',
      gatewayToolsMessage: labels => {
        const list =
          labels.length === 1
            ? labels[0]
            : labels.length === 2
              ? `${labels[0]} و${labels[1]}`
              : `${labels.slice(0, -1).join('، ')}، و${labels[labels.length - 1]}`

        return `ستعمل ${list} الآن عبر اشتراكك في Nous، ولا تحتاج إلى مفاتيح واجهات منفصلة.`
      },
      gatewayToolLabels: {
        browser: 'أتمتة المتصفح',
        image_gen: 'توليد الصور',
        tts: 'تحويل النص إلى كلام',
        video_gen: 'توليد الفيديو',
        web: 'البحث في الويب واستخراج محتواه'
      },
      providerUnavailable: detail =>
        detail
          ? `تم الاتصال، لكن هرمس لم يستطع تعيين مزوّد صالح للاستخدام. ${detail}`
          : 'تم الاتصال، لكن هرمس لم يستطع تعيين مزوّد صالح للاستخدام.',
      couldNotStartSignIn: detail => `تعذر بدء تسجيل الدخول: ${detail}`,
      signInStatus: status => `انتهى تسجيل الدخول بالحالة: ${status}.`,
      pollingFailed: detail => `فشل التحقق من تسجيل الدخول: ${detail}`,
      tokenExchangeFailed: 'فشل استبدال رمز الدخول.',
      externalUnavailable: (provider, command) =>
        `ما زال هرمس لا يستطيع الوصول إلى ${provider}. شغّل الأمر ${command} في الطرفية أولًا.`,
      enterValueFirst: 'أدخل قيمة أولًا.',
      enterEndpointFirst: 'أدخل رابط نقطة النهاية أولًا.',
      endpointUnreachable: url => `تعذر الوصول إلى ${url}.`,
      endpointNoModels: url =>
        `تم الاتصال بـ ${url}، لكنه لم يعلن عن أي نماذج. شغّل نموذجًا على نقطة النهاية ثم أعد المحاولة.`,
      endpointSavedButUnreachable: url => `حُفظ الإعداد، لكن هرمس ما زال لا يستطيع الوصول إلى ${url}.`,
      localEndpoint: 'نقطة نهاية محلية أو مخصصة',
      couldNotSaveProvider: provider => `تعذر حفظ ${provider}`,
      couldNotSaveEndpoint: 'تعذر حفظ نقطة النهاية المحلية',
      couldNotChangeModel: 'تعذر تغيير النموذج',
      unexpectedError: detail => (detail ? `فشلت العملية: ${detail}` : 'فشلت العملية. أعد المحاولة.')
    }
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
    approvalMode: {
      title: 'وضع الموافقة',
      ariaLabel: mode => `وضع الموافقة: ${mode}`,
      manual: 'يدوي',
      manualDescription: 'السؤال قبل الإجراءات التي تتطلب موافقة',
      smart: 'ذكي',
      smartDescription: 'تقييم الإجراءات تلقائيًا والسؤال عند الحاجة',
      off: 'دون موافقة',
      offDescription: 'التشغيل دون مطالبات بالموافقة'
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
      reasoningShort: {
        none: 'إيقاف',
        minimal: 'أدنى',
        low: 'منخفض',
        medium: 'متوسط',
        high: 'عالٍ',
        xhigh: 'فائق',
        max: 'أقصى',
        ultra: 'خارق'
      },
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
          mcp: 'بروتوكول سياق النموذج',
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
        'وضع التجاوز التلقائي مفعّل؛ ستتم الموافقة تلقائيًا على الأوامر الخطرة. انقر لتعطيله، أو انقر مع مفتاح التبديل لتبديله عامًا.',
      yoloOff:
        'وضع التجاوز التلقائي معطّل؛ انقر للموافقة تلقائيًا على الأوامر الخطرة، أو انقر مع مفتاح التبديل لتبديله عامًا.',
      modelNone: 'لا شيء',
      noModel: 'لا نموذج',
      switchModel: 'تبديل النموذج',
      openModelPicker: 'فتح منتقي النموذج',
      modelPinned: 'مثبّت باختيارك؛ تستخدم المحادثات الجديدة هذا بدل الإعداد الافتراضي',
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
    linkHint: 'النقر مع مفتاح الأوامر أو التحكم لفتح لوحة المعاينة',
    sourceLineTitle: 'انقر للتحديد · انقر مع مفتاح التبديل للتوسيع · اسحب إلى محرر الرسالة',
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
      stillWorking:
        'لا يزال هرمس يعمل، لكن لم تصل نتيجة إعادة التشغيل بعد. ربما يعمل أمر الخادم في المقدمة لا في الخلفية.',
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

  zones: {
    showHeader: 'إظهار الترويسة',
    hideHeader: 'إخفاء الترويسة',
    minimize: 'تصغير',
    restore: 'استعادة',
    closeRunningTitle: 'إغلاق تبويب يعمل؟',
    closeRunningBody:
      'هذه المحادثة ما تزال تعمل (أو تنتظر إدخالك). إغلاق التبويب يخفيها — تحتفظ الجلسة بتقدّمها ويمكن إعادة فتحها من الشريط الجانبي.',
    closeRunningConfirm: 'إغلاق التبويب',
    closeOthers: 'إغلاق البقية',
    closeToRight: 'إغلاق ما على اليمين',
    closeAll: 'إغلاق الكل',
    split: dir => `تقسيم ${dir}`,
    move: dir => `نقل ${dir}`,
    dirUp: 'لأعلى',
    dirDown: 'لأسفل',
    dirLeft: 'لليسار',
    dirRight: 'لليمين',
    pluginDisabled: pluginId => `عُطّلت الإضافة "${pluginId}"`,
    pluginDisabledBody: 'أعد تفعيلها من الإعدادات ← الإضافات لإرجاع اللوح.',
    missingPane: paneId => `لوح مفقود: ${paneId}`,
    editTitle: 'التخطيطات',
    editHint: 'اختر تخطيطًا، أو اسحب الألواح بين المناطق. انقر منطقة بالزر الأيمن لتقسيمها.',
    reset: 'إعادة تعيين',
    templates: 'القوالب',
    custom: 'مخصّص',
    newGridLayout: 'تخطيط شبكي جديد',
    saveCurrentAs: 'حفظ الترتيب الحالي قالبًا',
    nameLayoutPlaceholder: 'سمِّ هذا التخطيط…',
    deletePreset: name => `حذف ${name}`,
    zoneEditorTitle: 'محرر المناطق',
    editorHintPre: 'انقر للتقسيم · ',
    editorHintPost: ' يقلب الخط · اسحب عبر المناطق للدمج · اسحب الحواف المشتركة لتغيير الحجم',
    templateColumns: 'أعمدة',
    templateRows: 'صفوف',
    templateGrid: 'شبكة',
    templatePriority: 'أولوية',
    zoneTag: index => `منطقة ${index}`,
    mergeZones: count => (count === 2 ? 'دمج منطقتين' : count <= 10 ? `دمج ${count} مناطق` : `دمج ${count} منطقة`),
    customZoneName: count =>
      count === 2 ? 'مخصّص من منطقتين' : count <= 10 ? `مخصّص من ${count} مناطق` : `مخصّص من ${count} منطقة`,
    layoutNamePlaceholder: fallback => `اسم التخطيط (${fallback})`,
    saveApply: 'حفظ وتطبيق',
    notExpressible: 'هذا الترتيب متشابك (كالمروحة) — لا يمكن تمثيله بتقسيمات متداخلة بعد',
    zoneCount: count =>
      count === 1 ? 'منطقة واحدة' : count === 2 ? 'منطقتان' : count <= 10 ? `${count} مناطق` : `${count} منطقة`
  },

  assistant: {
    alerts: {
      caution: 'تنبيه',
      important: 'مهم',
      note: 'ملاحظة',
      tip: 'نصيحة',
      warning: 'تحذير'
    },
    embedTitle: provider => `تضمين ${provider}`,
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
      attachingFile: 'جارٍ إرفاق الملف…',
      timeline: 'خط زمن المحادثة',
      steered: 'وُجّهت'
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
      standardOutput: 'المخرجات القياسية',
      standardError: 'الأخطاء القياسية',
      payload: 'حمولة الأداة',
      copyActivity: 'نسخ النشاط',
      recoveredOne: 'نجح بعد محاولة فاشلة واحدة',
      recoveredMany: count =>
        count === 2
          ? 'نجح بعد محاولتين فاشلتين'
          : count <= 10
            ? `نجح بعد ${count} محاولات فاشلة`
            : `نجح بعد ${count} محاولة فاشلة`,
      failedOne: 'فشلت خطوة واحدة',
      failedMany: count => (count === 2 ? 'فشلت خطوتان' : count <= 10 ? `فشلت ${count} خطوات` : `فشلت ${count} خطوة`),
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
    yoloArmed: 'فُعّل وضع التجاوز التلقائي لهذه المحادثة',
    yoloOff: 'وضع التجاوز التلقائي معطّل',
    yoloSystem: active => `وضع التجاوز التلقائي ${active ? 'مفعّل' : 'معطّل'} لهذه الجلسة`,
    yoloTitle: 'التجاوز التلقائي',
    yoloToggleFailed: 'تعذر تبديل وضع التجاوز التلقائي',
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
    generatedImage: 'صورة مولّدة',
    openImage: 'فتح الصورة',
    overlayMessage: 'رسالة…',
    openInHermes: 'فتح في هرمس',
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
