import type { TranslationOverrides } from './define-locale'

export const arBoot = {
  boot: {
    ready: 'Hermes Desktop جاهز',
    desktopBootFailedWithMessage: message => `فشل تشغيل سطح المكتب: ${message}`,
    steps: {
      connectingGateway: 'جار الاتصال ببوابة سطح المكتب',
      loadingSettings: 'جار تحميل إعدادات Hermes',
      loadingSessions: 'جار تحميل الجلسات الأخيرة',
      retryingRemoteBackend: 'جارٍ إعادة الاتصال بخادم Hermes البعيد…',
      startingDesktopConnection: 'جار بدء اتصال سطح المكتب',
      startingHermesDesktop: 'جار تشغيل Hermes Desktop...'
    },
    errors: {
      backgroundExited: 'خرجت عملية Hermes الخلفية.',
      backgroundExitedDuringStartup: 'خرجت عملية Hermes الخلفية أثناء بدء التشغيل.',
      backendStopped: 'توقف الخلفية',
      desktopBootFailed: 'فشل تشغيل سطح المكتب',
      gatewayConnectionLost: 'انقطع الاتصال بالبوابة',
      gatewayConnectionLostDetail:
        'Still retrying in the background. You can keep reading and drafting — open Gateway settings if this persists.',
      gatewaySignInRequired: 'تسجيل الدخول للبوابة مطلوب',
      ipcBridgeUnavailable: 'جسر IPC لسطح المكتب غير متاح.'
    },
    failure: {
      title: 'تعذر تشغيل Hermes',
      description: 'لم تعمل البوابة الخلفية. جرب إحدى خطوات الاسترداد أدناه. لن يحذف ذلك محادثاتك أو إعداداتك.',
      remoteTitle: 'تسجيل الدخول للبوابة البعيدة مطلوب',
      remoteDescription: 'انتهت جلسة البوابة البعيدة. سجل الدخول مرة أخرى لإعادة الاتصال.',
      retry: 'إعادة المحاولة',
      repairInstall: 'إصلاح التثبيت',
      useLocalGateway: 'استخدام البوابة المحلية',
      cloudDownTitle: 'عامل Nous Cloud معطّل',
      cloudDownDescription:
        'يعيد عامل السحابة المُدار من Nous الذي يتصل به هذا البوابة خطأً من الخادم. لا يمكن إعادة تشغيله من هنا — تحقق من حالته، أو بدّل إلى البوابة المحلية، أو احصل على الدعم.',
      cloudDownHint: 'تفتح الأزرار أدناه بوابة Nous (حالة المثيل وعناصر التحكم) أو Discord للحصول على الدعم.',
      cloudDownCheckPortal: 'التحقق من حالة البوابة',
      cloudDownDiscord: 'الحصول على مساعدة عبر Discord',
      openLogs: 'فتح السجلات',
      repairHint: 'يعيد الإصلاح تشغيل المثبت وقد يستغرق بضع دقائق على جهاز جديد.',
      remoteSignInHint: signInLabel =>
        `يسجّل الخروج من جلسة المتصفح البعيدة المحفوظة، ثم يفتح ${signInLabel}. استخدم البوابة المحلية للتبديل إلى الخلفية المضمنة.`,
      signOutAndSignIn: 'تسجيل الخروج وإعادة تسجيل الدخول',
      hideRecentLogs: 'إخفاء السجلات الأخيرة',
      showRecentLogs: 'إظهار السجلات الأخيرة',
      signedInTitle: 'تم تسجيل الدخول',
      signedInMessage: 'جار إعادة الاتصال بالبوابة البعيدة...',
      signInIncompleteTitle: 'تسجيل الدخول غير مكتمل',
      signInIncompleteMessage: 'أغلقت نافذة تسجيل الدخول قبل اكتمال المصادقة.',
      signInFailed: 'فشل تسجيل الدخول',
      signInToRemoteGateway: 'تسجيل الدخول للبوابة البعيدة',
      signInWithProvider: provider => `تسجيل الدخول عبر ${provider}`,
      identityProvider: 'مزود الهوية'
    }
  },
  remoteDisplayBanner: {
    message: reason => `العرض البرمجي نشط — تم اكتشاف شاشة بعيدة (${reason}). تم تعطيل تسريع GPU لمنع الوميض.`
  },
  updates: {
    stages: {
      idle: 'جار التحضير...',
      prepare: 'جار التحضير...',
      fetch: 'جار التنزيل...',
      pull: 'أوشكنا على الانتهاء...',
      pydeps: 'جار الإنهاء...',
      update: 'جار تحديث Hermes...',
      rebuild: 'جار إعادة بناء تطبيق سطح المكتب...',
      restart: 'جار إعادة تشغيل Hermes...',
      done: 'اكتمل التحديث',
      manual: 'التحديث من الطرفية',
      guiSkew: 'تحديث تطبيق سطح المكتب',
      error: 'تم إيقاف التحديث مؤقتا'
    },
    checking: 'جار البحث عن تحديثات...',
    checkFailedTitle: 'تعذّر التحقق من التحديثات',
    tryAgain: 'إعادة المحاولة',
    notAvailableTitle: 'التحديث غير متاح',
    unsupportedMessage: 'لا يمكن لهذا الإصدار من Hermes تحديث نفسه من داخل التطبيق.',
    connectionRetry: 'تحقق من اتصالك وأعد المحاولة.',
    gitUnusable: 'لم يتمكن Hermes من تشغيل Git على هذا الجهاز، لذا لم يتمكن من التحقق من التحديثات.',
    latestBody: 'أنت تستخدم أحدث إصدار.',
    latestBodyBackend: 'الواجهة الخلفية تعمل بأحدث إصدار.',
    allSetTitle: 'كل شيء جاهز',
    availableTitle: 'يتوفر تحديث جديد',
    availableBody: 'إصدار جديد من Hermes جاهز للتثبيت.',
    availableTitleBackend: 'يتوفر تحديث للواجهة الخلفية',
    availableBodyBackend: 'إصدار أحدث من واجهة Hermes الخلفية المتصلة جاهز للتثبيت.',
    availableBodyNoChangelog: 'إصدار أحدث جاهز. ملاحظات الإصدار غير متاحة لنوع التثبيت هذا.',
    updateNow: 'التحديث الآن',
    maybeLater: 'ربما لاحقا',
    moreChanges: count => `+ ${count} تغيير${count === 1 ? '' : 'ات'} إضافي مُضمَّن.`,
    manualTitle: 'التحديث من الطرفية',
    manualBody: 'لقد ثبّتت Hermes من سطر الأوامر، لذا تُجرى التحديثات من هناك أيضا. الصق هذا في طرفيتك:',
    manualPickedUp: 'سيلتقط Hermes الإصدار الجديد في المرة التالية التي تشغّله فيها.',
    guiSkewTitle: 'تحديث تطبيق سطح المكتب',
    guiSkewBody:
      'تم تحديث الواجهة الخلفية، لكن حزمة تطبيق سطح المكتب هذه لم تتغير. حدّث أو أعد تثبيت تطبيق Hermes لسطح المكتب (ملف AppImage / ‎.deb / ‎.rpm) لمطابقته.',
    copy: 'نسخ',
    copied: 'تم النسخ',
    done: 'تم',
    applyingBody:
      'يتولّى مُحدِّث Hermes المهمة في نافذته الخاصة ويعيد فتح Hermes تلقائيا عند الانتهاء. الرجاء عدم إعادة فتح Hermes بنفسك أثناء التحديث.',
    applyingBodyBackend:
      'تطبّق الواجهة الخلفية البعيدة التحديث وستعيد التشغيل. يعيد Hermes الاتصال تلقائيا عند عودتها.',
    applyingClose: 'ستُغلق هذه النافذة أثناء تشغيل التحديث، ثم يعيد Hermes فتح نفسه تلقائيا.',
    errorTitle: 'لم يكتمل التحديث',
    errorBody: 'لا داعي للقلق — لم يُفقد شيء. يمكنك إعادة المحاولة الآن.',
    blockerTitle: 'إغلاق المعاينات المحلية لتحديث Hermes؟',
    blockerBody: 'يحتاج Hermes إلى إيقاف هذه المعاينات المحلية قبل التحديث. لن يؤدي ذلك إلى تعديل ملفاتك أو حذفها.',
    foreignBlockerTitle: 'أغلق العمليات الأخرى لتحديث Hermes',
    foreignBlockerBody:
      'لا يمكن لـ Hermes إغلاق هذه العمليات تلقائيًا بأمان. أغلق التطبيق أو الطرفية أو الخدمة التي تشغّل كل عملية، ثم حاول التحديث مرة أخرى.',
    mixedBlockerBody:
      'يمكن لـ Hermes إغلاق المعاينات المحلية المدرجة أدناه. يجب إغلاق العمليات الأخرى يدويًا قبل متابعة التحديث.',
    closePreviewsAndUpdate: 'إغلاق المعاينات والتحديث',
    closePreviewsAndCheckAgain: 'إغلاق المعاينات والتحقق مجددًا',
    localPreview: 'معاينة محلية',
    portLabel: port => `المنفذ ${port}`,
    pidLabel: pid => `معرّف العملية ${pid}`,
    technicalDetails: 'التفاصيل التقنية',
    notNow: 'ليس الآن',
    clientAlsoBehindTitle: 'تطبيق سطح المكتب متأخر',
    clientAlsoBehindMessage:
      'الخادم الخلفي محدث، لكن تطبيق سطح المكتب هذا لا يزال على إصدار أقدم. حدّثه للحصول على أحدث الإصلاحات.',
    clientAlsoBehindAction: 'تحديث تطبيق سطح المكتب',
    everythingDispatched: 'تم إرسال التحديث',
    everythingSkipped: 'تم التخطي',
    everythingRowFailed: 'فشل التحديث',
    everythingFanoutFailedTitle: 'تعذر تحديث المثيلات الأخرى',
    applyStatus: {
      preparing: 'جار تحديث الواجهة الخلفية...',
      pulling: 'جار تحديث الواجهة الخلفية...',
      restarting: 'تعيد الواجهة الخلفية التشغيل لتحميل التحديث...',
      notAvailable: 'التحديث غير متاح لهذه الواجهة الخلفية.',
      failed: 'فشل تحديث الواجهة الخلفية.',
      noReturn: 'لم تعد الواجهة الخلفية إلى الاتصال. قد لا يكون التحديث قد اكتمل — تحقق من مضيف الواجهة الخلفية.'
    }
  },
  guidedGreeting: {
    line: 'أهلا، تفضل بالدخول. أنا Hermes. امنحني دقيقتين لأرتب المكان حولك، ثم نبدأ بشيء تريد إنجازه فعلا.\n\nبداية، بماذا أناديك؟',
    nameSuggestion: (name: string) => `(يمكنني أن أناديك ${name} إن كنت تفضل ذلك.)`
  },
  install: {
    stageStates: {
      pending: 'قيد الانتظار',
      running: 'جار التثبيت',
      succeeded: 'تم',
      skipped: 'تم التخطي',
      failed: 'فشل'
    },
    oneTimeTitle: 'يحتاج Hermes إلى تثبيت لمرة واحدة',
    unsupportedDesc: platform =>
      `التثبيت التلقائي عند أول تشغيل غير متاح على ${platform} بعد. افتح الطرفية وشغّل الأمر أدناه، ثم أعد تشغيل هذا التطبيق. ستتخطى عمليات التشغيل اللاحقة هذه الخطوة.`,
    installCommand: 'أمر التثبيت',
    copyCommand: 'نسخ الأمر',
    viewDocs: 'عرض مستندات التثبيت',
    installTo: 'سيتم التثبيت في',
    retryAfterRun: 'لقد شغّلته -- إعادة المحاولة',
    failedTitle: 'فشل التثبيت',
    settingUpTitle: 'جار إعداد وكيل Hermes',
    finishingTitle: 'جار الإنهاء',
    failedDesc:
      'فشلت إحدى خطوات التثبيت. على Windows، قد يحدث هذا إذا كان هناك نسخة أخرى من Hermes CLI أو تطبيق سطح المكتب قيد التشغيل. أوقف أي نسخ Hermes قيد التشغيل، ثم أعد المحاولة. تحقق من التفاصيل أدناه أو من سجل سطح المكتب للحصول على النص الكامل.',
    activeDesc:
      'هذا إعداد لمرة واحدة. يقوم مثبّت Hermes بتنزيل التبعيات وتهيئة جهازك. ستتخطى عمليات التشغيل اللاحقة هذه الخطوة.',
    progress: (completed, total) => `اكتملت ${completed} من ${total} خطوة`,
    currentStage: stage => ` -- الآن: ${stage}`,
    fetchingManifest: 'جار جلب بيان المثبّت...',
    error: 'خطأ',
    hideOutput: 'إخفاء مخرجات المثبّت',
    showOutput: 'إظهار مخرجات المثبّت',
    lines: count => `${count} سطر`,
    noOutput: 'لا توجد مخرجات بعد.',
    cancelling: 'جار الإلغاء...',
    cancelInstall: 'إلغاء التثبيت',
    transcriptSaved: 'تم حفظ النص الكامل في',
    copiedOutput: 'تم النسخ!',
    copyOutput: 'نسخ المخرجات',
    reloadRetry: 'إعادة التحميل وإعادة المحاولة'
  },
  onboarding: {
    headerTitle: 'لنُعِدّ لك Hermes Agent',
    headerDesc: 'اربط مزوّد نماذج لبدء المحادثة. معظم الخيارات تتطلب نقرة واحدة.',
    preparingInstall: 'يُكمل Hermes التثبيت. عادة ما يستغرق ذلك أقل من دقيقة في أول تشغيل.',
    starting: 'جار بدء Hermes...',
    lookingUpProviders: 'جار البحث عن المزوّدين...',
    collapse: 'طي',
    otherProviders: 'مزودون آخرون',
    haveApiKey: 'لديك مفتاح API',
    chooseLater: 'سأختار مزوّدا لاحقا',
    recommended: 'موصى به',
    connected: 'متصل',
    featuredPitch: 'اشتراك واحد، أكثر من 300 نموذج متقدم — الطريقة الموصى بها لتشغيل Hermes',
    fireworksPitch: 'نماذج مفتوحة سريعة مع استضافة Fireworks.',
    openRouterPitch: 'مفتاح واحد لمئات النماذج — خيار افتراضي جيد',
    apiKeyOptions: {
      openrouter: {
        short: 'مفتاح واحد، نماذج كثيرة',
        description: 'يستضيف مئات النماذج خلف مفتاح واحد. خيار افتراضي جيد للتثبيتات الجديدة.'
      },
      openai: {
        short: 'نماذج من فئة GPT',
        description: 'وصول مباشر إلى نماذج OpenAI.'
      },
      gemini: {
        short: 'نماذج Gemini',
        description: 'وصول مباشر إلى نماذج Google Gemini.'
      },
      xai: {
        short: 'نماذج Grok',
        description: 'وصول مباشر إلى نماذج xAI Grok.'
      },
      local: {
        short: 'مستضاف ذاتيا',
        description:
          'وجّه Hermes إلى نقطة نهاية محلية أو مستضافة ذاتيا متوافقة مع OpenAI (vLLM، llama.cpp، Ollama، إلخ).'
      }
    },
    backToSignIn: 'العودة إلى تسجيل الدخول',
    getKey: 'الحصول على مفتاح',
    replaceCurrent: 'استبدال القيمة الحالية',
    pasteApiKey: 'ألصق مفتاح API',
    localApiKeyPlaceholder: 'مفتاح API (اختياري — فقط إذا كانت نقطة النهاية تتطلبه)',
    couldNotSave: 'تعذر حفظ بيانات الاعتماد.',
    connecting: 'جار الاتصال',
    update: 'تحديث',
    flowSubtitles: {
      pkce: 'يفتح المتصفح لتسجيل الدخول ثم يتابع هنا',
      device_code: 'يفتح صفحة تحقق في المتصفح — يتصل Hermes تلقائياً',
      loopback: 'يفتح المتصفح لتسجيل الدخول — يتصل Hermes تلقائياً',
      external: 'سجل الدخول مرة واحدة في الطرفية ثم عد إلى المحادثة'
    },
    startingSignIn: provider => `جار بدء تسجيل الدخول لـ ${provider}...`,
    verifyingCode: provider => `جار التحقق من الرمز عبر ${provider}...`,
    connectedProvider: provider => `تم ربط ${provider}`,
    connectedPicking: provider => `تم ربط ${provider}. جار اختيار نموذج افتراضي...`,
    signInFailed: 'فشل تسجيل الدخول. حاول مرة أخرى.',
    signInExpired:
      'انتهت مهلة انتظار التفويض. السبب الأكثر شيوعًا هو تعطّل صفحة تسجيل الدخول في تبويب المتصفح (مشكلة من جهة الخادم) — أكمل تسجيل الدخول هناك ثم أعد المحاولة. إذا استمر الفشل، استخدم مفتاح API أو واجهة سطر الأوامر بدلاً من ذلك.',
    pickDifferentProvider: 'اختر مزوداً آخر',
    signInWith: provider => `تسجيل الدخول عبر ${provider}`,
    openedBrowser: provider => `فتحنا ${provider} في المتصفح.`,
    authorizeThere: 'صرّح لـ Hermes هناك.',
    copyAuthCode: 'انسخ رمز التفويض وألصقه أدناه.',
    pasteAuthCode: 'ألصق رمز التفويض',
    reopenAuthPage: 'إعادة فتح صفحة التفويض',
    autoBrowser: provider => `فتحنا ${provider} في المتصفح. صرّح لـ Hermes هناك وسيتم الاتصال تلقائياً دون نسخ أو لصق.`,
    reopenSignInPage: 'إعادة فتح صفحة تسجيل الدخول',
    waitingAuthorize: 'بانتظار التفويض...',
    externalPending: provider =>
      `${provider} يسجل الدخول عبر أداة سطر الأوامر الخاصة به. شغّل هذا الأمر في الطرفية، ثم عد واختر "سجلت الدخول":`,
    signedIn: 'سجلت الدخول',
    deviceCodeOpened: provider => `فتحنا ${provider} في المتصفح. أدخل هذا الرمز هناك:`,
    reopenVerification: 'إعادة فتح صفحة التحقق',
    copy: 'نسخ',
    defaultModel: 'النموذج الافتراضي',
    freeTier: 'الطبقة المجانية',
    pro: 'مدفوع',
    free: 'مجاني',
    price: (input, output) => `${input} إدخال / ${output} إخراج لكل مليون رمز`,
    change: 'تغيير',
    startChatting: 'ابدأ',
    docs: provider => `وثائق ${provider}`
  }
} satisfies Pick<
  TranslationOverrides,
  'boot' | 'remoteDisplayBanner' | 'updates' | 'guidedGreeting' | 'install' | 'onboarding'
>
