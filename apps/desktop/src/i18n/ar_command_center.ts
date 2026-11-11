import { arCronControl } from './ar_cron_control'
import type { TranslationOverrides } from './define-locale'
import { desktopTerms } from './desktop-terms'

export const arCommandCenter = {
  commandCenter: {
    close: 'إغلاق',
    paletteTitle: 'لوحة الأوامر',
    back: 'رجوع',
    searchPlaceholder: 'ابحث عن أمر أو إعداد...',
    goTo: 'انتقال إلى',
    goToSession: 'الانتقال إلى الجلسة',
    branches: 'الفروع',
    startInBranch: branch => `محادثة جديدة في ${branch}`,
    commandCenter: 'مركز الأوامر',
    appearance: 'المظهر',
    settings: 'الإعدادات',
    changeTheme: 'تغيير الثيم',
    changeColorMode: 'تغيير نمط الألوان',
    pets: {
      title: 'الحيوانات الأليفة',
      placeholder: 'البحث في الحيوانات الأليفة...',
      loading: 'جار تحميل معرض petdex...',
      error: 'تعذّر الوصول إلى معرض petdex.',
      staleBackend: 'أعد تشغيل Hermes لاستخدام الحيوانات الأليفة — الخادم الخلفي أقدم من هذه الميزة.',
      empty: 'لا توجد حيوانات أليفة مطابقة.',
      turnOff: 'إيقاف التشغيل',
      turnOn: 'تشغيل',
      installed: 'مثبّت',
      generatedTag: 'مُولّد',
      adoptFailed: 'تعذّر تبنّي ذلك الحيوان الأليف.',
      toggleFailed: enabled => `تعذّر ${enabled ? 'تشغيل' : 'إيقاف'} الحيوان الأليف.`,
      noneAvailable: 'لا توجد حيوانات أليفة متاحة — اختر واحدا أدناه لتثبيته.'
    },
    generatePet: {
      title: 'توليد حيوان أليف',
      placeholder: 'صف حيوانا أليفا لتوليده...',
      promptHint: 'اكتب وصفا، ثم اضغط Enter لرسم أربعة مظاهر.',
      readyHint: 'اضغط Enter لرسم أربعة مظاهر من وصفك.',
      generate: 'توليد',
      generating: 'جار التوليد...',
      retry: 'إعادة المحاولة',
      hatch: 'تفقيس',
      spawning: 'جار الإنشاء...',
      hatching: 'جار تفقيس حيوانك الأليف...',
      hatchingSub: 'جار بثّ الحياة فيه...',
      hatched: 'تم التفقيس!',
      hatchRow: (_state, done, total) => `جار رسم الإطار ${done} من ${total}...`,
      hatchComposing: 'جار تجميع الأجزاء...',
      hatchSaving: 'أوشكنا على الانتهاء...',
      namePlaceholder: 'سمِّ حيوانك الأليف',
      staleBackend: 'حدّث Hermes لتوليد الحيوانات الأليفة.',
      backgroundHint: 'يمكنك إغلاق هذا — سيُعلِمك Hermes عند الانتهاء.',
      slowProviderHint: 'قد يستغرق هذا عدة دقائق',
      remix: 'إعادة مزج',
      remixConfirmTitle: 'إعادة مزج هذا المظهر؟',
      remixConfirmBody: 'يولّد هذا مجموعة جديدة من المسوّدات باستخدام هذا كنقطة بداية. قد يستغرق عدة دقائق.',
      genericError: 'فشل التوليد — حاول مجددا أو اختر اقتراحا.',
      referenceImageTooLarge: 'صورة المرجع كبيرة جدا. استخدم واحدة أقل من 16 MB.',
      referenceImageInvalid: 'تعذّرت قراءة صورة المرجع تلك. جرّب PNG أو JPG أو WebP أو GIF.',
      adopt: 'تبنّي',
      startOver: 'البدء من جديد'
    },
    installTheme: {
      title: 'تثبيت سمة...',
      placeholder: 'البحث في VS Code Marketplace...',
      loading: 'جار البحث في Marketplace...',
      error: 'تعذّر الوصول إلى Marketplace.',
      empty: 'لا توجد سمات مطابقة.',
      install: 'تثبيت',
      installing: 'جار التثبيت...',
      installed: 'مثبّت',
      installs: count => `${count} عملية تثبيت`
    },
    settingsFields: 'حقول الإعدادات',
    mcpServers: 'خوادم MCP',
    archivedChats: 'المحادثات المؤرشفة',
    commands: 'الأوامر',
    sections: {
      sessions: 'الجلسات',
      system: 'النظام',
      usage: 'الاستخدام'
    },
    nav: {
      newChat: {
        title: 'جلسة جديدة',
        detail: 'بدء جلسة جديدة'
      },
      settings: {
        title: 'الإعدادات',
        detail: 'تكوين Hermes desktop'
      },
      capabilities: {
        title: 'المهارات والأدوات',
        detail: 'تفعيل المهارات ومجموعات الأدوات والمزوّدين'
      },
      messaging: {
        title: 'المراسلة',
        detail: 'إعداد Telegram وSlack وDiscord والمزيد'
      },
      artifacts: {
        title: 'العناصر',
        detail: 'استعراض المخرجات المولّدة'
      }
    },
    sectionEntries: {
      sessions: {
        title: 'لوحة الجلسات',
        detail: 'البحث في الجلسات وتثبيتها وإدارتها'
      },
      system: {
        title: 'لوحة النظام',
        detail: 'حالة Hermes والسجلات وإعادة التشغيل/التحديث'
      },
      usage: {
        title: 'لوحة الاستخدام',
        detail: 'نشاط الرموز والتكلفة والمهارات'
      }
    },
    providerNavigate: 'فتح المزود',
    providerSessions: 'جلسات المزود',
    refresh: 'تحديث',
    refreshing: 'جار التحديث...',
    noResults: 'لا توجد نتائج',
    pinSession: 'تثبيت الجلسة',
    unpinSession: 'إلغاء تثبيت الجلسة',
    exportSession: 'تصدير الجلسة',
    deleteSession: 'حذف الجلسة',
    noSessions: 'لا توجد جلسات',
    gatewayRunning: 'المراسلة تعمل',
    gatewayStopped: 'المراسلة متوقفة',
    hermesActiveSessions: (version, count) => `Hermes ${version} لديه ${count} جلسة نشطة`,
    restartGateway: 'إعادة تشغيل Hermes',
    openBrowser: 'تبديل المتصفح',
    toggleBrowser: 'تبديل المتصفح',
    gatewayRestartFailed: 'فشل إعادة تشغيل Hermes.',
    sharedGatewayRestartTitle: 'إعادة تشغيل Hermes المشترك؟',
    sharedGatewayRestartDescription: bots => `تتم إعادة اتصال جميع الوكلاء على هذا الجهاز: ${bots}`,
    sharedGatewayRestartConfirm: 'إعادة تشغيل الكل',
    sharedGatewayRestarted: count => `تمت إعادة تشغيل Hermes المشترك (عدد الوكلاء: ${count})`,
    updateHermes: 'تحديث Hermes',
    reloadWindow: 'إعادة تحميل النافذة',
    actionRunning: 'الإجراء قيد التشغيل',
    actionDone: 'اكتمل الإجراء',
    actionFailed: 'فشل الإجراء',
    actionStartedWaiting: 'بدأ الإجراء، جار الانتظار...',
    loadingStatus: 'جار تحميل الحالة',
    recentLogs: 'السجلات الأخيرة',
    logSearchPlaceholder: 'البحث في سطور السجل...',
    noLogs: 'لا توجد سجلات',
    days: count => `${count} يوم`,
    statSessions: 'الجلسات',
    statApiCalls: 'نداءات API',
    statTokens: 'الرموز',
    statCost: 'التكلفة',
    actualCost: cost => `التكلفة الفعلية ${cost}`,
    loadingUsage: 'جار تحميل الاستخدام',
    noUsage: period => `لا يوجد استخدام خلال ${period} يوم`,
    retry: 'إعادة المحاولة',
    dailyTokens: 'الرموز اليومية',
    input: 'إدخال',
    output: 'إخراج',
    noDailyActivity: 'لا يوجد نشاط يومي',
    topModels: 'أكثر النماذج استخداما',
    noModelUsage: 'لا يوجد استخدام نماذج',
    topSkills: 'أكثر المهارات استخداما',
    noSkillActivity: 'لا يوجد نشاط مهارات',
    actions: count => `${count} إجراء`
  },
  messaging: {
    search: 'بحث',
    statusFilter: {
      all: 'الكل',
      bad: 'أخطاء',
      good: 'متصل',
      muted: 'غير نشط',
      warn: 'يحتاج انتباهًا'
    },
    loading: 'جار التحميل...',
    loadFailed: 'فشل التحميل',
    states: {
      connected: 'متصل',
      connecting: 'جار الاتصال',
      disabled: 'معطّل',
      fatal: 'خطأ',
      gateway_stopped: 'المراسلة متوقفة',
      not_configured: 'يحتاج إعدادا',
      pending_restart: 'يلزم إعادة التشغيل',
      retrying: 'جار إعادة المحاولة',
      startup_failed: 'فشل بدء التشغيل'
    },
    unknown: 'غير معروف',
    hintPendingRestart: 'أعد تشغيل Hermes لتطبيق التغييرات.',
    sharedListenerUrl: 'يُخدم عبر مستمع Hermes المشترك على',
    hintGatewayStopped: 'المراسلة متوقفة.',
    restartNeeded: 'تم الحفظ. أعد تشغيل Hermes لتطبيق الإعدادات الجديدة.',
    restartNow: 'إعادة التشغيل الآن',
    restarting: 'جارٍ إعادة التشغيل…',
    restartFailedManual: 'فشلت إعادة تشغيل Hermes — أعد تشغيله يدويًا وتحقق من سجلات Hermes.',
    relayLink: {
      title: 'Team Gateway',
      badge: 'لا حاجة لرمز بوت',
      intro: (platform: string) =>
        `يصل ${platform} إلى هذا الوكيل عبر بوت Hermes المشترك. اربط حسابك في ${platform} ثم راسل البوت للتحدث مع وكيلك.`,
      link: (platform: string) => `ربط ${platform}`,
      linkAnother: (platform: string) => `ربط حساب ${platform} آخر`,
      linkedAccounts: 'الحسابات المرتبطة',
      unlink: 'إلغاء الربط',
      unlinkFailed: 'تعذر إلغاء ربط هذا الحساب',
      stepOpen: (platform: string) => `1. امسح الرمز بهاتفك، أو افتح البوت في ${platform}.`,
      stepSend: '2. أرسل هذه الرسالة إلى البوت من الحساب الذي تريد ربطه:',
      open: (platform: string) => `فتح ${platform}`,
      copy: 'نسخ الرسالة',
      copied: 'تم النسخ',
      copyFailed: (command: string) => `تعذر النسخ. اكتب ${command} بدلًا من ذلك.`,
      waiting: 'بانتظار وصول رسالتك إلى البوت. الرمز يعمل مرة واحدة.',
      linked: (platform: string) => `تم ربط حساب ${platform}. راسل البوت للتحدث مع وكيلك.`,
      expired: 'انتهت صلاحية الرمز قبل استخدامه. احصل على رمز جديد وحاول مرة أخرى.',
      refused: (platform: string) => `رفض ${platform} الربط. احصل على رمز جديد وحاول مرة أخرى.`,
      qrAlt: (platform: string) => `رمز QR لبوت ${platform} المشترك`,
      managed: 'تتم إدارته عبر Team Gateway',
      managedNote: (platform: string) =>
        `يتولى Team Gateway المراسلة في هذا المثيل، لذا يوقف Hermes الاتصال المباشر بـ ${platform} عند كل تشغيل. لتشغيله بجانب المرحّل، اضبط GATEWAY_RELAY_ALLOW_DIRECT_PLATFORMS=true في بيئة Hermes (الاتصال المباشر يمنع المثيل من التقلص إلى الصفر وقد يسلّم الرسائل مرتين).`,
      managedIntro: 'تُربط حسابات هذا الوكيل من Nous Portal.',
      openPortal: 'اربط الحسابات في Nous Portal',
      noNewAccount: (platform: string) =>
        `لم يعد الرمز نشطًا ولم يظهر حساب جديد. إذا لم يكن حسابك في ${platform} مدرجًا، فاحصل على رمز جديد وحاول مرة أخرى.`
    },
    telegramQr: {
      title: 'اختر طريقة ربط بوت Telegram',
      subtitle: 'كلا الخيارين يربط بوتًا تتحكم به ويحفظ بياناته في هذا التثبيت من Hermes فقط.',
      quickSetup: 'إعداد سريع',
      recommended: 'موصى به',
      quickHelp: 'امسح رمز QR وأكّد في Telegram. سينشئ Hermes البوت ويكتشف معرّف مستخدم Telegram الخاص بك تلقائيًا.',
      createWithQr: 'إنشاء عبر QR',
      starting: 'جارٍ البدء…',
      replaceWarning: 'بيانات Telegram مُعدّة بالفعل. سيحل إعداد QR الجديد أو رمز البوت محل البوت الحالي عند الحفظ.',
      scanHint: 'امسح بتطبيق Telegram على هاتفك، أو افتح الرابط على هذا الجهاز.',
      waiting: 'في انتظار Telegram…',
      expiresIn: remaining => `ينتهي خلال ${remaining}`,
      expired: 'منتهي',
      openTelegram: 'افتح Telegram',
      ready: 'تم إنشاء البوت',
      allowedUsers: 'المستخدمون المسموح لهم',
      ownerDetected: 'تم اكتشاف المالك',
      addAtLeastOne: 'أضف معرّف مستخدم Telegram واحدًا على الأقل.',
      userIdPlaceholder: 'معرّف مستخدم Telegram',
      add: 'إضافة',
      numericOnly: 'يجب أن تكون معرّفات مستخدمي Telegram أرقامًا.',
      saveAndRestart: 'حفظ وإعادة التشغيل',
      applying: 'جارٍ الحفظ…',
      pairingExpired: 'انتهت صلاحية اقتران Telegram. ابدأ إعداد QR جديدًا.',
      stillWaiting: detail => `ما زلنا ننتظر Telegram. إعادة المحاولة بعد: ${detail}`,
      savedRestarting: 'تم حفظ Telegram؛ تجري إعادة تشغيل Hermes…',
      savedRestartFailed: detail => `تم حفظ Telegram؛ فشلت إعادة تشغيل Hermes${detail}`
    },
    credentialsSet: 'بيانات الاعتماد مضبوطة',
    needsSetup: 'يحتاج إعدادا',
    gatewayStopped: 'المراسلة متوقفة',
    getCredentials: 'الحصول على بيانات الاعتماد',
    openSetupGuide: 'فتح دليل الإعداد',
    required: 'مطلوب',
    recommended: 'موصى به',
    advanced: count => `${count} إعدادات متقدمة`,
    noTokenNeeded: 'لا يحتاج رمز',
    enabled: 'مفعل',
    disabled: 'معطل',
    unsavedChanges: 'تغييرات غير محفوظة',
    saving: 'جار الحفظ...',
    saveChanges: 'حفظ التغييرات',
    saved: 'تم الحفظ',
    replaceValue: 'استبدال القيمة',
    openDocs: 'فتح الوثائق',
    clearField: key => `مسح ${key}`,
    addListEntry: 'إضافة آخر',
    removeListEntry: 'إزالة',
    listEntryPlaceholder: 'أدخل معرّفًا',
    enableAria: name => `تفعيل ${name}`,
    disableAria: name => `تعطيل ${name}`,
    platformEnabled: name => `تم تفعيل ${name}`,
    platformDisabled: name => `تم تعطيل ${name}`,
    restartToApply: 'أعد تشغيل Hermes لتطبيق التغييرات.',
    setupSaved: name => `تم حفظ إعداد ${name}`,
    restartToReconnect: 'أعد تشغيل Hermes لإعادة الاتصال.',
    appliedLive: 'تم التطبيق على Hermes قيد التشغيل.',
    connectingLive: 'يتصل Hermes قيد التشغيل باستخدام بيانات الاعتماد الجديدة.',
    keyCleared: key => `تم مسح ${key}`,
    setupUpdated: name => `تم تحديث إعداد ${name}`,
    failedUpdate: name => `فشل تحديث ${name}`,
    failedSave: name => `فشل حفظ ${name}`,
    failedClear: key => `فشل مسح ${key}`,
    fieldCopy: {
      TELEGRAM_BOT_TOKEN: {
        label: 'رمز البوت (token)',
        help: 'أنشئ بوتا عبر @BotFather، ثم الصق الرمز الذي يمنحك إياه.',
        placeholder: 'الصق رمز بوت Telegram'
      },
      TELEGRAM_ALLOWED_USERS: {
        label: 'معرّفات مستخدمي Telegram المسموح بهم',
        help: 'موصى به. معرّفات رقمية (واحد في كل حقل) من @userinfobot. بدون ذلك، يمكن لأي شخص مراسلة بوتك مباشرة.'
      },
      TELEGRAM_PROXY: {
        label: 'رابط الـ Proxy',
        help: 'مطلوب فقط على الشبكات التي يكون فيها Telegram محجوبا.'
      },
      DISCORD_BOT_TOKEN: {
        label: 'رمز البوت (token)',
        help: 'أنشئ تطبيقا في Discord Developer Portal، وأضف بوتا، ثم الصق رمزه.'
      },
      DISCORD_ALLOWED_USERS: {
        label: 'معرّفات مستخدمي Discord المسموح بهم',
        help: 'موصى به. معرّفات مستخدمي Discord (واحد في كل حقل).'
      },
      DISCORD_REPLY_TO_MODE: {
        label: 'نمط الرد',
        help: 'first أو all أو off.'
      },
      DISCORD_ALLOW_ALL_USERS: {
        label: 'السماح لكل مستخدمي Discord',
        help: 'للتطوير فقط. عند التفعيل، يمكن لأي شخص مراسلة البوت مباشرة دون قائمة سماح.'
      },
      DISCORD_HOME_CHANNEL: {
        label: 'معرّف القناة الرئيسية',
        help: 'القناة التي يرسل فيها البوت الرسائل الاستباقية (نتائج الجداول، التذكيرات).'
      },
      DISCORD_HOME_CHANNEL_NAME: {
        label: 'اسم القناة الرئيسية',
        help: 'الاسم المعروض للقناة الرئيسية في السجلات ومخرجات الحالة.'
      },
      BLUEBUBBLES_ALLOW_ALL_USERS: {
        label: 'السماح لكل مستخدمي iMessage',
        help: 'عند التفعيل، يتم تخطي قائمة سماح BlueBubbles.'
      },
      MATTERMOST_ALLOW_ALL_USERS: {
        label: 'السماح لكل مستخدمي Mattermost'
      },
      MATTERMOST_HOME_CHANNEL: {
        label: 'القناة الرئيسية'
      },
      QQ_ALLOW_ALL_USERS: {
        label: 'السماح لكل مستخدمي QQ'
      },
      QQBOT_HOME_CHANNEL: {
        label: 'قناة QQ الرئيسية',
        help: 'القناة أو المجموعة الافتراضية لتسليم نتائج الجداول.'
      },
      QQBOT_HOME_CHANNEL_NAME: {
        label: 'اسم قناة QQ الرئيسية'
      },
      SLACK_BOT_TOKEN: {
        label: 'رمز بوت Slack',
        help: 'استخدم رمز البوت من OAuth & Permissions بعد تثبيت تطبيق Slack الخاص بك.',
        placeholder: 'الصق رمز بوت Slack'
      },
      SLACK_APP_TOKEN: {
        label: 'رمز تطبيق Slack',
        help: 'استخدم الرمز على مستوى التطبيق المطلوب لـ Socket Mode.',
        placeholder: 'الصق رمز تطبيق Slack'
      },
      SLACK_ALLOWED_USERS: {
        label: 'معرّفات مستخدمي Slack المسموح بهم',
        help: 'موصى به. معرّفات مستخدمي Slack (واحد في كل حقل).'
      },
      MATTERMOST_URL: {
        label: 'رابط الخادم',
        placeholder: 'https://mattermost.example.com'
      },
      MATTERMOST_TOKEN: {
        label: 'رمز البوت (token)'
      },
      MATTERMOST_ALLOWED_USERS: {
        label: 'معرّفات المستخدمين المسموح بهم',
        help: 'موصى به. معرّفات مستخدمي Mattermost (واحد في كل حقل).'
      },
      MATRIX_HOMESERVER: {
        label: 'رابط Homeserver',
        placeholder: 'https://matrix.org'
      },
      MATRIX_ACCESS_TOKEN: {
        label: 'رمز الوصول'
      },
      MATRIX_USER_ID: {
        label: 'معرّف مستخدم البوت',
        placeholder: '@hermes:example.org'
      },
      MATRIX_ALLOWED_USERS: {
        label: 'معرّفات مستخدمي Matrix المسموح بهم',
        help: 'موصى به. معرّفات مستخدمين (واحد في كل حقل) بصيغة @user:server.'
      },
      SIGNAL_HTTP_URL: {
        label: 'رابط جسر Signal',
        placeholder: 'http://127.0.0.1:8080',
        help: 'رابط جسر signal-cli REST قيد التشغيل.'
      },
      SIGNAL_ACCOUNT: {
        label: 'رقم الهاتف',
        help: 'الرقم المسجّل مع جسر signal-cli الخاص بك.'
      },
      SIGNAL_ALLOWED_USERS: {
        label: 'مستخدمو Signal المسموح بهم',
        help: 'موصى به. معرّفات Signal (واحد في كل حقل).'
      },
      WHATSAPP_ENABLED: {
        label: 'تفعيل جسر WhatsApp',
        help: 'يُضبط تلقائيا عبر المفتاح أدناه. اتركه دون تغيير ما لم تكن متأكدا من حاجتك إليه.'
      },
      WHATSAPP_MODE: {
        label: 'وضع الجسر'
      },
      WHATSAPP_ALLOWED_USERS: {
        label: 'مستخدمو WhatsApp المسموح بهم',
        help: 'موصى به. أرقام هواتف أو معرّفات WhatsApp (واحد في كل حقل).'
      }
    },
    platformIntro: {}
  },
  profiles: {
    close: 'إغلاق',
    nameHint: 'اسم الوكيل',
    title: desktopTerms.ar.agents,
    count: count => `عدد الوكلاء: ${count}`,
    search: 'البحث في الوكلاء…',
    loading: 'جار التحميل...',
    newProfile: 'وكيل جديد',
    importProfile: 'استيراد وكيل…',
    exportProfile: 'تصدير وكيل…',
    exportMenu: 'تصدير…',
    imported: 'تم استيراد الوكيل',
    exported: 'تم تصدير الوكيل',
    failedImport: 'فشل استيراد الوكيل',
    failedExport: 'فشل تصدير الوكيل',
    exportNeedsHost: name => `لا يمكن للمتصفح حفظ أرشيف الوكيل. على مضيف Hermes، شغّل: hermes profile export ${name}`,
    allProfiles: 'كل الوكلاء',
    showAllProfiles: 'إظهار كل الوكلاء',
    notServedHere: 'غير متاح هنا',
    switchToProfile: name => `التبديل إلى ${name}`,
    switchToConnection: name => `التبديل إلى ${name}`,
    switchConnectionFailed: name => `تعذّر الاتصال بـ ${name}`,
    manageProfiles: 'إدارة الوكلاء',
    fleet: {
      allOnGateway: 'كل الوكلاء على هذا الجهاز',
      gateway: gateway => `الوكلاء على ${gateway}`,
      localDevice: 'هذا الجهاز (خلفية محلية — تثبّت Hermes إن كان مفقودًا، وإلا تفتح جلسة جديدة)',
      installDeviceTitle: 'تثبيت Hermes على هذا الجهاز؟',
      installDeviceDesc: 'سيُثبَّت Hermes محليًا ثم تُفتح جلسة جديدة على هذا الحاسوب. لا يبدأ التثبيت قبل التأكيد.',
      installDeviceConfirm: 'تثبيت محليًا',
      connectExistingInstead: 'الاتصال بموجود بدلًا من ذلك'
    },
    remoteOverride: {
      menuItem: 'الاتصال بمضيف بعيد…',
      badge: (host: string) => `يعمل على ${host}`,
      title: (profile: string) => `ربط ${profile} بمضيف بعيد`,
      description: 'ستعمل جلسات هذا الوكيل على خادم Hermes البعيد الذي تحدده، بدلاً من هذا الجهاز.',
      urlLabel: 'العنوان البعيد',
      urlPlaceholder: 'https://hermes.example.com',
      urlInvalid: 'أدخل عنواناً كاملاً يبدأ بـ http:// أو https://',
      tokenLabel: 'رمز الوصول',
      tokenPlaceholder: 'الصق رمز الجلسة البعيد',
      tokenSavedHint: 'يوجد رمز محفوظ بالفعل. اتركه فارغاً للاحتفاظ به.',
      plainTextOptIn:
        'لا يتوفر تخزين مفاتيح آمن على هذا الجهاز، لذا سيُحفظ الرمز على القرص دون تشفير. احفظه على أي حال.',
      collisionWarning: (label: string) =>
        `يوجد جهاز باسم «${label}» في الإعدادات بالفعل. اتصال هذا الوكيل منفصل ولن يغيّره.`,
      confirmTitle: 'ربط هذا الوكيل بمضيف بعيد؟',
      confirmNote: (profile: string, host: string) =>
        `ستعمل المحادثات الجديدة في ${profile} على ${host}. سيقوم ذلك الجهاز بتنفيذ الأوامر وقراءة الملفات هناك، وليس هنا. اتصل فقط بمضيف تثق به.`,
      confirmBack: 'رجوع',
      connect: 'اتصال',
      connecting: 'جارٍ الاتصال…',
      disconnect: 'إزالة الاتصال البعيد',
      savedTitle: 'تم ربط الوكيل',
      savedMessage: (profile: string, host: string) => `${profile} يعمل الآن على ${host}`,
      removedTitle: 'تمت إزالة الاتصال البعيد',
      removedMessage: (profile: string) => `${profile} يعمل الآن على هذا الجهاز`,
      removeFailed: 'تعذّرت إزالة الاتصال البعيد',
      authFailedTitle: 'رفض المضيف البعيد الرمز المحفوظ',
      authFailedMessage: (profile: string, host: string) =>
        `رفض ${host} الرمز المحفوظ لـ ${profile}. ربما تم تغييره على الجانب البعيد.`,
      updateToken: 'أدخل رمزاً جديداً…'
    },
    actions: 'إجراءات',
    color: 'اللون',
    colorFor: 'اللون',
    openInNewWindow: 'فتح في نافذة جديدة',
    setAsDefault: 'تعيين كافتراضي',
    defaultProfile: 'الوكيل الافتراضي',
    defaultSet: name => `أصبح ${name} الوكيل الافتراضي`,
    defaultDescription: 'يُستخدم عند فتح Hermes وللمحادثات الجديدة. تبقى المحادثات الحالية مع وكلائها.',
    failedSetDefault: 'تعذّر تعيين الوكيل الافتراضي',
    setColor: color => `ضبط اللون ${color}`,
    autoColor: 'لون تلقائي',
    noProfiles: 'لا يوجد وكلاء',
    selectPrompt: 'اختر وكيلاً',
    refresh: 'تحديث',
    refreshing: 'جار التحديث...',
    default: 'الافتراضي',
    skills: count => `${count} مهارة`,
    env: 'البيئة',
    defaultBadge: 'افتراضي',
    rename: 'إعادة تسمية',
    copySetup: 'نسخ الإعداد',
    copying: 'جار النسخ...',
    modelLabel: 'النموذج',
    skillsLabel: 'المهارات',
    notSet: 'غير مضبوط',
    soulDesc: 'تعليمات هذا الوكيل وشخصيته.',
    soulMissing:
      'لا يوجد ملف SOUL.md لهذا الوكيل بعد. أضف التعليمات أدناه واحفظ لإنشائه. تُدار إعدادات الشخصية في config.yaml بشكل منفصل.',
    soulOptional: 'اختياري',
    soulPlaceholder: mode => `تعليمات هذا الوكيل وشخصيته.\nاتركه فارغا للإبقاء على افتراضي ${mode}.`,
    soulPlaceholderCloned: 'مستنسخ',
    soulPlaceholderEmpty: 'فارغ',
    unsavedChanges: 'تغييرات غير محفوظة',
    loadingSoul: 'جار تحميل SOUL.md...',
    emptySoul: 'SOUL.md فارغ — ابدأ بكتابة الشخصية...',
    saving: 'جار الحفظ...',
    saveSoul: 'حفظ التعليمات',
    deleteTitle: 'حذف الوكيل',
    deleteDescPrefix: 'سيؤدي هذا إلى حذف ',
    deleteDescMid: ' وإزالة ',
    deleteDescSuffix: ' الخاص به. لا يمكن التراجع عن هذا.',
    deleting: 'جار الحذف...',
    createDesc: 'أنشئ وكيلاً له تعليماته وإعداداته ومحادثاته الخاصة.',
    nameLabel: 'الاسم',
    cloneFrom: 'استنساخ من',
    cloneFromNone: 'لا شيء (فارغ)',
    cloneFromDesc: 'ينسخ الإعدادات والمهارات وSOUL.md من الوكيل المحدد.',
    cloneFromDefault: 'نسخ إعداد الافتراضي',
    cloneFromDefaultDesc: 'ابدأ من إعدادات الوكيل الافتراضي.',
    invalidName: hint => `اسم غير صالح: ${hint}`,
    nameRequired: 'الاسم مطلوب',
    creating: 'جار الإنشاء...',
    createAction: 'إنشاء وكيل',
    renameTitle: 'إعادة تسمية الوكيل',
    renameDescPrefix: 'تؤدي إعادة التسمية إلى تحديث مجلد الوكيل وأي سكربتات تغليف في ',
    renameDescSuffix: '.',
    displayNameTitle: 'تسمية هذا الوكيل',
    displayNameDesc: 'يحدد اسم العرض في التطبيق. يبقى المعرّف الداخلي "default".',
    displayNameLabel: 'اسم العرض',
    newNameLabel: 'الاسم الجديد',
    renaming: 'جار إعادة التسمية...',
    created: 'تم إنشاء الوكيل',
    renamed: 'تمت إعادة التسمية',
    deleted: 'تم الحذف',
    setupCopied: 'تم نسخ الإعداد',
    soulSaved: 'تم حفظ التعليمات',
    failedLoad: 'فشل تحميل الوكلاء',
    failedDelete: 'فشل الحذف',
    failedCopy: 'فشل النسخ',
    failedLoadSoul: 'فشل تحميل التعليمات',
    failedSaveSoul: 'فشل حفظ التعليمات',
    failedCreate: 'فشل الإنشاء',
    failedRename: 'فشل إعادة التسمية'
  },
  cron: {
    control: arCronControl,
    title: desktopTerms.ar.schedules,
    count: count => `عدد الجداول: ${count}`,
    tabs: { jobs: desktopTerms.ar.schedules },
    editJob: 'تحرير الجدول',
    scheduleRequired: 'وقت التشغيل مطلوب.',
    scriptOnlyEditHint: 'يشغّل هذا الجدول برنامجًا نصيًا، وليس رسالة للذكاء الاصطناعي. معرّف الجدول:',
    close: 'إغلاق',
    search: 'البحث في الجداول…',
    loading: 'جار تحميل الجداول…',
    states: {
      enabled: 'مُفعّل',
      scheduled: 'مجدول',
      running: 'قيد التشغيل',
      paused: 'متوقف مؤقتا',
      disabled: 'معطّل',
      error: 'خطأ',
      completed: 'مكتمل'
    },
    deliveryLabels: {
      local: 'سطح المكتب هذا',
      telegram: 'Telegram',
      discord: 'Discord',
      slack: 'Slack',
      email: 'البريد الإلكتروني'
    },
    scheduleLabels: {
      daily: 'يوميا',
      weekdays: 'أيام الأسبوع',
      weekly: 'أسبوعيا',
      monthly: 'شهريا',
      hourly: 'كل ساعة',
      'every-15-minutes': 'كل 15 دقيقة',
      custom: 'مخصص'
    },
    scheduleHints: {
      daily: 'كل يوم في الساعة 9:00 صباحا',
      weekdays: 'من الاثنين إلى الجمعة في الساعة 9:00 صباحا',
      weekly: 'كل اثنين في الساعة 9:00 صباحا',
      monthly: 'أول يوم من كل شهر في الساعة 9:00 صباحا',
      hourly: 'في بداية كل ساعة',
      'every-15-minutes': 'كل 15 دقيقة',
      custom: 'صياغة cron أو لغة طبيعية'
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
    dayFallback: value => `اليوم ${value}`,
    everyDayAt: time => `كل يوم في ${time}`,
    weekdaysAt: time => `أيام الأسبوع في ${time}`,
    everyDayOfWeekAt: (day, time) => `كل ${day} في ${time}`,
    monthlyOnDayAt: (dayOfMonth, time) => `شهريا في اليوم ${dayOfMonth} في ${time}`,
    topOfHour: 'في بداية كل ساعة',
    everyHourAt: minute => `كل ساعة عند :${minute}`,
    newCron: 'جدول جديد',
    emptyDescNew: 'أنشئ جدولاً لتشغيل الوكيل تلقائيًا.',
    emptyDescSearch: 'لا توجد جداول تطابق البحث.',
    emptyTitleNew: 'لا توجد جداول',
    emptyTitleSearch: 'لا توجد نتائج',
    last: 'آخر تشغيل',
    next: 'التالي',
    noRuns: 'لا توجد تشغيلات',
    queuedRun: 'تشغيل في قائمة الانتظار',
    manage: 'إدارة',
    showRuns: 'إظهار التشغيلات',
    hideRuns: 'إخفاء التشغيلات',
    runHistory: 'سجل التشغيل',

    actionsTitle: 'الإجراءات',
    resume: 'استئناف',
    pause: 'إيقاف مؤقت',
    resumeTitle: 'استئناف الجدول',
    pauseTitle: 'إيقاف الجدول مؤقتًا',
    triggerNow: 'تشغيل الآن',
    edit: 'تحرير',
    deleteTitle: 'حذف الجدول',
    deleteDescPrefix: 'سيؤدي هذا إلى إزالة ',
    deleteDescSuffix: ' نهائيا. سيتوقف عن العمل فورا.',
    deleting: 'جار الحذف...',
    resumed: 'تم الاستئناف',
    paused: 'تم الإيقاف مؤقتا',
    triggered: 'تم التشغيل',
    deleted: 'تم الحذف',
    created: 'تم الإنشاء',
    updated: 'تم التحديث',
    failedLoad: 'فشل تحميل الجداول',
    failedUpdate: 'فشل التحديث',
    failedTrigger: 'فشل التشغيل',
    failedDelete: 'فشل الحذف',
    failedSave: 'فشل الحفظ',
    editTitle: 'تحرير الجدول',
    createTitle: 'إنشاء جدول',
    editDesc: 'عدّل وقت التشغيل والرسالة.',
    createDesc: 'حدد ما يفعله الوكيل ومتى يعمل تلقائيًا.',
    nameLabel: 'الاسم',
    namePlaceholder: 'مثال: الملخص الصباحي',
    promptLabel: 'الرسالة',
    scriptLabel: 'البرنامج النصي',
    scriptBadge: 'برنامج نصي',
    promptPlaceholder: 'ماذا تريد من الوكيل أن يفعل؟',
    frequencyLabel: 'التكرار',
    deliverLabel: 'التسليم',
    customScheduleLabel: 'جدول مخصص',
    customPlaceholder: 'تعبير cron',
    customHint: 'استخدم صيغة cron القياسية.',
    optional: 'اختياري',
    promptScheduleRequired: 'الرسالة والجدول مطلوبان',
    saveChanges: 'حفظ التغييرات',
    createAction: 'إنشاء'
  }
} satisfies Pick<TranslationOverrides, 'commandCenter' | 'messaging' | 'profiles' | 'cron'>
