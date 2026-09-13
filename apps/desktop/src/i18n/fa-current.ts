import type { TranslationOverrides } from './define-locale'

/** Persian copy for desktop surfaces added after the original locale work.
 * Keeping this delta explicit makes upstream English additions auditable. */
export const faCurrent = {
  fileMenu: {
    download: 'دانلود',
    downloadSaved: 'ذخیره شد',
    downloadFailed: 'دانلود ناموفق بود'
  },
  boot: {
    steps: {
      retryingRemoteBackend: 'در حال اتصال دوباره به پشتیبان راه‌دور Hermes…'
    },
    errors: {
      gatewayConnectionLostDetail:
        'تلاش برای اتصال در پس‌زمینه ادامه دارد. می‌توانید به خواندن و نوشتن پیش‌نویس ادامه دهید؛ اگر مشکل ماندگار شد تنظیمات دروازه را باز کنید.'
    },
    failure: {
      cloudDownTitle: 'عامل Nous Cloud از دسترس خارج است',
      cloudDownDescription:
        'عامل ابری مدیریت‌شدهٔ Nous که این دروازه به آن متصل است خطای سرور برمی‌گرداند. راه‌اندازی دوبارهٔ آن از اینجا ممکن نیست؛ وضعیتش را بررسی کنید، به دروازهٔ محلی بروید یا از پشتیبانی کمک بگیرید.',
      cloudDownHint: 'دکمه‌های زیر پرتال Nous برای وضعیت و کنترل نمونه و Discord را برای پشتیبانی باز می‌کنند.',
      cloudDownCheckPortal: 'بررسی وضعیت در پرتال',
      cloudDownDiscord: 'دریافت کمک در Discord'
    }
  },
  notifications: {
    updateReadyMessageUnknown: 'به‌روزرسانی جدیدی در دسترس است.',
    mcp: {
      needsAuthTitle: 'سرور MCP به ورود دوباره نیاز دارد',
      needsAuthMessage: name => `سرور MCPِ ${name} به ورود دوباره نیاز دارد.`,
      errorTitle: 'سرور MCP در دسترس نیست',
      errorMessage: name => `بررسی سلامت MCPِ ${name} ناموفق بود.`,
      signIn: 'ورود',
      view: 'مشاهده'
    }
  },
  sendDiagnostics: {
    title: 'ارسال اطلاعات عیب‌یابی به Nous',
    privacyNotice:
      'این کار یک بستهٔ اشکال‌زدایی را در فضای داخلی و خصوصی Nous بارگذاری می‌کند. بسته شامل اطلاعات سیستم و گزارش‌های کامل عامل، دروازه و دسکتاپ تا سقف ۵۱۲ کیلوبایت برای هرکدام است و ممکن است محتوای گفت‌وگو، خروجی ابزارها و مسیر فایل‌ها را دربرگیرد. خود کلیدهای API هرگز فرستاده نمی‌شوند و رازها پیش از بارگذاری حذف می‌شوند. فقط کارکنان Nous و مدیران مجاز Discord به آن دسترسی دارند و پس از ۱۴ روز خودکار پاک می‌شود.',
    upload: 'بارگذاری',
    uploading: 'در حال بارگذاری…',
    cancel: 'انصراف',
    close: 'بستن',
    copyLink: 'رونوشت پیوند',
    uploadIdFallback: id => `پیوند مشاهده برگردانده نشد؛ شناسهٔ بارگذاری ${id} را به پشتیبانی بدهید`,
    doneTitle: 'اطلاعات عیب‌یابی ارسال شد',
    doneDescription:
      'بسته به‌صورت خصوصی بارگذاری شد. پیوند زیر را در گفت‌وگوی پشتیبانی بفرستید تا تیم گزارش‌ها را ببیند.',
    failedTitle: 'بارگذاری ناموفق بود',
    failedHint:
      'می‌توانید در پایانه دستور `hermes debug share --nous` یا برای چاپ گزارش بدون بارگذاری، `hermes debug share --local` را هم اجرا کنید.',
    handoffLead: 'ادامهٔ گفت‌وگو در:',
    links: { github: 'مسائل GitHub', portal: 'پشتیبانی پرتال Nous', discord: 'Discord' }
  },
  titlebar: {
    unreadSessions: count => `${count} نشست خوانده‌نشده`,
    resetHudLayout: 'بازنشانی اندازه و جایگاه HUD',
    layoutEditorTitle: mod => `ویرایشگر چیدمان — کلیک با ${mod} چیدمان را بازنشانی می‌کند`
  },
  keybinds: {
    actions: {
      'session.archive': 'بایگانی نشست جاری',
      'view.toggleTabStrip': 'تغییر وضعیت نوار زبانه‌ها',
      'view.showBrowser': 'باز کردن مرورگر',
      'view.selectionToComposer': 'فرستادن بخش انتخاب‌شده به کادر نوشتن'
    }
  },
  settings: {
    nav: { gateway: 'دروازه‌ها' },
    plugins: {
      agent: { appliesTo: 'اعمال روی:' },
      installModal: {
        title: 'نصب افزونه',
        description: 'پیش از نصب، محتوای این مخزن را بررسی کنید.',
        repoLabel: 'مخزن',
        includesHeading: 'این بسته شامل است',
        agentLabel: 'افزونهٔ عامل',
        desktopLabel: 'رابط دسکتاپ',
        agentTargetLocal: profile => `در پشتیبان ${profile} نصب می‌شود (~/.hermes/plugins/)`,
        agentTargetRemote: profile => `در پشتیبان متصل ${profile} نصب می‌شود`,
        desktopTarget: 'در پوشهٔ محلی desktop-plugins این برنامه نصب می‌شود',
        desktopOnlyNote: 'بسته‌های ویژهٔ دسکتاپ افزونه‌ای روی عامل پشتیبان نصب نمی‌کنند.',
        insecureWarning:
          'این نشانی از طرح محلی یا ناامن استفاده می‌کند. برای محیط عملیاتی https:// یا git@ را ترجیح دهید.',
        securityHeading: 'پیش از نصب',
        securityIntro: 'فقط از منبع مورداعتماد نصب کنید؛ برای دیدن موارد افزوده‌شونده مخزن را بررسی کنید.',
        sourceHeading: 'کد منبع',
        viewRepository: 'مشاهدهٔ مخزن',
        viewPluginFiles: 'مشاهدهٔ فایل‌های افزونه',
        gitCloneLabel: 'نشانی Git clone',
        enableAgent: 'فعال‌کردن افزونهٔ عامل پس از نصب',
        forceReinstall: 'نصب اجباری دوباره (جای‌گزینی نسخهٔ موجود)',
        install: 'نصب',
        installing: 'در حال نصب…',
        probing: 'در حال بررسی مخزن…',
        probeUnavailable: 'بررسی افزونه در این محیط در دسترس نیست.',
        desktopUnavailable: 'نصب افزونهٔ دسکتاپ در این محیط در دسترس نیست.',
        selectComponent: 'دست‌کم یک مؤلفه را برای نصب انتخاب کنید.',
        agentSuccess: name => `افزونهٔ عامل ${name} نصب شد`,
        desktopSuccess: name => `افزونهٔ دسکتاپ ${name} نصب شد`,
        agentFailed: 'نصب افزونهٔ عامل ناموفق بود',
        desktopFailed: 'نصب افزونهٔ دسکتاپ ناموفق بود',
        missingEnv: vars => `متغیرهای محیطی موجود نیستند: ${vars}. آن‌ها را در تنظیمات ← کلیدها اضافه کنید.`
      }
    },
    appearance: {
      reasoningCollapsedTitle: 'جمع‌کردن پیش‌فرض فرایند فکر',
      reasoningCollapsedDesc: 'استدلال جریانی را نگه می‌دارد اما تا وقتی خودتان بازش نکنید آن را گسترش نمی‌دهد.',
      sessionDensityTitle: 'تراکم فهرست نشست‌ها',
      sessionDensityDesc: 'میزان جزئیات زیر عنوان نشست‌ها در نوار کناری را انتخاب کنید.',
      sessionDensityCompact: 'فشرده',
      sessionDensityComfortable: 'راحت',
      sessionDensityDetailed: 'با جزئیات',
      tabStripTitle: 'نوار زبانه‌ها',
      tabStripDesc: 'زبانه‌ها را بالای هر ناحیه نشان می‌دهد؛ در ناحیه‌های تک‌صفحه به‌طور خودکار پنهان می‌شود.',
      tabStripAuto: 'خودکار',
      tabStripAlways: 'همیشه',
      tabStripNever: 'هرگز',
      translucencyDesc:
        'پس‌زمینهٔ میزکار را از سراسر پنجره، حتی زیر متن، ببینید. برای حالت روشن و تیره جدا تنظیم می‌شود.',
      translucencyGlassDesc:
        'شیشهٔ مات: میزکار با تاری نرم دیده می‌شود و متن واضح می‌ماند. برای حالت روشن و تیره جدا تنظیم می‌شود.',
      translucencyModeClear: 'شفاف',
      translucencyModeGlass: 'شیشه‌ای',
      translucencyTintTitle: 'ته‌رنگ',
      translucencyFadeTitle: 'محو',
      translucencyFrostTitle: 'ماتی',
      translucencyFrost: { 'under-window': 'عمیق', popover: 'نرم', titlebar: 'روشن', header: 'درخشان' },
      translucencyScopeTitle: 'ناحیه',
      translucencyScope: { window: 'تمام پنجره', sidebar: 'فقط نوار کناری' },
      introSplashTitle: 'صفحهٔ آغازین',
      introSplashDesc: 'نشان‌واژه و پیشنهاد نمایشی در گفت‌وگوی خالی.',
      tipsTitle: 'نکته‌های درون‌برنامه‌ای',
      tipsDesc:
        'حباب کوچکی که گاهی هنگام بی‌کاری یا در زمان مناسب به بخشی از برنامه اشاره می‌کند. بستن هر نکته آن را برای همیشه کنار می‌گذارد.',
      tipsReset: count => `بازگرداندن ${count} نکتهٔ بسته‌شده`,
      toursTitle: 'تورهای راهنما',
      toursDesc: 'اجازه دهید Hermes با کم‌نورکردن صفحه و برجسته‌کردن هر مرحله، برنامه را معرفی کند.',
      composerPopoutTitle: 'کادر نوشتن شناور',
      composerPopoutDesc: 'اجازه می‌دهد کادر نوشتن را از جایگاهش بیرون بکشید؛ با خاموش‌کردن، پایین صفحه قفل می‌شود.',
      vibeHeartsTitle: 'قلب‌های واکنشی',
      vibeHeartsDesc:
        'وقتی تشکر می‌کنید، می‌گویید «دوستت دارم» یا قلب می‌فرستید، قلب‌های شناور نشان می‌دهد؛ مستقل از واکنش به پیام است.'
    },
    about: {
      bundleOutOfSync: 'نسخهٔ برنامه قدیمی است',
      bundleOutOfSyncDesc:
        'محیط اجرایی Hermes به‌روز شده، اما خود برنامهٔ دسکتاپ هنوز قدیمی است؛ تا به‌روزرسانی رابط، قابلیت‌های تازه دیده نمی‌شوند. به‌روزرسانی زیر را اجرا کنید و اگر هشدار برطرف نشد، آخرین نصب‌کننده را دوباره نصب کنید.',
      bundleOutOfSyncAction: 'دریافت نصب‌کننده',
      updateReadyUnknown: 'به‌روزرسانی جدیدی آماده است.'
    },
    config: {
      toolsetsWipeConfirm:
        'همهٔ مجموعه‌ابزارهای فعال حذف شوند؟ حافظه، پایانه، جست‌وجوی وب، واگذاری و بیشتر ابزارها تا فعال‌کردن دوباره از کار می‌افتند.',
      disableF12Title: 'غیرفعال‌کردن DevTools با F12',
      disableF12Desc: 'بازشدن ابزار توسعه با F12 را مسدود می‌کند. Ctrl+Shift+I یا Cmd+Opt+I همچنان کار می‌کند.'
    },
    connections: {
      title: 'دروازه‌های ثبت‌شده',
      intro: 'این دستگاه و همهٔ دروازه‌های Hermes قابل‌دسترسی از راه دور، SSH یا Cloud را مدیریت کنید.',
      stagedNote:
        'دروازه را از نشست‌ها تغییر دهید. پروفایل‌ها، گفت‌وگوها، پیام‌رسانی و زمان‌بندی‌ها همراه دروازه می‌مانند و کار روی سایر دروازه‌ها ادامه پیدا می‌کند.',
      launchModeTitle: 'هنگام اجرا، بازگشت به نشست‌های آخرین دروازهٔ استفاده‌شده',
      launchModeDesc: 'اگر خاموش باشد، نشست‌ها روی دروازهٔ اصلی باز می‌شوند.',
      searchPlaceholder: 'جست‌وجوی دروازه‌ها…',
      noSearchResults: 'دروازه‌ای با جست‌وجوی شما پیدا نشد.',
      loadFailed: 'بارگذاری اتصال‌ها ممکن نشد',
      currentPill: 'جاری',
      primaryPill: 'اصلی',
      managedPill: 'مدیریت‌شده با برنامه',
      addConnection: 'افزودن اتصال',
      editConnection: 'ویرایش',
      removeConnection: 'حذف',
      removeConfirmTitle: 'این اتصال حذف شود؟',
      removeConfirmDesc: label =>
        `«${label}» از این برنامه حذف می‌شود، اما خود نمونه دست‌نخورده می‌ماند و هر زمان می‌توانید دوباره اضافه‌اش کنید.`,
      makePrimary: 'انتخاب به‌عنوان اصلی',
      testConnection: 'آزمایش',
      testOk: 'در دسترس',
      testFailed: 'آزمایش اتصال ناموفق بود',
      saveFailed: 'ذخیرهٔ اتصال ممکن نشد',
      removeFailed: 'حذف اتصال ممکن نشد',
      updateAll: 'به‌روزرسانی همهٔ نمونه‌ها',
      updateAllRunning: 'در حال به‌روزرسانی همهٔ نمونه‌ها…',
      updateAllDone: 'به‌روزرسانی‌ها ارسال شدند',
      updateAllFailed: 'ارسال گروهی به‌روزرسانی ناموفق بود',
      updateSkippedCloud: 'تحت مدیریت Hermes Cloud',
      kindLocal: 'محلی',
      kindRemote: 'دروازهٔ راه‌دور',
      kindCloud: 'Hermes Cloud',
      kindSsh: 'SSH',
      kindLocalDesc: 'محیط اجرایی Hermes که این برنامه مدیریت می‌کند.',
      kindRemoteDesc: 'دروازهٔ Hermes در دسترس با HTTP(S)، در شبکهٔ محلی، Tailscale یا اینترنت.',
      kindCloudDesc: 'نمونهٔ میزبانی‌شده‌ای که از حساب Hermes Cloud پیدا شده است.',
      kindSshDesc: 'نصب Hermes که از راه SSH در دسترس است.',
      labelTitle: 'نام',
      labelDesc: 'الزامی و یکتا؛ در همه‌جای برنامه نمایش داده می‌شود، مانند «سرور خانه» یا «رایانهٔ کار».',
      labelPlaceholder: 'سرور خانه',
      urlTitle: 'نشانی دروازه',
      sshHostTitle: 'میزبان SSH',
      headersTitle: 'سرآیندهای اضافی دروازه',
      headersDesc:
        'با هر درخواست HTTP و WebSocket برای پراکسی‌هایی مانند Cloudflare Access فرستاده می‌شوند. مقدارها رمزگذاری می‌شوند و سرآیندهای تحت مدیریت Hermes نادیده گرفته خواهند شد.',
      headerValuePlaceholder: 'مقدار',
      headerValueSaved: 'ذخیره شده؛ برای نگه‌داشتن خالی بگذارید',
      headerAdd: 'افزودن سرآیند',
      headerRemove: 'حذف',
      duplicateLocal: 'این برنامه از قبل یک اتصال محلی دارد و بیش از یکی ممکن نیست.',
      duplicateUrl: label => `اتصالی به این نشانی دروازه از قبل وجود دارد («${label}»).`,
      duplicateSsh: label => `اتصالی به این میزبان SSH از قبل وجود دارد («${label}»).`,
      sameBackendHint: label => `همان پشتیبان «${label}»`,
      localAddHint: 'اتصال محلی در دسترس نیست؛ تنها اتصال محلی مجاز از قبل وجود دارد.',
      cloudAddHint:
        'نکته: ورود به Hermes Cloud عامل‌ها را خودکار پیدا می‌کند؛ این فرم برای ثبت دستی نشانی یک نمونه است.',
      save: 'ذخیرهٔ اتصال',
      saving: 'در حال ذخیره…',
      cancel: 'انصراف',
      empty: 'هنوز اتصالی ثبت نشده است.'
    },
    managedUpdates: {
      title: 'به‌روزرسانی‌های مدیریت‌شده',
      intro:
        'نصب‌های SSH تحت مدیریت دسکتاپ را تراکنشی به‌روز کنید: نشست‌ها تخلیه، نسخهٔ راه‌دور به‌روز و همهٔ پروفایل‌ها با رسید مرتبط بازیابی می‌شوند.',
      sshConnection: 'نصب SSH تحت مدیریت دسکتاپ',
      update: 'به‌روزرسانی',
      updating: 'در حال به‌روزرسانی…',
      progress: 'در حال تخلیهٔ نشست‌ها، به‌روزرسانی نصب راه‌دور و بازیابی پروفایل‌ها…',
      updated: 'به‌روز شد',
      partial: 'به‌روز شد؛ بازیابی ناموفق بود',
      refused: 'رد شد',
      failed: 'به‌روزرسانی ناموفق بود',
      alreadyRunning: 'به‌روزرسانی از قبل در حال اجراست',
      receipt: (id, outcome) => `رسید ${id} · ${outcome}`,
      receiptVersions: (pre, post) => `${pre} ← ${post}`,
      scopesRestored: profiles => `پروفایل‌های بازیابی‌شده: ${profiles}`,
      scopeNotRestored: (profile, error) => `پروفایل «${profile}» بازیابی نشد: ${error}`
    },
    gateway: {
      intro:
        'به‌طور پیش‌فرض محلی است. اگر برنامه باید پشتیبان Hermes دیگری را کنترل کند حالت راه‌دور را انتخاب کنید. اتصال‌های دروازه در سطح دستگاه‌اند و پروفایل‌ها از دروازه‌های متصل پیدا می‌شوند.',
      keychainEncryptionTitle: 'رمزگذاری رازهای ذخیره‌شده با مدیر کلید سیستم‌عامل',
      keychainEncryptionDesc:
        'به‌طور پیش‌فرض خاموش است. با روشن‌کردن، توکن دروازه و اطلاعات ورود با Keychain، GNOME Keyring یا Windows DPAPI رمزگذاری می‌شوند و ممکن است سیستم اجازه یا گذرواژه بخواهد. در حالت خاموش، فایل‌ها فقط برای حساب کاربری شما خواندنی‌اند.',
      keychainEncryptionFailed: 'تغییر رمزگذاری رازها ممکن نشد'
    },
    search: { placeholder: 'جست‌وجو در همهٔ تنظیمات…', pill: 'جست‌وجو' },
    profileScope: {
      appliesTo: 'اعمال روی',
      editsProfile: profile => `تغییرهای این صفحه روی پروفایل «${profile}» اعمال می‌شوند.`
    },
    mcp: {
      costTokens: tokens => `حدود ${tokens} توکن در هر فراخوانی`,
      usage30d: uses => `${uses} استفاده در ۳۰ روز`,
      unusedPill: 'استفاده‌نشده',
      noOutput: 'هنوز خروجی‌ای نیست.',
      deepLinkTitle: 'سرور MCP اضافه شود؟',
      deepLinkDescription:
        'یک پیوند درخواست افزودن این سرور MCP را داده است. پیکربندی دقیق زیر را بررسی کنید؛ این اطلاعات از پیوند آمده‌اند، نه از Hermes.',
      deepLinkStdioWarning:
        'این سرور با دستور زیر یک فرایند محلی روی دستگاه اجرا می‌کند. فقط در صورت اعتماد به منبع ادامه دهید.',
      deepLinkConfirm: 'افزودن سرور',
      deepLinkNameInvalid: 'نام باید ۱ تا ۶۴ حرف، رقم، نقطه، خط تیره یا زیرخط داشته باشد.',
      deepLinkNameConflict: name => `سروری با نام ${name} وجود دارد؛ نام دیگری انتخاب کنید یا انصراف دهید.`,
      deepLinkErrorTitle: 'پیوند نصب MCP رد شد',
      deepLinkErrorName: 'نام سرور در پیوند موجود یا معتبر نیست.',
      deepLinkErrorConfig: 'پیکربندی پیوند JSON معتبر با کدگذاری base64 نیست.',
      deepLinkErrorShape: 'پیکربندی باید شیء JSON با فیلد متنی `url` یا `command` باشد.',
      deepLinkErrorUrl: 'فقط نشانی‌های http:// و https:// مجازند.',
      deepLinkErrorTooLarge: 'حجم پیکربندی از سقف ۳۲ کیلوبایت بیشتر است.',
      importButton: 'درون‌ریزی',
      importPlaceholder: 'قطعهٔ mcp.json، دستور npx/docker، دستور claude mcp add، نشانی یا پیوند Cursor را بچسبانید…',
      importNoMatch: 'پیکربندی سرور شناخته‌شده‌ای در متن پیدا نشد.',
      importConfirm: 'افزودن به mcp.json',
      importConfirmMany: count => `افزودن ${count} سرور به mcp.json`
    },
    model: { tasks: { review: { label: 'بازبینی', hint: 'زیراَعامل بازبین /review' } } }
  },
  skills: {
    configuringProfile: 'در حال پیکربندی:',
    hub: {
      alreadyInstalled: name => `«${name}» از قبل نصب شده است`,
      pickerTitle: 'مرکز مهارت‌ها',
      pickerBrowse: 'مرور کامل مرکز',
      pickerHide: 'پنهان‌کردن مرورگر مرکز',
      pickerHint: 'روی «افزودن به این عامل» در هر مهارت بزنید تا نصب شود و در فهرست بالا ظاهر شود.'
    }
  },
  commandCenter: { openBrowser: 'باز کردن مرورگر', reloadWindow: 'بارگذاری دوبارهٔ پنجره' },
  profiles: {
    switchToConnection: name => `رفتن به ${name}`,
    switchConnectionFailed: name => `اتصال به ${name} ممکن نشد`,
    connectGateway: 'مدیریت دروازه‌ها…',
    fleet: {
      allOnGateway: 'همهٔ پروفایل‌های این دروازه',
      gateway: gateway => `پروفایل‌های ${gateway}`,
      gatewayUnreachable: gateway => `${gateway} · خارج از دسترس`,
      onGateway: (name, gateway) => `${name} · ${gateway}`,
      switchTo: (name, gateway) => `رفتن به ${name} روی ${gateway}`,
      deleteOn: gateway => ` روی ${gateway}`
    },
    remoteOverride: {
      menuItem: 'اتصال به میزبان راه‌دور…',
      badge: host => `اجرا روی ${host}`,
      title: profile => `اتصال ${profile} به میزبان راه‌دور`,
      description: 'نشست‌های این پروفایل به‌جای این رایانه روی Hermes راه‌دوری که تعیین می‌کنید اجرا می‌شوند.',
      urlLabel: 'نشانی راه‌دور',
      urlPlaceholder: 'https://hermes.example.com',
      urlInvalid: 'نشانی کامل با http:// یا https:// وارد کنید.',
      tokenLabel: 'توکن دسترسی',
      tokenPlaceholder: 'توکن نشست راه‌دور را بچسبانید',
      tokenSavedHint: 'توکنی ذخیره شده است؛ برای نگه‌داشتن خالی بگذارید.',
      plainTextOptIn:
        'این رایانه محل امنی برای کلید ندارد؛ توکن بدون رمزگذاری روی دیسک ذخیره می‌شود. بااین‌حال ذخیره شود.',
      collisionWarning: label =>
        `دروازه‌ای با نام «${label}» در تنظیمات وجود دارد. اتصال این پروفایل جداست و آن را تغییر نمی‌دهد.`,
      confirmTitle: 'این پروفایل به میزبان راه‌دور متصل شود؟',
      confirmNote: (profile, host) =>
        `گفت‌وگوهای جدید ${profile} روی ${host} اجرا می‌شوند. همان رایانه فرمان‌ها را اجرا و فایل‌ها را می‌خواند؛ فقط به میزبان مورداعتماد وصل شوید.`,
      confirmBack: 'بازگشت',
      connect: 'اتصال',
      connecting: 'در حال اتصال…',
      disconnect: 'حذف اتصال راه‌دور',
      savedTitle: 'پروفایل متصل شد',
      savedMessage: (profile, host) => `${profile} اکنون روی ${host} اجرا می‌شود`,
      removedTitle: 'اتصال راه‌دور حذف شد',
      removedMessage: profile => `${profile} اکنون روی این رایانه اجرا می‌شود`,
      removeFailed: 'حذف اتصال راه‌دور ممکن نشد',
      authFailedTitle: 'میزبان راه‌دور توکن ذخیره‌شده را نپذیرفت',
      authFailedMessage: (profile, host) =>
        `${host} توکن ذخیره‌شدهٔ ${profile} را رد کرد؛ ممکن است در سمت میزبان تغییر کرده باشد.`,
      updateToken: 'واردکردن توکن تازه…'
    },
    displayNameTitle: 'نام‌گذاری این عامل',
    displayNameDesc: 'نام نمایشی عامل را در سراسر برنامه تعیین می‌کند؛ شناسهٔ داخلی پروفایل همچنان «default» می‌ماند.',
    displayNameLabel: 'نام نمایشی'
  },
  cron: {
    modelImpact: {
      saveFailed: 'Hermes تغییر مدل را ذخیره نکرد.',
      confirmTitle: 'هشدار انتخاب مدل',
      confirmDetail: 'فقط اگر این موازنه را می‌پذیرید تأیید کنید.',
      confirmAction: 'تأیید',
      declined: 'تغییر مدل لغو شد؛ هشدار سطح آموزش با داده را نپذیرفتید.'
    }
  },
  sidebar: {
    messageCount: count => `${count} پیام`,
    toolCallCount: count => `${count} فراخوانی ابزار`,
    projects: {
      worktreeStaleBackend:
        'برای ساخت worktree روی این اتصال راه‌دور، پشتیبان Hermes را به‌روز کنید؛ نسخهٔ فعلی پیش از API مربوط به git worktree ساخته شده است.'
    },
    row: {
      markUnread: 'علامت‌گذاری به‌عنوان خوانده‌نشده',
      markRead: 'علامت‌گذاری به‌عنوان خوانده‌شده',
      unreadFailed: 'به‌روزرسانی وضعیت خواندن ممکن نشد',
      openInTerminal: 'باز کردن در پایانه',
      deleteTitle: 'نشست حذف شود؟',
      deleteDesc: title => `«${title}» برای همیشه حذف می‌شود و بازگشت‌پذیر نیست.`,
      deleting: 'در حال حذف…',
      deleted: 'نشست حذف شد',
      messageCount: count => `${count} پیام`
    },
    markAllRead: 'خوانده‌شدن همه'
  },
  composer: {
    voiceControls: 'صدا',
    githubSuggestions: {
      label: 'راه‌اندازی GitHub',
      tip: 'GitHub از طریق مهارت‌های gh CLI کار می‌کند؛ برای اتصال حساب کلیک کنید',
      done: '/github-auth افزوده شد',
      doneTip: 'پیام را بفرستید تا عامل مرحله‌به‌مرحله ورود به GitHub را انجام دهد'
    }
  },
  updates: {
    blockerTitle: 'پیش‌نمایش‌های محلی برای به‌روزرسانی Hermes بسته شوند؟',
    blockerBody:
      'Hermes باید این پیش‌نمایش‌های محلی را پیش از به‌روزرسانی متوقف کند؛ فایل‌های شما تغییر نمی‌کنند و حذف نمی‌شوند.',
    foreignBlockerTitle: 'برای به‌روزرسانی Hermes فرایندهای دیگر را ببندید',
    foreignBlockerBody:
      'Hermes نمی‌تواند این فرایندها را با اطمینان ببندد. برنامه، پایانه یا سرویس مالک هرکدام را ببندید و دوباره تلاش کنید.',
    mixedBlockerBody: 'Hermes می‌تواند پیش‌نمایش‌های محلی زیر را ببندد؛ سایر فرایندها باید دستی بسته شوند.',
    closePreviewsAndUpdate: 'بستن پیش‌نمایش‌ها و به‌روزرسانی',
    closePreviewsAndCheckAgain: 'بستن پیش‌نمایش‌ها و بررسی دوباره',
    localPreview: 'پیش‌نمایش محلی',
    portLabel: port => `درگاه ${port}`,
    pidLabel: pid => `PID ${pid}`,
    technicalDetails: 'جزئیات فنی',
    clientAlsoBehindTitle: 'برنامهٔ دسکتاپ عقب است',
    clientAlsoBehindMessage:
      'پشتیبان به‌روز است اما این برنامهٔ دسکتاپ هنوز نسخهٔ قدیمی دارد. برای دریافت آخرین اصلاح‌ها آن را به‌روز کنید.',
    clientAlsoBehindAction: 'به‌روزرسانی برنامهٔ دسکتاپ',
    everythingDispatched: 'به‌روزرسانی ارسال شد',
    everythingSkipped: 'رد شد',
    everythingRowFailed: 'به‌روزرسانی ناموفق بود',
    everythingFanoutFailedTitle: 'به‌روزرسانی سایر نمونه‌ها ممکن نشد'
  },
  shell: {
    gatewayMenu: { reconnectGateway: 'اتصال دوبارهٔ دروازه' },
    statusbar: { gatewayUnavailable: 'استنتاج در دسترس نیست' }
  },
  preview: {
    openInExternal: 'باز کردن در برنامهٔ بیرونی',
    popIn: 'انتقال به داخل',
    popOut: 'انتقال به بیرون',
    web: {
      remoteLoopback:
        'این نشانی به رایانهٔ عامل اشاره می‌کند، نه این دستگاه. صفحهٔ مرورگر محلی بارگذاری می‌شود؛ سرور توسعهٔ راه‌دور به انتقال درگاه یا نام میزبان قابل‌دسترسی نیاز دارد.',
      goBack: 'عقب',
      goForward: 'جلو',
      reload: 'بارگذاری دوباره',
      address: 'نشانی',
      addressPlaceholder: 'نشانی را وارد کنید',
      blankPageBody: 'نشانی را بالا وارد کنید یا از Hermes بخواهید صفحه‌ای باز کند.'
    }
  },
  zones: {
    showTabStrip: 'نمایش زبانه‌ها',
    hideTabStrip: 'پنهان‌کردن زبانه‌ها',
    showStripTab: title => `نمایش ${title}`,
    hideStripTab: title => `پنهان‌کردن ${title}`,
    lastTabKeptTitle: 'آخرین زبانه می‌ماند',
    lastTabKeptBody:
      'این ناحیه دست‌کم به یک زبانهٔ نمایان نیاز دارد. نخست زبانهٔ دیگری را نشان دهید یا کل نوار کناری را جمع کنید.',
    toggleStripTab: title => `تغییر وضعیت زبانهٔ ${title}`,
    newTab: 'زبانهٔ جدید'
  },
  contextMenu: {
    link: {
      openInApp: 'باز کردن در مرورگر داخلی',
      openExternal: 'باز کردن در مرورگر بیرونی',
      copyUrl: 'رونوشت نشانی',
      copyResolvedUrl: 'رونوشت نشانی نهایی'
    },
    image: { copyImage: 'رونوشت تصویر', copyImageAddress: 'رونوشت نشانی تصویر', saveImageAs: 'ذخیرهٔ تصویر با نام…' },
    edit: { cut: 'برش', paste: 'چسباندن', selectAll: 'انتخاب همه', addToDictionary: 'افزودن به واژه‌نامه' },
    page: { copyPageUrl: 'رونوشت نشانی صفحه', inspectElement: 'بررسی عنصر' }
  },
  assistant: {
    thread: {
      turnDuration: duration => `این نوبت ${duration} طول کشید`,
      errorLayers: {
        auth: 'خطای احراز هویت',
        billing: 'اعتبار تمام شده',
        disk: 'دیسک پر است',
        endpoint: 'خطای نقطهٔ پایانی سفارشی',
        gateway: 'خطای دروازه',
        generic: 'نوبت ناموفق بود',
        provider: 'خطای ارائه‌دهنده',
        runtime: 'خطای محیط اجرایی محلی',
        streaming: 'خطای اتصال جریانی'
      },
      errorRetry: 'تلاش دوباره',
      errorSwitchProvider: 'تغییر ارائه‌دهنده',
      errorOpenLogs: 'باز کردن گزارش‌ها',
      errorOpenLogsFailed: 'باز کردن پوشهٔ گزارش‌ها ممکن نشد',
      errorOpenDesktopLogs: 'باز کردن گزارش‌های دسکتاپ',
      errorCopyDiagnostics: 'رونوشت جزئیات خطا',
      errorSendDiagnostics: 'ارسال اطلاعات عیب‌یابی'
    },
    clarify: {
      confirmAndContinueLabel: 'تأیید و ادامه',
      answeredBadge: 'پاسخ‌داده‌شده',
      questionProgress: (answered, total) => `${answered} از ${total} پاسخ داده شد`
    }
  },
  desktop: {
    editTurnUnavailable: 'این نوبت دیگر در تاریخچهٔ سرور نیست و ممکن است فشرده شده باشد.',
    readOnlyTranscriptTitle: 'به‌صورت فقط‌خواندنی باز شد',
    readOnlyTranscriptBody:
      'هیچ پشتیبان متصلی هنوز مالک این گفت‌وگوی قدیمی نیست؛ تاریخچه سالم است اما تا زمانی که پشتیبانی آن را بپذیرد ارسال غیرفعال می‌ماند.',
    readOnlyTranscriptSendBlocked: 'این گفت‌وگو فقط‌خواندنی است و ارسال غیرفعال است.',
    hydrationSyncing: profile => `در حال همگام‌سازی ${profile}…`
  },
  tips: {
    close: 'این نکته دوباره نشان داده نشود',
    items: {
      'new-session': { title: 'شروع تازه', text: 'هر گفت‌وگوی جدید زمینه، پایانه و پوشهٔ کاری مستقل دارد.' },
      skills: {
        title: 'یک بار آموزش بدهید',
        text: 'مهارت‌ها پوشه‌های راهنمایی‌اند که Hermes هنگام نیاز بارگذاری می‌کند.'
      },
      messaging: {
        title: 'Hermes دور از میز کار',
        text: 'Telegram، Discord، Slack و دیگر پیام‌رسان‌ها را با همان عامل و حافظه متصل کنید.'
      },
      artifacts: {
        title: 'همهٔ ساخته‌های Hermes',
        text: 'تصاویر، فایل‌ها و پیوندهای همهٔ نشست‌ها یک‌جا نمایه می‌شوند.'
      },
      cron: { title: 'کار خودکار', text: 'یک درخواست را ساعتی، شبانه یا با عبارت cron زمان‌بندی کنید.' },
      'command-palette': {
        title: 'یک کادر برای همه‌چیز',
        text: 'نشست‌ها، تنظیمات، مهارت‌ها و فرمان‌ها همگی از پالت در دسترس‌اند.'
      },
      profiles: { title: 'پروفایل‌ها مستقل‌اند', text: 'هرکدام Hermes جداگانه با کلیدها، حافظه و نشست‌های خود است.' },
      'composer-mentions': {
        title: 'پیوست و فرمان',
        text: 'برای آوردن فایل به گفت‌وگو @ و برای اجرای فرمان / بنویسید.'
      },
      'right-pane': { title: 'صفحهٔ کار', text: 'فایل‌ها، پایانه، بازبینی و مرورگر داخلی در سمت کار مشترک‌اند.' }
    }
  }
} satisfies TranslationOverrides
