// Arabic translations batch 2 (machine-translated). Glossary already
// applied; do not re-translate. Deep-merged over the section overlays in ar.ts.
import { arBatch2Settings } from './ar_batch_2_settings'
import type { TranslationOverrides } from './define-locale'

export const arBatch2 = {
  agents: {
    ageDays: days => `منذ ${days} يوم`
  },
  appTour: {
    capabilities: {
      text: 'المهارات والأدوات والإضافات التي يستطيع Hermes استخدامها. أضف المزيد هنا.',
      title: 'الإمكانات'
    },
    composer: {
      text: 'اكتب ما تريد إنجازه. اكتب @ لإدراج ملف.',
      title: 'اسأل هنا'
    },
    messaging: {
      text: 'تواصل مع Hermes من Telegram وSlack وDiscord وغيرها.',
      title: 'المراسلة'
    },
    model: {
      text: 'يحدد النموذج الذي يجيبك.',
      title: 'منتقي النموذج'
    },
    modelLocal: 'يمكن لهذا الحاسوب تشغيل نموذج محليًا: الإعدادات > المزوّدون > النماذج المحلية.',
    newSession: {
      text: 'للجلسة الجديدة سياقها الخاص. استخدم جلسة لكل مهمة.',
      title: 'ابدأ من جديد'
    },
    rightPane: {
      text: 'يفتح الملفات وTerminal والمراجعة والمتصفح داخل التطبيق على اليمين.',
      title: 'لوحة العمل'
    },
    sessions: {
      text: 'كل محادثة توجد هنا. ابحث فيها أو ثبّتها أو أعد فتحها.',
      title: 'محادثاتك'
    }
  },
  assistant: {
    tool: {
      titleTemplates: {
        actionCommand: (action, command) => `${action} ${command}`,
        actionQuoted: (action, value) => `${action} \u201C${value}\u201D`,
        actionTarget: (action, target) => `${action} ${target}`,
        prefixedDone: (prefix, action) => `${prefix} ${action}`
      }
    }
  },
  billingBlock: {
    addCredits: 'إضافة أرصدة',
    dismiss: 'تجاهل',
    fallbackMessage: 'نفدت أرصدة حسابك. أضف أرصدة للمتابعة.',
    openBilling: 'فتح الفوترة',
    titleNous: 'نفدت أرصدة Nous',
    titleProvider: provider => `نفدت الأرصدة \u2014 ${provider}`
  },
  boot: {
    causes: {
      diskFull: 'القرص ممتلئ، لذا لم يتمكن Hermes من البدء.',
      exitedEarly: 'توقفت خدمة Hermes الخلفية فور بدئها.',
      installMissing: 'جزء من تثبيت Hermes مفقود. اختر «إصلاح التثبيت» لإعادته.',
      permission: 'لم يتمكن Hermes من الكتابة في مجلد بياناته (مشكلة صلاحيات).',
      portInUse: 'برنامج آخر يستخدم منفذ الشبكة الذي يحتاجه Hermes.',
      timedOut: 'لم تستجب خدمة Hermes الخلفية في الوقت المناسب.'
    },
    errors: {
      connectionSettings: 'إعدادات الاتصال',
      gatewaySignInRequiredDetail: 'سجّل الدخول مجددًا لإعادة الاتصال. محادثاتك وإعداداتك بأمان.',
      openLogs: 'فتح السجلات',
      reconnectNow: 'إعادة الاتصال الآن',
      restartHermes: 'إعادة تشغيل Hermes',
      signInAgain: 'تسجيل الدخول مجددًا'
    },
    failure: {
      back: 'رجوع',
      bundledReinstallHint:
        'لا يستطيع هذا التثبيت المُضمَّن إصلاح نفسه من داخل التطبيق — أعد تثبيت التطبيق لاستعادة خدمته الخلفية.',
      details: 'التفاصيل',
      gatewaySettings: 'إعدادات البوابة',
      reinstallApp: 'إعادة تثبيت التطبيق',
      remoteFailureHint: 'تحقق من عنوان URL للبوابة وتسجيل الدخول في إعدادات البوابة، أو انتقل إلى البوابة المحلية.',
      webappDescription: 'شغّل hermes webapp على هذا الحاسوب لتسجيل دخول هذا المتصفح.',
      webappTitle: 'تسجيل الدخول إلى Hermes'
    },
    updateHold: {
      checkAgain: 'تحقق مرة أخرى',
      confirmBody:
        'قد تكون عملية التحديث المتبقية ما زالت تغيّر ملفات Hermes. البدء الآن قد يحمّل تثبيتًا محدَّثًا جزئيًا، وقد لا يعمل حتى تشغّل التحديث مجددًا. يسجّل Hermes هذا الاختيار في سجله ويترك علامة التحديث في مكانها.',
      confirmKeepWaiting: 'متابعة الانتظار',
      confirmStart: 'البدء على أي حال',
      confirmTitle: 'هل تريد بدء Hermes بينما التحديث ما زال يمنعه؟',
      description:
        'يؤجل Hermes البدء كي لا يحمّل ملفات قد يكون التحديث ما زال يغيّرها. سيبدأ من تلقاء نفسه بمجرد انتهاء التعليق.',
      heldByProcess: pid => `انتهت عملية التحديث (العملية ${pid})، لكن عملية بدأتها ما زالت تحتجز تثبيت Hermes.`,
      heldUnknown: 'انتهى تحديث، لكن عملية بدأها ما زالت تحتجز تثبيت Hermes.',
      lastChecked: time => `آخر تحقق ${time}`,
      openLogs: 'فتح السجلات',
      quit: 'إنهاء Hermes',
      recoveryHint:
        'يزول هذا عادةً خلال دقائق. إن لم يحدث: أنهِ Hermes، وأغلق عمليات git أو hermes المتبقية (أو أعد تشغيل الحاسوب)، ثم افتح Hermes مجددًا.',
      since: time => `الانتظار منذ ${time}`,
      startAnyway: 'البدء على أي حال…',
      startAnywayRefused: 'تغيّر ما يحتجز التثبيت قبل أن يتمكن Hermes من البدء. راجعه وحاول مجددًا.',
      title: 'تحديث سابق ما زال يحتجز Hermes',
      titleUnverified: 'لا يستطيع Hermes تأكيد اكتمال التحديث الأخير',
      unverified: 'لم يتمكن مساعد التحديث من التحقق من مالك تثبيت Hermes الآن. يواصل Hermes التحقق.'
    }
  },
  catalog: {
    developer: {
      docs: 'وثائق المطورين',
      heading: 'وضع المطور',
      pitch: 'هل أنشأت أداة أو موصلًا أو إضافة أو Mod؟',
      pitchFollowUp: 'تريد ذلك لكن لا تعرف كيف؟',
      submit: 'إرسال Mod',
      title: 'ابنِ لـ Hermes'
    },
    discoverKind: kind => `استكشف ${kind}`,
    moreToExplore: 'المزيد للاستكشاف',
    moreToExploreBlurb: 'أجزاء أصغر من الكتالوج، في مكان واحد.',
    nextSlide: 'المميز التالي',
    previousSlide: 'المميز السابق',
    skillsFor: category => `مهارات ${category.toLowerCase()}.`,
    slideOf: (index, count) => `${index} من ${count}`,
    tier: 'المستوى',
    tierLabels: {
      community: 'المجتمع',
      official: 'رسمي',
      verified: 'موثَّق'
    },
    tiers: {
      official: {
        blurb: 'من إنشاء Nous Research وصيانتها.',
        title: 'إضافات رسمية'
      },
      verified: {
        blurb: 'من الشركة المالكة للتكامل، موثَّقة من Nous.',
        title: 'إضافات موثَّقة'
      }
    },
    toolsets: {
      callCount: (count, calls) =>
        calls === 1
          ? 'استدعاء واحد'
          : calls === 2
            ? 'استدعاءان'
            : calls <= 10
              ? `${count} استدعاءات`
              : `${count} استدعاء`,
      essentials: 'الأساسيات',
      essentialsBlurb: 'ما يعتمد عليه معظم عمل Hermes.',
      mostUsed: 'الأكثر استخدامًا',
      mostUsedBlurb: 'مجموعات الأدوات التي يستخدمها Hermes أكثر على هذا الملف الشخصي.',
      themes: {
        connect: {
          blurb: 'الحسابات والوكلاء الآخرون والمنزل الذكي والبحث خارج الويب.',
          label: 'الاتصال والتواصل'
        },
        create: {
          blurb: 'شاهد الصور والفيديو، وفرّغ الصوت نصيًا، وأنشئ وسائط.',
          label: 'شاهد واستمع وأنشئ'
        },
        more: {
          blurb: 'كل ما يمكن لهذا الملف الشخصي تفعيله.',
          label: 'أدوات إضافية'
        },
        recall: {
          blurb: 'ما يتذكره Hermes، والمهارات والسجل الذي يمكنه البحث فيه.',
          label: 'الذاكرة والمعرفة'
        },
        work: {
          blurb: 'شغّل المتصفح وسطح المكتب، وحرّر الملفات، ونفّذ التعليمات البرمجية، وجدول المهام.',
          label: 'التصفح والعمل والأتمتة'
        }
      },
      toolCount: count =>
        count === 1 ? 'أداة واحدة' : count === 2 ? 'أداتان' : count <= 10 ? `${count} أدوات` : `${count} أداة`
    }
  },
  commandCenter: {
    installTheme: {
      pageTitle: 'تثبيت السمة'
    },
    logFile: 'ملف السجل',
    logLevel: 'المستوى',
    maintenance: {
      actionFailed: name => `فشل بدء ${name}`,
      actionStarted: name => `بدأ ${name} \u2014 تتم متابعة السجل...`,
      backup: 'إنشاء نسخة احتياطية',
      backupDesc: 'ضغط الإعدادات والذكريات والمهارات والجلسات في ملف ZIP',
      bytes: size => size,
      copyLink: 'نسخ الرابط',
      curator: 'منسّق المهارات',
      curatorActive: 'نشط',
      curatorDesc: 'مراجعة خلفية تؤرشف المهارات القديمة التي أنشأها الوكيل',
      curatorDisabled: 'معطّل',
      curatorLastRun: when => `آخر تشغيل ${when}`,
      curatorNeverRan: 'لم يُشغَّل قط',
      curatorPaused: 'متوقف مؤقتًا',
      debugShare: 'مشاركة التشخيص',
      debugShareDesc: 'ارفع تقريرًا منقّحًا مع السجلات، واحصل على روابط قابلة للمشاركة (تُحذف تلقائيًا بعد 6 ساعات)',
      debugShareFailed: 'فشلت مشاركة التشخيص',
      debugShareLinks: 'روابط المشاركة',
      debugShareRunning: 'جارٍ رفع تقرير التشخيص...',
      doctor: 'تشغيل الفحص',
      doctorDesc: 'فحص سلامة التثبيت والإعدادات والمزوّدين',
      empty: 'فارغ',
      linkCopied: 'تم نسخ الرابط',
      memoryData: 'بيانات الذاكرة',
      memoryFile: 'ذاكرة الوكيل (MEMORY.md)',
      pause: 'إيقاف مؤقت',
      resetAll: 'إعادة تعيين كليهما',
      resetConfirm: target => `حذف ${target}؟ لا يمكن التراجع عن هذا.`,
      resetDone: files => `تم حذف ${files}.`,
      resetFailed: 'فشلت إعادة تعيين الذاكرة',
      resetMemory: 'إعادة تعيين الذاكرة',
      resetUser: 'إعادة تعيين الملف الشخصي',
      resume: 'استئناف',
      runNow: 'تشغيل الآن',
      runOps: 'التشخيص',
      running: 'جارٍ التشغيل...',
      securityAudit: 'تدقيق أمني',
      securityAuditDesc: 'فحص الإعدادات والمهارات بحثًا عن إعدادات محفوفة بالمخاطر',
      userFile: 'ملف المستخدم (USER.md)',
      viewLog: 'سجل الإجراءات'
    },
    newSessionInProject: project => `جلسة جديدة في ${project}`,
    openFolder: 'فتح مجلد كمشروع…',
    openFolderAt: path => `فتح مجلد كمشروع \u2014 ${path}`,
    projects: 'المشاريع',
    sections: {
      maintenance: 'الصيانة'
    }
  },
  composer: {
    themeTryPost: '.'
  },
  connectorOffer: {
    body: 'سجّل الدخول بحساب Nous مجاني للإبقاء على التطبيقات التي وصّلتها، مثل Gmail أو Slack، في حسابك.',
    heading: 'احتفظ بتطبيقاتك المتصلة'
  },
  connectors: {
    authorizedToolsUnavailable: 'مُصرَّح. الأدوات غير متاحة.',
    cancel: 'إيقاف الانتظار',
    checking: 'جارٍ التحقق من تطبيقاتك…',
    connect: 'اتصال',
    connectError: 'تعذّر بدء التصريح. حاول مجددًا.',
    connectErrorFor: app => `تعذّر بدء التصريح لـ ${app}.`,
    connected: 'متصل',
    disabled: 'غير متاح',
    disclaimer: 'الاتصال اختياري. لا تصرّح إلا للتطبيقات التي تريد أن يستخدمها Hermes.',
    empty: 'لا توجد تطبيقات مطابقة',
    execution: 'أدوات الموصلات',
    failed: 'تعذّر الاتصال',
    grant: 'إعادة الاتصال',
    namedAccount: (app, alias) => `سمّى Hermes حساب ${app} هذا ${alias}.`,
    needsAuth: 'انتهت صلاحية الوصول',
    notConnected: 'غير متصل',
    openInBrowser: 'فتح في المتصفح',
    opening: 'جارٍ فتح تسجيل الدخول…',
    ownerMissing: 'أعد فتح هذه المحادثة لإدارة اتصالاتها.',
    refresh: 'تحديث الحالة',
    renameAccount: 'إعادة تسمية',
    required: 'مطلوب',
    retry: 'حاول مجددًا',
    search: 'ابحث عن تطبيق',
    setup: server => `إعداد ${server}`,
    setupCancel: 'إلغاء',
    skip: 'ليس الآن',
    skipped: 'تم التخطي',
    timeout: 'ما زال في انتظار التصريح.',
    title: 'وصّل تطبيقاتك',
    unavailable: 'الموصلات غير متاحة لهذه الجلسة.',
    waiting: 'في انتظار متصفحك…'
  },
  connectorsPage: {
    accounts: {
      add: 'اتصال',
      addAnother: 'إضافة حساب آخر',
      aliasHint: 'أحرف صغيرة وأرقام وشرطات، حتى 32 حرفًا. مثلًا work أو home.',
      aliasInvalid: 'استخدم أحرفًا صغيرة وأرقامًا وشرطات، تبدأ بحرف أو رقم.',
      aliasLabel: 'اسم الحساب',
      aliasTaken: 'هذا الاسم مستخدم بالفعل.',
      count: count =>
        count === 1 ? 'حساب واحد' : count === 2 ? 'حسابان' : count <= 10 ? `${count} حسابات` : `${count} حساب`,
      heading: 'الحسابات',
      remove: 'إزالة',
      removeBody: 'يتوقف Hermes عن العمل بهذا الحساب. تبقى حساباتك الأخرى متصلة.',
      removeTitle: (app, name) => `هل تريد إزالة حساب ${name} في ${app}؟`,
      rename: 'إعادة تسمية',
      renameGone: 'لم يعد هذا الحساب موجودًا. أُزيل أو استُبدل بإعادة اتصال.',
      renameLabel: name => `اسم جديد لـ ${name}`,
      retiredCount: count =>
        count === 1
          ? 'حساب واحد مُستبدَل'
          : count === 2
            ? 'حسابان مُستبدَلان'
            : count <= 10
              ? `${count} حسابات مُستبدَلة`
              : `${count} حسابًا مُستبدَلًا`,
      status: {
        expired: 'انتهت صلاحية الوصول',
        failed: 'تعذّر الاتصال',
        inactive: 'غير نشط',
        pending: 'في انتظار تسجيل الدخول',
        retired: 'مُستبدَل بإعادة اتصال',
        revoked: 'أُلغيت صلاحية الوصول'
      }
    },
    add: {
      action: 'أضف موصلًا خاصًا بك',
      addArg: '+ إضافة وسيط',
      addEnvVar: '+ إضافة متغير بيئة',
      addHeader: '+ إضافة ترويسة',
      addPassthrough: '+ إضافة متغير',
      args: 'الوسائط',
      auth: 'المصادقة',
      authBearer: 'رمز Bearer',
      authNone: 'بلا',
      authOauth: 'OAuth',
      command: 'أمر التشغيل',
      cwd: 'دليل العمل',
      editJson: 'تحرير mcp.json',
      envVars: 'متغيرات البيئة',
      headers: 'الترويسات',
      hint: 'مدخل جديد واحد في mcp.json على هذا الجهاز',
      keyPlaceholder: 'KEY',
      name: 'الاسم',
      nameTaken: 'هذا الاسم مستخدم بالفعل.',
      passthrough: 'تمرير متغيرات البيئة',
      pasteLabel: 'الصق أمرًا أو مقتطفًا',
      pasteNoMatch: 'لا شيء هنا يبدو خادمًا. املأ الحقول أدناه بدلًا من ذلك.',
      pastePlaceholder: 'npx -y @modelcontextprotocol/server-filesystem /path/to/dir',
      removeRow: 'إزالة هذا الصف',
      saveFailed: 'لم يُحفظ ذلك الخادم.',
      title: 'الاتصال بـ MCP مخصص',
      type: 'النوع',
      typeHttp: 'HTTP قابل للبث',
      typeStdio: 'STDIO',
      url: 'URL',
      valuePlaceholder: 'القيمة'
    },
    card: {
      alsoLocal: 'يعمل أيضًا على هذا الجهاز',
      fact: {
        tools: count =>
          count === 1 ? 'أداة واحدة' : count === 2 ? 'أداتان' : count <= 10 ? `${count} أدوات` : `${count} أداة`,
        toolsOff: count =>
          count === 1
            ? 'أداة واحدة متوقفة'
            : count === 2
              ? 'أداتان متوقفتان'
              : count <= 10
                ? `${count} أدوات متوقفة`
                : `${count} أداة متوقفة`,
        toolsOn: count =>
          count === 1
            ? 'أداة واحدة مفعّلة'
            : count === 2
              ? 'أداتان مفعّلتان'
              : count <= 10
                ? `${count} أدوات مفعّلة`
                : `${count} أداة مفعّلة`,
        toolsSomeOn: (total, on) => `${total} أداة، ${on} مفعّلة`
      },
      hostedTwin: 'تتوفر نسخة مُدارة',
      inCatalog: 'في كتالوج Hermes',
      kindCatalog: 'MCP · كتالوج',
      kindCustom: 'MCP · مخصص',
      kindManaged: 'مُدار',
      kindPlugin: plugin => `MCP \xB7 إضافة ${plugin}`,
      open: name => `فتح ${name}`,
      reason: {
        finishSignIn: 'أكمل تسجيل الدخول في متصفحك.',
        reconnect: 'أعد الاتصال ليبقى هذا التطبيق يعمل.',
        serverError: 'رفض الخادم الاتصال.',
        serverNeedsAuth: 'سجّل الدخول ليتمكن هذا الخادم من الرد.'
      },
      state: {
        accessExpired: 'انتهت صلاحية الوصول',
        available: 'متاح',
        connected: 'متصل',
        connecting: 'جارٍ الاتصال',
        connectionUnknown: 'الحالة غير معروفة',
        couldNotConnect: 'تعذّر الاتصال',
        offByYourOrganisation: 'معطّل من مؤسستك',
        offForYou: 'معطّل لك',
        serverConnecting: 'جارٍ الاتصال…',
        serverError: 'خطأ',
        serverNeedsAuth: 'يتطلب مصادقة',
        serverOff: 'متوقف',
        serverOn: 'مفعّل',
        serverOnUnused: 'مفعّل، غير مستخدم'
      },
      turnServerOff: name => `إيقاف ${name}`,
      turnServerOn: name => `تفعيل ${name}`,
      verb: {
        authenticate: 'مصادقة',
        connect: 'اتصال',
        install: 'تثبيت',
        openLogs: 'فتح السجلات',
        reconnect: 'إعادة الاتصال',
        stopWaiting: 'إيقاف الانتظار',
        tryAgain: 'حاول مجددًا',
        turnBackOn: 'إعادة التفعيل'
      }
    },
    categoryAll: 'كل الفئات',
    dialog: {
      advanced: 'متقدم',
      advancedHint: 'مدخل mcp.json والسجلات',
      appSwitch: name => `يمكن لـ Hermes استخدام ${name}`,
      bothOn: name => `كلاهما مفعّل، لذا يرى Hermes كل أداة ${name} مرتين.`,
      connectEnded: 'لم يكتمل تسجيل الدخول.',
      connectOpenAgain: 'افتح الرابط مجددًا',
      disconnect: 'قطع الاتصال',
      disconnectBody: 'يتوقف Hermes عن العمل بهذا الحساب. يمكنك الاتصال مجددًا في أي وقت.',
      disconnectTitle: name => `قطع اتصال ${name}؟`,
      menuRefreshTools: 'تحديث الأدوات',
      moreActions: 'مزيد من الإجراءات',
      nousLine: 'تطبيقات Nous تتبع حسابك، لا الملف الشخصي.',
      openPlugins: 'فتح تبويب الإضافات',
      orgLink: 'فتح إدارة الموصلات',
      orgNote: count =>
        count === 1
          ? 'أوقفت مؤسستك أداة واحدة.'
          : count === 2
            ? 'أوقفت مؤسستك أداتين.'
            : count <= 10
              ? `أوقفت مؤسستك ${count} أدوات.`
              : `أوقفت مؤسستك ${count} أداة.`,
      providedByPlugin: plugin => `مقدَّم من الإضافة ${plugin}`,
      removeServerBody: 'يزيل المدخل من mcp.json على هذا الحاسوب. لا يُحذف شيء آخر.',
      removeServerTitle: name => `إزالة ${name}؟`,
      rulesAppOff: name => `فعّل ${name} لتغيير أدواته.`,
      rulesReadOnly: 'لا يمكن تغيير القواعد الآن.',
      rulesSignIn: 'سجّل الدخول لتغيير ما يمكن لـ Hermes فعله هنا.',
      tokensPerCall: 'Tokens لكل استدعاء',
      turnOffLocal: 'إيقاف الخادم المحلي',
      usesPerMonth: 'استخدام خلال 30 يومًا',
      wayHosted: 'مُدار',
      wayNotConnected: name => `غير متصل بعد. سجّل الدخول إلى ${name} في متصفحك.`,
      waysTitle: name => `أين يعمل ${name}`
    },
    filterCategory: 'الفئة',
    group: {
      available: 'متاح',
      connected: 'متصل',
      connectedNote: 'الاتصالات المعطلة أولًا.',
      off: 'معطّل',
      offNote: 'يُحتفظ بتسجيلات الدخول.'
    },
    page: {
      clearSearch: 'مسح البحث',
      disconnectNoAccount: 'ليس لدى Hermes حساب لقطع اتصاله هنا. حدّث الصفحة وحاول مجددًا.',
      disconnectRefused:
        'لم يتمكن Nous من إزالة تسجيل الدخول هذا الآن. أوقف التطبيق بالمفتاح بدلًا من ذلك، أو حاول لاحقًا.',
      emptyTitle: 'لا توجد تطبيقات هنا بعد. أضف خادمًا على هذا الحاسوب للبدء.',
      freeTierNote: 'تبقى الاتصالات على هذا الحاسوب حتى تسجّل الدخول.',
      hostedFailedBody: 'الخوادم على هذا الحاسوب لم تتأثر وما زالت تعمل. لم يُوقف أي شيء.',
      hostedFailedTitle: 'تعذّر الوصول إلى التطبيقات المُدارة.',
      loading: 'جارٍ قراءة الكتالوج والخوادم على هذا الحاسوب',
      managedUnavailable: 'التطبيقات المُدارة غير متاحة لهذا الحساب بعد.',
      matchesElsewhere: count =>
        count === 1
          ? 'تطابق واحد آخر في مجموعات أخرى.'
          : count === 2
            ? 'تطابقان آخران في مجموعات أخرى.'
            : count <= 10
              ? `${count} تطابقات أخرى في مجموعات أخرى.`
              : `${count} تطابقًا آخر في مجموعات أخرى.`,
      noMatchBody: 'لا شيء هنا مطابق. وجّه Hermes إلى خادم MCP الخاص بك لإضافته.',
      noMatchTitle: 'لا توجد تطبيقات مطابقة',
      refreshFailed: 'لم يتم تحديث قائمة الأدوات.',
      retry: 'إعادة المحاولة',
      segmentNoMatch: segment => `لا يوجد تطابق في ${segment}، لذا تُعرض كل المطابقات.`,
      showAllMatches: 'عرض كل المطابقات',
      signIn: 'تسجيل الدخول',
      signInLine: 'سجّل الدخول إلى Nous لاستخدام التطبيقات المُدارة.',
      writeFailed: 'لم يُحفظ ذلك التغيير.'
    },
    residencyLocal: 'على هذا الجهاز',
    searchPlaceholder: count =>
      count === 1
        ? 'ابحث في تطبيق واحد'
        : count === 2
          ? 'ابحث في تطبيقين'
          : count <= 10
            ? `ابحث في ${count} تطبيقات`
            : `ابحث في ${count} تطبيقًا`,
    segment: {
      all: 'الكل',
      available: 'متاح',
      connected: 'متصل',
      off: 'معطّل'
    },
    title: 'الموصلات',
    tools: {
      allToolsSwitch: 'تفعيل أو إيقاف كل الأدوات',
      categorySelect: count => `${count} فئات`,
      conflictBody: (theyOff, theyOn) => {
        const they = [
          theyOff > 0 ? `أوقفوا ${theyOff} من الأدوات التي فعّلتها` : '',
          theyOn > 0 ? `تركوا ${theyOn} من الأدوات التي أوقفتها مفعّلة` : ''
        ].filter(Boolean)

        return `${they.length > 0 ? `لقد ${they.join('، و')}. ` : ''}تبقى تعديلاتك على الشاشة؛ لم يُكتب شيء.`
      },
      conflictReload: 'إعادة تحميل نسختهم',
      conflictSave: 'الحفظ فوق نسختهم',
      conflictTitle: 'غيّر أحدهم هذه القاعدة أثناء تحريرك.',
      discard: 'تجاهل',
      facetSwitch: facet => `تفعيل أو إيقاف أدوات ${facet}`,
      footerDirty: (off, backOn) => `${off} أداة متوقفة، و${backOn === 0 ? 'لا شيء' : backOn} أُعيد تفعيلها`,
      goneBody: 'لم يعد Hermes قادرًا على استدعائه. يبقى الصف حتى تزيله، فلا يختفي شيء.',
      goneTitle: name => `${name} غادر الكتالوج.`,
      hideDeprecated: count =>
        count === 1
          ? 'إخفاء أداة واحدة مهملة'
          : count === 2
            ? 'إخفاء أداتين مهملتين'
            : count <= 10
              ? `إخفاء ${count} أدوات مهملة`
              : `إخفاء ${count} أداة مهملة`,
      hideDetails: tool => `إخفاء ما تفعله ${tool}`,
      loading: 'جارٍ قراءة قائمة الأدوات',
      lockedHint: 'معطّل من مؤسستك',
      moreHints: count => `+${count}`,
      needsAuthBody: 'يبقى تسجيل الدخول على هذا الحاسوب. لا شيء يغادره.',
      needsAuthTitle: name => `سجّل الدخول إلى ${name} لقراءة أدواته.`,
      noMatch: 'لا توجد أداة مطابقة لهذه المرشحات.',
      notInstalledBody: 'ثبّته على هذا الجهاز لرؤية الأدوات التي يضيفها.',
      offBody: 'فعّله بالمفتاح أعلاه لقراءة الأدوات التي يضيفها.',
      offTitle: name => `${name} متوقف.`,
      quickEverythingOn: 'تفعيل الكل',
      quickNoDestructive: 'إيقاف الأدوات التدميرية',
      quickReadOnly: 'قراءة فقط',
      remove: 'إزالة',
      retry: 'إعادة المحاولة',
      save: 'حفظ التغييرات',
      saveFailed: 'لم تُحفظ قواعد الأدوات تلك.',
      saving: 'جارٍ الحفظ...',
      searchCountPlaceholder: count =>
        count === 1
          ? 'ابحث في أداة واحدة'
          : count === 2
            ? 'ابحث في أداتين'
            : count <= 10
              ? `ابحث في ${count} أدوات`
              : `ابحث في ${count} أداة`,
      showAllTools: count =>
        count === 1
          ? 'عرض الأداة الوحيدة'
          : count === 2
            ? 'عرض الأداتين'
            : count <= 10
              ? `عرض كل ${count} أدوات`
              : `عرض كل ${count} أداة`,
      showDeprecated: count =>
        count === 1
          ? 'عرض أداة واحدة مهملة'
          : count === 2
            ? 'عرض أداتين مهملتين'
            : count <= 10
              ? `عرض ${count} أدوات مهملة`
              : `عرض ${count} أداة مهملة`,
      showDetails: tool => `عرض ما تفعله ${tool}`,
      showSummary: 'عرض الملخص',
      signedOutBody: 'خوادمك على هذا الحاسوب لم تتأثر.',
      signedOutTitle: 'سجّل الدخول إلى Nous لقراءة قائمة الأدوات.',
      staleSignIn: 'سجّل الدخول لقراءة أحدث قائمة أدوات.',
      summaryAllOn: 'الكل مفعّل',
      summaryAllTools: 'كل الأدوات',
      summaryCount: count =>
        count === 1 ? 'أداة واحدة' : count === 2 ? 'أداتان' : count <= 10 ? `${count} أدوات` : `${count} أداة`,
      summaryOff: 'متوقف',
      summaryOther: 'أخرى',
      summaryPreviewTitle: name => `ما يمكن لـ Hermes فعله بـ ${name} بعد اتصالك`,
      summarySomeOn: (on, total) => `${on} من ${total} مفعّلة`,
      summaryTitle: name => `ما يمكن لـ Hermes فعله بـ ${name}`,
      title: 'الأدوات',
      toolList: name => `أدوات ${name}`,
      turnToolOff: tool => `إيقاف ${tool}`,
      turnToolOn: tool => `تفعيل ${tool}`,
      unavailableLine: 'قائمة الأدوات غير متاحة.'
    },
    uncategorised: 'غير مصنّف',
    vocabulary: {
      facetDestructive: {
        label: 'تدميري',
        long: 'يمكنه حذف شيء في هذا التطبيق نهائيًا.'
      },
      facetRead: {
        label: 'قراءة',
        long: 'يقرأ بيانات من هذا التطبيق. لا يغيّر شيئًا.'
      },
      facetUnclassified: {
        label: 'تأثير غير معروف',
        long: 'لم يذكر التطبيق ما تفعله هذه الأداة.'
      },
      facetWrite: {
        label: 'كتابة',
        long: 'ينشئ أو يغيّر شيئًا في هذا التطبيق.'
      },
      hintCreate: {
        label: 'ينشئ',
        long: 'يصنع شيئًا جديدًا.'
      },
      hintDelete: {
        label: 'يحذف',
        long: 'يزيل شيئًا.'
      },
      hintDestructive: {
        label: 'تدميري',
        long: 'لا يمكن التراجع عن التغيير الذي يُحدثه هنا.'
      },
      hintIdempotent: {
        label: 'قابل للتكرار',
        long: 'تشغيله مرتين يفعل ما يفعله تشغيله مرة واحدة.'
      },
      hintOpenWorld: {
        label: 'خارجي',
        long: 'يصل إلى شيء خارج هذا التطبيق.'
      },
      hintReadOnly: {
        label: 'قراءة فقط',
        long: 'تعلن الأداة أنها تقرأ فقط.'
      },
      hintUpdate: {
        label: 'يحدّث',
        long: 'يغيّر شيئًا موجودًا بالفعل.'
      }
    }
  },
  cron: {
    blueprints: {
      custom: 'مخصص',
      dialogDesc: 'املأ التفاصيل وجدولها.',
      emptyDesc: 'لا توجد مخططات أتمتة متاحة على هذا الخادم الخلفي.',
      emptyTitle: 'لا توجد مخططات متاحة',
      failedLoad: 'تعذّر تحميل المخططات',
      loading: 'جارٍ تحميل المخططات…',
      scheduleIt: 'جدولة',
      scheduled: 'تمت جدولة المخطط',
      scheduling: 'جارٍ الجدولة…',
      startFrom: 'ابدأ من',
      subtitle: 'أتمتة جاهزة',
      tab: 'المخططات'
    },
    count: count =>
      count === 1 ? 'مهمة واحدة' : count === 2 ? 'مهمتان' : count <= 10 ? `${count} مهام` : `${count} مهمة`,
    deliverNeedsHomeChannel: 'عيّن قناة رئيسية أولًا',
    deliveryLabels: {
      discord: 'Discord',
      slack: 'Slack',
      telegram: 'Telegram'
    },
    editJob: 'تعديل المهمة',
    lastRunFailed: 'فشل التشغيل الأخير:',
    modelDefault: 'الافتراضي (النموذج العام)',
    modelLabel: 'النموذج',
    overdueSince: 'متأخرة منذ:',
    promptRequired: 'الموجّه مطلوب.',
    runAgain: 'تشغيل مرة أخرى',
    scheduleRequired: 'الجدولة مطلوبة.',
    scriptOnlyEditHint: 'مهمة نصية فقط (دون موجّه AI). معرّف المهمة:',
    tabs: {
      blueprints: 'المخططات',
      jobs: 'المهام'
    },
    title: 'المهام المجدولة'
  },
  desktop: {
    handoff: {
      startMessaging: 'بدء المراسلة'
    }
  },
  externalOpenFailed: {
    missing: {
      message: 'هذا الملف غير موجود — ربما حُذف أو نُقل، أو أنه على جهاز آخر.',
      title: 'الملف غير موجود'
    }
  },
  fileMenu: {
    revealUnavailable: 'هذا المسار ليس على هذا الحاسوب — إنه على جهاز الخادم الخلفي. استخدم «إظهار في شجرة الملفات».'
  },
  findInPage: {
    next: 'التطابق التالي',
    previous: 'التطابق السابق'
  },
  freeTier: {
    alreadySignedInBody: 'هذه النسخة من Hermes مسجّلة الدخول بالفعل إلى حساب Nous.',
    alreadySignedInHeading: 'تم تسجيل الدخول بالفعل.',
    begin: 'ابدأ',
    busyBody: wait =>
      `تعذّر على Hermes إكمال تسجيل دخولك لأن خدمة Nous مشغولة. حاول مرة أخرى خلال ${wait}. جلستك لا تزال محفوظة في الأثناء.`,
    busyHeading: 'اقتربنا',
    change: 'تغيير',
    completedBody: 'أصبح حسابك الآن يتضمن الاستدلال والأدوات.',
    defaultModel: 'النموذج الافتراضي',
    didNotComplete: 'لم يكتمل تسجيل الدخول',
    dismiss: 'تجاهل',
    doNotShare: 'لا تشاركه.',
    done: 'تم',
    errorBody: 'لم يكتمل تسجيل الدخول. حاول مرة أخرى متى شئت.',
    finishInBrowser: 'أكمل في متصفحك',
    finishingBody: 'تمت الموافقة في المتصفح. جارٍ استلام Tokens حسابك.',
    finishingHeading: 'جارٍ إكمال تسجيل الدخول…',
    notNow: 'ليس الآن',
    openConnectors: 'فتح الموصلات',
    openModelPicker: 'فتح منتقي النموذج',
    openPageAgain: 'فتح الصفحة مرة أخرى',
    otherProviders: 'مزوّدون آخرون',
    providerName: 'Nous',
    providerRowPitch: 'سجّل الدخول بحساب Nous لفتح مزيد من النماذج والأدوات.',
    providerRowTitle: 'Nous · الخطة المجانية',
    readyCaption: 'مجاني · الموصلات مشمولة',
    readyTitle: 'Hermes جاهز.',
    reconnect: 'إعادة الاتصال',
    settingUp: 'جارٍ إعداد الاستدلال المجاني…',
    setupFailed: {
      gateClosed:
        'لا يمكن لهذه النسخة من Hermes البدء دون حساب Nous. سجّل الدخول أو أنشئ حسابًا، فهو مجاني ولا يستغرق سوى دقيقة.',
      generic: 'تعذّر على Hermes إعداد الوصول المجاني دون تسجيل الدخول. تسجيل الدخول مجاني، أو يمكنك توصيل مزوّد آخر.',
      locked: 'لا يمكن متابعة هذه الجلسة دون تسجيل الدخول. سجّل الدخول أو أنشئ حساب Nous مجانيًا للمتابعة.',
      paused:
        'استخدام Hermes دون تسجيل الدخول متوقف مؤقتًا. سيواصل Hermes التحقق. تسجيل الدخول مجاني ويتيح لك البدء الآن.',
      powRequired:
        'طلب خادم Nous إثبات عمل، لكن ذلك غير مدمج في Agent لديك بعد. سجّل الدخول أو أنشئ حساب Nous مجانيًا للمتابعة.',
      rateLimited: wait =>
        `هناك إقبال كبير على البدء الآن، لذا سيعيد Hermes المحاولة خلال ${wait}. تسجيل الدخول مجاني ويختصر الانتظار.`,
      retrying: 'جارٍ إعادة المحاولة…',
      serverError: 'حدث خلل في خدمة Nous. اضغط «إعادة المحاولة» بعد لحظات، أو وصّل مزوّدًا آخر مؤقتًا.',
      signInBelow: 'تسجيل الدخول مجاني. اختر Nous أدناه.',
      tryAgain: 'إعادة المحاولة',
      unreachable:
        'تعذّر على Hermes الوصول إلى خدمة Nous. تحقق من اتصال الإنترنت ثم اضغط «إعادة المحاولة». أو وصّل مزوّدًا آخر مؤقتًا.'
    },
    signIn: 'تسجيل الدخول',
    signInHeading: 'سجّل الدخول بحساب Nous لفتح مزيد من النماذج والأدوات.',
    signInInstead: 'تسجيل الدخول بحساب Nous بدلًا من ذلك',
    signedIn: 'تم تسجيل الدخول.',
    signedInAsLabel: 'مسجّل الدخول باسم',
    statusLabel: model => `Nous · ${model}`,
    stripBody: 'افتح منتقي النموذج لتجربتها، أو سجّل الدخول بحساب Nous.',
    stripTitle: 'أصبح الاستدلال والموصلات المجانية من Nous متاحة الآن.',
    timedOutBody: 'ابدأ من جديد متى شئت. ما زلت على خدمة Nous المجانية.',
    timedOutHeading: 'انتهت صلاحية رابط تسجيل الدخول',
    tryAgain: 'إعادة المحاولة',
    unreachableBody:
      'تعذّر على Hermes الوصول إلى خدمة Nous لإكمال تسجيل دخولك. تحقق من اتصال الإنترنت وحاول مرة أخرى. جلستك لا تزال محفوظة.',
    waiting: 'بانتظار تسجيل الدخول…'
  },
  install: {
    applyRemote: 'تطبيق وإعادة الاتصال',
    authNeedsOauth: provider => `سجّل الدخول عبر ${provider} قبل اختبار هذه البوابة.`,
    authSignedIn: 'اكتمل تسجيل الدخول عبر المتصفح.',
    authTitle: 'المصادقة',
    backToSetup: 'رجوع',
    bundledLocalDesc: 'استخدم بيئة تشغيل Hermes المضمّنة في هذا التطبيق — الخادم الخلفي المضمّن هو التثبيت المحلي.',
    connectExistingDesc: 'استخدم خادمًا خلفيًا بعيدًا مع رمز جلسة أو تسجيل دخول عبر المتصفح. لن يبدأ أي تثبيت محلي.',
    connectExistingShort: 'الاتصال بموجود',
    connectExistingTitle: 'الاتصال بـ Hermes موجود',
    connected: 'متصل',
    enterUrlFirst: 'أدخل عنوان URL للبوابة أولًا.',
    identityProvider: 'مزوّد الهوية لديك',
    incompleteSignInTest: 'سجّل الدخول قبل اختبار هذه البوابة المحمية بـ OAuth.',
    incompleteTokenTest: 'أدخل رمز جلسة قبل اختبار هذه البوابة.',
    installLocalDesc: 'نزّل Hermes وأنشئ بيئة Python الخاصة به وشغّل الخادم الخلفي على هذا الحاسوب.',
    installLocalTitle: 'تثبيت Hermes محليًا',
    localStartUnavailable: 'تعذّر بدء التثبيت المحلي. أعد تشغيل Hermes Desktop وحاول مرة أخرى.',
    openLogs: 'فتح السجلات',
    pasteSessionToken: 'لصق رمز الجلسة',
    probeError:
      'تعذّر على Hermes الوصول إلى ذلك العنوان. تحقق من URL ومن تشغيل Hermes على الحاسوب الآخر — تظهر خيارات تسجيل الدخول بمجرد استجابته.',
    probeErrorDetails: 'التفاصيل',
    probing: 'جارٍ اكتشاف مصادقة البوابة…',
    remoteSetupDesc: 'أدخل عنوان URL للبوابة. سيكتشف Hermes Desktop ما إذا كانت تحتاج رمزًا أو تسجيل دخول عبر المتصفح.',
    remoteSetupTitle: 'الاتصال بـ Hermes موجود',
    remoteUrlDesc: 'استخدم عنوان URL الأساسي لبوابة Hermes، مع تضمين https:// عند الاتصال عن بُعد.',
    remoteUrlPlaceholder: 'https://gateway.example.com/hermes',
    remoteUrlTitle: 'عنوان URL للبوابة',
    setupChoiceDesc: 'وصّل هذا التطبيق ببوابة Hermes تشغّلها بالفعل، أو ثبّت Hermes محليًا على هذا الحاسوب.',
    setupChoiceDescLocal: 'ثبّت Hermes على هذا الحاسوب، أو اتصل ببوابة Hermes تشغّلها بالفعل.',
    setupChoiceTitle: 'إعداد Hermes Desktop',
    signIn: 'تسجيل الدخول',
    signInIncomplete: 'أُغلقت نافذة تسجيل الدخول قبل اكتمال المصادقة.',
    signInWith: provider => `تسجيل الدخول عبر ${provider}`,
    testConnection: 'اختبار الاتصال',
    testSucceeded: (baseUrl, version) => `تم الاتصال بـ ${baseUrl}${version ? ` (${version})` : ''}.`,
    tokenDesc: 'الصق رمز الجلسة من ملف .env للبوابة البعيدة.',
    tokenTitle: 'رمز الجلسة',
    useLocalDesc: 'بيئة تشغيل Hermes مثبتة هنا بالفعل — شغّلها بنقرة واحدة. لن يُنزّل أي شيء.',
    useLocalTitle: 'استخدام Hermes على هذا الحاسوب'
  },
  intro: {
    custom: () => []
  },
  keybinds: {
    actions: {
      'composer.queue': 'إدراج الرسالة في الانتظار',
      'hud.snapToPointer': 'نقل HUD إلى المؤشر (عام، أثناء فتح HUD)',
      'session.archive': 'أرشفة الجلسة الحالية',
      'view.closeTerminal': 'إغلاق Terminal',
      'view.cycleSidebarGrouping': 'تبديل تجميع الجلسات',
      'view.findInPage': 'بحث في الصفحة',
      'view.findNext': 'البحث عن التطابق التالي',
      'view.findPrevious': 'البحث عن التطابق السابق',
      'view.newTerminal': 'Terminal جديد',
      'view.nextTerminal': 'Terminal التالي',
      'view.prevTerminal': 'Terminal السابق',
      'view.tabSlot.1': 'التبديل إلى علامة التبويب 1',
      'view.tabSlot.2': 'التبديل إلى علامة التبويب 2',
      'view.tabSlot.3': 'التبديل إلى علامة التبويب 3',
      'view.tabSlot.4': 'التبديل إلى علامة التبويب 4',
      'view.tabSlot.5': 'التبديل إلى علامة التبويب 5',
      'view.tabSlot.6': 'التبديل إلى علامة التبويب 6',
      'view.tabSlot.7': 'التبديل إلى علامة التبويب 7',
      'view.tabSlot.8': 'التبديل إلى علامة التبويب 8',
      'view.tabSlot.9': 'التبديل إلى علامة التبويب 9',
      'view.toggleHud': 'تبديل وضع HUD',
      'view.toggleProfileRail': 'تبديل شريط الملفات الشخصية',
      'view.toggleSimpleMode': 'تبديل الوضع المبسّط',
      'view.toggleStatusbar': 'تبديل شريط الحالة',
      'view.toggleTabStrip': 'تبديل علامات التبويب'
    },
    clear: 'مسح'
  },
  messaging: {
    approve: 'موافقة',
    approvedHint: 'سيتم التعرف عليهم تلقائيًا في رسالتهم التالية.',
    approvedUser: name => `تمت الموافقة على ${name}`,
    approvedUsers: count => `المستخدمون الموافق عليهم (${count})`,
    approving: 'جارٍ الموافقة…',
    failedApprove: name => `تعذّرت الموافقة على ${name}`,
    failedRevoke: name => `تعذّر إلغاء وصول ${name}`,
    fieldCopy: {
      MATRIX_HOMESERVER: {
        placeholder: 'https://matrix.org'
      },
      MATRIX_USER_ID: {
        placeholder: '@hermes:example.org'
      },
      MATTERMOST_URL: {
        placeholder: 'https://mattermost.example.com'
      },
      SIGNAL_HTTP_URL: {
        placeholder: 'http://127.0.0.1:8080'
      }
    },
    openLogs: 'فتح السجلات',
    pairingLockedOut: 'محاولات موافقة فاشلة كثيرة — تم قفل هذه المنصة. حاول لاحقًا.',
    pendingAria: count =>
      count === 1
        ? 'طلب اقتران واحد معلّق'
        : count === 2
          ? 'طلبا اقتران معلّقان'
          : count <= 10
            ? `${count} طلبات اقتران معلّقة`
            : `${count} طلب اقتران معلّق`,
    pendingRequests: count => `الطلبات المعلّقة (${count})`,
    relayLink: {
      title: 'بوابة الفريق'
    },
    restartAgain: 'إعادة التشغيل مرة أخرى',
    restartFailedManualDetail: 'جرّب «إعادة التشغيل مرة أخرى»؛ وإذا استمر الفشل، افتح السجلات وأرسل بيانات التشخيص.',
    revoke: 'إلغاء الوصول',
    revokeAria: name => `إلغاء وصول ${name}`,
    revokeDesc: name => `سيفقد ${name} الوصول ولن يتم التعرف عليه في رسالته التالية.`,
    revokeTitle: 'إلغاء الوصول',
    revokedUser: name => `تم إلغاء وصول ${name}`,
    revoking: 'جارٍ إلغاء الوصول…',
    waitingSince: minutes => (minutes < 1 ? 'الآن' : `قبل ${minutes} د`)
  },
  modelPicker: {
    downloading: 'جارٍ التنزيل',
    loadingIntoMemory: 'جارٍ التحميل إلى الذاكرة',
    localDownloadsHeading: 'محلي',
    pro: 'Pro',
    wasPrice: 'كان'
  },
  notifications: {
    actions: {
      openGateways: 'فتح البوابات',
      openKeys: 'فتح المفاتيح',
      openMaintenance: 'فتح الصيانة',
      restartHermes: 'إعادة تشغيل Hermes'
    },
    compressDeferredDone: 'اكتمل ضغط السياق',
    errors: {
      gatewayAuthFailed:
        'لم تعد هذه النسخة من Hermes تقبل تسجيل دخولك المحفوظ. افتح البوابات وسجّل الدخول مجددًا (أو الصق رمز وصول جديدًا)، ثم أعد المحاولة.',
      restartHermesFailed: 'تعذّرت إعادة تشغيل Hermes',
      rpcOutOfSync: 'التطبيق والخادم الخلفي بإصدارين مختلفين. حدّث كليهما.',
      storageFailure: 'تعذّر على Hermes الحفظ في مجلد بياناته. افتح الصيانة للتحقق والإصلاح.'
    },
    installMethodUnsupportedTitle: 'طريقة تثبيت غير مدعومة',
    native: {
      creditsTitle: 'الأرصدة'
    },
    updateReadyMessageAppInstaller: 'إصدار جديد من Hermes جاهز. حدّث الآن وسيكمل Windows العملية نيابة عنك.',
    voice: {
      liveDelegationFailed: 'تعذّر تسليم الطلب إلى Hermes',
      liveEnded: 'انتهت جلسة الصوت المباشر',
      liveEndedClosed: 'أُغلقت جلسة الصوت المباشر من الخدمة.',
      liveEndedConnectionLost: 'فقدت جلسة الصوت المباشر اتصالها.',
      liveError: 'الصوت المباشر',
      liveUnavailable: reason => `دردشة GPT-Live الصوتية غير متاحة: ${reason}. سيُستخدم تحويل الكلام إلى نص بدلًا منها.`
    }
  },
  onboarding: {
    apiKeyOptions: {
      fireworks: {
        description: 'وصول مباشر إلى النماذج المستضافة لدى Fireworks AI.',
        short: 'API مباشر للنماذج'
      }
    },
    errorDetails: 'التفاصيل',
    localModelNamePlaceholder: 'اسم النموذج (مثل command-a-plus-05-2026)',
    localModelsPitch: 'لا حاجة إلى حساب — نزّل نموذجًا وشغّله على هذا الجهاز',
    localModelsTitle: 'تشغيل النماذج محليًا',
    signInDidNotFinish: provider =>
      `لم يكتمل تسجيل الدخول عبر ${provider}. تحقق من اتصال الإنترنت وحاول مرة أخرى، أو اختر مزوّدًا آخر.`,
    tryAgain: 'إعادة المحاولة',
    useApiKeyInstead: 'استخدام مفتاح API'
  },
  preview: {
    missingTarget: 'هذا المسار غير موجود على هذا الحاسوب',
    web: {
      addComments: count =>
        count === 1
          ? 'إضافة تعليق واحد'
          : count === 2
            ? 'إضافة تعليقين'
            : count <= 10
              ? `إضافة ${count} تعليقات`
              : `إضافة ${count} تعليقًا`,
      annotate: 'تعليق توضيحي',
      annotateFailed: 'تعذّر بدء وضع التعليقات التوضيحية',
      annotateNeedPage: 'افتح صفحة في المتصفح داخل التطبيق أولًا.',
      annotateOn: 'إيقاف التعليقات التوضيحية',
      cancelComment: 'إلغاء التعليق',
      commentPlaceholder: 'أضف تعليقًا…',
      commentTitle: n => `تعليق ${n}`,
      commenting: 'جارٍ التعليق',
      saveComment: 'حفظ'
    }
  },
  profiles: {
    connectGateway: 'إدارة البوابات…',
    displayNameDesc: 'يحدد اسم عرض يظهر في أنحاء التطبيق. يبقى معرّف الملف الشخصي الداخلي «default».',
    displayNameLabel: 'اسم العرض',
    displayNameTitle: 'سمِّ هذا الوكيل',
    editSoul: 'تعديل SOUL.md…',
    fleet: {
      allOnGateway: 'كل الملفات الشخصية على هذه البوابة',
      deleteOn: gateway => ` على ${gateway}`,
      gateway: gateway => `الملفات الشخصية على ${gateway}`,
      gatewayUnreachable: gateway => `${gateway} · غير قابل للوصول`,
      onGateway: (name, gateway) => `${name} · ${gateway}`,
      switchTo: (name, gateway) => `فتح ${name} على ${gateway}`
    },
    remoteOverride: {
      urlPlaceholder: 'https://hermes.example.com'
    },
    renameDescSuffix: '.',
    renameMenu: 'إعادة تسمية…',
    search: 'بحث في الملفات الشخصية…',
    status: {
      needsInput: count =>
        count === 1
          ? 'جلسة واحدة تحتاج إجابتك'
          : count === 2
            ? 'جلستان تحتاجان إجابتك'
            : count <= 10
              ? `${count} جلسات تحتاج إجابتك`
              : `${count} جلسة تحتاج إجابتك`,
      unread: count =>
        count === 1
          ? 'جلسة واحدة غير مقروءة'
          : count === 2
            ? 'جلستان غير مقروءتين'
            : count <= 10
              ? `${count} جلسات غير مقروءة`
              : `${count} جلسة غير مقروءة`,
      working: count =>
        count === 1
          ? 'جلسة واحدة قيد التشغيل'
          : count === 2
            ? 'جلستان قيد التشغيل'
            : count <= 10
              ? `${count} جلسات قيد التشغيل`
              : `${count} جلسة قيد التشغيل`
    }
  },
  promptNotices: {
    desktopCommands: {
      historyHermes: 'Hermes'
    }
  },
  prompts: {
    reconnect: 'إعادة الاتصال',
    vaultSaveIdentifierPlaceholder: 'you@example.com'
  },
  rightSidebar: {
    folderTip: cwd => cwd
  },
  sendDiagnostics: {
    links: {
      discord: 'Discord',
      github: 'GitHub Issues'
    }
  },
  settings: arBatch2Settings,
  shell: {
    modelOptions: {
      daybreak: 'Daybreak'
    },
    statusbar: {
      commit: sha => `commit ${sha}`
    }
  },
  sidebar: {
    filter: {
      pr: 'PR'
    }
  },
  skillDeepLink: {
    destinationChanged: 'تغيّرت الوجهة. أغلق هذا الحوار وافتح رابط التثبيت مرة أخرى.',
    installComplete: name => `تم تثبيت «${name}»`,
    installDescription: 'ستكون هذه المهارة متاحة في الجلسات الجديدة. ثبّت فقط من مصادر تثق بها.',
    installTitle: name => `هل تريد تثبيت «${name}»؟`,
    installTo: 'التثبيت في',
    installed: 'مثبَّت',
    installing: 'جارٍ التثبيت…',
    source: 'المصدر',
    thisComputer: 'هذا الحاسوب'
  },
  skills: {
    archive: 'أرشفة',
    bulkNoChange: 'لا يوجد ما يُغيَّر.',
    bulkUpdated: count =>
      count === 1
        ? 'حُدِّث عنصر واحد للجلسات الجديدة.'
        : count === 2
          ? 'حُدِّث عنصران للجلسات الجديدة.'
          : count <= 10
            ? `حُدِّثت ${count} عناصر للجلسات الجديدة.`
            : `حُدِّث ${count} عنصرًا للجلسات الجديدة.`,
    changesApplyNewSessions: 'تُطبَّق التغييرات على الجلسات الجديدة.',
    configuringProfile: 'جارٍ الإعداد:',
    disableAll: 'تعطيل الكل',
    disableUnused: 'تعطيل غير المستخدمة',
    edit: 'تعديل',
    emptyNoneAvailable: noun => `لا توجد ${noun} متاحة بعد.`,
    emptyNoneFound: noun => `لم يُعثر على ${noun}`,
    emptyNothingMatches: query => `لا شيء يطابق «${query}».`,
    enableAll: 'تفعيل الكل',
    hub: {
      actionFailed: 'فشل إجراء المهارة',
      actionLog: 'سجل الإجراءات',
      alreadyInstalled: name => `«${name}» مثبَّتة بالفعل`,
      close: 'إغلاق',
      connectedHubs: 'المراكز المتصلة:',
      connectingHubs: 'جارٍ الاتصال بمراكز المهارات...',
      featured: 'مهارات مميزة',
      files: 'الملفات',
      findings: count =>
        count === 1 ? 'ملاحظة واحدة' : count === 2 ? 'ملاحظتان' : count <= 10 ? `${count} ملاحظات` : `${count} ملاحظة`,
      install: 'تثبيت',
      installBlockedMessage: (findings, unverified) =>
        `أشار الفحص الأمني إلى ${findings > 0 ? (findings === 1 ? 'عنصر واحد' : findings === 2 ? 'عنصرين' : findings <= 10 ? findings + ' عناصر' : findings + ' عنصرًا') : 'أنماط خطرة'} للمراجعة${unverified ? '، كما أن المهارة تأتي من مصدر غير موثّق' : ''}. اقرأ نتائج الفحص قبل أن تقرر مدى الثقة بالمؤلف.`,
      installBlockedTitle: name => `تعذّر تثبيت ${name}`,
      installStarted: name => `جارٍ تثبيت ${name}...`,
      installed: 'مثبَّت',
      installing: 'جارٍ التثبيت...',
      landingHint: 'ابحث في المركز لتصفح المهارات القابلة للتثبيت من الفهرس الرسمي وGitHub والمصادر المجتمعية.',
      loadFailed: 'تعذّر تحميل مركز المهارات',
      noFindings: 'لا توجد ملاحظات أمنية.',
      noReadme: 'لا تتوفر معاينة SKILL.md لهذه المهارة.',
      noResults: 'لم يُعثر على مهارات مطابقة في المركز.',
      openLog: 'فتح السجل',
      pickerBrowse: 'تصفح المركز بالكامل',
      pickerHide: 'إخفاء متصفح المركز',
      pickerHint: 'اضغط «+ إضافة إلى هذا الوكيل» على أي مهارة — ستُثبَّت وتظهر في القائمة أعلاه.',
      pickerTitle: 'مركز المهارات',
      policyAllow: 'التثبيت مسموح',
      policyAsk: 'المراجعة قبل التثبيت',
      policyBlock: 'التثبيت محظور بالسياسة',
      preview: 'معاينة',
      previewFailed: 'فشلت معاينة المهارة',
      resultCount: (count, ms) =>
        `${count === 1 ? 'نتيجة واحدة' : count === 2 ? 'نتيجتان' : count <= 10 ? count + ' نتائج' : count + ' نتيجة'}${ms !== null ? ' في ' + ms + 'ms' : ''}`,
      scan: 'فحص',
      scanFailed: 'فشل الفحص الأمني',
      scanning: 'جارٍ الفحص...',
      search: 'بحث',
      searchFailed: 'فشل البحث في المركز',
      searchPlaceholder: 'البحث في مركز المهارات',
      searching: 'جارٍ البحث...',
      timedOut: sources => `انتهت المهلة: ${sources}`,
      trust: {
        builtin: 'مدمج',
        community: 'مجتمعي',
        trusted: 'موثوق'
      },
      uninstall: 'إزالة التثبيت',
      uninstallStarted: name => `جارٍ إزالة تثبيت ${name}...`,
      uninstalling: 'جارٍ إزالة التثبيت...',
      updateAll: 'تحديث المثبَّتة',
      updateStarted: 'جارٍ تحديث المهارات المثبَّتة...',
      updating: 'جارٍ التحديث...',
      verdictCaution: 'تنبيه',
      verdictDangerous: 'خطير',
      verdictSafe: 'آمن',
      viewScan: 'عرض الفحص'
    },
    officialCatalog: 'متاحة للتثبيت',
    officialPill: 'رسمي',
    plugins: {
      agentBlurb: 'وسّع الوكيل لملف التعريف المحدد — الأدوات والخطافات والمزوّدون. يُطبَّق بعد إعادة تشغيل البوابة.',
      agentTitle: 'إضافات الوكيل',
      alreadyInstalled: name => `${name} مثبَّت بالفعل في ملف التعريف هذا.`,
      catalogProvenance: sha => `ثُبِّت من كتالوج Hermes${sha ? ' عند التثبيت ' + sha : ''}.`,
      deepLinkCatalogInvalidName: 'اسم الكتالوج في الرابط مفقود أو غير صالح.',
      deepLinkCatalogUnavailable: 'تعذّر تحميل كتالوج إضافات Hermes. تحقق من اتصالك وافتح الرابط مرة أخرى.',
      deepLinkCatalogUnknown: name => `«${name}» غير موجود في كتالوج إضافات Hermes. لم يُثبَّت أي شيء.`,
      deepLinkErrorTitle: 'تم رفض رابط تثبيت الإضافة',
      defaultProfile: 'Hermes (الافتراضي)',
      desktopHalfPending: 'جارٍ النسخ…',
      desktopHalfPendingTip:
        'تتضمن هذه الحزمة جزءًا لسطح المكتب لم يُنسخ بعد إلى التطبيق. استخدم إعادة الفحص، أو أعد تشغيل التطبيق.',
      desktopHalfRemote: 'غير متاح (واجهة خلفية بعيدة)',
      desktopHalfRemoteTip:
        'الجزء الخاص بسطح المكتب في هذه الحزمة موجود على قرص الواجهة الخلفية البعيدة، ولا يستطيع هذا التطبيق قراءته. لاستخدامه هنا، شغّل «تثبيت من Git» باستخدام رابط مستودع الحزمة مع تحديد هدف سطح المكتب — إذ يستنسخ ذلك الجزء الخاص بسطح المكتب على هذا الجهاز.',
      empty: 'لا توجد إضافات وكيل مثبَّتة لملف التعريف هذا.',
      emptyAll: 'لا توجد إضافات بعد.',
      emptyHint: 'تصفح الكتالوج أدناه وثبّت إضافة مراجَعة بنقرة واحدة.',
      halfAgent: 'الوكيل',
      halfAgentIn: profile => `الوكيل في ${profile}`,
      halfDesktop: 'سطح المكتب',
      halfDesktopHint: 'هذا التطبيق، الأمر نفسه لكل ملف تعريف',
      installAgentHere: 'التثبيت هنا',
      installAgentHereNoOrigin:
        'الجزء الخاص بالوكيل غير مثبَّت في ملف التعريف هذا، وقد نُسخت هذه الحزمة يدويًا (لا يوجد مدخل في الكتالوج ولا مستودع Git بعيد)، لذا لا يمكن تثبيتها من هنا. انسخ مجلدها إلى ملف التعريف أو أعد التثبيت من Git.',
      installAgentHereTip: profile =>
        `الجزء الخاص بسطح المكتب محمَّل في هذا التطبيق، لكن الجزء الخاص بالوكيل غير مثبَّت في ${profile}. ثبّته هناك.`,
      kindAgent: 'الوكيل',
      kindBoth: 'الوكيل + سطح المكتب',
      kindDesktop: 'سطح المكتب',
      legacyBackend:
        'هذه الواجهة الخلفية أقدم من مفاتيح تبديل الإضافات المعتمدة على المفاتيح — حدّث Hermes لإدارتها من هنا.',
      loadFailed: 'تعذّر تحميل إضافات الوكيل',
      pinnedBadge: sha => `مثبَّتة عند ${sha}`,
      pinnedProvenance: sha => `مثبَّتة عند commit ${sha}. تُرفض التحديثات حتى يُعاد تثبيتها بمرجع جديد.`,
      portableBadge: 'محمولة',
      serverStates: {
        app_not_running: 'التطبيق لا يعمل',
        connected: 'متصل',
        endpoint_unavailable: 'نقطة النهاية غير متاحة',
        hermes_not_connected: 'اتصال MCP مفقود',
        missing_app: 'التطبيق مفقود',
        no_interactive_session: 'لا توجد جلسة تفاعلية',
        unknown: 'الحالة غير معروفة',
        unsupported_gpu: 'GPU غير مدعوم',
        version_too_old: 'الإصدار قديم جدًا'
      },
      settingsForm: {
        optional: '(اختياري)',
        required: 'مطلوب',
        save: 'حفظ الإعدادات',
        saveFailed: name => `تعذّر حفظ إعدادات ${name}`,
        saved: name => `تم حفظ إعدادات ${name}.`,
        secretSet: '•••••••• (محدد)',
        secretStoredAs: env =>
          `يُخزَّن في ملف .env الخاص بملف التعريف باسم ${env}، ولا يُخزَّن في config.yaml أبدًا؛ اتركه فارغًا للإبقاء على القيمة الحالية.`
      },
      settingsToggle: name => `الإعدادات: ${name}`,
      toggleFailed: name => `تعذّر تبديل ${name}`,
      toolsetOff: (name, profile) => `أدوات وكيل ${name} معطّلة لـ${profile}`,
      toolsetOn: (name, profile) => `أدوات وكيل ${name} مفعّلة لـ${profile}`,
      toolsetToggleFailed: name => `تعذّر تبديل أدوات وكيل ${name}؛ لم تتغير لوحة سطح المكتب`,
      uninstall: 'إزالة التثبيت',
      uninstallConfirmBody: (name, profile) =>
        `سيحذف هذا ملفات الإضافة من ملف التعريف ${profile}. وسيُزال معها أي جزء خاص بسطح المكتب كانت تتضمنه. يمكنك إعادة تثبيتها من الكتالوج أو من Git في أي وقت.`,
      uninstallConfirmTitle: name => `هل تريد إزالة تثبيت ${name}؟`,
      uninstallDesktopConfirmBody: name =>
        `سيحذف هذا ${name} من مجلد desktop-plugins على هذا الحاسوب ويُفرغ تحميله الآن. يمكنك إعادة تثبيته من Git أو إعادة المجلد في أي وقت.`,
      uninstallDesktopTip: name => `إزالة تثبيت ${name} من هذا التطبيق`,
      uninstallFailed: name => `تعذّرت إزالة تثبيت ${name}`,
      uninstallTip: (name, profile) => `إزالة تثبيت ${name} من ${profile}`,
      uninstalled: name => `تمت إزالة تثبيت ${name}. أعد تشغيل البوابة لإلغاء تحميلها.`,
      uninstalledDesktop: name => `تمت إزالة تثبيت ${name}.`,
      updateConsentBody: (name, sha) =>
        `يضيف تثبيت الكتالوج الجديد لـ${name} (${sha}) واجهات لا تمتلكها النسخة المثبَّتة. طبّقه فقط إذا كنت تثق بها:`,
      updateConsentConfirm: 'تطبيق التحديث',
      updateConsentTitle: name => `${name} يطلب المزيد من الصلاحيات`,
      updateFailed: name => `تعذّر تحديث ${name}`,
      updateScanBlocked: name => `حجب الفحص الأمني تحديث ${name} (حكم: خطير). لا يمكن تجاوزه.`,
      updateScanBody: (name, count) =>
        `حصل تحديث ${name} على حكم تنبيه (${count === 1 ? 'ملاحظة واحدة' : count === 2 ? 'ملاحظتان' : count <= 10 ? count + ' ملاحظات' : count + ' ملاحظة'}). يُفحص كل تحديث. طبّقه فقط إذا كنت تثق بالمصدر:`,
      updateScanConfirm: 'التحديث على أي حال',
      updateScanMore: count => `+ ${count} أخرى`,
      updateScanTitle: name => `أشار الفحص الأمني إلى ${name}`,
      updateToPin: sha => `التحديث إلى ${sha}`,
      updated: name => `تم تحديث ${name} إلى تثبيت الكتالوج الحالي. أعد تشغيل البوابة للتطبيق.`,
      updatedGit: name => `تم تحديث ${name} إلى أحدث commit.`
    },
    provenance: {
      agent: 'مُتعلَّمة',
      bundled: 'مدمجة',
      hub: 'المركز'
    },
    skillArchivedMessage: 'يمكن استعادتها عبر hermes curator restore.',
    skillArchivedTitle: 'تمت أرشفة المهارة',
    skillUpdated: 'تم تحديث المهارة',
    sortAlpha: 'أ–ي',
    sortLeastUsedAsc: '↑ الأقل استخدامًا',
    sortMostUsed: 'الأكثر استخدامًا',
    sortMostUsedDesc: '↓ الأكثر استخدامًا',
    tabPlugins: 'الإضافات',
    usageCount: count => `استُخدمت ${count} مرة`,
    visionModelHint:
      'تستخدم الرؤية إعدادات نموذجك المساعد — يُختار النموذج القادر على معالجة الصور هناك، وليس لكل مزوّد هنا.',
    visionModelLink: 'اختر نموذج الرؤية في الإعدادات ← النماذج'
  },
  starmap: {
    close: 'إغلاق مخطط الذاكرة',
    copied: 'تم النسخ!',
    copy: 'نسخ شيفرة المخطط',
    emptyDesc: 'بينما يبني Hermes المهارات والذكريات الخاصة بعملك، تظهر هنا.',
    emptyTitle: 'لم يُتعلَّم شيء بعد',
    filterAll: 'الكل',
    filterLearned: 'مُتعلَّم',
    filterUsed: 'مستخدم',
    importBtn: 'تحميل',
    importEmpty: 'الصق شيفرة مخطط لتحميله.',
    importMap: 'استيراد مخطط',
    importSuccess: nodes =>
      `تم تحميل مخطط يضم ${nodes === 1 ? 'عقدة واحدة' : nodes === 2 ? 'عقدتين' : nodes <= 10 ? nodes + ' عقد' : nodes + ' عقدة'}.`,
    importedBadge: 'مخطط مستورد',
    loadFailed: 'تعذّر تحميل مخطط الذاكرة',
    loading: 'جارٍ التحميل…',
    memory: 'الذاكرة',
    refresh: 'تحديث',
    resetToMine: 'العودة إلى مخططي',
    share: 'مشاركة المخطط',
    shareHint: 'انسخ الشيفرة لمشاركة هذا المخطط، أو الصق شيفرة لتحميله. تتضمن التخطيط فقط، لا نص ذاكرتك أو مهاراتك.',
    sharePlaceholder: 'الصق شيفرة مخطط…',
    shareTitle: 'استيراد / تصدير المخطط',
    subtitle: (nodes, clusters) => `${nodes} مهارة في ${clusters} فئة`,
    title: 'مخطط الذاكرة',
    viewGraph: 'المخطط'
  },
  statusStack: {
    control: {
      dueWaitingForIdle: 'مستحقة — بانتظار الخمول'
    }
  },
  updates: {
    applyStatus: {
      owed: steps => `تم تحديث الواجهة الخلفية، لكن ما زال مطلوبًا: ${steps}. أعد تشغيل \`hermes update\` لإكمالها.`
    },
    applyingBodyAppInstaller:
      'سيُغلق Hermes وسيُكمل Windows التحديث. سيُعاد فتح Hermes عند الانتهاء — لا تحتاج إلى فعل أي شيء.',
    applyingCloseAppInstaller: 'ستُغلق هذه النافذة، ويُكمل Windows التحديث، ويُعاد فتح Hermes تلقائيًا.',
    availableBodyAppInstaller:
      'إصدار جديد من Hermes جاهز. سيُغلق Hermes، ويُكمل Windows التحديث، وسيُعاد فتح Hermes تلقائيًا.',
    checkUnknownBodyAppInstaller:
      'تعذّر على Windows التحقق من التحديثات الآن. تُثبَّت التحديثات أيضًا تلقائيًا عند إعادة تشغيل Hermes.',
    checkUnknownTitleAppInstaller: 'تعذّر التحقق من التحديثات',
    connectionSettings: 'إعدادات الاتصال',
    openDownloadPage: 'فتح صفحة التنزيل',
    releaseAvailable: tag => `الإصدار ${tag} متاح.`,
    versionDetailsDistributionStore: 'Microsoft Store'
  },
  webhooks: {
    all: '(الكل)',
    copy: 'نسخ',
    create: 'إنشاء',
    createFailed: detail => `فشل الإنشاء: ${detail}`,
    created: 'تم الإنشاء',
    createdSecretHint: 'انسخ السر الآن — يُعرض مرة واحدة فقط.',
    createdTitle: 'تم إنشاء الاشتراك',
    creating: 'جارٍ الإنشاء...',
    delete: 'حذف',
    deleteDescPrefix: 'سيؤدي هذا إلى إزالة ',
    deleteDescSuffix: ' نهائيًا. لا يمكن التراجع عن ذلك.',
    deleteFailed: name => `فشل حذف «${name}»`,
    deleteTitle: 'حذف Webhook',
    deleted: 'تم حذف Webhook',
    deleting: 'جارٍ الحذف...',
    deliverOnly: 'التسليم فقط',
    deliverOptions: {
      discord: 'Discord',
      email: 'البريد الإلكتروني',
      github_comment: 'تعليق GitHub',
      log: 'السجل',
      slack: 'Slack',
      telegram: 'Telegram'
    },
    disableRow: 'تعطيل',
    disabled: name => `تم التعطيل: «${name}»`,
    disabledBody:
      'تُعد Webhooks منصة بوابة مستقلة. فعّلها هنا لاستقبال أحداث HTTP الواردة؛ ولا تكون قنوات الدردشة ضرورية إلا عندما يسلّم اشتراك إلى Telegram أو Discord أو Slack أو قناة أخرى.',
    disabledTitle: 'مستقبِل Webhook معطّل',
    done: 'تم',
    empty: 'لا توجد اشتراكات Webhook بعد.',
    enable: 'تفعيل Webhooks',
    enableRow: 'تفعيل',
    enabled: name => `تم التفعيل: «${name}»`,
    enabledRestarting: 'تم تفعيل Webhooks؛ جارٍ إعادة تشغيل البوابة...',
    enabling: 'جارٍ التفعيل...',
    fieldDeliver: 'التسليم إلى',
    fieldDeliverOnly: 'تسليم الحمولة فقط',
    fieldDescription: 'الوصف',
    fieldDescriptionPlaceholder: 'وظيفة هذا Webhook (اختياري)',
    fieldEvents: 'الأحداث',
    fieldEventsPlaceholder: 'مفصولة بفواصل، اتركها فارغة للكل',
    fieldName: 'الاسم',
    fieldNamePlaceholder: 'مثال: github-push',
    fieldPrompt: 'الموجِّه',
    fieldPromptPlaceholder: 'تعليمات للوكيل عند تفعيل هذا Webhook (اختياري)',
    fieldSkills: 'المهارات',
    fieldSkillsPlaceholder: 'أسماء مهارات مفصولة بفواصل (اختياري)',
    hint: 'تُعاد تحميل تغييرات الاشتراك فورًا بمجرد تشغيل المستقبِل. الاشتراكات المعطّلة ترفض الأحداث الواردة.',
    loadFailed: 'تعذّر تحميل Webhooks',
    loading: 'جارٍ تحميل Webhooks...',
    nameRequired: 'الاسم مطلوب',
    newSubscription: 'اشتراك جديد',
    restartFailed: detail => `فشلت إعادة تشغيل البوابة${detail}`,
    restartGateway: 'إعادة تشغيل البوابة',
    restartNeeded: 'تم تفعيل Webhooks، لكن البوابة ما زالت تحتاج إلى إعادة تشغيل قبل أن يتمكن المستقبِل من العمل.',
    restarting: 'جارٍ إعادة تشغيل البوابة...',
    restartingGateway: 'جارٍ إعادة التشغيل...',
    search: 'البحث في Webhooks...',
    secretOnce: 'السر (يُعرض مرة واحدة)',
    subscriptions: count => `الاشتراكات (${count})`,
    toggleFailed: (name, enabled) => `فشل ${enabled ? 'تفعيل' : 'تعطيل'} «${name}»`,
    webhookUrl: 'رابط Webhook'
  },
  zones: {
    newSessionTab: 'تبويب جلسة جديد',
    newTab: 'تبويب جديد'
  }
} satisfies TranslationOverrides
