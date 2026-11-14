import type { RedesignOverlay } from './types'

// Arabic wording for the redesigned shell: only the keys whose text differs from the base catalog.
export const arRedesign: RedesignOverlay = {
  sharedMetrics: {
    collectedMilestones: 'الإعداد: الخطوات المكتملة، واتصالات المزوّدين، وعدد المهارات والإضافات والمهام الروتينية',
    sending:
      'إذا اخترت المشاركة، يرسل Hermes الإحصاءات إلى Nous مرة يوميًا مع معرّف عشوائي لهذا الوكيل. باستثناء ملاحظة لمرة واحدة بأن Hermes قد ثُبّت، لا تُرسل أبدًا الإحصاءات السابقة لموافقتك. يمكنك تغيير ذلك في أي وقت من الإعدادات.',
    sendDesc:
      'إحصاءات فقط، تُرسل مرة يوميًا مع معرّف عشوائي لهذا الوكيل. لا تشمل أبدًا رسائلك أو ملفاتك أو مساراتك أو نصوص الأخطاء.',
    consentBody:
      'يمكن لـ Hermes إحصاء طريقة استخدامك له: طول المحادثات، والنماذج والأدوات التي تعمل، ومتى يفشل شيء ما. لا يسجّل أبدًا رسائلك أو ملفاتك أو مساراتك أو نصوص الأخطاء.',
    collectedActivity: 'المحادثات: الطول، والنتيجة، ونوع الخطأ، ووقت النشاط في اليوم'
  },
  common: {
    bots: 'الوكلاء'
  },
  boot: {
    steps: {
      connectingGateway: 'جار الاتصال بالجهاز',
      loadingSessions: 'جار تحميل المحادثات الأخيرة'
    },
    errors: {
      gatewayConnectionLost: 'انقطع الاتصال بالجهاز',
      gatewayConnectionLostDetail:
        'ما زلنا نحاول إعادة الاتصال في الخلفية. يمكنك متابعة القراءة والكتابة — افتح إعدادات الأجهزة إذا استمرت المشكلة.',
      gatewaySignInRequired: 'تسجيل الدخول إلى الجهاز مطلوب'
    },
    failure: {
      description: 'لم يبدأ Hermes في الخلفية. جرب إحدى خطوات الاسترداد أدناه. لن يحذف ذلك محادثاتك أو إعداداتك.',
      remoteTitle: 'تسجيل الدخول إلى الجهاز البعيد مطلوب',
      remoteDescription:
        'انتهت جلسة الاتصال بالجهاز البعيد. سجّل الدخول مرة أخرى لإعادة الاتصال. لن يؤدي أي شيء هنا إلى حذف محادثاتك أو إعداداتك.',
      useLocalGateway: 'استخدام الجهاز المحلي',
      remoteSignInHint: signInLabel =>
        `يسجّل الخروج من جلسة المتصفح البعيدة المحفوظة، ثم يفتح ${signInLabel}. استخدم الجهاز المحلي للتبديل إلى الخلفية المضمنة.`,
      cloudDownDescription:
        'يعيد الوكيل السحابي المُدار من Nous والمتصل بهذا الجهاز خطأً من الخادم. لا يمكن إعادة تشغيله من هنا — تحقق من حالته، أو بدّل إلى الجهاز المحلي، أو احصل على الدعم.',
      signedInMessage: 'جار إعادة الاتصال بالجهاز البعيد...',
      signInToRemoteGateway: 'تسجيل الدخول إلى الجهاز البعيد'
    }
  },
  notifications: {
    sharedProfileWarning:
      'تستخدم نسخة أخرى من Hermes هذا الوكيل. تتشارك النسختان إعداداته وبياناته، لذا قد تتعارض التغييرات. يمكنك المتابعة أو إغلاق النسخة الأخرى قبل إجراء تغييرات.',
    voice: {
      couldNotStartSession: 'تعذر بدء محادثة صوتية'
    },
    native: {
      inputBody: 'ينتظر الوكيل ردّك.',
      turnDoneTitle: 'أنهى الوكيل'
    }
  },
  promptNotices: {
    desktopCommands: {
      configFooter:
        'غيّر الإعدادات من الإعدادات، أو حرّر config.yaml باستخدام `hermes config edit` على الجهاز الذي يشغّل Hermes.',
      saveLegacyRefused:
        'لم يتم الحفظ: لا يستطيع Hermes على هذا الجهاز تصدير نسخة منقّحة أو باسم محدد أو بغير JSON. حدّث Hermes، أو شغّل /save دون خيارات لكتابة JSON خام على ذلك الجهاز.',
      savedOnHost: (path: string) => `تم الحفظ على الجهاز الذي يشغّل Hermes: ${path}`,
      exportRemote:
        'يحزم /export وكيلاً من الجهاز الذي يشغّل Hermes، وهو ليس هذا الجهاز. شغّل `hermes profile export` على ذلك الجهاز.',
      importRemote:
        'يثبّت /import وكيلاً على الجهاز الذي يشغّل Hermes، وهو ليس هذا الجهاز. شغّل `hermes profile import <archive>` على ذلك الجهاز.',
      historyHermes: 'الوكيل',
      checkpointsDisabled: 'نقاط الاستعادة غير مفعّلة لهذه المحادثة (اضبط checkpoints.enabled في config.yaml).'
    },
    sessionCommands: {
      busy: (command: string) =>
        `المحادثة مشغولة — ما زال الوكيل يرد. أوقف الرد الحالي أولاً (زر الإيقاف، أو Ctrl+C في الطرفية)، ثم شغّل /${command}.`,
      busyCompressing: (command: string) => `المحادثة مشغولة — يجري تنفيذ /compress؛ شغّل /${command} عند انتهائه.`,
      titleSet: (title, queued) =>
        `تم تعيين عنوان المحادثة: ${title}${queued ? ' (في الانتظار حتى تبدأ المحادثة)' : ''}`,
      titleCleared: 'تم مسح عنوان المحادثة.'
    }
  },
  sendDiagnostics: {
    privacyNotice:
      'سيؤدي هذا إلى رفع حزمة تصحيح إلى التخزين الداخلي لدى Nous (ليست لصيقة عامة). تتضمن معلومات النظام (نظام التشغيل، الإصدارات، المزوّد، وأنواع مفاتيح API المُهيأة — وليس المفاتيح نفسها أبداً) والسجلات الكاملة للوكيل وخدمة Hermes وسطح المكتب (حتى 512 كيلوبايت لكل منها، ومن المرجح أن تحتوي على محتوى المحادثات ومخرجات الأدوات ومسارات الملفات). تُحجب الأسرار قبل الرفع. لا يمكن الاطلاع عليها إلا لموظفي Nous ومشرفي Discord المعتمدين، وتُحذف تلقائياً بعد 14 يوماً.'
  },
  keybinds: {
    categories: {
      profiles: 'الوكلاء',
      session: 'المحادثة'
    },
    actions: {
      'nav.profiles': 'فتح الوكلاء',
      'nav.cron': 'فتح المهام الروتينية',
      'profile.default': 'الانتقال إلى الوكيل الافتراضي',
      'profile.switch.1': 'الانتقال إلى الوكيل 1',
      'profile.switch.2': 'الانتقال إلى الوكيل 2',
      'profile.switch.3': 'الانتقال إلى الوكيل 3',
      'profile.switch.4': 'الانتقال إلى الوكيل 4',
      'profile.switch.5': 'الانتقال إلى الوكيل 5',
      'profile.switch.6': 'الانتقال إلى الوكيل 6',
      'profile.switch.7': 'الانتقال إلى الوكيل 7',
      'profile.switch.8': 'الانتقال إلى الوكيل 8',
      'profile.switch.9': 'الانتقال إلى الوكيل 9',
      'profile.switch.10': 'الانتقال إلى الوكيل 10',
      'profile.switch.11': 'الانتقال إلى الوكيل 11',
      'profile.switch.12': 'الانتقال إلى الوكيل 12',
      'profile.switch.13': 'الانتقال إلى الوكيل 13',
      'profile.switch.14': 'الانتقال إلى الوكيل 14',
      'profile.switch.15': 'الانتقال إلى الوكيل 15',
      'profile.switch.16': 'الانتقال إلى الوكيل 16',
      'profile.switch.17': 'الانتقال إلى الوكيل 17',
      'profile.switch.18': 'الانتقال إلى الوكيل 18',
      'profile.next': 'الوكيل التالي',
      'profile.prev': 'الوكيل السابق',
      'profile.toggleAll': 'تبديل عرض كل الوكلاء',
      'profile.create': 'إنشاء وكيل',
      'agents.new': 'وكيل جديد (صفحة الوكلاء)',
      'nav.artifacts': 'فتح المكتبة',
      'session.new': 'محادثة جديدة',
      'session.newTab': 'علامة تبويب محادثة جديدة',
      'session.newWindow': 'نافذة جديدة',
      'session.next': 'المحادثة التالية',
      'session.prev': 'المحادثة السابقة',
      'session.slot.1': 'الانتقال إلى المحادثة الأخيرة 1',
      'session.slot.2': 'الانتقال إلى المحادثة الأخيرة 2',
      'session.slot.3': 'الانتقال إلى المحادثة الأخيرة 3',
      'session.slot.4': 'الانتقال إلى المحادثة الأخيرة 4',
      'session.slot.5': 'الانتقال إلى المحادثة الأخيرة 5',
      'session.slot.6': 'الانتقال إلى المحادثة الأخيرة 6',
      'session.slot.7': 'الانتقال إلى المحادثة الأخيرة 7',
      'session.slot.8': 'الانتقال إلى المحادثة الأخيرة 8',
      'session.slot.9': 'الانتقال إلى المحادثة الأخيرة 9',
      'session.focusSearch': 'البحث في المحادثات',
      'session.togglePin': 'تثبيت / إلغاء تثبيت المحادثة الحالية',
      'view.toggleSidebar': 'تبديل الشريط الجانبي للمحادثات'
    }
  },
  settings: {
    subpages: {
      gatewayConnection: 'الجهاز الأساسي'
    },
    nav: {
      gateway: 'الأجهزة',
      sessions: 'المحادثات'
    },
    plugins: {
      installModal: {
        installsAndEnables: profile => `يُثبَّت للوكيل ${profile} ويعمل فورًا.`,
        installsOnly: profile => `يُثبَّت للوكيل ${profile}.`
      }
    },
    searchPlaceholder: {
      gateway: 'اتصال الجهاز...'
    },
    appearance: {
      themeProfileNote: profile => `تم الحفظ للوكيل ${profile} — يحتفظ كل وكيل بثيمه الخاص.`,
      reactionsDesc: 'تفاعلات إيموجي بأسلوب iMessage — تفاعل مع الرسائل، ويمكن للوكيل التفاعل مع رسائلك.',
      tipsDesc:
        'نصائح تظهر أحيانًا من التطبيق ووكيلك. تظهر كل نصيحة مرة واحدة. تُعطّل تلقائيًا بعد أول 30 يومًا من الاستخدام، ويمكنك تفعيلها مجددًا.',
      toursDesc:
        'دع الوكيل يرشدك في التطبيق مع إبراز كل خطوة. تُعطّل الجولات تلقائيًا بعد أول 30 يومًا من الاستخدام، ويمكنك تفعيلها مجددًا.',
      pet: {
        intro:
          'تبنَّ تعويذة petdex متحركة تطفو فوق التطبيق وتتفاعل مع ما يفعله الوكيل — تجري أثناء تنفيذ الأدوات، وتحتفل عند النجاح، وتعبس عند الأخطاء.'
      }
    },
    about: {
      channel: {
        heldBody: (v: string) =>
          `هذا الإصدار أحدث من الإصدار المستقر ${v}، ومحادثاتك تستخدم بالفعل بيئة تشغيل Hermes الخاصة به التي لا يستطيع ${v} فتحها. سيبقى Hermes على هذا الإصدار حتى يصدر إصدار مستقر أحدث.`
      }
    },
    gateway: {
      loading: 'جار تحميل إعدادات الأجهزة...',
      unavailableTitle: 'إعدادات الأجهزة غير متاحة',
      unavailableDesc: 'جسر IPC في سطح المكتب لا يوفّر إعدادات الأجهزة.',
      webappHostDesc:
        'يعمل Webapp دائمًا على الجهاز الذي يقدّمه. الاتصالات المحفوظة أدناه هي قائمة هذا الجهاز، مشتركة مع تطبيق Hermes Desktop؛ ويتم الوصول إلى الأجهزة الأخرى عبر هذا الجهاز.',
      webappOauthPaste: {
        description:
          'إذا انتهت علامة تبويب تسجيل الدخول على عنوان لا يُحمَّل (هذا المتصفح ليس على الجهاز الذي يشغّل Hermes)، فالصق ذلك العنوان هنا.'
      },
      webAccess: {
        pendingOn: 'تم الحفظ. سيقدّم Hermes تطبيق الويب بعد إعادة تشغيله.',
        pendingOff: 'تم الإيقاف، لكن Hermes يواصل تقديم تطبيق الويب حتى إعادة تشغيله.',
        unavailableDetail: reason => `تعذّر على Hermes تقديم تطبيق الويب: ${reason}`,
        restartBusy:
          'تم الحفظ. توجد محادثة قيد التشغيل، لذا لم تتم إعادة تشغيل Hermes. اختر «إعادة تشغيل Hermes» عند انتهائها.',
        restartFailed: 'تم الحفظ، لكن تعذّرت إعادة تشغيل Hermes. اختر «إعادة تشغيل Hermes» للمحاولة مجددًا.',
        restartFailedBecause: reason => `تعذّرت إعادة تشغيل Hermes: ${reason}`,
        startTimedOut: 'لم يقدّم Hermes تطبيق الويب خلال 90 ثانية.',
        stopTimedOut: 'لم يتوقف Hermes عن تقديم تطبيق الويب خلال 90 ثانية.',
        startStages: {
          restarting: 'جارٍ إعادة تشغيل Hermes لتقديم تطبيق الويب.'
        },
        restartGateway: 'إعادة تشغيل Hermes'
      },
      title: 'اتصال الجهاز',
      intro:
        'محلي افتراضيًا. استخدم الجهاز البعيد عندما ينبغي لهذا التطبيق تشغيل Hermes في مكان آخر. صِل جهازًا لاكتشاف وكلائه.',
      localTitle: 'جهاز محلي',
      remoteTitle: 'جهاز بعيد',
      remoteDesc: 'صل واجهة سطح المكتب هذه بخلفية Hermes بعيدة.',
      remoteUrlTitle: 'رابط الجهاز البعيد',
      probing: 'جار فحص طريقة مصادقة هذا الجهاز...',
      probeError: 'تعذر الوصول إلى هذا الجهاز الآن. تحقق من الرابط وستظهر طريقة المصادقة عند الاستجابة.',
      authSignedInPassword: 'يستخدم هذا الجهاز اسم مستخدم وكلمة مرور. أنت مسجل الدخول والجلسة تتجدد تلقائياً.',
      authSignedInOauth: 'يستخدم هذا الجهاز OAuth. أنت مسجل الدخول والجلسة تتجدد تلقائياً.',
      authNeedsPassword: 'يستخدم هذا الجهاز اسم مستخدم وكلمة مرور. سجل الدخول لتفويض تطبيق سطح المكتب.',
      authNeedsOauth: provider => `يستخدم هذا الجهاز OAuth. سجل الدخول عبر ${provider} لتفويض تطبيق سطح المكتب.`,
      keychainEncryptionDesc:
        'معطّل افتراضياً. عند التفعيل، تُشفَّر رموز الجهاز وبيانات تسجيل الدخول باستخدام سلسلة مفاتيح النظام (Keychain Access أو GNOME Keyring أو Windows DPAPI) — وقد يطلب النظام إذناً أو كلمة مرور. عند التعطيل، تُخزَّن كملفات عادية لا يقرؤها سوى حساب المستخدم الحالي.',
      diagnosticsDesc: 'اعرض desktop.log في مدير الملفات، وهذا مفيد عند فشل تشغيل Hermes.',
      incompleteTitle: 'إعداد الجهاز البعيد غير مكتمل',
      incompleteSignIn: 'أدخل رابط الجهاز البعيد وسجل الدخول قبل التبديل إلى البعيد.',
      incompleteToken: 'أدخل رابط الجهاز البعيد ورمز الجلسة قبل التبديل إلى البعيد.',
      incompleteSignInTest: 'أدخل رابط الجهاز البعيد وسجل الدخول قبل الاختبار.',
      incompleteTokenTest: 'أدخل رابط الجهاز البعيد ورمز الجلسة قبل الاختبار.',
      enterUrlFirst: 'أدخل رابط الجهاز البعيد أولاً.',
      restartingTitle: 'جار إعادة الاتصال بالجهاز',
      savedTitle: 'تم حفظ إعدادات الجهاز',
      reachableTitle: 'يمكن الوصول إلى الجهاز البعيد',
      signedOutMessage: 'تم مسح جلسة الجهاز البعيد.',
      failedLoad: 'فشل تحميل إعدادات الجهاز',
      testFailed: 'فشل اختبار الجهاز البعيد',
      applyFailed: 'تعذر تطبيق إعدادات الجهاز',
      saveFailed: 'تعذر حفظ إعدادات الجهاز',
      webappHostTitle: 'جهاز Hermes',
      envOverrideTitle: 'تم تثبيت هذا الاتصال بحسب طريقة تشغيل Hermes.'
    },
    profileScope: {
      editsProfile: profile => `تنطبق التغييرات في هذه الصفحة على الوكيل «${profile}».`
    },
    model: {
      tasks: {
        profile_describer: {
          label: 'واصف الوكيل',
          hint: 'أوصاف تلقائية للوكلاء'
        },
        title_generation: {
          hint: 'عناوين المحادثات'
        }
      },
      subagents: {
        routing: {
          modes: {
            profile_routes: {
              label: 'مسارات الوكلاء',
              description: 'يعمل الوكلاء الفرعيون بإعدادات أحد وكلائك، بنموذجه ومهاراته ومفاتيحه الخاصة.'
            }
          },
          pluginMissing: 'مسارات النماذج ومسارات الوكلاء مصدرها الإضافة Advanced Delegation.',
          pluginDisabled: 'الإضافة Advanced Delegation مثبتة لكنها متوقفة لهذا الوكيل.',
          blockedDisabled: 'لن يبدأ الوكلاء الفرعيون حتى تُفعَّل الإضافة Advanced Delegation لهذا الوكيل.',
          profileRoutesTitle: 'الوكلاء الذين يمكن استخدام إعداداتهم للوكلاء الفرعيين',
          profileRoutesHint: 'يوفر كل وكيل محدد نموذجه ومهاراته وشخصيته ومفاتيحه الخاصة.',
          profileEmpty: 'لا يوجد وكلاء آخرون بعد. أنشئ وكيلاً من شريط الوكلاء.',
          profileMissing: name => `${name} (لم يعد الوكيل موجودًا)`,
          inactiveProfiles: profile =>
            `تحدد مسارات الوكلاء إعدادات كل وكيل فرعي. من لا يختار وكيلاً يعمل بإعدادات «${profile}».`,
          inactiveProfilesNoDefault:
            'تحدد مسارات الوكلاء إعدادات كل وكيل فرعي. من لا يختار وكيلاً يستخدم نموذج الوكيل الرئيسي.'
        }
      },
      appliesDesc: 'ينطبق على المحادثات الجديدة. استخدم منتقي النموذج في صندوق الإنشاء لتبديل المحادثة النشطة فورا.'
    },
    vault: {
      otpHint: '«مفتاح الإعداد» الذي يعرضه الموقع عند تفعيل المصادقة الثنائية. بحفظه يولّد الوكيل الرموز بنفسه.'
    },
    notifications: {
      kinds: {
        input: {
          description: 'طرح الوكيل سؤالا أو يحتاج إلى كلمة مرور أو سر.'
        }
      }
    },
    fieldDescriptions: {
      'display.personality': 'أسلوب المساعد الافتراضي للمحادثات الجديدة.',
      'memory.memoryEnabled': 'يحفظ ذكريات دائمة يمكن أن تساعد المحادثات القادمة.'
    },
    memoryLayers: {
      selectedDesc: current => `غير نشط بعد: تواصل المحادثات استخدام ${current} حتى يُحفظ هذا الاختيار.`,
      noProviderDesc:
        'يستخدم الوكيل الطبقة المدمجة أدناه فقط. اختر مزوّدًا خارجيًا من الإعدادات › الذاكرة والسياق لإضافة واحد.',
      providerDesc: name => `يضيف ${name} استرجاعه الخاص إلى كل محادثة، فوق الطبقة المدمجة.`,
      builtinDesc:
        'MEMORY.md وUSER.md هما ذاكرة الوكيل الخاصة، مستقلتان عن أي مزوّد خارجي. يُحمَّل كلاهما في المحادثات افتراضيًا؛ ويعطّل مفتاحا الذاكرة وملف المستخدم في الإعدادات › الذاكرة والسياق كلًّا منهما. يضيف المزوّد إلى هذه الطبقة ولا يستبدلها أبدًا.'
    },
    sessions: {
      loading: 'جار تحميل المحادثات المؤرشفة...',
      archivedTitle: 'المحادثات المؤرشفة',
      defaultDirDesc: 'تبدأ المحادثات الجديدة في هذا المجلد ما لم تختر غيره. اتركه غير مضبوط لاستخدام مجلدك الرئيسي.',
      failedLoad: 'تعذر تحميل المحادثات المؤرشفة'
    }
  },
  skills: {
    plugins: {
      autoUpdateTip: (hours: number) =>
        `تطبيق تحديثات الإضافات تلقائيًا عندما يفحصها Hermes (كل ${hours} ساعة). تُحمَّل الشيفرة المحدّثة في المحادثات الجديدة.`
    },
    remotePluginHint: 'افتح هذا الوكيل على جهازه لإدارة إضافاته. يستخدم تثبيت الإضافات الجهاز المتصل.',
    openAgentSettings: 'إعدادات الوكيل',
    appliesToNewSessions: name => `ينطبق على المحادثات الجديدة في ${name}`,
    openToolLibrary: 'تصفح كتالوج الأدوات'
  },
  gatewayGroupOpen: {
    lookupFailed: 'تعذّر العثور على هذه المجموعة على الجهاز. حاول مجددًا.'
  },
  commandCenter: {
    sectionEntries: {
      system: {
        detail: 'حالة الجهاز والسجلات وإعادة التشغيل/التحديث'
      },
      sessions: {
        title: 'لوحة المحادثات',
        detail: 'البحث في المحادثات وتثبيتها وإدارتها'
      }
    },
    gatewayRunning: 'المراسلة تعمل',
    gatewayStopped: 'المراسلة متوقفة',
    restartGateway: 'إعادة تشغيل Hermes',
    gatewayRestartFailed: 'فشل إعادة تشغيل Hermes.',
    sharedGatewayRestartTitle: 'إعادة تشغيل Hermes لجميع الوكلاء؟',
    sharedGatewayRestartDescription: bots => `تتم إعادة اتصال جميع الوكلاء على هذا الجهاز: ${bots}`,
    sharedGatewayRestarted: count => `تمت إعادة تشغيل Hermes (عدد الوكلاء: ${count})`,
    searchPlaceholder: 'ابحث في المحادثات والعروض والإجراءات',
    goToSession: 'الانتقال إلى المحادثة',
    sections: {
      sessions: 'المحادثات'
    },
    nav: {
      newChat: {
        title: 'محادثة جديدة',
        detail: 'بدء محادثة جديدة'
      },
      artifacts: {
        title: 'المكتبة'
      }
    },
    providerSessions: 'محادثات المزوّد',
    pinSession: 'تثبيت المحادثة',
    unpinSession: 'إلغاء تثبيت المحادثة',
    exportSession: 'تصدير المحادثة',
    deleteSession: 'حذف المحادثة',
    noSessions: 'لا توجد محادثات بعد.',
    hermesActiveSessions: (version, count) => `Hermes ${version} · المحادثات النشطة ${count}`,
    statSessions: 'المحادثات'
  },
  messaging: {
    states: {
      gateway_stopped: 'المراسلة متوقفة'
    },
    hintPendingRestart: 'أعد تشغيل Hermes لتطبيق التغييرات.',
    sharedListenerUrl: 'يُخدم عبر مستمع Hermes المشترك على',
    hintGatewayStopped: 'المراسلة متوقفة.',
    gatewayStopped: 'المراسلة متوقفة',
    restartToApply: 'أعد تشغيل Hermes لتطبيق التغييرات.',
    restartToReconnect: 'أعد تشغيل Hermes لإعادة الاتصال.',
    appliedLive: 'تم التطبيق على Hermes قيد التشغيل.',
    connectingLive: 'يتصل Hermes قيد التشغيل باستخدام بيانات الاعتماد الجديدة.',
    restartNeeded: 'تم الحفظ. أعد تشغيل Hermes لتطبيق الإعدادات الجديدة.',
    restartFailedManual: 'فشلت إعادة تشغيل Hermes — أعد تشغيله يدويًا وتحقق من سجلات Hermes.',
    relayLink: {
      managedNote: (platform: string) =>
        `يتولى Team Gateway المراسلة في هذا المثيل، لذا يوقف Hermes الاتصال المباشر بـ ${platform} عند كل تشغيل. لتشغيله بجانب المرحّل، اضبط GATEWAY_RELAY_ALLOW_DIRECT_PLATFORMS=true في بيئة Hermes (الاتصال المباشر يمنع المثيل من التقلص إلى الصفر وقد يسلّم الرسائل مرتين).`
    },
    telegramQr: {
      savedRestarting: 'تم حفظ Telegram؛ تجري إعادة تشغيل Hermes…',
      savedRestartFailed: detail => `تم حفظ Telegram؛ فشلت إعادة تشغيل Hermes${detail}`
    },
    fieldCopy: {
      DISCORD_HOME_CHANNEL: {
        help: 'القناة التي يرسل فيها البوت الرسائل الاستباقية (نتائج المهام الروتينية، التذكيرات).'
      },
      QQBOT_HOME_CHANNEL: {
        help: 'القناة أو المجموعة الافتراضية لتسليم نتائج المهام الروتينية.'
      }
    }
  },
  profiles: {
    nameHint: 'اسم الوكيل',
    title: 'الوكلاء',
    count: count => `عدد الوكلاء: ${count}`,
    search: 'البحث في الوكلاء…',
    newProfile: 'وكيل جديد',
    importProfile: 'استيراد وكيل…',
    exportProfile: 'تصدير وكيل…',
    imported: 'تم استيراد الوكيل',
    exported: 'تم تصدير الوكيل',
    failedImport: 'فشل استيراد الوكيل',
    failedExport: 'فشل تصدير الوكيل',
    exportNeedsHost: name =>
      `لا يمكن للمتصفح حفظ أرشيف الوكيل. على الجهاز الذي يشغّل Hermes، شغّل: hermes profile export ${name}`,
    allProfiles: 'كل الوكلاء',
    showAllProfiles: 'إظهار كل الوكلاء',
    manageProfiles: 'إدارة الوكلاء',
    fleet: {
      allOnGateway: 'كل الوكلاء على هذا الجهاز',
      gateway: gateway => `الوكلاء على ${gateway}`,
      localDevice: 'هذا الجهاز (خلفية محلية — تثبّت Hermes إن كان مفقودًا، وإلا تفتح محادثة جديدة)',
      installDeviceDesc: 'سيُثبَّت Hermes محليًا ثم تُفتح محادثة جديدة على هذا الحاسوب. لا يبدأ التثبيت قبل التأكيد.'
    },
    remoteOverride: {
      description: 'ستعمل محادثات هذا الوكيل على Hermes البعيد الذي تحدده، بدلاً من هذا الجهاز.',
      collisionWarning: (label: string) =>
        `يوجد جهاز باسم «${label}» في الإعدادات بالفعل. اتصال هذا الوكيل منفصل ولن يغيّره.`,
      confirmTitle: 'ربط هذا الوكيل بجهاز بعيد؟',
      savedTitle: 'تم ربط الوكيل',
      menuItem: 'الاتصال بجهاز بعيد…',
      title: (profile: string) => `ربط ${profile} بجهاز بعيد`,
      confirmNote: (profile: string, host: string) =>
        `ستعمل المحادثات الجديدة في ${profile} على ${host}. سيقوم ذلك الجهاز بتنفيذ الأوامر وقراءة الملفات هناك، وليس هنا. اتصل فقط بجهاز تثق به.`,
      authFailedTitle: 'رفض الجهاز البعيد الرمز المحفوظ'
    },
    defaultProfile: 'الوكيل الافتراضي',
    defaultSet: name => `أصبح ${name} الوكيل الافتراضي`,
    defaultDescription: 'يُستخدم عند فتح Hermes وللمحادثات الجديدة. تبقى المحادثات الحالية مع وكلائها.',
    failedSetDefault: 'تعذّر تعيين الوكيل الافتراضي',
    noProfiles: 'لا يوجد وكلاء',
    selectPrompt: 'اختر وكيلاً',
    soulDesc: 'تعليمات هذا الوكيل وشخصيته.',
    soulMissing:
      'لا يوجد ملف SOUL.md لهذا الوكيل بعد. أضف التعليمات أدناه واحفظ لإنشائه. تُدار إعدادات الشخصية في config.yaml بشكل منفصل.',
    soulPlaceholder: mode => `تعليمات هذا الوكيل وشخصيته.\nاتركه فارغا للإبقاء على افتراضي ${mode}.`,
    deleteTitle: 'حذف الوكيل',
    createDesc: 'أنشئ وكيلاً له تعليماته وإعداداته ومحادثاته الخاصة.',
    cloneFromDesc: 'ينسخ الإعدادات والمهارات وSOUL.md من الوكيل المحدد.',
    cloneFromDefaultDesc: 'ابدأ من إعدادات الوكيل الافتراضي.',
    createAction: 'إنشاء وكيل',
    renameTitle: 'إعادة تسمية الوكيل',
    renameDescPrefix: 'تؤدي إعادة التسمية إلى تحديث مجلد الوكيل وأي سكربتات تغليف في ',
    displayNameTitle: 'تسمية هذا الوكيل',
    displayNameDesc: 'يحدد اسم العرض في التطبيق. يبقى المعرّف الداخلي "default".',
    displayNameLabel: 'اسم العرض',
    created: 'تم إنشاء الوكيل',
    failedLoad: 'فشل تحميل الوكلاء'
  },
  cron: {
    title: 'المهام الروتينية',
    count: count => `عدد المهام الروتينية: ${count}`,
    search: 'البحث في المهام الروتينية…',
    loading: 'جار تحميل المهام الروتينية…',
    editJob: 'تحرير المهمة الروتينية',
    newCron: 'مهمة روتينية جديدة',
    emptyDescNew: 'أنشئ مهمة روتينية لتشغيل الوكيل تلقائيًا.',
    emptyDescSearch: 'لا توجد مهام روتينية تطابق البحث.',
    emptyTitleNew: 'لا توجد مهام روتينية',
    resumeTitle: 'استئناف المهمة الروتينية',
    pauseTitle: 'إيقاف المهمة الروتينية مؤقتًا',
    deleteTitle: 'حذف المهمة الروتينية',
    failedLoad: 'فشل تحميل المهام الروتينية',
    editTitle: 'تحرير المهمة الروتينية',
    createTitle: 'إنشاء مهمة روتينية',
    editDesc: 'عدّل وقت التشغيل والرسالة.',
    createDesc: 'شغّل موجّهًا تلقائيًا وفق جدول. استخدم صياغة cron أو عبارة طبيعية مثل «كل 15 دقيقة».',
    promptPlaceholder: 'ماذا تريد من الوكيل أن يفعل؟',
    scheduleRequired: 'وقت التشغيل مطلوب.',
    scriptOnlyEditHint: 'تشغّل هذه المهمة الروتينية برنامجًا نصيًا، وليس رسالة للذكاء الاصطناعي. معرّف المهمة:',
    tabs: {
      jobs: 'المهام الروتينية'
    },
    customPlaceholder: '0 9 * * * أو أيام الأسبوع الساعة 9 صباحًا',
    customHint: 'تعبير cron، أو عبارات مثل «كل ساعة» أو «أيام الأسبوع الساعة 9 صباحًا».'
  },
  sidebar: {
    filter: {
      profile: 'الوكيل'
    },
    gatewayGroups: {
      grouping: 'الجهاز والوكيل',
      aliasHint: 'يتغير اسم العرض فقط؛ تبقى أسماء الجهاز والوكيل دون تغيير.'
    },
    profileRail: 'شريط الوكلاء',
    nav: {
      cron: 'المهام الروتينية',
      'new-session': 'محادثة جديدة',
      artifacts: 'المكتبة'
    },
    cronJobs: 'المهام الروتينية',
    projects: {
      createdInPreviousContext:
        'أُنشئ المشروع على الاتصال السابق أو لدى الوكيل السابق. عُد إليه؛ لم يُكتب ملف IDEA.md.',
      showAllSessions: 'عرض جميع المحادثات',
      toggle: (label, open) => `${open ? 'إظهار' : 'إخفاء'} محادثات ${label}`,
      showAllCount: count => `إظهار كل المحادثات (${count})`
    },
    row: {
      ownedByProfile: profile => `تابعة للوكيل ${profile}`,
      copyIdFailed: 'فشل نسخ معرّف المحادثة',
      sessionActions: 'إجراءات المحادثة',
      sessionRunning: 'المحادثة تعمل',
      renameTitle: 'إعادة تسمية المحادثة',
      untitledPlaceholder: 'محادثة بلا عنوان',
      deleteTitle: 'حذف المحادثة؟',
      deleted: 'تم حذف المحادثة'
    },
    searchAria: 'البحث في المحادثات',
    searchPlaceholder: 'البحث في المحادثات...',
    noMatch: query => `لا توجد محادثات تطابق "${query}"`,
    sessions: 'المحادثات',
    groupAriaGrouped: 'عرض المحادثات كقائمة واحدة',
    groupAriaUngrouped: 'تجميع المحادثات حسب مساحة العمل',
    showSessions: 'عرض المحادثات',
    groupTitleGrouped: 'إلغاء تجميع المحادثات',
    groupTitleUngrouped: 'التجميع حسب مساحة العمل',
    allPinned: 'كل ما هنا مثبّت. ألغِ تثبيت محادثة لإظهارها في الأخيرة.',
    projectEmpty: 'لا توجد محادثات بعد',
    projectLoadFailed: 'تعذر تحميل المحادثات',
    noSessions: 'لا توجد محادثات بعد',
    noFilterMatches: 'لا توجد محادثات تطابق عوامل التصفية هذه',
    newSessionIn: label => `محادثة جديدة في ${label}`
  },
  composer: {
    commandDescs: {
      '/export': 'تصدير الوكيل بإعداداته ومهاراته وسمته إلى أرشيف قابل للمشاركة',
      '/import': 'استيراد أرشيف وكيل كوكيل جديد',
      '/profile': 'تبديل الوكيل النشط',
      '/platform': 'إيقاف منصة مراسلة متعثرة مؤقتًا أو استئنافها أو عرضها',
      '/clear': 'بدء محادثة جديدة',
      '/resume': 'استئناف محادثة محفوظة',
      '/title': 'إعادة تسمية المحادثة الحالية',
      '/handoff': 'نقل هذه المحادثة إلى منصة مراسلة',
      '/bg': 'تشغيل طلب في محادثة خلفية منفصلة',
      '/goal': 'تحديد هدف مستمر يعمل عليه الوكيل حتى إنجازه',
      '/heartbeat': 'تحديد طلب متكرر يعود إلى هذه المحادثة عندما تكون خاملة',
      '/loop': 'تكرار طلب على فترات منتظمة في هذه المحادثة',
      '/status': 'عرض حالة المحادثة الحالية',
      '/model': 'تبديل نموذج هذه المحادثة'
    },
    queueLostDiscardTip: 'أُعيد تشغيل Hermes أثناء هذا الدور ولا يمكن إكماله. تجاهله لتستمر الأدوار المنتظرة خلفه.',
    newSessionPlaceholders: ['اسأل Hermes عن شيء...', 'كلّف الوكيل بمهمة...', 'ابدأ محادثة جديدة...'],
    queueDroppedBody:
      'أُسقط هذا العنصر في الخلفية لأن محادثته تعذّر استئنافها بعد محاولات متكررة. بقية قائمة الانتظار لم تتأثر.',
    dropSession: 'أفلت لربط هذه المحادثة',
    browseAllSessions: 'تصفح كل المحادثات…'
  },
  updates: {
    bundleOutOfSyncDesc:
      'تم تحديث وقت تشغيل Hermes، لكن تطبيق سطح المكتب نفسه لا يزال إصدارًا قديمًا — لن تظهر ميزات الواجهة الجديدة (مثل وضع الوكيل) حتى يتم تحديث التطبيق. شغّل التحديث أدناه لإعادة بناء التطبيق. إذا لم يختفِ هذا التحذير، فأعد التثبيت من أحدث مثبّت لسطح المكتب.',
    applyStatus: {
      noReturn: 'لم تعد الواجهة الخلفية إلى الاتصال. قد لا يكون التحديث قد اكتمل — تحقق من الجهاز الذي يشغّل Hermes.'
    }
  },
  shell: {
    gatewayMenu: {
      gateway: 'الجهاز',
      reconnectGateway: 'إعادة الاتصال بالجهاز',
      machineSettings: 'الأجهزة',
      restartTitle: 'إعادة تشغيل Hermes على هذا الجهاز؟'
    },
    statusbar: {
      gateway: 'الجهاز',
      gatewayNeedsSetup: 'الجهاز يحتاج إعدادا',
      gatewayChecking: 'جار فحص الجهاز',
      gatewayConnecting: 'جار اتصال الجهاز',
      gatewayOffline: 'الجهاز غير متصل',
      gatewayTitle: 'الجهاز',
      cron: 'المهام الروتينية',
      openCron: 'فتح المهام الروتينية'
    }
  },
  rightSidebar: {
    terminalGatewayRemoved: 'جهاز محذوف',
    terminalGatewayRemovedBody:
      'حُذف الجهاز الذي كانت تعمل عليه هذه الطرفية، لذا لا يمكن إعادة فتحها. أغلق علامة التبويب هذه.',
    terminalGatewayNoShellBody:
      'لا يوفر هذا الجهاز وصولاً إلى الصدفة، لذا لا يمكن إعادة فتح هذه الطرفية. أغلق علامة التبويب هذه.'
  },
  preview: {
    saveScopeChanged: 'عُد إلى الاتصال والوكيل الأصليين لحفظ هذه المسودة.',
    web: {
      restarting: 'يعيد الوكيل تشغيل الخادم...',
      askRestart: 'اطلب من الوكيل إعادة تشغيل الخادم',
      lookingRestart: taskId => `يبحث الوكيل عن خادم معاينة لإعادة تشغيله (${taskId})`,
      restartingMessage: 'يعمل الوكيل في الخلفية. راقب كونسول المعاينة لمتابعة التقدم.',
      blankPageBody: 'اكتب عنوانًا في الأعلى للتصفح، أو اطلب من الوكيل فتح صفحة.',
      finishedRestarting: message => `أنهى الوكيل إعادة تشغيل خادم المعاينة${message ? `: ${message}` : ''}`,
      restartFailedMessage: 'تعذّر على الوكيل إعادة تشغيل الخادم.',
      stillWorking:
        'لا يزال الوكيل يعمل، لكن لم تصل نتيجة إعادة التشغيل بعد. قد يكون أمر الخادم قيد التشغيل في المقدمة.'
    }
  },
  assistant: {
    thread: {
      errorCodes: {
        cron_session_read_only: {
          title: 'هذه المحادثة تابعة لمهمة روتينية',
          body: 'إنها مخصصة لتشغيل تلك المهمة الروتينية، لذا لم تُنفَّذ رسالتك. ابدأ محادثة جديدة لإرسالها.'
        },
        no_reply: {
          body: 'أنهى الوكيل هذه الجولة، لكن لم يصل أي رد إلى هذه النافذة. أعد المحاولة لإرسالها مجددًا.'
        },
        loop_error: {
          title: 'تعذّر على الوكيل إنهاء هذه الخطوة'
        },
        launch_credentials_unavailable: {
          body: 'بدأت هذه المحادثة بمفتاح API تم تغييره أو حذفه منذ ذلك الحين، لذا لم تُرسل رسالتك. المحادثة المفتوحة لا تنتقل أبدًا إلى مفتاح مختلف من تلقاء نفسها. استخدم «بدء محادثة جديدة» لاستخدام مفتاحك الحالي.'
        }
      },
      loadingSession: 'جار تحميل المحادثة...',
      errorLayers: {
        generic: 'تعذّر على الوكيل إكمال هذا الرد',
        runtime: 'واجه الوكيل مشكلة'
      },
      errorStartNewSession: 'بدء محادثة جديدة'
    },
    approval: {
      gatewayDisconnected: 'الجهاز غير متصل',
      allowSession: 'السماح لهذه المحادثة',
      alwaysDescription: pattern =>
        `يضيف هذا النمط «${pattern}» إلى قائمة السماح الدائمة (~/.hermes/config.yaml). لن يسأل الوكيل مجددًا عن أوامر كهذه — في هذه المحادثة أو في أي محادثة مستقبلية.`
    },
    clarify: {
      gatewayDisconnected: 'الجهاز غير متصل'
    },
    catalogInstall: {
      targetProfile: profile => `يُثبَّت للوكيل ${profile}`
    },
    tool: {
      titles: {
        cronjob: {
          done: 'تم إعداد المهمة الروتينية',
          pending: 'جار إعداد المهمة الروتينية',
          pendingAction: 'جار الإعداد'
        },
        session_search_recall: {
          done: 'تم البحث في سجل المحادثات',
          pending: 'جار البحث في سجل المحادثات'
        }
      }
    },
    themePicker: {
      basicDetail: 'للتحدث مع وكيلك.'
    }
  },
  prompts: {
    gatewayDisconnected: 'الجهاز غير متصل',
    sudoInstallDesc:
      'يحتاج الوكيل إلى كلمة مرور sudo لتثبيت حزم شاشة الوكيل (TigerVNC + Xfce) على جهازه. تُرسل إلى ذلك الجهاز فقط.',
    sudoDesc:
      'راجع الأمر قبل إدخال كلمة مرور sudo. تُرسل كلمة المرور إلى الوكيل الذي ينفّذه وتُحفظ مؤقتًا لهذه المحادثة.',
    secretDesc: 'يحتاج الوكيل إلى بيانات اعتماد للمتابعة.',
    vaultSaveDesc: origin =>
      `وصل الوكيل إلى صفحة تسجيل الدخول في ${origin} ولا توجد بيانات دخول محفوظة لها. أدخلها هنا مرة واحدة؛ تُشفَّر على هذا الجهاز وتُملأ في الصفحة مباشرة، ولا يرى النموذج كلمة المرور أبدًا.`,
    vaultCodeDesc: site =>
      `يطلب ${site} رمزًا لمرة واحدة (رسالة نصية أو بريد إلكتروني أو تطبيق مصادقة). أدخله هنا وسيكتبه الوكيل في الصفحة؛ لا يراه النموذج أبدًا.`,
    vaultCodeFootnote:
      'تلميح: احفظ مفتاح المصادقة مع بيانات الدخول هذه في الإعدادات ← كلمات المرور وتسجيلات الدخول وسيُدخل الوكيل الرموز نيابةً عنك.'
  },
  desktop: {
    profileStatus: current =>
      `الوكيل الحالي: ${current}. استخدم /profile <name> أو منتقي «محادثة جديدة» لبدء محادثة مع وكيل آخر.`,
    unknownProfile: 'وكيل غير معروف',
    noProfileNamed: (target, available) => `لا يوجد وكيل باسم ${target}. المتاح: ${available}`,
    newChatsProfile: name => `ستستخدم المحادثات الجديدة الوكيل ${name}.`,
    setProfileFailed: 'فشل تعيين الوكيل',
    resumeStrandedBody:
      'فشل الاتصال بهذه المحادثة وتوقفت إعادة المحاولة التلقائية. تأكد من تشغيل Hermes على الجهاز، ثم حاول مجددا.',
    poolSlotTimeoutBody:
      'يعمل عدد كبير جدًا من الوكلاء في وقت واحد بالنسبة إلى حد هذا الحاسوب. ارفع الحد من الإعدادات ← متقدم، أو انتظر حتى ينتهي أحدهم ثم أعد المحاولة.',
    handoff: {
      timedOut: 'انتهت المهلة في انتظار Hermes. هل `hermes gateway` قيد التشغيل؟'
    },
    sessionUnavailable: 'المحادثة غير متاحة',
    createSessionFailed: 'فشل إنشاء محادثة جديدة',
    yoloSystem: active => (active ? 'YOLO مفعّل لهذه المحادثة' : 'YOLO معطّل لهذه المحادثة'),
    resumeStrandedTitle: 'تعذّر تحميل هذه المحادثة',
    sessionBusy: 'المحادثة مشغولة',
    sessionExported: 'تم تصدير المحادثة',
    sessionExportFailed: 'فشل تصدير المحادثة',
    sessionOpenFailed: 'تعذّر فتح هذه المحادثة'
  },
  tips: {
    items: {
      cron: {
        title: 'المهام الروتينية تعمل تلقائيًا',
        text: 'تشغّل المهام الروتينية طلبًا كل ساعة أو كل ليلة أو وفق تعبير cron.'
      },
      profiles: {
        title: 'لكل وكيل مساحته الخاصة',
        text: 'لكل وكيل تعليماته وإعداداته ومحادثاته الخاصة.'
      },
      skills: {
        text: 'المهارات مجلدات من التعليمات يحمّلها الوكيل عندما يقتضي العمل ذلك.'
      },
      messaging: {
        title: 'وكيلك بعيدًا عن مكتبك'
      },
      artifacts: {
        title: 'كل ما صنعه وكيلك',
        text: 'الصور والملفات والروابط من كل المحادثات، مجموعة في مكتبتك.'
      },
      'command-palette': {
        text: 'المحادثات والإعدادات والمهارات والأوامر كلها تستجيب للوحة الأوامر.'
      }
    }
  },
  firstRun: {
    guide: {
      upgradeRequired:
        'يحتاج Hermes على هذا الجهاز إلى تحديث للإعداد دون محادثة. شغّل hermes update على ذلك الجهاز، وأعد تشغيل Hermes، ثم أعد المحاولة. محادثاتك لم تتغير.'
    },
    intro: {
      greeting: 'مرحبًا، أنا وكيلك.'
    }
  },
  catalog: {
    installDescription: 'ستتوفر هذه المهارة في المحادثات الجديدة. ثبّت من المصادر التي تثق بها فقط.',
    installHint: 'راجع المصدر قبل التثبيت. تسري التغييرات على المحادثات الجديدة.'
  },
  sessionImport: {
    action: 'استيراد محادثة',
    search: 'البحث في المحادثات المحمّلة',
    scanError: 'تعذر العثور على المحادثات',
    emptyHelp: 'ستظهر هنا محادثات Claude Code وCodex الموجودة على هذا الجهاز.',
    searchHelp: 'جرّب عنوانًا أو مجلدًا آخر، أو حمّل المزيد من المحادثات.',
    more: 'تحميل المزيد من المحادثات',
    chooseHelp: 'اختر محادثة لقراءة سجلها قبل نقلها إلى Hermes.'
  },
  titlebar: {
    searchTitle: 'البحث في المحادثات والعروض والإجراءات',
    unreadSessions: count => (count === 1 ? 'محادثة واحدة غير مقروءة' : `${count} محادثات غير مقروءة`)
  },
  artifacts: {
    search: 'البحث في المكتبة...',
    refresh: 'تحديث المكتبة',
    refreshing: 'جار تحديث المكتبة...',
    indexing: 'جار فهرسة مخرجات المحادثات الأخيرة...',
    noArtifactsTitle: 'لا يوجد شيء في مكتبتك بعد',
    failedLoad: 'فشل تحميل المكتبة',
    colSession: 'المحادثة',
    partialIndex: (failed, total) => `تم تخطي ${failed} من ${total} محادثات حديثة أثناء فهرسة المكتبة.`
  },
  artifactPreview: {
    missingTitle: 'العنصر غير متاح',
    missingBody: 'لم يعد هذا العنصر في المكتبة.'
  },
  statusStack: {
    control: {
      waitSession: target => `في انتظار المحادثة ${target}`,
      loopDeferredNotice: 'الهدف النشط يتحكم بالمحادثة حالياً.',
      continuationBusy: 'تم استئناف الهدف — المحادثة مشغولة، أوقف الرد الحالي أولاً (زر الإيقاف أو Esc) للمتابعة',
      controlUnavailable: msg => `عناصر تحكم المحادثة غير متاحة: ${msg}`
    },
    coding: {
      agentShip: 'اطلب من الوكيل فتح PR'
    }
  },
  onboarding: {
    headerTitle: 'لنُعِدّ وكيلك'
  },
  interfaceMode: {
    hint: 'يغيّر ما يظهر، وليس ما يستطيع الوكيل فعله.',
    simple: {
      description: 'للتحدث مع وكيلك. الشريط الجانبي والدردشة؛ بلا طرفية أو لوحات ملفات أو فروقات.'
    }
  },
  zones: {
    closeRunningBody:
      'هذه المحادثة ما زالت تعمل (أو تنتظر إدخالك). إغلاق التبويب يخفيها فقط — ستحتفظ المحادثة بتقدمها ويمكن إعادة فتحها من الشريط الجانبي.'
  }
}
