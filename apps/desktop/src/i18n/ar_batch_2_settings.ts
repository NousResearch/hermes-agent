// Arabic settings overrides, split to keep catalog modules reviewable.
import type { TranslationOverrides } from './define-locale'

export const arBatch2Settings = {
  appearance: {
    appActionsDesc: 'مكان «الإعدادات» و«التخطيط» وHUD في شريط العنوان. «يمين» يترك مساحة لعلامات التبويب على اليسار.',
    appActionsLeft: 'يسار',
    appActionsRight: 'يمين',
    appActionsTitle: 'إجراءات التطبيق',
    chatFontDesc:
      'اختر خطًا مثبتًا للدردشة وبقية التطبيق. مفيد لخطوط سهولة القراءة مثل OpenDyslexic؛ اتركه فارغًا لاستخدام خط السمة.',
    chatFontPlaceholder: 'OpenDyslexic أو مجموعة خطوط CSS',
    chatFontPreview: 'معاينة',
    chatFontReset: 'استخدام خط السمة',
    chatFontSample: 'أبجد هوز حطي كلمن سعفص قرشت ثخذ ضظغ. 0123456789',
    chatFontTitle: 'خط الدردشة',
    pet: {
      roamDesc: 'دع الحيوان الأليف يتجول في النافذة من تلقاء نفسه أثناء الخمول.',
      roamTitle: 'التجول'
    },
    sessionDensityComfortable: 'مريح',
    sessionDensityCompact: 'مضغوط',
    sessionDensityDesc: 'اختر مقدار السياق الظاهر أسفل عناوين الجلسات في الشريط الجانبي.',
    sessionDensityDetailed: 'مفصّل',
    sessionDensityTitle: 'كثافة قائمة الجلسات',
    tabStripAlways: 'دائمًا',
    tabStripAuto: 'تلقائي',
    tabStripDesc:
      'إظهار علامات التبويب أعلى المنطقة. يخفيها «تلقائي» عند وجود جزء واحد فقط ما لم تكن هناك دردشة أخرى أو منطقة بطاقات مفتوحة.',
    tabStripNever: 'أبدًا',
    tabStripTitle: 'شريط علامات التبويب',
    terminalFontDesc:
      'اختر خطًا مثبتًا لـ Terminal في تطبيق Desktop. تعرض خطوط Nerd Fonts أيقونات Powerlevel10k وأيقونات الصدفة؛ اتركه فارغًا لاستخدام JetBrains Mono المضمّن.',
    terminalFontPlaceholder: 'MesloLGS NF أو مجموعة خطوط CSS',
    terminalFontPreview: 'معاينة المحارف',
    terminalFontReset: 'استخدام الافتراضي',
    terminalFontTitle: 'خط Terminal',
    themeSearchPlaceholder: 'ابحث في سماتك أو في VS Code Marketplace…',
    uiScaleDesc: percent =>
      `يغيّر حجم النصوص وعناصر التحكم في التطبيق كله. يعمل أيضًا Cmd/Ctrl مع + و- و0. الحالي: ${percent}%.`,
    uiScaleTitle: 'مقياس الواجهة'
  },
  billing: {
    amountValidation: {
      decimal: label => `${label}: أدخل مبلغًا بالدولار بحد أقصى منزلتين عشريتين.`,
      greaterThanThreshold: 'يجب أن يكون مبلغ إعادة التعبئة أكبر من الحد الأدنى.',
      maximum: (label, amount) => `${label}: الحد الأقصى ${amount}.`,
      minimum: (label, amount) => `${label}: الحد الأدنى ${amount}.`,
      positive: label => `${label}: يجب أن يكون المبلغ أكبر من $0.`,
      reloadTo: 'التعبئة إلى'
    },
    autoReload: {
      cancel: 'إلغاء',
      disable: 'تعطيل',
      manage: 'إدارة',
      reloadTo: 'التعبئة إلى',
      reloadToAria: 'مبلغ إعادة التعبئة التلقائية',
      save: 'حفظ',
      saving: 'جارٍ الحفظ…',
      threshold: 'الحد الأدنى',
      thresholdAria: 'الحد الأدنى لإعادة التعبئة التلقائية',
      turnOff: 'إيقاف',
      turnOffConfirm: 'إيقاف إعادة التعبئة التلقائية؟',
      turnedOff: 'تم إيقاف إعادة التعبئة التلقائية.',
      updated: 'تم تحديث إعادة التعبئة التلقائية.'
    },
    buyCredits: {
      added: amount => `تمت إضافة ${amount}. جارٍ تحديث الرصيد.`,
      buyButton: 'شراء',
      customAmount: 'مبلغ أرصدة مخصص',
      openPortal: 'فتح البوابة',
      processing: 'جارٍ المعالجة… التحقق من التسوية',
      retry: 'إعادة المحاولة',
      title: 'شراء أرصدة الآن'
    },
    charge: {
      added: amount => (amount ? `تمت إضافة $${amount}.` : 'تمت إضافة أرصدة.'),
      authenticationRequired: 'يتطلب بنكك تحققًا (3DS). أكمله عبر البوابة لإتمام هذه العملية.',
      checkBody: 'تعذّر التحقق من العملية.',
      checkTitle: 'تعذّر التحقق من العملية',
      declined: 'رُفضت بطاقتك. جرّب بطاقة أخرى عبر البوابة.',
      expired: 'انتهت صلاحية بطاقتك. حدّثها عبر البوابة.',
      failedBody: reason => `لم تتم العملية (${reason}).`,
      failedTitle: 'فشلت العملية',
      timeoutBody: 'قد تُسوّى العملية لاحقًا. تحقق من البوابة قبل إعادة المحاولة.',
      timeoutTitle: 'ما زالت قيد المعالجة بعد 5 دقائق',
      unconfirmedBody: message => `${message} نتيجة عمليتك الأخيرة غير مؤكدة — تحقق من رصيدك/سجلك قبل إعادة المحاولة.`,
      unconfirmedTitle: 'نتيجة العملية غير مؤكدة',
      untrackedBody: 'قبلت خدمة الفوترة الطلب لكنها لم تُرجع معرّف عملية.',
      untrackedTitle: 'تعذّر تتبع العملية'
    },
    creditsPerMonth: amount => `${amount} أرصدة/شهر`,
    errors: {
      cliBillingDisabled: {
        message: 'الإنفاق عن بُعد معطّل لهذا الحساب — يمكن لمسؤول الفوترة تفعيله من صفحة Hermes Agent في البوابة.',
        title: 'الإنفاق عن بُعد معطّل'
      },
      consentRequired: {
        message: 'أكّد هذه البطاقة لرسوم Terminal في البوابة',
        title: 'يلزم تأكيد البطاقة'
      },
      default: {
        message: 'فشل طلب الفوترة.',
        title: 'فشل طلب الفوترة'
      },
      endpointUnavailable: {
        message: 'أعادت نقطة نهاية الفوترة استجابة غير JSON (قد لا تكون متاحة في هذا النشر).',
        title: 'نقطة نهاية الفوترة غير متاحة'
      },
      idempotencyConflict: {
        message: '🔴 استُخدم مفتاح العملية هذا لمبلغ مختلف. ابدأ عملية تعبئة جديدة.',
        title: 'ابدأ عملية تعبئة جديدة'
      },
      insufficientScope: {
        message: 'يتطلب هذا السماح بـ«الإنفاق عن بُعد». ابدأ عملية تعبئة للسماح به ثم أعد المحاولة.',
        title: 'الإنفاق عن بُعد يحتاج إلى موافقة'
      },
      monthlyCapExceeded: {
        messageHeadroom: remaining => `🔴 بلغت الحد الشهري للإنفاق — المتبقي ${remaining} من الهامش.`,
        messageReached: '🔴 بلغت الحد الشهري للإنفاق.',
        title: 'بلغت الحد الشهري للإنفاق'
      },
      noPaymentMethod: {
        message:
          '💳 لا توجد بطاقة محفوظة لرسوم Terminal بعد. أعدّ واحدة عبر البوابة (شراء الأرصدة لمرة واحدة لا يحفظ بطاقة قابلة لإعادة الاستخدام).',
        title: 'لا توجد بطاقة محفوظة'
      },
      orgAccessDenied: {
        message: 'رمز الوصول هذا غير مرتبط بمؤسسة يمكنك إدارتها',
        title: 'تم رفض الوصول إلى المؤسسة'
      },
      rateLimited: {
        message: mins =>
          mins > 0
            ? `🟡 عمليات كثيرة جدًا الآن (أعد المحاولة خلال ~${mins} دقيقة). هذا ليس فشل دفع.`
            : '🟡 عمليات كثيرة جدًا الآن. هذا ليس فشل دفع.',
        title: 'عمليات كثيرة جدًا الآن'
      },
      remoteSpendingReconnect: who => `${who} أعد الاتصال من الإعدادات -> البوابة لإعادة تفويض هذا الجهاز.`,
      remoteSpendingRevoked: {
        messageByAdmin: 'أوقف أحد المسؤولين الإنفاق عن بُعد لهذا الجهاز.',
        messageBySelf: 'أوقفتَ الإنفاق عن بُعد لهذا الجهاز.',
        title: 'تم إيقاف الإنفاق عن بُعد'
      },
      roleRequired: {
        message: 'تتطلب إضافة الأموال مسؤولًا أو مالكًا للمؤسسة. اطلب من أحد المسؤولين، أو أدرها عبر البوابة.',
        title: 'يلزم دور مسؤول'
      },
      sessionRevoked: {
        message: 'تم تسجيل خروج جلستك. سجّل الدخول مجددًا من الإعدادات → البوابة.',
        title: 'تم تسجيل خروج الجلسة'
      },
      stripeUnavailable: {
        message: mins =>
          mins > 0
            ? `يواجه Stripe مشكلة — أعد المحاولة خلال ~${mins} دقيقة`
            : 'يواجه Stripe مشكلة — أعد المحاولة بعد قليل',
        title: 'يواجه Stripe مشكلة'
      },
      timeout: {
        message: 'انتهت مهلة طلب الفوترة.',
        title: 'انتهت مهلة طلب الفوترة'
      },
      transport: {
        message: 'فشل طلب الفوترة قبل الوصول إلى البوابة.',
        title: 'فشل اتصال الفوترة'
      },
      upgradeCapExceeded: {
        message: 'بلغت حد تغيير الخطة اليومي — حاول غدًا',
        title: 'بلغت حد تغيير الخطة اليومي'
      }
    },
    freeTier: {
      caption:
        'يعمل على nous/welcome مع الموصلات المشمولة. تسجيل الدخول يحفظ موصلاتك ويضيف الأدوات التي تحتاج حسابًا وكل النماذج الأخرى.',
      connectors: 'الموصلات',
      footnote: 'الخطة المجانية بلا رصيد وبلا أي مبلغ مستحق. تظهر الدفعات والاستخدام عند تسجيل الدخول بحساب Nous.',
      included: 'مشمول',
      message: 'سجّل الدخول بحساب Nous لفتح مزيد من النماذج والأدوات.',
      model: 'النموذج',
      name: 'Nous · الخطة المجانية',
      plan: 'الخطة المجانية',
      signIn: 'تسجيل الدخول',
      title: 'أنت على خطة Nous المجانية'
    },
    perMonth: amount => `${amount}/شهر`,
    plan: {
      alreadyOn: name => `أنت بالفعل على ${name} — لا شيء لتغييره.`,
      backAria: 'العودة إلى الفوترة',
      cancel: 'إلغاء',
      cannotChange: 'لا يمكن إجراء هذا التغيير هنا.',
      changePlan: 'تغيير الخطة',
      checkingChange: 'جارٍ التحقق من هذا التغيير…',
      confirmDowngrade: 'تأكيد التخفيض',
      current: 'الخطة الحالية',
      downgrade: 'تخفيض الخطة',
      effectScheduled: (targetName, effectiveAt, creditsDelta) =>
        `التغيير إلى ${targetName} — يسري ${effectiveAt}. لا يوجد خصم الآن؛ وتبقى على خطتك الحالية حتى ذلك الحين.${creditsDelta ? ` التغيير الشهري في الأرصدة: ${creditsDelta}.` : ''}`,
      empty: 'لا توجد خطط متاحة للتغيير إليها الآن.',
      notScheduleable: 'لا يمكن جدولة هذا التغيير هنا.',
      scheduled: 'مجدولة',
      scheduling: 'جارٍ الجدولة…',
      title: 'الخطط',
      tryAgain: 'إعادة المحاولة',
      undo: 'تراجع',
      undoing: 'جارٍ التراجع…',
      viewPlans: 'عرض الخطط'
    },
    preview: 'معاينة',
    sections: {
      invoices: 'الفواتير',
      paymentAndCredits: 'الدفع والأرصدة',
      plan: 'الخطة',
      usage: 'الاستخدام'
    },
    state: {
      autoRefill: {
        chargesDescription: (reloadTo, threshold) =>
          `يخصم ${reloadTo} تلقائيًا عندما ينخفض رصيدك إلى ما دون ${threshold}.`,
        distinctCardCaption: cardLabel => `تخصم إعادة التعبئة التلقائية من ${cardLabel} — طابقها عبر البوابة`,
        distinctCardFallback: 'بطاقة أخرى',
        enabledPill: 'مفعّلة',
        genericDescription: 'أبقِ رصيدك معبأً عندما ينخفض إلى ما دون الحد الأدنى.',
        manageCaption: 'أدر إعادة التعبئة التلقائية من البوابة.',
        notAvailablePill: '—',
        offPill: 'معطّلة',
        reconcileAction: 'مطابقة ↗',
        title: 'التعبئة عند الانخفاض',
        turnOnCaption: 'فعّل إعادة التعبئة التلقائية من البوابة'
      },
      buyCredits: {
        description: 'خصم واحد على بطاقتك يُضاف إلى رصيدك اليوم.'
      },
      notice: {
        loggedOut: {
          action: 'تسجيل الدخول',
          message: 'سجّل الدخول بحساب Nous لعرض رصيدك وخطتك واستخدامك هنا.',
          title: 'وصّل حساب Nous الخاص بك'
        },
        noCard: {
          action: 'إضافة بطاقة ↗',
          message: 'يظل شراء أرصدة التعبئة وإعادة التعبئة التلقائية معطّلَين حتى تُسجَّل بطاقة. أضف واحدة عبر البوابة.',
          title: 'لا توجد طريقة دفع مسجّلة'
        },
        openPortal: 'فتح البوابة ↗'
      },
      paymentMethod: {
        addAction: 'إضافة طريقة دفع',
        description: 'أدر البطاقة المستخدمة للتعبئة وتجديدات الاشتراك.',
        provenance: {
          autoRefill: 'بطاقة إعادة التعبئة التلقائية',
          customerDefault: 'البطاقة الافتراضية للعميل',
          subPin: 'بطاقة الاشتراك',
          suffix: label => ` - ${label}`
        },
        title: 'طريقة الدفع',
        updateAction: 'تحديث'
      },
      planCard: {
        adjustPlanAction: 'تعديل الخطة ↗',
        cancellationCaption: when => `يُلغى في ${when}.`,
        chooseAction: 'اختيار ↗',
        downgradeCaption: (tierName, when) => `يتغيّر إلى ${tierName} في ${when}.`,
        freeTier: 'مجاني',
        noSubscriptionCaption: 'لا يوجد اشتراك نشط — النماذج المدفوعة تُخصم من أرصدة الشحن.',
        renewsCaption: date => `يتجدّد في ${date}`,
        unavailableCaption: 'تفاصيل الاشتراك غير متاحة؛ ويظل فتح البوابة متاحًا.'
      },
      usage: {
        monthlyCap: {
          barLabel: 'سقف الإنفاق الشهري المستخدم',
          captionDefault: 'الحد الافتراضي',
          captionSpending: 'الإنفاق الشهري عن بُعد',
          title: 'سقف الإنفاق الشهري',
          valueUsed: (spent, limit) => `${spent} من ${limit} مستخدم`
        },
        subscriptionCredits: {
          barLabel: 'أرصدة الاشتراك المتبقية',
          captionResets: date => `يتجدّد في ${date}`,
          title: 'أرصدة الاشتراك',
          valueOf: (remaining, monthly) => `${remaining} من ${monthly} متبقٍ`,
          valueOver: (remaining, monthly, over) => `${remaining} من ${monthly} متبقٍ · تجاوز ${over}`
        },
        topupCredits: {
          caption: 'لا تنتهي صلاحيتها',
          title: 'أرصدة الشحن الإضافي'
        }
      }
    },
    stepUp: {
      deniedBody: 'اكتمل التحقق دون السماح بالإنفاق عن بُعد لهذا Terminal.',
      deniedTitle: 'لم تتم الموافقة على التحقق',
      dismiss: 'تجاهل',
      openVerification: 'فتح صفحة التحقق',
      successBody: 'الإنفاق عن بُعد مسموح لهذا Terminal.',
      successTitle: 'اكتمل التحقق',
      verify: 'تحقّق للمتابعة',
      waiting: 'بانتظار رابط التحقق…'
    },
    summary: {
      autoRefill: 'إعادة الشحن التلقائي',
      balance: 'الرصيد',
      plan: 'الخطة'
    },
    title: 'الفوترة',
    usage: {
      title: 'الاستخدام'
    },
    usageLabel: label => `استخدام ${label}`
  },
  computerUse: {
    accessibility: 'إمكانية الوصول',
    driverHealth: 'حالة برنامج التشغيل',
    screenRecording: 'تسجيل الشاشة'
  },
  config: {
    alwaysExternalLinksDesc:
      'افتح كل رابط تنقر عليه في متصفح النظام بدلًا من المتصفح داخل التطبيق. خيار «فتح في المتصفح داخل التطبيق» في قائمة النقر بالزر الأيمن يظل يعمل.',
    alwaysExternalLinksTitle: 'فتح الروابط دائمًا في المتصفح الخارجي',
    attachmentSizeDesc:
      'أقصى حجم لملف محلي يحمّله Desktop للمعاينة وإرفاق الصور، بالميغابايت. الافتراضي 16. الإرفاق البعيد لغير الصور يستخدم حدًا منفصلًا مقداره 256 ميغابايت. ضبط قيمة عالية جدًا يحمّل الملف كاملًا في الذاكرة وقد يجمّد التطبيق أو ينهيه.',
    attachmentSizeLabel: 'أقصى حجم لتحميل المعاينة / الصور بالميغابايت',
    attachmentSizeTitle: 'أقصى حجم لتحميل المعاينة / الصور',
    attachmentSizeUnit: 'ميغابايت',
    builtinOnly: 'المدمج فقط',
    disableF12Desc: 'منع F12 من فتح أدوات المطوّر. يظل Ctrl+Shift+I (أو Cmd+Opt+I على Mac) يعمل.',
    disableF12Title: 'تعطيل أدوات المطوّر F12',
    keepAwakeAlways: 'دائمًا',
    keepAwakeDesc:
      'منع هذا الجهاز من الدخول في وضع السكون. «أثناء العمل» يبقيه مستيقظًا فقط ما دام دور قيد التنفيذ، لذا تستمر التشغيلات الليلية دون إبقاء الحاسوب مستيقظًا طوال الأسبوع. يمكن أن تخفت الشاشة مع ذلك.',
    keepAwakeOff: 'إيقاف',
    keepAwakeTitle: 'إبقاء الحاسوب مستيقظًا',
    keepAwakeWhileWorking: 'أثناء العمل',
    showOptions: 'إظهار الخيارات',
    toolsetsWipeConfirm:
      'إزالة كل مجموعات الأدوات المفعّلة؟ سيؤدي ذلك إلى تعطيل الذاكرة وTerminal والبحث على الويب والتفويض ومعظم الأدوات الأخرى إلى أن تعيد تفعيلها.'
  },
  connections: {
    addConnection: 'إضافة اتصال',
    cancel: 'إلغاء',
    cloudAddHint:
      'نصيحة: تسجيل الدخول ضمن Hermes Cloud أعلاه يكتشف وكلاءك تلقائيًا — استخدم هذا النموذج فقط لتسجيل عنوان URL لنسخة معروفة يدويًا.',
    currentPill: 'الحالي',
    duplicateLocal: 'هذا التطبيق يدير اتصالًا محليًا بالفعل — لا يمكن أن يوجد سوى اتصال واحد.',
    duplicateSsh: label => `يوجد بالفعل اتصال بمضيف SSH هذا («${label}»).`,
    duplicateUrl: label => `يوجد بالفعل اتصال بعنوان URL لهذه البوابة («${label}»).`,
    editConnection: 'تعديل',
    empty: 'لا توجد اتصالات مسجّلة بعد.',
    headerAdd: 'إضافة ترويسة',
    headerRemove: 'إزالة',
    headerValuePlaceholder: 'القيمة',
    headerValueSaved: 'محفوظة — اتركه فارغًا للإبقاء عليها',
    headersDesc:
      'تُرسل مع كل طلب HTTP وWebSocket إلى هذه البوابة — لوكلاء الوصول مثل Cloudflare Access (CF-Access-Client-Id / CF-Access-Client-Secret). تُخزّن القيم مشفّرة. تُتجاهل الترويسات التي يديرها Hermes (Authorization، Cookie، Host…).',
    headersTitle: 'ترويسات إضافية للبوابة',
    intro: 'أدر هذا الجهاز وكل بوابة Hermes يمكنه الوصول إليها عبر اتصالات بعيدة أو SSH أو Cloud.',
    kindCloud: 'Hermes Cloud',
    kindCloudDesc: 'نسخة مستضافة تُكتشف عبر حساب Hermes Cloud الخاص بك.',
    kindLocal: 'محلي',
    kindLocalDesc: 'بيئة تشغيل Hermes التي يديرها هذا التطبيق.',
    kindRemote: 'بوابة بعيدة',
    kindRemoteDesc: 'بوابة Hermes يمكن الوصول إليها عبر HTTP(S) — شبكة محلية أو Tailscale أو الإنترنت.',
    kindSsh: 'SSH',
    kindSshDesc: 'تثبيت Hermes يُتوصل إليه عبر SSH.',
    labelDesc: 'مطلوب. يظهر في كل موضع تظهر فيه هذه النسخة؛ ويجب أن يكون فريدًا (مثل «Homelab»، «Work laptop»).',
    labelPlaceholder: 'Homelab',
    labelTitle: 'الاسم',
    launchModeDesc: 'عند الإيقاف، تُفتح Sessions على البوابة الأساسية.',
    launchModeTitle: 'عند بدء التشغيل، العودة إلى Sessions على البوابة المستخدمة آخر مرة',
    loadFailed: 'تعذّر تحميل الاتصالات',
    localAddHint: 'المحلي غير متاح: الاتصال المحلي المُدار موجود بالفعل (لا يوجد سوى واحد دائمًا).',
    makePrimary: 'تعيين كأساسي',
    managedPill: 'يديره التطبيق',
    noSearchResults: 'لا توجد بوابات تطابق بحثك.',
    primaryPill: 'أساسي',
    removeConfirmDesc: label =>
      `ستُزال «${label}» من هذا التطبيق. لن تتأثر النسخة نفسها — يمكنك إضافتها مرة أخرى في أي وقت.`,
    removeConfirmTitle: 'إزالة هذا الاتصال؟',
    removeConnection: 'إزالة',
    removeFailed: 'تعذّرت إزالة الاتصال',
    sameBackendHint: label => `نفس الخادم الخلفي الخاص بـ«${label}»`,
    save: 'حفظ الاتصال',
    saveFailed: 'تعذّر حفظ الاتصال',
    saving: 'جارٍ الحفظ…',
    searchPlaceholder: 'ابحث عن بوابة…',
    sshHostTitle: 'مضيف SSH',
    stagedNote:
      'تبقى كل بوابة متصلة. تبقى الملفات الشخصية والمحادثات والمراسلة ومهام cron مع بوابتها، ويمكن لـ Sessions عرضها جميعًا في وقت واحد.',
    testConnection: 'اختبار',
    testFailed: 'فشل اختبار الاتصال',
    testOk: 'يمكن الوصول إليه',
    title: 'البوابات المسجّلة',
    updateAll: 'تحديث كل النسخ',
    updateAllDone: 'تم إرسال التحديثات',
    updateAllFailed: 'فشل توزيع التحديث',
    updateAllRunning: 'جارٍ تحديث كل النسخ…',
    updateSkippedCloud: 'تُدار عبر Hermes Cloud',
    urlTitle: 'عنوان URL للبوابة'
  },
  customEndpoints: {
    activationFailed: 'فشل التفعيل',
    active: 'نشط',
    addTitle: 'إضافة نقطة نهاية',
    apiKeySet: 'تم تعيين مفتاح API',
    apiMode: 'وضع API',
    autoDetect: 'اكتشاف تلقائي',
    contextPlaceholder: 'تلقائي',
    couldNotLoad: 'تعذّر تحميل نقاط النهاية المخصصة',
    deleteConfirm: name => `حذف ${name}؟`,
    deleteEndpoint: 'حذف نقطة النهاية',
    deleteFailed: 'فشل الحذف',
    editTitle: 'تعديل نقطة النهاية',
    emptyDescription: 'أضف نقطة نهاية متوافقة مع OpenAI أدناه.',
    emptyTitle: 'لا توجد نقاط نهاية مخصصة',
    endpointReachable: 'يمكن الوصول إلى نقطة النهاية.',
    endpointReachableModels: (reachable, count) => `${reachable} تمّ العثور على ${count} نموذج.`,
    endpointReachableTransport: transport => `يمكن الوصول إلى نقطة النهاية (تم تقديم مسار ${transport}).`,
    endpointSaved: 'تم حفظ نقطة النهاية المخصصة.',
    endpointValidationFailed: 'فشل التحقق من نقطة النهاية.',
    fields: {
      apiKey: 'مفتاح API',
      apiKeyNewPlaceholder: 'اتركه فارغًا للإبقاء على المفتاح الحالي',
      apiKeyPlaceholder: 'اختياري',
      context: 'السياق',
      defaultModel: 'النموذج الافتراضي',
      discoverModels: 'اكتشاف النماذج',
      endpointUrl: 'عنوان URL لنقطة النهاية',
      name: 'الاسم',
      providerId: 'معرّف المزوّد',
      useNewChats: 'استخدام للمحادثات الجديدة'
    },
    namePlaceholder: 'Axet Proxy',
    newEndpoint: 'نقطة نهاية جديدة',
    save: 'حفظ',
    saveFailed: 'فشل الحفظ',
    test: 'اختبار',
    title: 'نقاط النهاية المخصصة',
    use: 'استخدام',
    validationFailed: 'فشل التحقق'
  },
  envActions: {
    manageInKeys: 'الإدارة في مفاتيح API'
  },
  fieldDescriptions: {
    'browser.useRealProfile':
      'التصفح المحلي يستخدم تسجيلات دخولك الحقيقية. ينسخ Hermes ملف تعريف متصفحك الافتراضي (ملفات تعريف الارتباط وتسجيلات الدخول والتفضيلات) إلى نسخة مُدارة ويشغّلها بمتصفح Chromium المضمّن — فلا يُفتح ملفك الشخصي الحقيقي مباشرة أبدًا، وتُحدّث النسخة منه في كل تشغيل. يتيح ذلك أيضًا للوكيل فتح جلسة محلية بملف تعريف حقيقي عند الطلب حتى مع تهيئة خادم متصفح سحابي. المتصفحات المدعومة هي Chromium فقط (Chrome، Edge، Brave، Brave Origin، Chromium)؛ وإذا كان الافتراضي غير Chromium يفشل الأمر برسالة واضحة. معطّل افتراضيًا.',
    'desktop.repoScanEnabled': 'فحص المجلدات المحلية بحثًا عن مستودعات Git لعرضها في Projects.',
    'desktop.repoScanExcludePaths': 'المجلدات وتوابعها التي تُستثنى أثناء اكتشاف المستودعات.',
    'desktop.repoScanRoots': 'المجلدات المراد فحصها. اتركه فارغًا لفحص مجلدك الرئيسي.',
    'stt.echoTranscripts': 'نشر النص الخام 🎙️ للرسائل الصوتية في المحادثة.',
    'stt.streaming': 'إظهار النص أثناء تحدثك (OpenAI، xAI، ElevenLabs). يعود إلى التسجيل عند أي فشل.',
    'tts.xai.autoSpeechTags': 'السماح لنموذج لغوي بإدراج وسوم صوتية تعبيرية ([laughing]، [sighs]) في النص قبل التوليد.',
    'tts.xai.bitRate': 'معدل بت MP3 بالبت في الثانية. يُطبّق فقط عندما يكون الترميز mp3.',
    'tts.xai.optimizeStreamingLatency': 'مقايضة بين زمن الاستجابة والجودة. 0 = أفضل جودة، 2 = أقل زمن استجابة.',
    'tts.xai.sampleRate': 'معدل عينات الصوت بالهرتز. أعلى = جودة أفضل وملفات أكبر.',
    'tts.xai.speed': 'سرعة التشغيل. 0.7 = أبطأ، 1.0 = عادي، 1.5 = أسرع.',
    'voice.gptLive.instructions':
      'جُمل إضافية لشخصية الصوت المباشر (النبرة، الإيقاع، اللغة). يحتفظ Hermes بمطالبه النظامية الخاصة.',
    'voice.gptLive.voice': 'الصوت لوضع GPT-Live. تُقبل معرّفات الأصوات المخصصة.',
    'voice.voiceChatMode':
      'chained: تحويل الكلام إلى نص ← Hermes ← تحويل النص إلى كلام باستخدام المزوّدين أدناه. gpt-live: نموذج صوتي واحد ثنائي الاتجاه من OpenAI (gpt-live-1) يستمع ويتحدث، ويحيل كل طلب حقيقي إلى Hermes — وأي نموذج مختار يجيب بكامل مجموعة الأدوات. يحتاج إلى مفتاح API من OpenAI؛ وتُحتسب طبقة الصوت 0.05 دولار لكل دقيقة.'
  },
  fieldLabels: {
    'browser.useRealProfile': 'استخدام ملف تعريف متصفحي الحقيقي',
    'desktop.repoScanEnabled': 'الاكتشاف التلقائي للمستودعات',
    'desktop.repoScanExcludePaths': 'مسارات المستودعات المستثناة',
    'desktop.repoScanRoots': 'جذور اكتشاف المستودعات',
    'stt.deepinfra.model': 'نموذج DeepInfra STT',
    'stt.echoTranscripts': 'إظهار النصوص',
    'stt.openai.streamingModel': 'نموذج النسخ المباشر من OpenAI',
    'stt.streaming': 'النسخ المباشر',
    'stt.xai.model': 'نموذج xAI STT',
    'tts.deepinfra.model': 'نموذج DeepInfra TTS',
    'tts.deepinfra.voice': 'صوت DeepInfra',
    'tts.xai.autoSpeechTags': 'الوسوم الصوتية التلقائية في xAI',
    'tts.xai.bitRate': 'معدل البت في xAI',
    'tts.xai.optimizeStreamingLatency': 'تحسين زمن استجابة البث في xAI',
    'tts.xai.sampleRate': 'معدل العينات في xAI',
    'tts.xai.speed': 'سرعة التشغيل في xAI',
    'voice.gptLive.instructions': 'شخصية GPT-Live',
    'voice.gptLive.voice': 'صوت GPT-Live',
    'voice.voiceChatMode': 'وضع المحادثة الصوتية'
  },
  gateway: {
    cloudActive: 'نشط في هذه النافذة',
    cloudAgentProvisioning: 'جارٍ التهيئة…',
    cloudAgentsTitle: 'وكلاؤك',
    cloudConnect: 'اتصال',
    cloudConnectFailed: 'تعذّر الاتصال بذلك الوكيل',
    cloudConnectedPill: 'متصل',
    cloudConnectedTitle: 'متصل',
    cloudConnectedTo: name => `تم الاتصال بـ${name}.`,
    cloudConnecting: 'جارٍ الاتصال…',
    cloudDesc: 'سجّل الدخول مرة واحدة إلى Hermes Cloud واختر من الوكلاء في حسابك — دون لصق أي URL.',
    cloudDiscoverFailed: 'تعذّر تحميل وكلاء Hermes Cloud',
    cloudLoadingAgents: 'جارٍ تحميل وكلائك…',
    cloudNeedsSignIn: 'سجّل الدخول إلى Hermes Cloud لاكتشاف الوكلاء في حسابك.',
    cloudNoAgents: {
      after: '، ثم أعد التحديث.',
      before: 'لم يُعثر على وكلاء في هذا الحساب. أنشئ وكيلًا في ',
      linkText: 'بوابة Nous'
    },
    cloudOrgChange: 'تغيير المؤسسة',
    cloudOrgPickerTitle: 'اختر مؤسسة',
    cloudOrgRole: role => `الدور: ${role}`,
    cloudOrgSelect: 'اختيار',
    cloudRefresh: 'تحديث',
    cloudSavedDesc:
      'استخدم بوابة محفوظة دون تغيير افتراضيك. سجّل الدخول أدناه لإضافة نسخ. أدر الأسماء وتسجيل الدخول من قائمة الاتصالات المحفوظة.',
    cloudSavedTitle: 'بوابات Cloud المحفوظة',
    cloudSignIn: 'تسجيل الدخول إلى Hermes Cloud',
    cloudSignInFailed: 'فشل تسجيل الدخول إلى Hermes Cloud',
    cloudSignInTitle: 'Hermes Cloud',
    cloudSignedIn: 'تم تسجيل الدخول إلى Hermes Cloud',
    cloudSignedInDesc: 'أنت مسجّل الدخول. اختر وكيلًا أدناه؛ وتُحدّث الجلسة تلقائيًا.',
    cloudSignedOutMessage: 'تم مسح جلسة Hermes Cloud.',
    cloudSignedOutTitle: 'تم تسجيل الخروج من Hermes Cloud',
    cloudStatusLabel: status => `الحالة: ${status}`,
    cloudTitle: 'Hermes Cloud',
    cloudUseSaved: 'استخدام البوابة',
    modeTitle: 'وضع الاتصال',
    plainTextConfirmAction: 'الحفظ كنص عادي',
    plainTextConfirmDesc:
      'لم يُعثر على خدمة سلسلة مفاتيح في نظام التشغيل على هذا الجهاز، لذا سيُحفظ الرمز غير مشفّر في ملف إعدادات الاتصال الخاص بالتطبيق، ويمكن لأي عملية تعمل باسم هذا المستخدم قراءته. ثبّت سلسلة مفاتيح نظامك أو فعّلها (GNOME Keyring أو KWallet على Linux) لتخزين مشفّر.',
    plainTextConfirmTitle: 'حفظ رمز البوابة كنص عادي؟',
    plainTextStoredDesc:
      'التخزين الآمن غير متاح، لذا يُخزّن الرمز المحفوظ غير مشفّر في ملف إعدادات الاتصال بالتطبيق على هذا الجهاز. ثبّت سلسلة مفاتيح نظامك أو فعّلها (GNOME Keyring أو KWallet على Linux) لتشفيره.',
    plainTextStoredTitle: 'الرمز مخزّن كنص عادي',
    remoteAuthHint:
      'البوابات المستضافة تستخدم OAuth أو اسم مستخدم وكلمة مرور؛ أما المستضافة ذاتيًا فقد تستخدم رمز جلسة.',
    sshButtonsHint: 'يُطبّق الحفظ عند التشغيل التالي. أما اتصال فيعيد الاتصال الآن.',
    sshConnect: 'اتصال',
    sshDesc:
      'يُشغّل Hermes على الخادم البعيد عبر SSH ويُنفّق إلى هذا التطبيق — لا شيء تشغّله أو تعرّضه بنفسك. يتطلب وصول SSH عاملاً بالمفاتيح إلى المضيف.',
    sshErrAuth:
      'فشلت مصادقة SSH. حمّل مفتاحك إلى ssh-agent (ssh-add) أو عيّن IdentityFile في ~/.ssh/config — فـHermes يشغّل ssh دون تفاعل.',
    sshErrHostKey: 'تغيّر مفتاح المضيف منذ آخر اتصال. تأكد من أن ذلك متوقع، ثم شغّل ssh-keygen -R <host> وأعد الاتصال.',
    sshErrInteractiveAuth:
      'يتطلب Tailscale SSH تحققًا تفاعليًا في المتصفح. في Terminal، شغّل `ssh <host> true`، وأكمل التحقق، ثم أعد المحاولة — فـHermes يشغّل SSH دون تفاعل.',
    sshErrNotInstalled:
      'Hermes غير مثبّت على المضيف البعيد. ثبّته هناك (curl -fsSL https://hermes-agent.nousresearch.com/install.sh | sh) أو عيّن مسار Hermes.',
    sshErrPlatform: 'منصة بعيدة غير مدعومة. يدعم وضع SSH في Hermes Desktop مضيفات Linux وmacOS وWindows.',
    sshErrTimeout: 'انتهت مهلة اتصال SSH. قد يكون المضيف غير قابل للوصول أو في وضع السكون.',
    sshErrUnknown: 'فشل اتصال SSH.',
    sshErrUnreachable: 'تعذّر الوصول إلى ذلك المضيف عبر SSH. تحقق من المضيف والمنفذ وشبكتك.',
    sshErrUpdateRequired: 'حدّث Hermes على المضيف البعيد قبل الاتصال عبر Desktop SSH.',
    sshHermesPathDesc: 'المسار الكامل إلى ملف hermes التنفيذي البعيد. فارغ = اكتشاف تلقائي.',
    sshHermesPathPlaceholder: 'اكتشاف تلقائي',
    sshHermesPathTitle: 'مسار Hermes (اختياري)',
    sshHostCustom: 'مخصص (إدخال يدوي)…',
    sshHostDesc: 'user@host، أو اسم مستعار Host من ~/.ssh/config.',
    sshHostPick: 'اختر مضيفًا…',
    sshHostPickDesc: 'اسم مستعار Host من ~/.ssh/config، أو مخصص لكتابته.',
    sshHostPickTitle: 'المضيف',
    sshHostTitle: 'المضيف',
    sshIncompleteHost: 'أدخل مضيف SSH قبل الاتصال.',
    sshKeyDesc: 'مسار المفتاح الخاص. فارغ = ssh-agent أو ~/.ssh/config.',
    sshKeyTitle: 'ملف الهوية',
    sshPortDesc: 'فارغ = 22 أو المنفذ المحدد في ~/.ssh/config.',
    sshPortTitle: 'المنفذ',
    sshReachable: (host, platform) => `يمكن الوصول: ${host} (${platform}) — تم العثور على Hermes`,
    sshTestConnection: 'اختبار SSH',
    sshTitle: 'الاتصال عبر SSH',
    sshTrustHint: 'يُوثق بمفتاح المضيف الأول ويُثبّت؛ وأي تغيير لاحق يُفشل الاتصال.',
    sshUserDesc: 'فارغ = ~/.ssh/config أو مستخدمك الحالي.',
    sshUserPlaceholder: 'من ~/.ssh/config',
    sshUserTitle: 'المستخدم',
    webappOauthPaste: {
      placeholder: 'http://127.0.0.1:…/webapp/gateway-connections/oauth/callback?…'
    }
  },
  localModels: {
    activateDoneToast: model => `المحادثات الجديدة تستخدم ${model}.`,
    activateFailed: model => `تعذّر التبديل إلى ${model}`,
    activating: 'جارٍ البدء…',
    activeDetail: 'المحادثات الجديدة تستخدم هذا النموذج — يُحمّل عند إرسال أول رسالة',
    activeNotLoaded: 'يُحمّل عند أول رسالة',
    activePill: 'افتراضي',
    addedByYou: 'أضفتها بنفسك',
    browseAlreadyDownloaded: 'تم تنزيله بالفعل.',
    browseDownloadAria: 'تنزيل {name}',
    browseDownloadStarted: 'جارٍ تنزيل {name}',
    browseDownloads: 'مرات التنزيل',
    browseFitUnknown: 'التوافق غير معروف',
    browseGated: 'يتطلب تسجيل الدخول إلى Hugging Face',
    browseHint: 'ابحث في كل Hugging Face. النماذج التي تنزّلها هنا تُقاس تلقائيًا وفق جهازك، لكننا لا نختبرها.',
    browseLikes: 'إعجابات',
    browseListing: 'جارٍ قراءة ملفات النموذج',
    browseNoGguf: 'لم يُعثر على ملفات نموذج متوافقة.',
    browsePlaceholder: 'ابحث عن النماذج بالاسم أو المؤلف…',
    browseRefresh: 'تحديث',
    browseSearching: 'جارٍ البحث في Hugging Face',
    browseShowFiles: 'إظهار الملفات',
    browseTitle: 'العثور على المزيد من النماذج',
    connectionChanged: 'تغيّر اتصال النماذج المحلية',
    deleteAction: 'حذف النموذج',
    deleteConfirm: model => `حذف ${model} من القرص؟`,
    deleteFailed: 'فشل الحذف',
    deleted: model => `تم حذف ${model}.`,
    downloadAction: size => `تنزيل · ${size}`,
    downloadDoneToast: model => `${model} جاهز.`,
    downloadEta: time => `متبقٍ ~${time}`,
    downloadEtaHours: (hours, minutes) => (minutes ? `${hours} س ${minutes} د` : `${hours} س`),
    downloadEtaMinutes: count => `${count} د`,
    downloadEtaSeconds: count => `${count} ث`,
    downloadFailed: model => `فشل تنزيل ${model}`,
    downloadPauseAction: 'إيقاف مؤقت',
    downloadPauseFailed: model => `تعذّر إيقاف تنزيل ${model} مؤقتًا`,
    downloadPausedLabel: 'متوقف مؤقتًا',
    downloadProgress: (done, total) => `${done} من ${total}`,
    downloadResumeAction: 'استئناف',
    downloadResumeFailed: model => `تعذّر استئناف تنزيل ${model}`,
    downloadSpeed: rate => `${rate}`,
    downloadStatusRunning: 'جارٍ التنزيل',
    downloaded: 'تم التنزيل',
    ejectFailed: 'تعذّر إفراغ النموذج من الذاكرة',
    ejectTip: 'تفريغ ذاكرة GPU (يُحمّل مجددًا مع الرسالة التالية)',
    ejected: 'تم إفراغ النموذج — تم تحرير ذاكرة GPU.',
    hardwareLoading: 'جارٍ فحص عتادك…',
    hardwareTitle: 'هذا الجهاز',
    installAction: 'تثبيت بيئة التشغيل',
    installDetail:
      'ينزّل محرك الاستدلال llama.cpp (بضع مئات من الميغابايت). النماذج التي تنزّلها تعمل بالكامل على هذا الجهاز — بلا حساب، ولا يغادر أي شيء حاسوبك.',
    installDoneToast: 'تم تثبيت بيئة التشغيل المحلية وهي جاهزة.',
    installFailed: 'فشل تثبيت بيئة التشغيل',
    installTitle: 'تثبيت بيئة التشغيل المحلية',
    installing: 'جارٍ تثبيت بيئة التشغيل…',
    loadedPill: 'في الذاكرة',
    loadingPill: 'جارٍ التحميل…',
    modelsTitle: 'النماذج',
    noRecommendationAction: 'تصفح النماذج',
    noRecommendationDetail:
      'يتطلب الإعداد التلقائي نموذجًا منسّقًا يعمل بالكامل في ذاكرة GPU أو الذاكرة الموحدة بالسرعة الكاملة. يمكنك مع ذلك اختيار نموذج أدناه أو تصفح المزيد.',
    noRecommendationTitle: 'لا توصية تلقائية لهذا الجهاز',
    pillFitsGpu: 'يناسب ذاكرة GPU',
    pillFullContext: max => `سياق كامل ${max}`,
    pillFullContextTip: 'يعمل بنافذة السياق الكاملة للنموذج من البداية',
    pillGrowsTip: 'يتوسّع تلقائيًا حسب حاجة محادثتك إلى مساحة أكبر',
    pillTooBig: 'أكبر من طاقة هذا الجهاز',
    pillUpTo: max => `حتى ${max} سياق`,
    pillUsesRam: 'يستخدم RAM النظام',
    pillVision: 'يرى الصور',
    placementResident: 'كله على GPU',
    placementResidentTip: 'يعمل بالكامل في ذاكرة GPU عند نافذة السياق هذه — بالسرعة الكاملة.',
    placementSpilled: 'جزئيًا في RAM',
    placementSpilledTip:
      'جزء من هذا النموذج يعمل من RAM النظام — يعمل لكنه أبطأ. نسخة أكثر إحكامًا أو سياق أصغر سيتسع بالكامل.',
    quickstartAction: 'أعدّه لي',
    quickstartConfigure: 'سأختار بنفسي',
    quickstartDetail: (model, size) =>
      `بنقرة واحدة يُعدّ كل شيء: المحرك المحلي، و${model} (تنزيل ${size})، والافتراضي لمحادثاتك الجديدة. لا شيء يغادر هذا الحاسوب.`,
    quickstartDetailReady: model => `بنقرة واحدة يصبح ${model} افتراضيك للمحادثات الجديدة. كل شيء يعمل على هذا الجهاز.`,
    quickstartDoneToast: model => `تم إعداد ${model} — المحادثات الجديدة تعمل على هذا الجهاز.`,
    quickstartFailed: 'فشل إعداد النموذج المحلي',
    quickstartStageEngine: 'المحرك',
    quickstartStageFinish: 'إنهاء',
    quickstartStageModel: 'النموذج',
    quickstartTitle: 'تشغيل نموذج على هذا الجهاز',
    ram: label => `ذاكرة ${label}`,
    recommended: 'موصى به',
    recommendedReason: {
      'best-quality-resident':
        'أعلى النماذج جودةً التي تعمل بالكامل على GPU بالسرعة الكاملة. يوازن الاختيار بين الجودة والسرعة المتوقعة على هذا العتاد.',
      'product-default': 'النموذج الافتراضي لهذا الجهاز، اختاره مصنّعه.',
      'speed-gated-quality':
        'نموذج أعلى جودة يناسب هذا الجهاز لكنه سيستجيب ببطء شديد بسبب نطاق تردد ذاكرته — هذا أفضل نموذج يبقى سريعًا.'
    },
    runtimeInstalled: 'تم تثبيت بيئة llama.cpp',
    runtimeInstalledDetail: (tag, backend) => `إصدار ${tag}، خادم ${backend}. يشغّل Hermes الخادم ويديره لك.`,
    runtimeReady: backend => `جاهز · ${backend}`,
    runtimeRunningDetail:
      'الخادم المحلي يعمل. إيقافه يحرّر كل ذاكرة GPU ويمنع المحادثات الجديدة من استخدام النماذج المحلية حتى تعيد تشغيله.',
    runtimeTitle: 'بيئة التشغيل المحلية',
    serverRunning: 'يعمل',
    serverStartFailed: 'تعذّر تشغيل الخادم المحلي',
    serverStarted: 'الخادم المحلي يعمل.',
    serverStopFailed: 'تعذّر إيقاف الخادم المحلي',
    serverStopped: 'تم إيقاف الخادم المحلي — تم تحرير ذاكرة GPU.',
    sideloadAlreadyPresent: 'موجود بالفعل في مكتبتك.',
    sideloadButton: 'إضافة ملف نموذج',
    sideloadDone: 'تمت إضافة {name}.',
    sideloadTitle: 'اختر ملف نموذج GGUF',
    startServer: 'تشغيل',
    stopServer: 'إيقاف',
    title: 'النماذج المحلية',
    unifiedMemory: 'ذاكرة موحدة',
    upToDateDetail: (tag, backend) => `يعمل llama.cpp ${tag} (${backend}).`,
    upToDateTitle: 'المحرك محدّث',
    updateAction: 'تحديث المحرك',
    updateDetail: (next, current) =>
      `إصدار أحدث من llama.cpp (${next}) جاهز للتثبيت — ولديك ${current}. تستمر النماذج بالعمل أثناء التنزيل.`,
    updateSameBuildDetail: tag =>
      `أصبح Hermes يضمّ llama.cpp \u2068${tag}\u2069 مع الملفات اللازمة لتشغيله. سيُعاد تنزيل المحرّك عند التحديث، وتبقى النماذج متاحة أثناء التنزيل.`,
    updateTitle: 'يتوفر تحديث للمحرك',
    updating: 'جارٍ تحديث المحرك…',
    useAction: 'استخدام',
    vram: label => `ذاكرة ${label} GPU`
  },
  managedUpdates: {
    alreadyRunning: 'التحديث قيد التنفيذ بالفعل',
    failed: 'فشل التحديث',
    intro:
      'حدّث تثبيتات SSH التي يديرها Desktop بشكل معاملاتي: تُصرّف الجلسات، وتُحدّث نسخة الشيفرة البعيدة، وتُستعاد كل الملفات الشخصية بإيصال مترابط.',
    partial: 'تم التحديث — فشلت الاستعادة',
    progress: 'جارٍ تصريف الجلسات وتحديث التثبيت البعيد واستعادة الملفات الشخصية…',
    receipt: (id, outcome) => `إيصال ${id} · ${outcome}`,
    receiptVersions: (pre, post) => `${pre} → ${post}`,
    refused: 'مرفوض',
    scopeNotRestored: (profile, error) => `لم يُستعد الملف الشخصي «${profile}»: ${error}`,
    scopesRestored: profiles => `الملفات الشخصية المستعادة: ${profiles}`,
    sshConnection: 'تثبيت SSH يديره Desktop',
    title: 'التحديثات المُدارة',
    update: 'تحديث',
    updated: 'تم التحديث',
    updating: 'جارٍ التحديث…'
  },
  mcp: {
    allServers: 'كل الخوادم',
    authenticate: 'المصادقة',
    authenticatedMessage: (server, count) =>
      `${server}: ${count} ${count === 1 ? 'أداة' : count === 2 ? 'أداتان' : count <= 10 ? 'أدوات' : 'أداة'}`,
    authenticatedTitle: 'تمت المصادقة',
    capabilitySummary: (tools, prompts, resources) =>
      `${[`${tools} أدوات`, ...(prompts ? [`${prompts} أوامر`] : []), ...(resources ? [`${resources} موارد`] : [])].join('، ')} مفعّلة`,
    catalogEnvRequired: 'املأ القيم المطلوبة قبل التثبيت.',
    catalogInstallFailed: name => `فشل تثبيت ${name}`,
    catalogLoading: 'جارٍ تحميل كتالوج MCP...',
    costTokens: tokens => `~${tokens} Tokens لكل استدعاء`,
    noOutput: 'لا مخرجات بعد.',
    statusConnecting: 'جارٍ الاتصال…',
    statusError: 'خطأ',
    statusNeedsAuth: 'يتطلب المصادقة',
    statusOff: 'متوقف',
    test: 'اختبار الاتصال',
    usage30d: uses =>
      `${uses} ${Number(uses) === 1 ? 'استخدام' : Number(uses) === 2 ? 'استخدامان' : Number(uses) <= 10 ? 'استخدامات' : 'استخدام'} خلال 30 يومًا`
  },
  model: {
    fallbackAdd: 'إضافة بديل احتياطي',
    fallbackEmpty: 'لا نماذج احتياطية — يُستخدم النموذج الافتراضي ما لم يفشل.',
    mainAppliedMessage: model => `ستستخدم الجلسات الجديدة ${model}.`,
    mainAppliedTitle: 'تم تحديث النموذج الرئيسي',
    notInCatalog: 'غير موجود في قائمة نماذج هذا المزوّد — قد تُحال الاستدعاءات إلى نسخة احتياطية.',
    subagents: {
      routing: {
        daybreakLabel: 'Daybreak'
      }
    },
    tasks: {
      mcp: {
        label: 'MCP'
      }
    }
  },
  nav: {
    billing: 'الفوترة',
    mcp: 'MCP',
    providerCustomEndpoints: 'نقاط النهاية المخصصة',
    providerLocalModels: 'النماذج المحلية'
  },
  notifications: {
    kinds: {
      credits: {
        description: 'تم إيقاف الوصول إلى الرصيد أو استعادته.',
        label: 'تنبيهات الرصيد'
      },
      plugin: {
        description: 'أرسلت إحدى إضافات سطح المكتب إشعارًا أثناء عمل Hermes في الخلفية.',
        label: 'إشعارات الإضافات'
      }
    },
    testTitle: 'Hermes'
  },
  plugins: {
    installModal: {
      agentFailed: 'فشل تثبيت إضافة الوكيل',
      agentLabel: 'إضافة الوكيل',
      agentSuccess: name => `تم تثبيت إضافة الوكيل ${name}`,
      agentTargetLocal: (profile, dir) => `يُثبَّت في الواجهة الخلفية ${profile} (${dir})`,
      agentTargetRemote: profile => `يُثبَّت في الواجهة الخلفية المتصلة ${profile}`,
      alreadyInstalled: name => `${name} مثبَّت بالفعل.`,
      description: 'راجع محتوى هذا المستودع قبل تثبيت أي شيء.',
      desktopFailed: 'فشل تثبيت إضافة سطح المكتب',
      desktopLabel: 'واجهة سطح المكتب',
      desktopOnlyNote: 'الحزم الخاصة بسطح المكتب فقط لا تثبّت إضافة وكيل في الواجهة الخلفية.',
      desktopSuccess: name => `تم تثبيت إضافة سطح المكتب ${name}`,
      desktopTarget: 'يُثبَّت في مجلد desktop-plugins المحلي لهذا التطبيق',
      desktopTargetFromPackage: 'يُحمَّل في هذا التطبيق من الحزمة أعلاه — الأمر نفسه لكل ملف تعريف',
      desktopUnavailable: 'تثبيت إضافات سطح المكتب غير متاح في هذه البيئة.',
      enableAgent: 'تفعيل إضافة الوكيل بعد التثبيت',
      forceReinstall: 'إعادة تثبيت قسرية (استبدال إذا كانت مثبَّتة)',
      includesHeading: 'تتضمن هذه الحزمة',
      insecureWarning:
        'يستخدم هذا الرابط مخططًا غير آمن أو محليًا. يُفضَّل استخدام https:// أو git@ لعمليات التثبيت الإنتاجية.',
      install: 'تثبيت',
      installFromGit: 'تثبيت من Git',
      installing: 'جارٍ التثبيت…',
      missingEnv: (name, vars) =>
        `تم تثبيت ${name} لكنه يحتاج إلى مفتاح ليعمل: ${vars}. أضفه الآن، وإلا فستفشل أدوات الإضافة.`,
      missingEnvAction: 'إعداده',
      pinToCommit: 'التثبيت على commit محدد (اختياري)',
      pinToCommitHint:
        'يحصل كل من يثبّت هذا الـSHA على الشيفرة نفسها؛ وترفض الإضافة التحديثات بعد ذلك حتى يُعاد تثبيتها على SHA جديد. اتركه فارغًا لاستخدام أحدث commit.',
      pinToCommitInvalid: 'يجب أن يكون SHA كاملًا من 40 حرفًا (لا تُقبل الفروع والوسوم).',
      pinToCommitPlaceholder: 'SHA كامل من 40 حرفًا',
      probeUnavailable: 'فحص الإضافات غير متاح في هذه البيئة.',
      probing: 'جارٍ فحص المستودع…',
      profileLabel: 'التثبيت لملف التعريف',
      repoLabel: 'المستودع',
      repoPlaceholder: 'https://github.com/owner/repo',
      reviewRepository: 'مراجعة المستودع',
      securityIntro: 'ثبّت فقط من مصادر تثق بها — راجع المستودع أدناه إذا أردت معرفة ما سيُضاف.',
      selectComponent: 'اختر مكوّنًا واحدًا على الأقل للتثبيت.',
      sourceHeading: 'الشيفرة المصدرية',
      title: 'تثبيت إضافة',
      viewPluginFiles: 'عرض ملفات الإضافة',
      viewRepository: 'عرض المستودع'
    }
  },
  search: {
    pill: 'بحث',
    placeholder: 'البحث في جميع الإعدادات…'
  },
  sessions: {
    autoArchiveDaysLabel: 'الأرشفة بعد',
    autoArchiveDaysUnit: 'يومًا من عدم النشاط',
    autoArchiveDesc:
      'أرشِف المحادثات التي لم تتفاعل معها منذ فترة تلقائيًا. المحادثات المثبَّتة لا تُؤرشف أبدًا، ولا يُحذف أي شيء — المحادثات المؤرشفة تنتقل إلى هنا فقط.',
    autoArchiveFailed: 'تعذّر تحديث الأرشفة التلقائية',
    autoArchiveTitle: 'الأرشفة التلقائية للمحادثات القديمة'
  },
  toolsets: {
    activeBackend: 'نشط',
    activeBackendHint: 'هذه هي الواجهة الخلفية النشطة لديك',
    browserRealProfile: {
      description:
        'ينسخ بيانات تسجيل الدخول وملفات تعريف الارتباط من متصفحك الافتراضي إلى لقطة مُدارة يتصفح بها الوكيل، ويسجّل الدخول بها أيضًا في تبويب المتصفح داخل التطبيق. لا يُفتح ملف تعريفك الحقيقي مباشرة أبدًا. يُطبَّق على الجلسات الجديدة.',
      disabledMessage:
        'ستُحذف لقطة ملف التعريف ويُسجَّل الخروج منها في المتصفح داخل التطبيق؛ وستستخدم الجلسات الجديدة متصفحًا نظيفًا.',
      disabledTitle: 'تصفح ملف التعريف الحقيقي متوقف',
      enabledMessage: 'ستتصفح الجلسات الجديدة باستخدام لقطة من ملف تعريف متصفحك الافتراضي.',
      enabledTitle: 'تصفح ملف التعريف الحقيقي مفعّل',
      failedSave: 'تعذّر حفظ إعداد ملف التعريف الحقيقي',
      label: 'استخدام ملف تعريف متصفحي الحقيقي',
      prompt: {
        body: 'اسمح لـHermes بالتصفح باستخدام لقطة من ملف تعريف متصفحك الافتراضي، لتُفتح المواقع مسجَّلة الدخول مسبقًا، سواء للوكيل أو في تبويب المتصفح داخل التطبيق.',
        bulletLiveProfile: 'لا يُفتح ملف تعريف متصفحك الحقيقي مباشرة أبدًا.',
        bulletLocal: 'لا يغادر أي شيء هذا الحاسوب.',
        bulletSnapshot: 'تُنسخ ملفات تعريف الارتباط وبيانات تسجيل الدخول إلى لقطة مُدارة.',
        dontShowAgain: 'عدم الإظهار مجددًا',
        enable: 'استخدام ملف تعريفي',
        notNow: 'ليس الآن',
        title: 'ابقَ مسجَّل الدخول إلى مواقعك'
      }
    },
    failedSelectCapability: provider => `تعذّر تعيين ${provider}`,
    failedSelectModel: model => `تعذّر اختيار ${model}`,
    loadingModels: 'جارٍ تحميل كتالوج النماذج...',
    modelCount: count =>
      count === 1 ? 'نموذج واحد' : count === 2 ? 'نموذجان' : count <= 10 ? `${count} نماذج` : `${count} نموذج`,
    modelDefault: 'افتراضي',
    modelInUse: 'قيد الاستخدام',
    modelInactiveHint: 'اختر هذه الواجهة الخلفية أولًا لتغيير نموذجها.',
    modelSectionTitle: 'النموذج',
    modelSelectedMessage: model => `سيُستخدم ${model} في الجلسات الجديدة.`,
    modelSelectedTitle: 'تم اختيار النموذج',
    needsSetup: 'يلزم الإعداد',
    needsSignIn: 'يلزم تسجيل الدخول',
    nousAuthDoneMessage: 'أصبحت الواجهات الخلفية لاشتراكك نشطة الآن.',
    nousAuthDoneTitle: 'تم ربط حساب Nous',
    nousAuthFailed: 'لم يكتمل تسجيل الدخول إلى Nous',
    nousAuthFailedMessage: 'حاول مرة أخرى.',
    nousAuthNeededMessage: provider => `تم حفظ ${provider} لكنه لن يعمل إلا بعد تسجيل الدخول بحساب Nous.`,
    nousAuthNeededTitle: 'سجّل الدخول بحساب Nous',
    nousAuthSignIn: 'تسجيل الدخول',
    nousAuthTryAgain: 'حاول مرة أخرى',
    postSetupInstalled: 'مثبَّت',
    postSetupInstalledHint: 'تم التثبيت. أعد تشغيل الإعداد فقط إذا كان هناك خلل.',
    postSetupOpenLogs: 'فتح السجلات',
    postSetupRerun: 'إعادة تشغيل الإعداد',
    postSetupRunAgain: 'تشغيل مرة أخرى',
    terminalBackend: {
      failedLoad: 'تعذّر تحميل الواجهات الخلفية لـTerminal',
      failedSelect: backend => `تعذّر اختيار ${backend}`,
      inUse: 'قيد الاستخدام',
      loading: 'جارٍ التحقق من واجهات التنفيذ الخلفية…',
      needsSetup: 'يحتاج إعدادًا',
      needsSetupConfirmAction: 'الاختيار على أي حال',
      needsSetupConfirmDescription: detail =>
        `${detail} لن تمتلك الجلسات التي تبدأ بعد هذا التغيير أدوات Terminal أو أدوات ملفات حتى يكتمل الإعداد.`,
      needsSetupConfirmDescriptionGeneric:
        'لم تُعدّ هذه الواجهة الخلفية بعد. لن تمتلك الجلسات التي تبدأ بعد هذا التغيير أدوات Terminal أو أدوات ملفات حتى يكتمل الإعداد.',
      needsSetupConfirmTitle: backend => `هل تريد اختيار ${backend} على أي حال؟`,
      needsSetupHint: 'هذه الواجهة الخلفية محددة حاليًا دون إعداد كامل — ستفشل الأوامر حتى يكتمل الإعداد.',
      openBackendSettings: 'فتح إعدادات Terminal',
      ready: 'جاهز',
      sectionTitle: 'واجهة التنفيذ الخلفية',
      selectedMessage: backend => `أصبحت أوامر Terminal تُنفَّذ عبر ${backend}. يُطبَّق على الجلسات الجديدة.`,
      selectedTitle: 'تم اختيار الواجهة الخلفية',
      switchedToLocal: 'أصبحت أوامر Terminal تُنفَّذ محليًا. يُطبَّق على الجلسات الجديدة.',
      unavailable: 'غير متاحة',
      unavailableMessage: backend =>
        `لا يستطيع Hermes تنفيذ أوامر الصدفة الآن: ${backend} غير جاهزة. انتقل إلى المحلي، أو أكمل إعداد ${backend} ثم حاول مرة أخرى.`,
      unavailableTitle: 'أوامر Terminal غير متاحة',
      useLocal: 'استخدام المحلي'
    },
    useBackend: 'استخدام هذه الواجهة الخلفية',
    webCapabilitySelectedMessage: (provider, capability) => `يتولى ${provider} الآن ${capability} الويب.`,
    webCapabilityUnset: 'غير محدد',
    webExtractActive: backend => `الاستخراج: ${backend}`,
    webSearchActive: backend => `البحث: ${backend}`,
    webUseForExtract: 'استخدام للاستخراج',
    webUseForSearch: 'استخدام للبحث',
    webUsedForExtract: 'واجهة الاستخراج الخلفية',
    webUsedForSearch: 'واجهة البحث الخلفية'
  },
  uninstallSection: {
    appLabel: 'التطبيق:',
    checkingInstalled: 'جارٍ التحقق مما هو مثبَّت…',
    chooseHowMuch:
      'اختر مقدار ما تريد إزالته. يُغلق التطبيق لإتمام المهمة؛ ويمكنك إعادة فتح المثبِّت في أي وقت للعودة.',
    confirmBody: what => `سيؤدي هذا إلى إزالة ${what}. لا يمكن التراجع عن ذلك.`,
    confirmUninstall: 'تأكيد إزالة التثبيت',
    couldNotStart: 'تعذّر بدء إزالة التثبيت.',
    dangerZone: 'منطقة الخطر',
    dataKept: path => `توجد إعداداتك ومحادثاتك وأسرارك في ${path}. إزالة التطبيق لا تحذفها.`,
    managedBody: 'هذا التثبيت يديره نظامك، لذا لا يستطيع Hermes إزالة نفسه.',
    openAppsSettings: 'فتح إعدادات التطبيقات',
    options: {
      full: {
        consequence: 'كل شيء — واجهة الدردشة ووكيل Hermes وجميع إعداداتك ومحادثاتك وأسرارك وسجلاتك',
        description:
          'إزالة التطبيق والوكيل وجميع بيانات المستخدم — الإعدادات والمحادثات والمهام المجدولة والأسرار والسجلات.',
        title: 'إزالة تثبيت كل شيء'
      },
      gui: {
        consequence: 'واجهة الدردشة لسطح المكتب (هذا التطبيق وبياناته)',
        description: 'إزالة تطبيق سطح المكتب هذا. يبقى وكيل Hermes وإعداداتك ومحادثاتك.',
        title: 'إزالة واجهة الدردشة فقط'
      },
      lite: {
        consequence: 'واجهة الدردشة ووكيل Hermes (تبقى الإعدادات والمحادثات والأسرار)',
        description:
          'إزالة التطبيق ووكيل Hermes، مع الإبقاء على الإعدادات والمحادثات والأسرار لإعادة التثبيت مستقبلًا.',
        title: 'إزالة الواجهة والوكيل مع الإبقاء على بياناتي'
      }
    },
    uninstallHermes: 'إزالة تثبيت Hermes',
    uninstalling: 'جارٍ إزالة التثبيت…',
    yesUninstall: 'نعم، أزل التثبيت'
  },
  vault: {
    cvcField: 'CVC',
    identifierShown: identifier => identifier,
    originPlaceholder: 'https://github.com',
    originPlaceholderCheckout: 'https://shop.example.com'
  }
} satisfies NonNullable<TranslationOverrides['settings']>
