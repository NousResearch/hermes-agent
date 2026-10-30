import { arThemePicker } from './ar_theme_picker'
import type { TranslationOverrides } from './define-locale'

export const arAssistant = {
  assistant: {
    catalogInstall: {
      preparing: 'جارٍ تجهيز التثبيت…',
      install: 'تثبيت',
      advanced: 'خيارات متقدمة',
      skip: 'تخطٍّ',
      installing: 'جارٍ التثبيت…',
      installed: 'مثبّت',
      notInstalled: 'غير مثبّت',
      failed: 'فشل',
      showNames: 'إظهار الأسماء',
      hideNames: 'إخفاء الأسماء',
      skill: name => `المهارة ${name}`,
      kind: { plugin: 'إضافة', skill: 'مهارة' },
      tier: { official: 'رسمي', verified: 'موثّق', community: 'مجتمعي' },
      targetProfile: profile => `يُثبَّت في ملفك الشخصي ${profile}`,
      sendFailed: 'تعذّر إرسال ردك. حاول مرة أخرى.',
      commitLabel: 'الإيداع',
      subdirLabel: 'المجلد',
      securityHeading: 'الأمان',
      scan: { passed: 'نجح الفحص', warnings: 'وجد الفحص تحذيرات', failed: 'فشل الفحص' },
      requirementsLabel: 'المتطلبات',
      credentialsHeading: 'بيانات الاعتماد'
    },
    thread: {
      loadingSession: 'جار تحميل الجلسة...',
      showEarlier: 'عرض الرسائل الأقدم',
      loadingResponse: 'جار تحميل الرد...',
      resumeWhenBackgroundDone: count =>
        count === 1 ? 'سيُستأنف عند انتهاء المهمة الخلفية' : `سيُستأنف عند انتهاء ${count} مهام خلفية`,
      thinking: 'يفكر...',
      thought: 'فكّر',
      thoughtBriefly: 'فكّر قليلاً',
      thoughtFor: duration => `فكّر لمدة ${duration}`,
      turnDuration: duration => `استغرقت هذه الجولة ${duration}`,
      today: time => `اليوم ${time}`,
      yesterday: time => `أمس ${time}`,
      copy: 'نسخ',
      refresh: 'تحديث',
      moreActions: 'إجراءات إضافية',
      branchNewChat: 'تفريع إلى محادثة جديدة',
      react: 'تفاعل',
      dismissError: 'تجاهل الخطأ',
      errorLayers: {
        auth: 'خطأ في المصادقة',
        billing: 'نفاد الرصيد',
        disk: 'القرص ممتلئ',
        endpoint: 'خطأ في نقطة النهاية المخصصة',
        gateway: 'تعذّر إرسال رسالتك',
        generic: 'فشلت الجولة',
        provider: 'خطأ من المزوّد',
        runtime: 'خطأ في بيئة التشغيل المحلية',
        streaming: 'خطأ في اتصال البث'
      },
      errorCodes: {
        auth: {
          title: provider => `رفض ${provider} مفتاح API الخاص بك`,
          body: () => 'المفتاح المحفوظ غير صالح أو أُلغي. حدّثه، ثم أرسل رسالتك مجددًا.'
        },
        auth_permanent: {
          title: provider => `رفض ${provider} مفتاح API الخاص بك`,
          body: () => 'المفتاح المحفوظ غير صالح أو أُلغي. حدّثه، ثم أرسل رسالتك مجددًا.'
        },
        billing: {
          title: provider => `نفد رصيدك في ${provider}`,
          body: provider => `أضف رصيدًا لمواصلة استخدام النماذج المدفوعة على ${provider}.`
        },
        rate_limit: {
          title: provider => `${provider} مشغول الآن`,
          body: provider => `بلغت حد الطلبات في ${provider}. انتظر قليلًا، ثم أعد المحاولة.`
        },
        upstream_rate_limit: {
          title: (_provider, model) => `${model || 'هذا النموذج'} مشغول الآن`,
          body: (provider, model) =>
            `${model || 'هذا النموذج'} مقيَّد بحد الطلبات لدى مصدر ${provider}، لا في حسابك. أعد المحاولة بعد قليل.`
        },
        overloaded: {
          title: provider => `${provider} مشغول الآن`,
          body: provider => `${provider} مثقل بالطلبات ولم يتمكن من معالجة هذا الطلب. أعد المحاولة بعد قليل.`
        },
        server_error: {
          title: provider => `${provider} يواجه مشكلة`,
          body: provider => `أعاد ${provider} خطأ في الخادم. أعد المحاولة بعد قليل.`
        },
        timeout: {
          title: provider => `تعذّر الوصول إلى ${provider}`,
          body: provider => `تعذّر على Hermes الوصول إلى ${provider}. تحقّق من اتصالك، ثم أعد المحاولة.`
        },
        no_reply: {
          title: 'لم يصل الرد',
          body: 'أنهى Hermes هذه الجولة، لكن لم يصل أي رد إلى هذه النافذة. أعد المحاولة لإرسالها مجددًا.'
        },
        upstream_blocked: {
          title: provider => `حظر ${provider} هذا الطلب`,
          body: provider =>
            `رفض جدار حماية أمام ${provider} الطلب قبل وصوله إلى النموذج. بدّل المزوّد، أو عيّن له User-Agent مخصصًا.`
        },
        ssl_cert_verification: {
          title: provider => `تعذّر التحقق من اتصال ${provider}`,
          body: provider =>
            `منع وكيل (proxy) أو مخزن شهادات قديم الاتصال الآمن بـ ${provider}. يوضح دليل الإعداد كيفية إصلاح ذلك.`
        },
        context_overflow: {
          title: 'هذه المحادثة طويلة جدًا',
          body: (_provider, model) => `لم تعد المحادثة تتسع في نافذة سياق ${model || 'النموذج'}. اضغطها للمتابعة.`
        },
        payload_too_large: {
          title: 'هذه الرسالة كبيرة جدًا',
          body: provider => `يتجاوز الطلب، بما فيه المرفقات، حد الحجم لدى ${provider}. اضغط المحادثة أو أزل المرفقات.`
        },
        model_not_found: {
          title: (_provider, model) => `${model || 'هذا النموذج'} غير متاح`,
          body: (provider, model) => `لا يوفّر ${provider} ${model || 'هذا النموذج'} في حسابك. اختر نموذجًا آخر.`
        },
        provider_policy_blocked: {
          title: (provider, model) => `إعدادات ${provider} تحظر ${model || 'هذا النموذج'}`,
          body: provider =>
            `تستبعد إعدادات الخصوصية في ${provider} جميع نقاط الوصول لهذا النموذج. غيّرها على ${provider}، أو اختر نموذجًا آخر.`
        },
        content_policy_blocked: {
          title: provider => `رفض ${provider} هذه الرسالة`,
          body: provider => `رفضها مرشّح الأمان في ${provider}. عدّل الرسالة وأرسلها مجددًا.`
        },
        format_error: {
          title: provider => `رفض ${provider} هذا الطلب`,
          body: provider =>
            `لم يقبل ${provider} طريقة بناء هذه المحادثة. بدّل المزوّد، أو ابدأ محادثة جديدة إذا تكرر ذلك.`
        },
        truncated: {
          title: 'انقطع الرد قبل اكتماله',
          body: (_provider, model) =>
            `توقف ${model || 'النموذج'} قبل أن يُنهي ردّه. اختر «متابعة» ليكمل من حيث توقف.`
        },
        invalid_response: {
          title: 'تعذّرت قراءة الرد',
          body: provider => `أعاد ${provider} ردًا فارغًا أو تالفًا. أعد المحاولة بعد قليل.`
        },
        empty_response: {
          title: 'تعذّر الحصول على رد',
          body: provider => `لم يُعِد ${provider} أي شيء لهذه الرسالة. أعد المحاولة بعد قليل.`
        },
        loop_error: {
          title: 'تعذّر على Hermes إنهاء هذه الخطوة',
          body: 'حدث خطأ أثناء معالجة الرد. أعد المحاولة؛ وإذا تكرر ذلك، أرسل التشخيصات من التفاصيل.'
        },
        agent_init_failed: {
          title: 'تعذّر بدء هذه المحادثة',
          body: provider =>
            `تعذّر على Hermes الاتصال بـ ${provider} لهذه المحادثة. أعد المحاولة؛ وإذا استمر الفشل، تحقّق من المزوّد في الإعدادات.`
        },
        SESSION_NOT_OWNED: {
          title: 'المحادثة مفتوحة في مكان آخر',
          body: 'هذه المحادثة مفتوحة في نافذة Hermes أخرى أو في الطرفية. ابدأ محادثة جديدة هنا، أو أغلقها هناك أولًا.'
        },
        disk_full: {
          title: 'القرص ممتلئ',
          body: 'تعذّر على Hermes حفظ هذه المحادثة. حرّر بعض المساحة، ثم أعد المحاولة.'
        },
        free_tier_disabled: {
          title: 'سجّل الدخول لمتابعة المحادثة',
          body: 'سجّل الدخول بحساب Nous لمتابعة المحادثة، فهو مجاني.'
        },
        free_tier_rate_limited: {
          title: 'نفدت الحصة المجانية',
          body: 'تتجدد قريبًا. سجّل الدخول بحساب Nous للحصول على حصة أكبر، فهو مجاني.'
        },
        free_tier_at_capacity: {
          title: 'الفئة المجانية مزدحمة',
          body: 'سجّل الدخول لتتجاوز الانتظار، فهو مجاني، أو حاول مجددًا بعد قليل.'
        },
        free_tier_model_not_free: {
          title: 'هذا النموذج يتطلب تسجيل الدخول',
          body: 'يستخدم Hermes النموذج المجاني حاليًا. سجّل الدخول بحساب Nous للوصول إلى نماذج أكثر، فهو مجاني.'
        },
        free_tier_route: {
          title: 'تعذّر الوصول إلى النموذج المجاني',
          body: 'سجّل الدخول بحساب Nous، فهو مجاني، أو تحقّق من إعداد NOUS_INFERENCE_BASE_URL.'
        },
        free_tier_outage: {
          title: 'النموذج المجاني يواجه مشكلة',
          body: 'حاول إرسال رسالتك مجددًا بعد دقيقة.'
        },
        free_tier_refused: {
          title: 'سجّل الدخول لمتابعة المحادثة',
          body: 'تسجيل الدخول بحساب Nous مجاني.'
        },
        cron_session_read_only: {
          title: 'هذه المحادثة تابعة لمهمة مجدولة',
          body: 'إنها تشغّل تلك المهمة فقط، لذا لم تُنفَّذ رسالتك. ابدأ محادثة جديدة لإرسالها.'
        },
        session_read_only: {
          title: 'هذه المحادثة للقراءة فقط',
          body: 'إنها سجلّ محفوظ يستطيع Hermes عرضه لكن لا يستطيع متابعته، لذا لم تُنفَّذ رسالتك. ابدأ محادثة جديدة لإرسالها.'
        },
        usage_window: {
          title: 'نفد الاستخدام المضمّن',
          body: 'نفد الاستخدام المضمّن في خطتك حتى إعادة التعيين. أضف رصيدًا عبر /topup للمتابعة.'
        },
        workspace_unavailable: {
          title: 'مجلد هذه المحادثة غير موجود',
          body: 'نُقل المجلد الذي تعمل فيه هذه المحادثة أو حُذف، لذا لم تُنفَّذ رسالتك. أعد إنشاء المجلد، أو استخدم «Move to project» لتعيين مجلد آخر للمحادثة، ثم أعد المحاولة.'
        },
        launch_credentials_unavailable: {
          title: 'تغيّر مفتاح API لهذه المحادثة',
          body: 'بدأت هذه المحادثة بمفتاح API تم تغييره أو حذفه منذ ذلك الحين، لذا لم تُرسل رسالتك. المحادثة المفتوحة لا تنتقل أبدًا إلى مفتاح مختلف من تلقاء نفسها. استخدم «بدء جلسة جديدة» لاستخدام مفتاحك الحالي.'
        }
      },
      errorAuthKinds: {
        api_key: {
          title: provider => `رفض ${provider} مفتاح API الخاص بك`,
          body: () => 'المفتاح المحفوظ غير صالح أو أُلغي. حدّثه، ثم أرسل رسالتك مجددًا.'
        },
        oauth: {
          title: provider => `سجّل الدخول إلى ${provider} مجددًا`
        }
      },
      errorDetails: 'عرض تفاصيل الخطأ',
      errorHideDetails: 'إخفاء تفاصيل الخطأ',
      errorDetailsCopy: 'نسخ التفاصيل',
      errorCopyRaw: 'نسخ الخطأ',
      errorContinue: 'متابعة',
      errorOpenProviderSettings: provider => `فتح إعدادات ${provider}`,
      errorOpenSetupGuide: 'فتح دليل الإعداد',
      errorSetupGuideLink: 'دليل الإعداد',
      errorOpenModelSettings: 'فتح إعدادات النموذج',
      errorStartNewChatLink: 'بدء محادثة جديدة',
      errorChooseModelLink: 'اختيار نموذج آخر',
      errorRetryAtResetShort: time => `إعادة المحاولة عند ${time}`,
      errorRateLimitResetBody: (provider, time) => `بلغت حد الطلبات في ${provider}. يُعاد ضبطه عند ${time}.`,
      errorRetry: 'إعادة المحاولة',
      errorSwitchToModel: (model, note) => (note ? `التبديل إلى ${model} (${note})` : `التبديل إلى ${model}`),
      errorSwitchModelFailed: 'تعذر تبديل النموذج',
      errorUseCredits: 'تفعيل الاستمرار بالرصيد',
      errorUseCreditsFailed: 'تعذر تفعيل الاستمرار بالرصيد',
      errorAddCredits: 'إضافة رصيد',
      errorOpenBillingSettings: 'فتح إعدادات الفوترة',
      errorRetryScheduled: (time, wait) => `ستتم إعادة المحاولة عند ${time} — بعد ${wait}`,
      errorRetryScheduledCancel: 'إلغاء',
      errorStartNewSession: 'بدء جلسة جديدة',
      errorSwitchProvider: 'تبديل المزوّد',
      errorSignInAgain: provider => `تسجيل الدخول إلى ${provider} مجدداً`,
      errorOauthExpired: provider =>
        `انتهت صلاحية تسجيل دخولك إلى ${provider} أو أُلغي. سجّل الدخول مجدداً لمتابعة المحادثة.`,
      errorOpenLogs: 'فتح السجلات',
      errorOpenLogsFailed: 'تعذّر فتح مجلد السجلات',
      errorOpenDesktopLogs: 'فتح سجلات سطح المكتب',
      errorSendDiagnostics: 'إرسال التشخيصات',
      filesChanged: count => `${count} ملفات تم تغييرها`,
      reviewChanges: 'مراجعة',
      readAloudFailed: 'فشلت القراءة بصوت عال',
      preparingAudio: 'جار تجهيز الصوت',
      stopReading: 'إيقاف القراءة',
      readAloud: 'قراءة بصوت عال',
      copyFullResponse: 'نسخ الرد الكامل',
      readAloudFullResponseHint: 'انقر مع الضغط على Shift: قراءة الرد الكامل بصوت عال',
      editMessage: 'تحرير الرسالة',
      scrollToBottom: 'التمرير إلى الأسفل',
      stop: 'إيقاف',
      restorePrevious: 'استعادة السابق',
      restoreCheckpoint: 'استعادة النقطة',
      restoreFromHere: 'استعادة نقطة التحقق — إعادة التشغيل من هذا الموجّه',
      restoreTitle: 'الاستعادة إلى نقطة التحقق هذه؟',
      restoreBody: 'يُزال كل ما يلي هذا الموجّه من المحادثة، ويُعاد تشغيل الموجّه من هنا.',
      restoreConfirm: 'استعادة وإعادة تشغيل',
      restoreNext: 'استعادة التالي',
      goForward: 'تقدم',
      sendEdited: 'إرسال التعديل',
      attachingFile: 'جار إرفاق الملف'
    },
    approval: {
      gatewayDisconnected: 'البوابة غير متصلة',
      sendFailed: 'فشل الإرسال',
      run: 'تشغيل',
      command: 'الأمر',
      moreOptions: 'خيارات إضافية',
      allowSession: 'السماح لهذه الجلسة',
      alwaysAllowMenu: 'السماح دائما',
      jumpToApproval: 'الموافقة مطلوبة',
      reject: 'رفض',
      alwaysTitle: 'السماح دائما',
      alwaysDescription: pattern => `السماح دائما بالأوامر المطابقة لـ ${pattern}`,
      alwaysAllow: 'السماح دائما'
    },
    clarify: {
      notReady: 'غير جاهز',
      gatewayDisconnected: 'البوابة غير متصلة',
      sendFailed: 'فشل الإرسال',
      loadingQuestion: 'جار تحميل السؤال...',
      other: 'غير ذلك',
      placeholder: 'اكتب إجابتك...',
      skip: 'تخطي',
      confirmAndContinueLabel: 'تأكيد ومتابعة',
      singleSelectHint: 'اختر واحدا',
      multiSelectHint: 'حدد كل ما ينطبق',
      oneQuestion: 'سؤال واحد',
      questionProgress: (answered, total) => `تمت الإجابة على ${answered} من ${total}`
    },
    themePicker: arThemePicker,
    tool: {
      copyCode: 'نسخ الكود',
      renderingImage: 'جار عرض الصورة...',
      copyOutput: 'نسخ الإخراج',
      copyCommand: 'نسخ الأمر',
      copyContent: 'نسخ المحتوى',
      copyUrl: 'نسخ الرابط',
      copyResults: 'نسخ النتائج',
      copyQuery: 'نسخ الاستعلام',
      copyFile: 'نسخ الملف',
      copyPath: 'نسخ المسار',
      failedCalls: (count: number) => `عدد استدعاءات الأدوات الفاشلة: ${count}`,
      skillActivity: {
        loading: 'جارٍ تحميل المهارة',
        loaded: 'تم تحميل المهارة',
        loadFailed: 'تعذر تحميل المهارة',
        readingResource: 'جارٍ قراءة مورد المهارة',
        readResource: 'تمت قراءة مورد المهارة',
        resourceFailed: 'تعذرت قراءة مورد المهارة',
        listing: 'جارٍ عرض المهارات',
        listed: 'تم عرض المهارات',
        listFailed: 'تعذر عرض المهارات',
        unavailable: 'نتيجة المهارة غير متاحة'
      },
      outputAlt: 'إخراج الأداة',
      rawResponse: 'الرد الخام',
      copyActivity: 'نسخ النشاط',
      recoveredOne: 'تم الاسترداد',
      recoveredMany: count => `تم استرداد ${count}`,
      failedOne: 'فشل',
      failedMany: count => `فشل ${count}`,
      statusRunning: 'يعمل',
      statusError: 'خطأ',
      statusRecovered: 'تم الاسترداد',
      statusDone: 'تم',
      resultUnavailable: 'النتيجة غير متاحة',
      resultInterrupted: 'تمت المقاطعة',
      memoryWriteNoted: 'تم تسجيل كتابة الذاكرة',
      actions: {
        read: 'قراءة',
        reading: 'جار القراءة',
        opened: 'تم الفتح',
        opening: 'جار الفتح',
        searched: 'تم البحث',
        searching: 'جار البحث',
        ran: 'تم التشغيل',
        running: 'جار التشغيل',
        ranCode: 'تم تشغيل الكود',
        runningCode: 'جار البرمجة'
      },
      prefixes: {
        browser: 'المتصفح',
        web: 'الويب'
      },
      titleTemplates: {
        actionCommand: (action, command) => `${action} ${command}`,
        actionQuoted: (action, value) => `${action} “${value}”`,
        actionTarget: (action, target) => `${action} ${target}`,
        prefixedDone: (prefix, action) => `${prefix} ${action}`,
        runningPrefixedTool: (prefix, action) => `جار تشغيل ${prefix.toLowerCase()} ${action.toLowerCase()}`,
        runningTool: action => `جار تشغيل ${action.toLowerCase()}`
      },
      titles: {
        browser_click: {
          done: 'تم النقر على عنصر الصفحة',
          pending: 'جار النقر على عنصر الصفحة',
          pendingAction: 'جار النقر'
        },
        browser_fill: {
          done: 'تم ملء حقل النموذج',
          pending: 'جار ملء حقل النموذج',
          pendingAction: 'جار الملء'
        },
        browser_navigate: {
          done: 'تم فتح الصفحة',
          pending: 'جار فتح الصفحة',
          pendingAction: 'جار الفتح'
        },
        browser_snapshot: {
          done: 'تم التقاط لقطة الصفحة',
          pending: 'جار التقاط لقطة الصفحة',
          pendingAction: 'جار الالتقاط'
        },
        browser_take_screenshot: {
          done: 'تم التقاط لقطة الشاشة',
          pending: 'جار التقاط لقطة الشاشة',
          pendingAction: 'جار الالتقاط'
        },
        browser_type: {
          done: 'تمت الكتابة على الصفحة',
          pending: 'جار الكتابة على الصفحة',
          pendingAction: 'جار الكتابة'
        },
        clarify: {
          done: 'تم طرح سؤال',
          pending: 'جار طرح سؤال',
          pendingAction: 'جار السؤال'
        },
        cronjob: {
          done: 'مهمة مجدولة',
          pending: 'جار جدولة المهمة',
          pendingAction: 'جار الجدولة'
        },
        edit_file: {
          done: 'تم تحرير الملف',
          pending: 'جار تحرير الملف',
          pendingAction: 'جار التحرير'
        },
        execute_code: {
          done: 'تم تشغيل الكود',
          pending: 'جار البرمجة',
          pendingAction: 'جار البرمجة'
        },
        image_generate: {
          done: 'تم إنشاء الصورة',
          pending: 'جار إنشاء الصورة',
          pendingAction: 'جار الإنشاء'
        },
        list_files: {
          done: 'تم سرد الملفات',
          pending: 'جار سرد الملفات',
          pendingAction: 'جار السرد'
        },
        memory: {
          done: 'تم الحفظ في الذاكرة',
          pending: 'جار الحفظ في الذاكرة',
          pendingAction: 'جار الحفظ'
        },
        patch: {
          done: 'تم تصحيح الملف',
          pending: 'جار تصحيح الملف',
          pendingAction: 'جار التصحيح'
        },
        read_file: {
          done: 'تمت قراءة الملف',
          pending: 'جار قراءة الملف',
          pendingAction: 'جار القراءة'
        },
        search_files: {
          done: 'تم البحث في الملفات',
          pending: 'جار البحث في الملفات',
          pendingAction: 'جار البحث'
        },
        session_search_recall: {
          done: 'تم البحث في سجل الجلسة',
          pending: 'جار البحث في سجل الجلسة',
          pendingAction: 'جار البحث'
        },
        terminal: {
          done: 'تم تشغيل الأمر',
          pending: 'جار تشغيل الأمر',
          pendingAction: 'جار التشغيل'
        },
        todo: {
          done: 'تم تحديث المهام',
          pending: 'جار تحديث المهام',
          pendingAction: 'جار التحديث'
        },
        vision_analyze: {
          done: 'تم تحليل الصورة',
          pending: 'جار تحليل الصورة',
          pendingAction: 'جار التحليل'
        },
        web_extract: {
          done: 'تمت قراءة صفحة الويب',
          pending: 'جار قراءة صفحة الويب',
          pendingAction: 'جار القراءة'
        },
        web_search: {
          done: 'تم البحث في الويب',
          pending: 'جار البحث في الويب',
          pendingAction: 'جار البحث'
        },
        write_file: {
          done: 'تم تحرير الملف',
          pending: 'جار تحرير الملف',
          pendingAction: 'جار التحرير'
        }
      }
    }
  }
} satisfies Pick<TranslationOverrides, 'assistant'>
