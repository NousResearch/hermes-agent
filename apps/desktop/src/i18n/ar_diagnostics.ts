import type { TranslationOverrides } from './define-locale'

export const arDiagnostics = {
  notifications: {
    region: 'الإشعارات',
    hide: 'إخفاء',
    show: 'إظهار',
    more: count => `${count} إشعار إضافي`,
    clearAll: 'مسح الكل',
    dismiss: 'إغلاق الإشعار',
    details: 'التفاصيل',
    copyDetail: 'نسخ التفاصيل',
    copyDetailFailed: 'تعذر نسخ تفاصيل الإشعار',
    backendOutOfDateTitle: 'الخلفية قديمة',
    backendOutOfDateMessage: 'خلفية Hermes أقدم من إصدار سطح المكتب الحالي وقد لا تعمل كما يجب. حدثهما ليتوافقا.',
    updateHermes: 'تحديث Hermes',
    updateReadyTitle: 'التحديث جاهز',
    updateReadyMessage: count => `${count} تغيير جديد متاح.`,
    updateReadyMessageUnknown: 'يتوفر تحديث جديد.',
    seeWhatsNew: 'عرض الجديد',
    mcp: {
      needsAuthTitle: 'خادم MCP يحتاج إلى إعادة المصادقة',
      needsAuthMessage: name => `يحتاج ${name} MCP إلى إعادة المصادقة.`,
      errorTitle: 'تعذر الوصول إلى خادم MCP',
      errorMessage: name => `فشل فحص سلامة ${name} MCP.`,
      signIn: 'تسجيل الدخول',
      view: 'عرض',
      disable: 'تعطيل',
      disabledMessage: name => `تم تعطيل ${name} MCP. يمكنك إعادة تفعيله في أي وقت من الإمكانات → MCP.`,
      disableFailed: name => `تعذّر تعطيل ${name} MCP.`
    },
    errors: {
      elevenLabsNeedsKey: 'يتطلب ElevenLabs STT المفتاح ELEVENLABS_API_KEY.',
      elevenLabsRejectedKey: 'رفض ElevenLabs مفتاح API (401).',
      diskFull: 'القرص ممتلئ — حرّر مساحة ثم أعد المحاولة.',
      methodNotAllowed: 'رفضت خلفية سطح المكتب هذا الطلب (405 Method Not Allowed). جرب إعادة تشغيل Hermes Desktop.',
      microphonePermission: 'تم رفض إذن الميكروفون.',
      openaiRejectedApiKey: 'رفض OpenAI مفتاح API.',
      openaiTtsNeedsKey: 'يتطلب OpenAI TTS المفتاح VOICE_TOOLS_OPENAI_KEY أو OPENAI_API_KEY.',
      codeSkewRestartRequired: 'بعد التحديث ما زال هذا الخلفية يشغّل كودا قديما. أعد تشغيله لتحميل الكود الجديد.'
    },
    voice: {
      configureSpeechToText: 'اضبط تحويل الكلام إلى نص لاستخدام وضع الصوت.',
      couldNotStartSession: 'تعذر بدء جلسة الصوت',
      microphoneAccessDenied: 'تم رفض الوصول إلى الميكروفون.',
      microphoneConstraintsUnsupported: 'قيود الميكروفون غير مدعومة على هذا الجهاز.',
      microphoneFailed: 'فشل الميكروفون',
      microphoneInUse: 'الميكروفون مستخدم من تطبيق آخر.',
      microphonePermissionDenied: 'تم رفض إذن الميكروفون.',
      microphoneStartFailed: 'تعذر بدء تسجيل الميكروفون.',
      microphoneUnsupported: 'هذا المتصفح لا يدعم تسجيل الميكروفون.',
      noMicrophone: 'لم يتم العثور على ميكروفون.',
      noSpeechDetected: 'لم يتم اكتشاف كلام',
      playbackFailed: 'فشل تشغيل الصوت',
      recordingFailed: 'فشل التسجيل',
      sayStopToEnd: phrase => `قل "${phrase}" لإنهاء المحادثة الصوتية.`,
      transcriptionFailed: 'فشل التفريغ النصي',
      transcriptionUnavailable: 'التفريغ النصي غير متاح.',
      tryRecordingAgain: 'حاول التسجيل مرة أخرى.',
      unavailable: 'الصوت غير متاح'
    },
    native: {
      approvalTitle: 'مطلوب موافقة',
      approvalTitleNamed: session => `مطلوب موافقة — ${session}`,
      approveAction: 'موافقة',
      rejectAction: 'رفض',
      inputTitle: 'مطلوب إدخال',
      inputTitleNamed: session => `مطلوب إدخال — ${session}`,
      inputBody: 'ينتظر Hermes ردّك.',
      turnDoneTitle: 'أنهى Hermes',
      turnDoneBody: '',
      turnErrorTitle: 'فشلت الجولة',
      backgroundDoneTitle: 'انتهت المهمة في الخلفية',
      backgroundFailedTitle: 'فشلت المهمة في الخلفية'
    }
  },
  sendDiagnostics: {
    title: 'إرسال التشخيصات إلى Nous',
    privacyNotice:
      'سيؤدي هذا إلى رفع حزمة تصحيح إلى التخزين الداخلي لدى Nous (ليست لصيقة عامة). تتضمن معلومات النظام (نظام التشغيل، الإصدارات، المزوّد، وأنواع مفاتيح API المُهيأة — وليس المفاتيح نفسها أبداً) والسجلات الكاملة للوكيل والبوابة وسطح المكتب (حتى 512 كيلوبايت لكل منها، ومن المرجح أن تحتوي على محتوى المحادثات ومخرجات الأدوات ومسارات الملفات). تُحجب الأسرار قبل الرفع. لا يمكن الاطلاع عليها إلا لموظفي Nous ومشرفي Discord المعتمدين، وتُحذف تلقائياً بعد 14 يوماً.',
    upload: 'رفع',
    uploading: 'جارٍ الرفع…',
    cancel: 'إلغاء',
    close: 'إغلاق',
    copyLink: 'نسخ الرابط',
    uploadIdFallback: id => `لم يتم إرجاع رابط عرض — اذكر معرّف الرفع ${id} للدعم`,
    doneTitle: 'تم إرسال التشخيصات',
    doneDescription: 'تم رفع الحزمة بشكل خاص. شارك الرابط أدناه في محادثة الدعم لكي يتمكن الفريق من رؤية سجلاتك.',
    failedTitle: 'فشل الرفع',
    failedHint:
      'يمكنك أيضاً تشغيل `hermes debug share --nous` من الطرفية، أو `hermes debug share --local` لعرض التقرير دون رفعه.',
    handoffLead: 'تابع النقاش في:',
    links: {
      github: 'GitHub Issues',
      portal: 'دعم بوابة Nous',
      discord: 'Discord'
    }
  },
  errors: {
    genericFailure: 'حدث خطأ',
    boundaryTitle: 'تعطل جزء من الواجهة',
    boundaryDesc: 'يمكنك إعادة تحميل النافذة أو فتح السجلات لمعرفة التفاصيل.',
    reloadWindow: 'إعادة تحميل النافذة',
    openLogs: 'فتح السجلات'
  }
} satisfies Pick<TranslationOverrides, 'notifications' | 'sendDiagnostics' | 'errors'>
