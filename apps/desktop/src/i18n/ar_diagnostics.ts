import type { TranslationOverrides } from './define-locale'

export const arDiagnostics = {
  notifications: {
    sharedProfileWarning:
      'تستخدم نسخة أخرى من Rabbit هذا الملف الشخصي. تتشارك النسختان إعداداته وبياناته، لذا قد تتعارض التغييرات. يمكنك المتابعة أو إغلاق النسخة الأخرى قبل إجراء تغييرات.',
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
    backendOutOfDateMessage: 'خلفية Rabbit أقدم من إصدار سطح المكتب الحالي وقد لا تعمل كما يجب. حدثهما ليتوافقا.',
    desktopOutOfDateTitle: 'التطبيق قديم',
    desktopOutOfDateMessage: 'تطبيق Rabbit أقدم من الخلفية المتصل بها وقد لا يعمل كما يجب. حدّث التطبيق ليتوافقا.',
    updateDesktopApp: 'تحديث التطبيق',
    updateRabbit: 'تحديث Rabbit',
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
      methodNotAllowed: 'رفضت خلفية سطح المكتب هذا الطلب (405 Method Not Allowed). جرب إعادة تشغيل Rabbit Desktop.',
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
      inputBody: 'ينتظر Rabbit ردّك.',
      turnDoneTitle: 'أنهى Rabbit',
      turnDoneBody: '',
      turnErrorTitle: 'فشلت الجولة',
      backgroundDoneTitle: 'انتهت المهمة في الخلفية',
      backgroundFailedTitle: 'فشلت المهمة في الخلفية'
    }
  },
  errors: {
    genericFailure: 'حدث خطأ',
    boundaryTitle: 'تعطل جزء من الواجهة',
    boundaryDesc: 'يمكنك إعادة تحميل النافذة أو فتح السجلات لمعرفة التفاصيل.',
    reloadWindow: 'إعادة تحميل النافذة',
    openLogs: 'فتح السجلات'
  }
} satisfies Pick<TranslationOverrides, 'notifications' | 'errors'>
