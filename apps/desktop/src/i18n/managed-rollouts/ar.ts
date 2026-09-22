import { mergeTranslations, type TranslationOverride } from '@hermes/shared/i18n'

import { managedRolloutsEn } from './en'
import type { ManagedRolloutMessages } from '../managed-rollouts-types'

const overrides: TranslationOverride<ManagedRolloutMessages> = {
  title: 'عمليات النشر المُدارة',
  noActive: 'لا توجد لقطة نشر نشطة.',
  history: 'سجل النشر',
  unresolved: count => `الحواجز غير المحلولة: ${count}`,
  archived: 'مؤرشف',
  active: 'نشط',
  sections: {
    fleet: 'أهداف النشر المُدار',
    preparation: 'إعداد النشر المُدار',
    configuration: 'إعدادات النشر المُدار',
    preflight: 'مراجعة ما قبل النشر المُدار',
    wavePreview: 'معاينة موجات النشر المُدار',
    active: 'النشر المُدار النشط',
    controls: 'عناصر تحكم النشر المُدار',
    recovery: 'استرداد النشر المُدار',
    history: 'سجل النشر المُدار',
    summary: 'ملخص النشر المُدار'
  },
  actions: {
    select: 'تحديد',
    selected: 'محدد',
    prepare: 'إعداد الأهداف المحددة',
    preparing: 'جار الإعداد…',
    recheckEligibility: 'إعادة فحص الأهلية',
    continueToPreflight: 'المتابعة إلى الفحص المسبق',
    start: 'بدء النشر',
    starting: 'جار البدء…',
    pause: 'إيقاف مؤقت',
    pausing: 'جار الإيقاف المؤقت…',
    stop: 'إيقاف',
    stopping: 'جار الإيقاف…',
    resume: 'استئناف',
    verifyBeforePromotion: 'التحقق قبل الترقية',
    verifyingBeforePromotion: 'جار التحقق قبل الترقية',
    recheckOutcome: 'إعادة فحص النتيجة',
    recoverConnections: 'استرداد الاتصالات',
    retry: 'إعادة المحاولة',
    exclude: 'استبعاد',
    stopAndPlanRetry: 'الإيقاف والتخطيط لإعادة المحاولة',
    archiveStoppedRollout: 'أرشفة النشر المتوقف'
  },
  status: {
    activePhase: phase => `المرحلة الحالية: ${phase}`,
    queued: 'في قائمة الانتظار',
    preparing: 'جار الإعداد',
    ready: 'جاهز',
    running: 'جار التشغيل',
    awaitingPromotion: 'في انتظار الترقية',
    attentionRequired: 'يتطلب الانتباه',
    paused: 'متوقف مؤقتًا',
    stopped: 'متوقف',
    completed: 'مكتمل',
    completedWithExclusions: 'مكتمل مع استبعادات',
    failed: 'فشل',
    refused: 'مرفوض',
    unknown: 'النتيجة غير معروفة',
    unverified: 'لم يتم التحقق',
    recoveryRequired: 'يتطلب الاسترداد',
    fenced: 'حاجز الاسترداد محفوظ',
    alreadyCurrent: 'هو الإصدار الحالي بالفعل',
    pending: 'في انتظار الدليل'
  },
  policy: {
    manual: 'موافقة يدوية',
    automatic: 'تقدم تلقائي بعد موافقة الاختبار المصغر أثناء السلامة',
    mode: mode => `الوضع: ${mode}`,
    canaryGate: gate => `حاجز الاختبار المصغر: ${gate}`
  },
  labels: {
    progressionMode: 'وضع التقدم',
    concurrency: 'التزامن',
    targetDetails: (label, alias, machineId, installId) =>
      `${label} · ${alias ? `${alias} · ` : ''}${machineId} · ${installId}`,
    wave: (number, targets) => `الموجة ${number}: ${targets || 'لا توجد أهداف'}`,
    emptyWave: number => `الموجة ${number}: لا توجد أهداف`,
    progress: (completed, total) => `التقدم: اكتمل ${completed} من أصل ${total} هدفًا.`,
    receipt: (outcome, correlationId) => `الإيصال: ${outcome} (${correlationId})`,
    readiness: value => `حالة الجاهزية: ${value}`,
    reason: value => `السبب: ${value}`,
    attempt: (installId, phase) => `${installId} · ${phase}`,
    historyEntry: (id, phase, updatedAt, unresolved, archived) =>
      `${id} · ${phase} · ${updatedAt} · الحواجز غير المحلولة ${unresolved} · ${archived ? 'مؤرشف' : 'نشط'}`,
    archive: archived => `الأرشيف: ${archived ? 'مؤرشف' : 'نشط'}`
  },
  warnings: {
    sharedMachine: 'آلة مشتركة؛ راجع الملكية قبل الإعداد.',
    unsupportedTarget: 'الهدف غير مدعوم.',
    preparation: 'يتبع الإعداد طرف الفرع المُهيأ؛ وليس نشرًا مثبت الهدف.',
    preparationMayDisconnect: 'قد يؤدي الإعداد إلى قطع الاتصالات مؤقتًا.',
    preparationInvalidatesReview: 'يلغي الإعداد المراجعة السابقة ويتطلب إعادة التأهل بعد تغيير الهدف أو الاسم المستعار أو المصدر أو النطاق.',
    changedPlan: 'تم تغيير الإعدادات؛ جدّد رمز المراجعة قبل المتابعة.',
    projectionUnavailable: 'معاينة الموجة غير متاحة؛ لا يمكن متابعة الفحص المسبق.',
    incompatible: 'الإمكانات الحالية غير متوافقة مع هذه الخطة؛ لا يتوفر البدء.',
    commitmentBoundary: 'يمنع الإيقاف عمليات التفويض الجديدة؛ وقد تواصل التحديثات الملتزم بها تفريغ الجلسات والتطبيق واسترداد النطاق.',
    unknownOutcome: 'النتيجة البعيدة غير معروفة؛ لا تعتبر غياب الإيصال دليلًا على النجاح.',
    recoveryFence: 'يبقى حاجز الاسترداد محفوظًا؛ وتظل الأعمال غير الآمنة الجديدة محظورة.',
    unknownAndFenced: 'النتيجة غير معروفة وحاجز الاسترداد محفوظ؛ وقد حُظرت الأعمال غير الآمنة الجديدة.',
    estimateUnavailable: 'التقدير غير متاح.',
    unavailable: 'بيانات النشر المُدار غير متاحة. لم يبدأ أي عمل جديد.',
    stale: 'قد تكون لقطة النشر هذه قديمة. أعد الفحص قبل اتخاذ إجراء.',
    reconnecting: 'جار إعادة الاتصال بحالة النشر المُدار…'
  },
  descriptions: {
    serialCapability: 'يتطلب عقد الهدف النشط قدرة تسلسلية. يظل هدف الاختبار المصغر المحدد ثابتًا حتى يتغير المسودّة.',
    canonicalPlan: 'تطابق المسودة المقدمة خطة الموجات القانونية التي تمت معاينتها تمامًا. تتطلب الصفوف المجددة أو المتغيرة تأكيدًا جديدًا.',
    manualPolicy: 'تُطلب موافقة يدوية بعد كل موجة تمت تسويتها.',
    automaticPolicy: 'لا يُسمح بالتقدم التلقائي إلا بعد موافقة صريحة على الاختبار المصغر وفحص دليل جديد مع بقاء النشر سليمًا.',
    commitmentBoundary: 'قد يكتمل التحديث الملتزم به بعد إيقاف النشر. يمنع الإيقاف عمليات التفويض الجديدة لكنه لا يستطيع إلغاء العمل الذي سُلّم إلى مُحدّث بعيد؛ فقد يواصل تفريغ الجلسات وتطبيق الإصدار واسترداد نطاق الإعدادات، ولا يعني ذلك أن النتيجة البعيدة أصبحت ناجحة.',
    lastSettledLocalFenceNotLive: 'هذه حالة آخر تسوية مع حاجز محلي وليست حالة مباشرة. لا يثبت هذا العرض أي تدهور بعيد لم يرصده Hermes Desktop.',
    archiveRetainsFence: 'تغير الأرشفة العرض والسجل فقط؛ ولا تحذف الدليل أو تطلق حاجزًا غير محلول.',
    noAutomaticResume: 'بعد إعادة تشغيل Desktop، أعد التوفيق أولًا ثم اطلب الاستئناف أو الترقية صراحةً؛ فالسياسة المحفوظة ليست إذنًا متجددًا.',
    historicalEstimate: 'التقديرات التاريخية للمعلومات فقط ولا تمنح إذنًا لبدء عمل جديد.'
  },
  summary: {
    outcome: phase => `النتيجة: ${phase}`,
    excludedTargets: count => `الأهداف المستبعدة: ${count}`,
    unresolvedFences: count => `الحواجز غير المحلولة: ${count}`,
    archive: archived => `الأرشيف: ${archived ? 'مؤرشف' : 'نشط'}`,
    reason: reason => `السبب: ${reason}`
  },
  a11y: {
    confirmPreparation: 'أؤكد أن هذه الأهداف هي مجموعة الإعداد المنفصلة المقصودة.',
    confirmPreflight: 'أؤكد مراجعة الفحص المسبق هذه.',
    selectTarget: label => `تحديد الهدف ${label}`,
    selectedTarget: label => `إلغاء تحديد الهدف ${label}`,
    warning: message => `تحذير: ${message}`,
    status: message => `الحالة: ${message}`,
    action: message => `الإجراء: ${message}`,
    historyEntry: id => `فتح تفاصيل النشر ${id}`,
    attemptToggle: (installId, expanded) => `${expanded ? 'طي' : 'توسيع'} تفاصيل محاولة الهدف ${installId}`
  }
}

export const managedRolloutsAr = mergeTranslations<ManagedRolloutMessages>(managedRolloutsEn, overrides)
export const managedRollouts = managedRolloutsAr
export default managedRolloutsAr
