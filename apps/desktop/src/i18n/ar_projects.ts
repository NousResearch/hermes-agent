import type { TranslationOverrides } from './define-locale'

export const arProjects = {
  projects: {
    search: 'ابحث في المشاريع...',
    refresh: 'تحديث المشاريع',
    refreshing: 'جارٍ تحديث المشاريع',
    loading: 'جارٍ تحميل المشاريع',
    emptyTitle: 'لا توجد مشاريع بعد',
    emptyDesc: 'أنشئ مشروعًا لتجمع مجلداته ومستودعاته وجلساته في مكان واحد.',
    noMatchesTitle: 'لا توجد مشاريع مطابقة',
    unavailableTitle: 'المشاريع غير متاحة',
    unavailableDesc: 'الخادم الخلفي لـ Hermes المتصل لا يدعم المشاريع بعد. حدّث Hermes لاستخدامها.',
    loadFailedTitle: 'تعذّر تحميل المشاريع',
    loadFailedDesc: 'لم يُرجع Hermes قائمة المشاريع. تحقّق من الاتصال ثم أعد المحاولة.',
    partialFailed: 'تعذّر تحديث كل تفاصيل المشاريع. تُعرض آخر معلومات محمّلة.',
    incompleteProfiles: 'تعذّرت قراءة بعض الملفات الشخصية، لذا قد تكون هذه القائمة غير مكتملة.',
    selectTitle: 'اختر مشروعًا',
    selectDesc: 'اختر مشروعًا لعرض مجلداته ومستودعاته وجلساته.',
    autoDiscovered: 'مستودع مكتشف تلقائيًا',
    sessionCount: count => (count === 1 ? 'جلسة واحدة' : `${count} جلسات`),
    primaryPath: 'المجلد الرئيسي',
    noPath: 'بلا مجلد',
    folders: 'المجلدات',
    primaryFolder: 'رئيسي',
    repositories: 'المستودعات',
    noRepositories: 'لا توجد مستودعات git في هذا المشروع بعد.',
    laneMain: 'النسخة الرئيسية',
    laneWorktree: 'شجرة عمل',
    laneKanban: 'أشجار عمل مهام Kanban',
    activeSessions: 'نشطة الآن',
    noActiveSessions: 'لا يعمل أي وكيل في هذا المشروع الآن.',
    activityUnknown: 'لا يمكن التأكد من الوكلاء الذين يعملون حتى تُحمَّل كل جلسات هذا المشروع.',
    sessions: 'الجلسات',
    noSessions: 'لا توجد جلسات في هذا المشروع بعد.',
    sessionsFailed: 'تعذّر تحميل كل جلسات هذا المشروع. تُعرض أحدث الجلسات.',
    allProfilesLimited:
      'لا يمكن لعرض «كل الملفات الشخصية» سرد جلسات مشروع واحد. اختر ملفًا شخصيًا واحدًا من الشريط الجانبي لعرض كل جلسات هذا المشروع.',
    untitledSession: 'جلسة بلا عنوان',
    status: {
      background: 'يعمل في الخلفية',
      'needs-input': 'يحتاج إدخالاً',
      stalled: 'متوقف',
      working: 'يعمل'
    },
    openArtifacts: 'العناصر',
    openKanban: 'Kanban',
    showInSidebar: 'إظهار في الشريط الجانبي',
    openOverview: 'فتح نظرة عامة على المشروع'
  }
} satisfies Pick<TranslationOverrides, 'projects'>
