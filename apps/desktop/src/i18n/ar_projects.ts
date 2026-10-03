import type { TranslationOverrides } from './define-locale'

export const arProjects = {
  projects: {
    search: 'ابحث في المشاريع...',
    refresh: 'تحديث المشاريع',
    refreshing: 'جارٍ تحديث المشاريع',
    loading: 'جارٍ تحميل المشاريع',
    emptyTitle: 'لا توجد مشاريع بعد',
    emptyDesc: 'أنشئ مشروعًا لتجمع مجلداته ومستودعاته وجلساته في مكان واحد.',
    newProject: 'مشروع جديد',
    noMatchesTitle: 'لا توجد مشاريع مطابقة',
    unavailableTitle: 'المشاريع غير متاحة',
    unavailableDesc: 'الخادم الخلفي لـ Hermes المتصل لا يدعم المشاريع بعد. حدّث Hermes لاستخدامها.',
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
    sessions: 'الجلسات',
    noSessions: 'لا توجد جلسات في هذا المشروع بعد.',
    sessionsFailed: 'تعذّر تحميل كل جلسات هذا المشروع. تُعرض أحدث الجلسات.',
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
