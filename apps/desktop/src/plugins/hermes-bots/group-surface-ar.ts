import { GROUP_CHAT_LOCALES } from './group-chat-locales'
import type { BotsMessages } from './i18n-messages'
import { ROSTER_CHROME } from './roster-chrome-locales'

export const GROUP_SURFACE_AR = {
  group: {
    ...GROUP_CHAT_LOCALES.ar,
    newTitle: 'محادثة مجموعة جديدة',
    manageDesc: 'يمكن للوكيل الانضمام إلى عدة محادثات مجموعات. تُزامن العضويات على جميع أجهزة الكمبيوتر.',
    manageTitle: 'إدارة المجموعات',
    settingsTitle: 'إعدادات المجموعة',
    settingsDesc: 'غيّر اسم المجموعة أو صورتها. يبقى الأعضاء والسجل محفوظين.',
    nameLabel: 'اسم المجموعة',
    holdDetection: 'اكتشاف تعليمات الإيقاف',
    holdDetectionHint: 'توقف تعليمات الإيقاف في محادثة المجموعة الأعضاء المعنيين مؤقتًا حتى تتم الإشارة إليهم مجددًا.',
    compressHistory: 'ضغط السجل',
    compressHistoryHint: member => `ضغط سجل محادثة المجموعة المخفي للعضو ${member} لتجنب الردود الفارغة`,
    compressing: member => `جارٍ ضغط سجل محادثة المجموعة للعضو ${member}…`,
    compressDone: (member, compressed, detail) =>
      `تم ضغط جلسات محادثة المجموعة للعضو ${member}، العدد: ${compressed}${detail ? ` — ${detail}` : ''}`,
    compressNothing: member => `لا يوجد ما يُضغط للعضو ${member} — لا توجد جلسات محادثة مجموعة بعد`,
    compressFailed: (member, error) => `تعذّر ضغط سجل محادثة المجموعة للعضو ${member}: ${error}`,
    searchToAdd: 'البحث عن وكلاء لإضافتها',
    searchToAddPlaceholder: 'ابحث عن وكلاء لإضافتها…',
    removeFromSelection: 'إزالة من التحديد',
    disbandTitle: 'حلّ محادثة المجموعة؟',
    deleteTitle: 'حذف محادثة المجموعة؟',
    deleteAction: 'حذف',
    composerPlaceholder: 'اكتب رسالة — سيراها كل وكيل في محادثة المجموعة هذه.',
    slashCommandsUnsupported:
      'الأوامر التي تبدأ بشرطة مائلة غير متاحة في محادثات المجموعات. افتح محادثة فردية مع وكيل لاستخدامها.',
    attachHint: 'إرفاق ملفات — يراها كل وكيل يرد',
    newThread: 'سلسلة جديدة',
    reply: 'رد',
    replyInThread: 'الرد في السلسلة',
    replyInThreadPlaceholder: 'الرد في السلسلة…',
    openThread: 'فتح هذه السلسلة',
    collapseThread: 'طي السلسلة',
    collapseThreadLabel: 'طي هذه السلسلة',
    activity: 'النشاط',
    noActivityYet: 'لا يوجد نشاط في هذا الدور بعد.',
    showActivity: 'إظهار نشاط محادثة المجموعة',
    hideActivity: 'إخفاء نشاط محادثة المجموعة',
    stop: 'إيقاف',
    stopHint: 'إيقاف هذا التشغيل — يقاطع العضو النشط ويوقف البقية مؤقتًا',
    heldMembersStatus: members => `متوقفون مؤقتًا: ${members}`,
    holdReleaseHint: 'أشر إلى وكيل متوقف مؤقتًا أو أرسل @all resume للاستئناف.',
    needsYourInput: 'يحتاج وكيل في محادثة المجموعة هذه إلى ردك',
    noMembersToSend: group =>
      `لا يوجد أعضاء لإرسال الرسالة إليهم في «${group}» — أضف وكيلًا أو أعد فتح محادثة المجموعة إذا كان تحميل الأعضاء مستمرًا.`,
    pictureGenerationFailed: 'تعذّر إنشاء صورة المجموعة',
    nameTaken: name => `توجد مجموعة باسم «${name}» بالفعل.`,
    you: 'أنت',
    settingsHint: group => `إعدادات المجموعة — تغيير اسم «${group}» أو صورتها`,
    settingsLabel: group => `إعدادات المجموعة «${group}»`,
    disbandHint: group => `حلّ محادثة المجموعة «${group}»`,
    disbandLabel: group => `حلّ «${group}»`,
    disbandAction: 'حلّ المجموعة',
    disbanding: 'جارٍ حلّ المجموعة…',
    disbandDone: 'تم حلّ المجموعة',
    disbanded: group => `تم حلّ «${group}»`,
    stopped: group => `تم إيقاف «${group}» — الأدوار المتبقية متوقفة مؤقتًا حتى الاستئناف`,
    removeAttachment: 'إزالة المرفق',
    threadFallback: 'سلسلة',
    replyCount: replies =>
      replies === 0
        ? 'لا ردود'
        : replies === 1
          ? 'رد واحد'
          : replies === 2
            ? 'ردان'
            : `${replies} ${replies % 100 >= 3 && replies % 100 <= 10 ? 'ردود' : replies % 100 >= 11 ? 'ردًا' : 'رد'}`,
    dropToThread: 'أفلت للإرفاق بهذا الرد في السلسلة',
    dropToRoom: 'أفلت للإرفاق — يراه كل وكيل يرد',
    waitingForAnswer: 'في انتظار ردك…',
    memberThinking: name => `${name} يفكر…`,
    roomWorking: 'محادثة المجموعة تعمل…',
    messageRoom: group => `إرسال رسالة إلى «${group}»`,
    newThreadPlaceholder: group => `سلسلة جديدة في «${group}»… (@الاسم لعضو، @everyone للجميع)`,
    everyoneMeta: 'كل وكيل في محادثة المجموعة',
    commandApproval: 'الموافقة على أمر',
    answerFailed: (handle, error) => `تعذّر إرسال الرد إلى @${handle}: ${error}`,
    wantsToRunCommand: handle => `يريد @${handle} تنفيذ أمر:`,
    asks: handle => `يسأل @${handle}:`,
    answerTo: member => `الرد على @${member}`
  },
  roster: {
    ...ROSTER_CHROME.ar,
    search: 'البحث عن وكلاء ومحادثات مجموعات',
    searchPlaceholder: 'ابحث عن وكلاء ومحادثات مجموعات…',
    newBotOrGroup: 'وكيل أو محادثة مجموعة جديدة',
    groupChats: 'محادثات المجموعات',
    emptyTitle: 'لا توجد وكلاء بعد',
    emptyDesc: 'أنشئ وكيلك الأول.',
    noMatchQuery: query => `لا توجد وكلاء أو محادثات مجموعات تطابق «${query}»`,
    noMatchQueryOn: (query, gateway) => `لا توجد وكلاء أو محادثات مجموعات تطابق «${query}» على ${gateway}`,
    noMatchFiltersOn: gateway => `لا توجد وكلاء أو محادثات مجموعات تطابق هذه المرشحات على ${gateway}`,
    noMatchFilters: 'لا توجد وكلاء أو محادثات مجموعات تطابق هذه المرشحات.',
    clearFilters: 'مسح المرشحات',
    allHidden: 'جميع الوكلاء مخفية',
    allHiddenDesc: 'تواصل العمل وتحتفظ بسجلها.',
    showHidden: 'إظهار الوكلاء المخفية',
    noHiddenMatch: 'لا توجد وكلاء مخفية تطابق هذه المرشحات.',
    hiddenFromRoster: 'مخفي من القائمة',
    pinned: 'مثبّت',
    needsAttention: 'يحتاج إلى انتباه',
    needsInput: 'يحتاج إلى ردك',
    botsAndGroups: 'الوكلاء ومحادثات المجموعات',
    botsOnly: 'الوكلاء فقط',
    groupsOnly: 'محادثات المجموعات فقط',
    anyActivity: 'أي نشاط',
    activeNow: 'نشط الآن',
    recentlyActive: 'نشط مؤخرًا',
    older: 'أقدم',
    gatewayRemoved: 'أُزيل الجهاز',
    onDemand: 'عند الطلب',
    ready: 'جاهز',
    statusUnknown: 'الحالة غير معروفة',
    unavailable: 'غير متاح',
    retryNow: 'إعادة المحاولة الآن',
    rosterUnavailable: reason =>
      `القائمة غير متاحة: ${reason}. إذا كان إصدار Hermes لا يدعم profiles.list، فحدّث Hermes وأعد تشغيله.`,
    waitingForGateway: 'في انتظار الاتصال بالجهاز… (قد تستغرق الأجهزة البعيدة بضع ثوانٍ؛ تُعاد المحاولة تلقائيًا)'
  },
  sections: {
    newSection: 'قسم جديد',
    newTitle: 'قسم جديد',
    renameTitle: 'تغيير اسم القسم',
    nameLabel: 'اسم القسم',
    namePlaceholder: 'مثلًا: العملاء',
    create: 'إنشاء',
    rename: 'تغيير الاسم…',
    moveUp: 'نقل لأعلى',
    moveDown: 'نقل لأسفل',
    unassigned: 'غير معيّن',
    options: name => `خيارات القسم «${name}»`,
    headingTip: 'أفلت الوكلاء هنا · انقر مرتين لتغيير الاسم',
    emptyHint: 'اسحب الوكلاء هنا',
    moveTo: 'نقل إلى قسم',
    newSectionEllipsis: 'قسم جديد…',
    removeFromSection: 'إزالة من القسم',
    deleted: (name, count) => (count === 0 ? `تم حذف «${name}»` : `تم حذف «${name}» — عدد الوكلاء دون قسم: ${count}`),
    undo: 'تراجع'
  },
  bot: {
    newChatWith: 'محادثة جديدة مع هذا الوكيل',
    newTitle: 'وكيل جديد',
    thisDevice: 'هذا الجهاز',
    removeFromAllGroups: 'إزالة من جميع المجموعات',
    openBotChat: 'فتح محادثة الوكيل',
    openBotChatInNewTab: 'فتح محادثة الوكيل في علامة تبويب جديدة',
    openingChat: 'جارٍ فتح المحادثة…',
    pinToTop: 'تثبيت في الأعلى',
    unpin: 'إلغاء التثبيت',
    pinnedToast: name => `تم تثبيت ${name} في الأعلى`,
    unpinnedToast: name => `تم إلغاء تثبيت ${name}`,
    hide: 'إخفاء',
    unhide: 'إظهار',
    hiddenToast: name => `تم إخفاء ${name} — استخدم زر العين أعلى قائمة الوكلاء لإظهاره`,
    unhiddenToast: name => `عاد ${name} إلى القائمة`,
    groupsMenu: groups => `المجموعات: ${groups}…`,
    manageGroups: 'إدارة المجموعات…',
    metadataLoadFailed: 'تعذّر تحميل بيانات الوكيل',
    loadFailed: 'تعذّر تحميل الوكيل',
    groupsLoadFailed: 'تعذّر تحميل مجموعات الوكيل',
    attentionFallback: 'يحتاج إلى انتباه',
    attentionProviderAuth: 'سجّل الدخول مجددًا لهذا الملف الشخصي',
    attentionQuota: 'نفدت الحصة أو الرصيد',
    attentionMissingConfig: 'لم يُضبط المزوّد — نفّذ hermes model',
    attentionBlocked: 'الوكيل محظور — راجع آخر رسالة له',
    editMenu: 'تعديل…',
    duplicate: 'إنشاء نسخة',
    duplicateFailed: 'تعذّر إنشاء نسخة'
  },
  avatar: {
    upload: 'رفع',
    generate: 'إنشاء',
    generating: 'جارٍ الإنشاء…'
  },
  screen: {
    autoOpenMenu: 'فتح الشاشة عندما يستخدمها الوكيل',
    autoOpenOnToast: name => `تُفتح شاشة ${name} عندما يبدأ الوكيل باستخدام سطح مكتبه`,
    autoOpenOffToast: name => `تبقى شاشة ${name} مغلقة حتى تفتحها`,
    menu: 'فتح الشاشة'
  }
} satisfies Pick<BotsMessages, 'group' | 'roster' | 'sections'> & {
  bot: Pick<
    BotsMessages['bot'],
    | 'newChatWith'
    | 'newTitle'
    | 'thisDevice'
    | 'removeFromAllGroups'
    | 'openBotChat'
    | 'openBotChatInNewTab'
    | 'openingChat'
    | 'pinToTop'
    | 'unpin'
    | 'pinnedToast'
    | 'unpinnedToast'
    | 'hide'
    | 'unhide'
    | 'hiddenToast'
    | 'unhiddenToast'
    | 'groupsMenu'
    | 'manageGroups'
    | 'metadataLoadFailed'
    | 'loadFailed'
    | 'groupsLoadFailed'
    | 'attentionFallback'
    | 'attentionProviderAuth'
    | 'attentionQuota'
    | 'attentionMissingConfig'
    | 'attentionBlocked'
    | 'editMenu'
    | 'duplicate'
    | 'duplicateFailed'
  >
  avatar: Pick<BotsMessages['avatar'], 'upload' | 'generate' | 'generating'>
  screen: Pick<BotsMessages['screen'], 'menu' | 'autoOpenMenu' | 'autoOpenOnToast' | 'autoOpenOffToast'>
}
