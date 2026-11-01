import type { DelegationRoutingTranslations } from './types_delegation_routing'

export const arDelegationRouting: DelegationRoutingTranslations = {
  title: 'التوجيه',
  settingsTitle: 'الإعدادات الافتراضية للوكلاء الفرعيين',
  intro: 'كيف يحصل كل وكيل فرعي على نموذجه.',
  modes: {
    default: { label: 'افتراضي', description: 'يعمل كل وكيل فرعي بنموذج الوكيل الفرعي أدناه.' },
    model_map: { label: 'مسارات النماذج', description: 'سمِّ بضعة نماذج، ويختار الوكيل واحدًا منها لكل وكيل فرعي.' },
    profile_routes: {
      label: 'مسارات الملفات الشخصية',
      description: 'يعمل الوكلاء الفرعيون بأحد ملفاتك الشخصية، بنموذجه ومهاراته ومفاتيحه الخاصة.'
    }
  },
  inUse: 'قيد الاستخدام',
  plugin: 'إضافة',
  pluginMissing: 'مسارات النماذج ومسارات الملفات الشخصية مصدرها الإضافة Advanced Delegation.',
  pluginDisabled: 'الإضافة Advanced Delegation مثبتة لكنها متوقفة لهذا الملف الشخصي.',
  pluginNotLoaded: 'الإضافة Advanced Delegation مفعّلة، لكن Hermes لم يحمّلها بعد. أعد تشغيل Hermes لإكمال ذلك.',
  installPlugin: 'تثبيت الإضافة',
  installUnavailable: 'الإضافة Advanced Delegation ليست في كتالوج الإضافات بعد. ثبّتها من المهارات ← الإضافات.',
  enablePlugin: 'تفعيل',
  enableFailed: 'تعذّر تفعيل Advanced Delegation',
  modeFailed: 'تعذّر تغيير توجيه الوكلاء الفرعيين',
  loadFailed: 'تعذّر تحميل توجيه الوكلاء الفرعيين',
  blockedPill: 'لا يعمل',
  blockedDetail: 'هذا الوضع محفوظ، لذا يرفض Hermes كل وكيل فرعي بدلًا من استخدام الوضع الافتراضي بصمت.',
  blockedMissing: 'لن يبدأ الوكلاء الفرعيون حتى تُثبَّت الإضافة Advanced Delegation.',
  blockedDisabled: 'لن يبدأ الوكلاء الفرعيون حتى تُفعَّل الإضافة Advanced Delegation لهذا الملف الشخصي.',
  blockedNotLoaded:
    'لن يبدأ الوكلاء الفرعيون حتى يحمّل Hermes الإضافة Advanced Delegation. أعد تشغيل Hermes لإكمال ذلك.',
  blockedUnavailable: label => `لن يبدأ الوكلاء الفرعيون حتى يتمكن ${label} من العمل هنا. تفيد إضافته بأنه لا يستطيع.`,
  blockedUnregistered: name => `لن يبدأ الوكلاء الفرعيون حتى تُثبَّت وتُفعَّل إضافة توفّر «${name}».`,
  thirdPartyDescription: 'من إضافة. إعداداته موجودة في تلك الإضافة.',

  modelRoutesTitle: 'مسارات النماذج',
  modelRoutesEmpty: 'لا توجد مسارات بعد. إلى أن تضيف مسارًا، يستخدم الوكلاء الفرعيون نموذج الوكيل الرئيسي.',
  addRoute: 'إضافة مسار',
  editRoute: 'تعديل',
  makeDefault: 'تعيين كافتراضي',
  deleteRoute: 'حذف',
  defaultBadge: 'افتراضي',
  routeActions: name => `إجراءات ${name}`,
  deleteConfirmTitle: name => `حذف المسار «${name}»؟`,
  deleteConfirmBody: 'لن يتمكن الوكلاء الفرعيون من اختياره بعد الآن.',
  deleteConfirmNextDefault: name => `هذا هو المسار الافتراضي، لذا يصبح «${name}» هو الافتراضي.`,
  deleteConfirmAction: 'حذف المسار',
  saveFailed: 'تعذّر حفظ المسارات',
  baseUrlRoute: url => `نقطة النهاية ${url}`,

  dialogNew: 'مسار نموذج جديد',
  dialogEdit: 'تعديل مسار النموذج',
  dialogDesc: 'يقرأ الوكيل الاسم والوصف ليقرر أي مسار يسلكه الوكيل الفرعي.',
  nameLabel: 'الاسم',
  namePlaceholder: 'مثال: fast',
  nameHint: 'قصير، بلا مسافات. يختار الوكيل المسارات بهذا الاسم.',
  descriptionLabel: 'متى يُستخدم',
  descriptionPlaceholder: 'مثال: رخيص وسريع: عمليات البحث، والملخصات، والتعديلات البسيطة.',
  descriptionHint: 'سطر واحد يقرؤه الوكيل قبل الاختيار.',
  providerLabel: 'المزوّد',
  providerPlaceholder: 'اختر مزوّدًا',
  modelLabel: 'النموذج',
  effortLabel: 'مستوى الاستدلال',
  effortInherit: 'الافتراضي للوكيل الفرعي',
  tierLabel: 'السرعة',
  tierNormal: 'عادية',
  tierFast: 'سريعة',
  tierUltrafast: 'فائقة السرعة',
  tierHint: 'فقط في النماذج التي توفرها.',
  optional: 'اختياري',
  errors: { required: 'مطلوب', spaces: 'بلا مسافات', duplicate: 'مسار آخر يستخدم هذا الاسم بالفعل' },
  saveRoute: 'حفظ المسار',
  createRoute: 'إضافة مسار',

  profileRoutesTitle: 'الملفات الشخصية التي يمكن للوكلاء الفرعيين العمل بها',
  profileRoutesHint: 'يجلب كل ملف شخصي محدد نموذجه ومهاراته وشخصيته ومفاتيحه الخاصة.',
  profileDescriptionPlaceholder: 'متى يُستخدم (اختياري)',
  profileDefault: 'افتراضي',
  profileEmpty: 'لا توجد ملفات شخصية أخرى بعد. أنشئ واحدًا من شريط الملفات الشخصية.',
  profileMissing: name => `${name} (لم يعد الملف الشخصي موجودًا)`,

  inactiveTitle: 'غير مستخدم في هذا الوضع',
  inactiveModelMap: route =>
    `تختار مسارات النماذج نموذج كل وكيل فرعي. الوكلاء الفرعيون الذين لا يختارون مسارًا يستخدمون «${route}».`,
  inactiveProfiles: profile =>
    `تختار مسارات الملفات الشخصية ملف كل وكيل فرعي. الوكلاء الفرعيون الذين لا يختارون ملفًا شخصيًا يعملون بـ «${profile}».`,
  inactiveModelMapNoDefault:
    'تختار مسارات النماذج نموذج كل وكيل فرعي. الوكلاء الفرعيون الذين لا يختارون مسارًا يستخدمون نموذج الوكيل الرئيسي.',
  inactiveProfilesNoDefault:
    'تختار مسارات الملفات الشخصية ملف كل وكيل فرعي. الوكلاء الفرعيون الذين لا يختارون ملفًا شخصيًا يستخدمون نموذج الوكيل الرئيسي.',
  inactivePlugin: label => `${label} يحدد نموذج كل وكيل فرعي، لذا لا ينطبق هذا الإعداد.`,
  useDefault: 'التبديل إلى الافتراضي'
}
