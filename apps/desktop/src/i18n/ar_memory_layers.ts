import type { MemoryLayersTranslations } from './types_memory_layers'

export const arMemoryLayers: MemoryLayersTranslations = {
  activeProvider: 'مزوّد الذاكرة النشط',
  selectedProvider: 'مزوّد الذاكرة المحدد',
  selectedDesc: current => `غير نشط بعد: تواصل الجلسات استخدام ${current} حتى يُحفظ هذا الاختيار.`,
  builtinMemory: 'الذاكرة المدمجة فقط',
  noProvider: 'لا يوجد — الذاكرة المدمجة فقط',
  noProviderDesc:
    'يستخدم Hermes الطبقة المدمجة أدناه فقط. اختر مزوّدًا خارجيًا من الإعدادات › الذاكرة والسياق لإضافة واحد.',
  providerDesc: name => `يضيف ${name} استرجاعه الخاص إلى كل جلسة، فوق الطبقة المدمجة.`,
  providerNotReadyDesc: name =>
    `${name} هو المزوّد المحفوظ لكنه لا يضيف أي استرجاع حتى يصبح جاهزًا. تعمل الطبقة المدمجة في كل الأحوال.`,
  providerUnknownDesc: name => `${name} هو المزوّد المحفوظ؛ تعذّر على Hermes معرفة ما إذا كان جاهزًا.`,
  statusUnavailable: 'حالة الذاكرة غير متاحة',
  statusUnavailableDesc: 'تعذّرت قراءة مزوّد الذاكرة النشط وما إذا كان جاهزًا.',
  retry: 'إعادة المحاولة',
  status: {
    ready: 'جاهز',
    needs_config: 'يحتاج إلى إعداد',
    unavailable: 'غير متاح',
    missing: 'غير مثبّت'
  },
  builtinLayer: 'الطبقة المدمجة',
  onByDefault: 'مفعّلة افتراضيًا',
  on: 'مفعّلة',
  off: 'معطّلة',
  onlyFile: file => `${file} فقط`,
  builtinDesc:
    'MEMORY.md وUSER.md هما ذاكرة Hermes الخاصة، مستقلتان عن أي مزوّد خارجي. يُحمَّل كلاهما في الجلسات افتراضيًا؛ ويعطّل مفتاحا الذاكرة وملف المستخدم في الإعدادات › الذاكرة والسياق كلًّا منهما. يضيف المزوّد إلى هذه الطبقة ولا يستبدلها أبدًا.'
}
