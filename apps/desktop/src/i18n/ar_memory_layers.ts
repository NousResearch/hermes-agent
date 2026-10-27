import type { MemoryLayersTranslations } from './types_memory_layers'

export const arMemoryLayers: MemoryLayersTranslations = {
  activeProvider: 'مزوّد الذاكرة النشط',
  noProvider: 'لا يوجد — الذاكرة المدمجة فقط',
  noProviderDesc: 'يستخدم Hermes الطبقة المدمجة أدناه فقط. اختر مزوّدًا خارجيًا في «مزوّد الذاكرة» لإضافته.',
  providerDesc: name => `يضيف ${name} استرجاعه الخاص إلى كل جلسة، فوق الطبقة المدمجة.`,
  status: {
    ready: 'جاهز',
    needs_config: 'يحتاج إلى إعداد',
    unavailable: 'غير متاح',
    missing: 'غير مثبّت'
  },
  builtinLayer: 'الطبقة المدمجة',
  alwaysOn: 'مفعّلة دائمًا',
  builtinDesc:
    'يُحمَّل MEMORY.md وUSER.md في كل جلسة، مع مزوّد خارجي أو بدونه. يضيف المزوّد إلى هذه الطبقة ولا يستبدلها أبدًا.'
}
