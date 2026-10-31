import type { QueueFairCopy } from './types_queue_fair'

export const arQueueFair: QueueFairCopy = {
  holdUnknown: reason => `في الانتظار: ${reason}`,
  hold: {
    session_busy: 'بانتظار انتهاء الدور الحالي',
    slot: 'بانتظار توفر خانة جلسة',
    editing: 'قيد التحرير، لن تبدأ بعد',
    restarting: 'بانتظار إعادة تشغيل البوابة',
    paused: 'متوقفة حتى يُحسم الدور المفقود',
    provider_limit: 'بانتظار إعادة تعيين حد المزوّد',
    memory: 'بانتظار توفر ذاكرة'
  },
  moveUp: 'نقل لأعلى',
  moveDown: 'نقل لأسفل',
  holdFailed: 'تعذّر إيقاف هذه الرسالة المنتظرة مؤقتًا أثناء تحريرها؛ قد تبدأ قبل الحفظ'
}
