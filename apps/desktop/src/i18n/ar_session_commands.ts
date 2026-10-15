// Typed /undo /retry /steer /queue lines (en_session_commands.ts is the source).
export const arSessionCommands = {
  undoUsage: 'الاستخدام: /undo [N]',
  undoNothing: 'لا يوجد ما يمكن التراجع عنه',
  undidOne: (count: string) => `تم التراجع عن ${count} رسالة — الرسالة المحذوفة موجودة في مربع الكتابة`,
  undidOther: (count: string) => `تم التراجع عن ${count} رسائل — الرسالة المحذوفة موجودة في مربع الكتابة`,
  retryNothing: 'لا يوجد ما يمكن إعادة محاولته',
  retryUnsupported: '/retry: لا يمكن إعادة محاولة الوسائط أو أجزاء المحتوى غير المعروفة',
  steerUsage: 'الاستخدام: /steer <الطلب>',
  steerQueued: (preview: string) => `أُضيف التوجيه إلى قائمة الانتظار — يصل بعد استدعاء الأداة التالي: "${preview}"`,
  steerNoActiveTurn: (preview: string) => `لا يوجد دور نشط — أُضيف إلى قائمة الانتظار للدور التالي: "${preview}"`,
  steerRejected: 'رُفض التوجيه — لا يوجد دور نشط، أُضيف إلى قائمة الانتظار للدور التالي',
  queued: (preview: string) => `في قائمة الانتظار: "${preview}"`,
  queueFailed: 'تعذّرت إضافة الرسالة إلى قائمة الانتظار',
  countOne: (count: string) => `${count} رسالة في قائمة الانتظار`,
  countOther: (count: string) => `${count} رسائل في قائمة الانتظار`
}
