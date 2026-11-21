import type { InboxCopy } from './types_inbox'

export const arInbox: InboxCopy = {
  title: 'صندوق الوارد',
  unreadCount: (count: number) => `${count} غير مقروءة`,
  loading: 'جارٍ تحميل صندوق الوارد…',
  emptyTitle: 'لا توجد نتائج بعد',
  emptyDesc: 'عند انتهاء مهمة مجدولة، تصل نتيجتها إلى هنا.',
  running: 'قيد التشغيل',
  unread: 'غير مقروءة',
  noPreview: 'انتهت دون رد.',
  refresh: 'تحديث',
  refreshing: 'جارٍ التحديث…',
  markReadFailed: 'تعذّر تعليم النتيجة كمقروءة',
  newResultsTitle: (count: number) => `${count} نتائج جديدة للمهام المجدولة`,
  newResultsBody: 'افتح صندوق الوارد لقراءتها.'
}
