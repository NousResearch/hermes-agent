import type { WebhookTestCopy } from './types_webhook_test'

export const arWebhookTest: Partial<WebhookTestCopy> = {
  title: 'اختبار هذا الـ Webhook',
  hint: 'الصقه في الطرفية لإرسال حدث موقّع واحد. لكل تشغيل معرّف تسليم خاص به.',
  placeholderHint: 'استبدل العنصر النائب بالسر الذي ظهر عند إنشاء الاشتراك، أو استخدم «إرسال اختبار».',
  copyCommand: 'نسخ الأمر',
  send: 'إرسال اختبار',
  sending: 'جارٍ الإرسال…',
  accepted: (status, deliveryId) => `تم تسليم الاختبار (${status}) · ${deliveryId}`,
  rejected: (status, detail) => `ردّ المستقبِل ${status}: ${detail}`,
  failed: 'تعذّر إرسال الاختبار'
}
