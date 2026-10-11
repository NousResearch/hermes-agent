import type { TranslationOverrides } from './define-locale'

// Shell notices (remote-display toast, butterbar), spread into ru.ts.
export const ruNotices = {
  remoteDisplayBanner: {
    message: reason =>
      `Включён программный рендеринг — обнаружен удалённый дисплей (${reason}). GPU-ускорение отключено, чтобы избежать мерцания.`
  },
  previewDraft: {
    discardTitle: 'Отменить несохранённые изменения?',
    discardBody: label => `В ${label} есть несохранённые изменения. Если закрыть вкладку, они пропадут.`,
    discardConfirm: 'Отменить изменения'
  },
  butterbar: {
    goTo: (index, total) => `Показать уведомление ${index} из ${total}`,
    legal: {
      before: 'Использование Hermes Agent регулируется нашими ',
      terms: 'Условиями обслуживания',
      between: ' и ',
      privacy: 'Политикой конфиденциальности',
      after: '.'
    }
  }
} satisfies Pick<TranslationOverrides, 'remoteDisplayBanner' | 'previewDraft' | 'butterbar'>
