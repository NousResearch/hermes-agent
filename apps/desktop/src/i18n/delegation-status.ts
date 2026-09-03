import type { Locale } from './types'

/** Delegation-only copy kept with bounded outcome presentation owners. */
export interface DelegationStatusCopy {
  dispatched: string
  partial: string
  unverified: string
  verificationRequired: string
}

export const DELEGATION_STATUS_COPY = {
  ar: {
    dispatched: 'تم الإرسال',
    partial: 'جزئي',
    unverified: 'غير مُتحقَّق منه',
    verificationRequired: 'التحقق مطلوب'
  },
  en: {
    dispatched: 'Dispatched',
    partial: 'Partial',
    unverified: 'Unverified',
    verificationRequired: 'Verification required'
  },
  ja: {
    dispatched: 'ディスパッチ済み',
    partial: '部分的',
    unverified: '未検証',
    verificationRequired: '検証が必要'
  },
  ru: {
    dispatched: 'Отправлено',
    partial: 'Частично',
    unverified: 'Не проверено',
    verificationRequired: 'Требуется проверка'
  },
  zh: {
    dispatched: '已分派',
    partial: '部分完成',
    unverified: '未验证',
    verificationRequired: '需要验证'
  },
  'zh-hant': {
    dispatched: '已分派',
    partial: '部分完成',
    unverified: '未驗證',
    verificationRequired: '需要驗證'
  }
} satisfies Record<Locale, DelegationStatusCopy>
