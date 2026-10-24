import type { DecisionTreeCopy } from './types_decision_tree'

export const arDecisionTree: DecisionTreeCopy = {
  title: 'القرارات',
  agentsPick: 'اختيار الوكيل',
  empty: 'لا يوجد قرار بانتظارك.',
  other: 'أخرى…',
  otherPlaceholder: 'اكتب تعليماتك…',
  pickBranch: 'اختر فرعًا',
  showNote: 'إظهار الملاحظة',
  nextDecision: 'القرار التالي',
  previousDecision: 'القرار السابق',
  risk: {
    high: 'خطورة عالية',
    low: 'خطورة منخفضة',
    medium: 'خطورة متوسطة'
  },
  chose: o => `تم اختيار ${o}`,
  instructionSent: 'تم إرسال التعليمات',
  pending: 'جارٍ الرد…'
}
