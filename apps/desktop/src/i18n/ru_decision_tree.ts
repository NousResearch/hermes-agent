import type { DecisionTreeCopy } from './types_decision_tree'

export const ruDecisionTree: DecisionTreeCopy = {
  title: 'Решения',
  agentsPick: 'выбор агента',
  empty: 'Нет решений, ожидающих ответа.',
  other: 'Другое…',
  otherPlaceholder: 'Введите указание…',
  pickBranch: 'Выберите ветку',
  showNote: 'Показать заметку',
  nextDecision: 'Следующее решение',
  previousDecision: 'Предыдущее решение',
  risk: {
    high: 'Высокий риск',
    low: 'Низкий риск',
    medium: 'Средний риск'
  },
  chose: o => `Выбрано: ${o}`,
  instructionSent: 'Указание отправлено',
  pending: 'Отвечает…'
}
