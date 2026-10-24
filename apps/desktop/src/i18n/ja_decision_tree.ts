import type { DecisionTreeCopy } from './types_decision_tree'

export const jaDecisionTree: DecisionTreeCopy = {
  title: '判断',
  agentsPick: 'エージェントの選択',
  empty: '待機中の判断はありません。',
  other: 'その他…',
  otherPlaceholder: '指示を入力…',
  pickBranch: '分岐を選ぶ',
  showNote: 'メモを表示',
  nextDecision: '次の判断',
  previousDecision: '前の判断',
  risk: {
    high: '高リスク',
    low: '低リスク',
    medium: '中リスク'
  },
  chose: o => `${o} を選択しました`,
  instructionSent: '指示を送信しました',
  pending: '回答中…'
}
