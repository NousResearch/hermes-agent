import type { DecisionTreeCopy } from './types_decision_tree'

export const zhHantDecisionTree: DecisionTreeCopy = {
  title: '決策',
  agentsPick: 'Agent 的選擇',
  empty: '沒有等待處理的決策。',
  other: '其他…',
  otherPlaceholder: '輸入你的指令…',
  pickBranch: '選擇分支',
  showNote: '查看說明',
  nextDecision: '下一個決策',
  previousDecision: '上一個決策',
  risk: {
    high: '高風險',
    low: '低風險',
    medium: '中風險'
  },
  chose: o => `已選擇：${o}`,
  instructionSent: '指令已送出',
  pending: '正在回答…'
}
