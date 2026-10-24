import type { DecisionTreeCopy } from './types_decision_tree'

export const zhDecisionTree: DecisionTreeCopy = {
  title: '决策',
  agentsPick: 'Agent 的选择',
  empty: '没有等待处理的决策。',
  other: '其他…',
  otherPlaceholder: '输入你的指令…',
  pickBranch: '选择分支',
  showNote: '查看说明',
  nextDecision: '下一个决策',
  previousDecision: '上一个决策',
  risk: {
    high: '高风险',
    low: '低风险',
    medium: '中风险'
  },
  chose: o => `已选择：${o}`,
  instructionSent: '指令已发送',
  pending: '正在回答…'
}
