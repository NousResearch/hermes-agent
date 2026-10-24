import type { DecisionTreeCopy } from './types_decision_tree'

export const enDecisionTree: DecisionTreeCopy = {
  title: 'Decisions',
  agentsPick: "the agent's pick",
  empty: 'Nothing is waiting on a decision.',
  other: 'Other…',
  otherPlaceholder: 'Type your instruction…',
  pickBranch: 'Pick a branch',
  showNote: 'Show the note',
  nextDecision: 'Next decision',
  previousDecision: 'Previous decision',
  risk: {
    high: 'High risk',
    low: 'Low risk',
    medium: 'Medium risk'
  },
  chose: o => `Chose ${o}`,
  instructionSent: 'Instruction sent',
  pending: 'Answering…'
}
