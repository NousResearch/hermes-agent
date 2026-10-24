import type { DecisionTreeCopy } from './types_decision_tree'

export const esDecisionTree: DecisionTreeCopy = {
  title: 'Decisiones',
  agentsPick: 'la opción del agente',
  empty: 'No hay ninguna decisión pendiente.',
  other: 'Otra…',
  otherPlaceholder: 'Escribe tu instrucción…',
  pickBranch: 'Elige una rama',
  showNote: 'Ver la nota',
  nextDecision: 'Siguiente decisión',
  previousDecision: 'Decisión anterior',
  risk: {
    high: 'Riesgo alto',
    low: 'Riesgo bajo',
    medium: 'Riesgo medio'
  },
  chose: o => `Elegido: ${o}`,
  instructionSent: 'Instrucción enviada',
  pending: 'Respondiendo…'
}
