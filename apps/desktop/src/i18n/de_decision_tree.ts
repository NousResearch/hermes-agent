import type { DecisionTreeCopy } from './types_decision_tree'

export const deDecisionTree: DecisionTreeCopy = {
  title: 'Entscheidungen',
  agentsPick: 'die Wahl des Agenten',
  empty: 'Keine Entscheidung offen.',
  other: 'Andere…',
  otherPlaceholder: 'Anweisung eingeben…',
  pickBranch: 'Zweig wählen',
  showNote: 'Notiz anzeigen',
  nextDecision: 'Nächste Entscheidung',
  previousDecision: 'Vorherige Entscheidung',
  risk: {
    high: 'Hohes Risiko',
    low: 'Geringes Risiko',
    medium: 'Mittleres Risiko'
  },
  chose: o => `Gewählt: ${o}`,
  instructionSent: 'Anweisung gesendet',
  pending: 'Antwortet…'
}
