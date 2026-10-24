import type { DecisionTreeCopy } from './types_decision_tree'

export const frDecisionTree: DecisionTreeCopy = {
  title: 'Décisions',
  agentsPick: 'le choix de l’agent',
  empty: 'Aucune décision en attente.',
  other: 'Autre…',
  otherPlaceholder: 'Écrivez votre instruction…',
  pickBranch: 'Choisissez une branche',
  showNote: 'Afficher la note',
  nextDecision: 'Décision suivante',
  previousDecision: 'Décision précédente',
  risk: {
    high: 'Risque élevé',
    low: 'Risque faible',
    medium: 'Risque moyen'
  },
  chose: o => `Choisi : ${o}`,
  instructionSent: 'Instruction envoyée',
  pending: 'Réponse en cours…'
}
