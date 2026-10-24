/** The decision tree's own chrome. Questions, option labels and consequences
 *  are content: they arrive from the agent, not from the catalog. */
export interface DecisionTreeCopy {
  title: string
  agentsPick: string
  empty: string
  other: string
  otherPlaceholder: string
  pickBranch: string
  showNote: string
  nextDecision: string
  previousDecision: string
  risk: { high: string; low: string; medium: string }
  /** Settled receipt for a pick; `option` is the label (a two-level route
   *  arrives as "branch · follow-up"). */
  chose: (option: string) => string
  instructionSent: string
  pending: string
}
