import type { ReactNode, Ref } from 'react'

/** A consequence the agent wants to explain: the line stays plain, the note
 *  carries the worry. `source` is shown under the note when the agent has one. */
export interface DecisionEffectFlag {
  text: string
  note: string
  source?: string
}

/** What a branch leads to, one line each. */
export type DecisionEffect = DecisionEffectFlag | string

/** A second question this branch leads to (one level deep). */
export interface DecisionFollowUp {
  question: string
  options: DecisionOption[]
}

export interface DecisionOption {
  label: string
  /** Logo of the tool the branch runs with. */
  icon?: ReactNode
  /** Logos of the systems the branch touches. */
  apps?: ReactNode[]
  effects?: DecisionEffect | DecisionEffect[]
  /** The agent's own preference. At most one per level. */
  pick?: boolean
  next?: DecisionFollowUp
}

export type DecisionRisk = 'high' | 'low' | 'medium'

export interface Decision {
  id: string
  question: string
  context?: string
  /** Who is asking — an agent name, a person, a service. */
  agent: string
  model?: string
  icon?: ReactNode
  risk?: DecisionRisk
  options: DecisionOption[]
}

/** A route the user picked. `then` carries the follow-up's pick, so a two-level
 *  route reaches the agent as one choice; `other` carries a written instruction
 *  (which arrives with `index: -1` and the instruction as the option's label). */
export interface DecisionChoice {
  index: number
  option: DecisionOption
  other?: string
  then?: DecisionOption
}

/** Drives the tree from outside — a demo, a replay, a canvas the agent paints.
 *  Every method acts on the decision the tree currently has active. */
export interface DecisionTreeHandle {
  hover: (index: number | null) => void
  peek: (branch?: number) => void
  choose: (index: number) => void
  next: () => void
  previous: () => void
  setOther: (text: string) => void
  send: () => void
}

export interface DecisionTreeProps {
  decisions: Decision[]
  /** Called with the whole route. A returned string is the agent's short answer
   *  and settles into the outcome line; a promise is awaited. */
  onChoose?: (decision: Decision, choice: DecisionChoice) => Promise<string | void> | string | void
  allowOther?: boolean
  title?: string
  className?: string
  ref?: Ref<DecisionTreeHandle>
}
