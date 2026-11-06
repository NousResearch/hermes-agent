/**
 * Shared vocabulary for Prompt Refinement: types plus the depth order. No imports, so
 * every module (and its tests) can depend on this file without pulling in the
 * app shell.
 */

/** `off` = sends pass straight through; `manual` (shown as "Approval-First";
 *  the stored value is kept so saved preferences carry over) = Enter refines the prompt
 *  and opens the review drawer, nothing is sent until the user decides;
 *  `auto` = Enter refines and sends, stopping for review only when refinement
 *  fails or would drop something from the prompt. */
export type RefinementMode = 'auto' | 'manual' | 'off'

/** Which model refines: the `prompt_refinement` auxiliary slot (Settings ▸
 *  Model ▸ Auxiliary), or the model of the chat being typed in. */
export type RefinementModelSource = 'chat' | 'dedicated'

/** What kind of rewrite (the Depth decides how far it goes). `custom` follows
 *  the user's own instruction text (`RefinementSettings.customStyle`). */
export type RefinementStyle = 'agentic' | 'auto' | 'clarity' | 'creative' | 'custom' | 'research' | 'technical'

/** Menu and settings order. */
export const REFINEMENT_STYLES: readonly RefinementStyle[] = [
  'auto',
  'clarity',
  'technical',
  'research',
  'creative',
  'agentic',
  'custom'
]

/** How far a rewrite may go, from least to most change: `polish` fixes how the
 *  draft reads and adds nothing; `clarify` makes what the draft implies
 *  explicit; `deepen` asks for the task the way a top professional would do it. */
export type RefinementDepth = 'clarify' | 'deepen' | 'polish'

/** Slider order, least to most change. */
export const REFINEMENT_DEPTHS: readonly RefinementDepth[] = ['polish', 'clarify', 'deepen']

/** What a request asks for: a fixed depth, or `auto` (the refiner picks one of
 *  the three for this draft and reports which). */
export type RequestDepth = 'auto' | RefinementDepth

export interface RefinementSettings {
  readonly version: 1
  readonly mode: RefinementMode
  readonly modelSource: RefinementModelSource
  readonly style: RefinementStyle
  readonly depth: RefinementDepth
  /** Let the refiner pick the depth per prompt (`depth` is kept for when the
   *  user switches back to a fixed depth). */
  readonly autoDepth: boolean
  /** The user's own style guidance, used when `style` is `custom`. */
  readonly customStyle: string
  /** Send a short excerpt of this chat's recent messages with the draft. */
  readonly useChatContext: boolean
  /** Typed sends with fewer words are sent as typed (greetings, "yes"). */
  readonly minWords: number
}

/** Why a refinement produced no usable text. Each maps to its own copy. */
export type RefinementFailure =
  | 'canceled'
  | 'empty'
  | 'failed'
  | 'malformed'
  | 'offline'
  | 'slot-missing'
  | 'timeout'
  | 'too-long'
  | 'unavailable'
  | 'unsupported'

/** The four questions the refiner answers before rewriting, one line each. */
export interface RefinementAnalysis {
  professional: string
  rawMaterial: string
  methods: string
  verification: string
}

export type RefineResult =
  | {
      ok: true
      text: string
      unchanged: boolean
      analysis: null | RefinementAnalysis
      /** The depth the refiner picked, for an `auto` request. */
      chosenDepth?: RefinementDepth
    }
  | { ok: false; failure: RefinementFailure; detail?: string }

export interface RefineRequest {
  text: string
  /** Attachment names only, for context. Contents never reach the refiner. */
  attachmentLabels: readonly string[]
  style: RefinementStyle
  depth: RequestDepth
  /** With `style: 'custom'`: the user's own guidance. */
  customStyle?: string
  /** Recent conversation from the same chat, already trimmed (opt-in). */
  context?: string
}

export type ReviewCancelReason = 'busy' | 'no-host' | 'scope-changed' | 'unloaded' | 'user'

export type ReviewDecision =
  { kind: 'approve'; text: string } | { kind: 'cancel'; reason: ReviewCancelReason } | { kind: 'original' }
