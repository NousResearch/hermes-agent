// Every structural change a scenario can undergo, as pure functions over
// { nodes, edges }.
//
// There is one rule this module exists to enforce: an agent edit and a hand
// edit are the SAME operation on the same document. The inspector, the + on a
// wire, the composer and a model calling `graph_connect` all land here, so
// there is no second implementation of "add a step" that can drift from the one
// the UI uses, and no mutation that skips undo. `graph-tools.ts` publishes these
// same functions as JSON Schema descriptors; it adds no behaviour of its own.
//
// Pure on purpose. Each op takes the current graph and returns the next one,
// which is what lets the page wrap the whole set in one snapshot/undo boundary
// and what lets the tools be tested without a canvas.
//
//   graph-core      the document, and how a step is named, found and wired
//   graph-steps     add / remove / update / rename / change kind
//   graph-arms      a gate's outputs, which the gate owns and a wire names
//   graph-wiring    connect / disconnect, and giving every gate wire an arm
//   graph-validate  what's wrong with the scenario as authored
//   graph-scenario  to and from the authored form the runner and disk speak
//
// Import from here rather than reaching into one of those: which file an op
// lives in is an implementation detail, and the one-door rule above is the
// point.

export { addArm, armLabel, armsOf, armTargets, removeArm, setBranch } from './graph-arms'
export {
  edgeIdFor,
  type Graph,
  isLoop,
  mintId,
  newEdge,
  type OpResult,
  reaches,
  resolveStep,
  stepById,
  stepNodes
} from './graph-core'
export { fromScenario, type RunPlan, runPlan, setScenario, toScenario } from './graph-scenario'
export { addStep, type AddStepInput, removeStep, renameStep, setKind, updateStep } from './graph-steps'
export { type Problem, validate } from './graph-validate'
export { armWires, connect, type ConnectInput, disconnect } from './graph-wiring'
