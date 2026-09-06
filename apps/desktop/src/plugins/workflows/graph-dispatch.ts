// Running a tool from the catalog.
//
// Each case validates its arguments and calls the corresponding primitive in
// graph.ts — the same primitive the inspector and the canvas call — so there is
// no privileged path. If a tool can do it, a person can do it by hand, and vice
// versa. The catalog these dispatch against lives in `graph-tools.ts`.

import {
  addArm,
  addStep,
  connect,
  disconnect,
  type Graph,
  type OpResult,
  removeArm,
  removeStep,
  renameStep,
  setBranch,
  setKind,
  setScenario,
  stepNodes,
  toScenario,
  updateStep,
  validate
} from './graph'
import { GRAPH_TOOLS } from './graph-tools'
import { DEFAULT_DIR, type FlowDir } from './layout'
import type { Predicate, Scenario, StepConfig, StepKind } from './scenario'

export type ToolArgs = Record<string, unknown>

/** Run control isn't a graph edit, so it can't return a graph. The player is
 *  handed in and these report back in the same shape as everything else. */
export interface RunControl {
  running: boolean
  paused: boolean
  start: () => void
  pause: () => void
  resume: () => void
  reset: () => void
}

const str = (a: ToolArgs, k: string) => (typeof a[k] === 'string' ? (a[k] as string) : undefined)

/** Run one op. `dir` is the canvas's current orientation — placement is the
 *  one thing an op can't work out from the graph alone, because a rank step
 *  runs down a vertical canvas and across a horizontal one. */
export function callTool(
  graph: Graph,
  run: RunControl,
  name: string,
  args: ToolArgs = {},
  dir: FlowDir = DEFAULT_DIR
): OpResult {
  const no = (m: string): OpResult => ({ ok: false, graph, message: m })

  switch (name) {
    case 'graph_get': {
      const s = toScenario(graph)

      return {
        ok: true,
        graph,
        message: `${s.steps.length} steps, ${s.edges.length} wires: ${s.steps.map(x => x.id).join(', ')}.`
      }
    }

    case 'graph_add_step': {
      const kind = str(args, 'kind')

      if (!kind) {
        return no('graph_add_step needs a kind.')
      }

      return addStep(graph, {
        kind: kind as StepKind,
        title: str(args, 'title'),
        goal: str(args, 'goal'),
        onEdge: str(args, 'on_edge'),
        after: str(args, 'after'),
        before: str(args, 'before'),
        config: args.config as Partial<StepConfig> | undefined,
        dir
      })
    }

    case 'graph_update_step': {
      const step = str(args, 'step')

      if (!step) {
        return no('graph_update_step needs a step.')
      }

      return updateStep(graph, step, (args.patch as Partial<StepConfig>) ?? {})
    }

    case 'graph_rename_step': {
      const step = str(args, 'step')
      const next = str(args, 'new_id')

      if (!step || !next) {
        return no('graph_rename_step needs a step and a new_id.')
      }

      return renameStep(graph, step, next)
    }

    case 'graph_set_kind': {
      const step = str(args, 'step')
      const kind = str(args, 'kind')

      if (!step || !kind) {
        return no('graph_set_kind needs a step and a kind.')
      }

      return setKind(graph, step, kind as StepKind)
    }

    case 'graph_remove_step': {
      const step = str(args, 'step')

      if (!step) {
        return no('graph_remove_step needs a step.')
      }

      return removeStep(graph, step)
    }

    case 'graph_connect': {
      const source = str(args, 'source')
      const target = str(args, 'target')

      if (!source || !target) {
        return no('graph_connect needs a source and a target.')
      }

      return connect(graph, { source, target, when: args.when as Predicate | undefined })
    }

    case 'graph_disconnect': {
      const source = str(args, 'source')
      const target = str(args, 'target')

      if (!source) {
        return no('graph_disconnect needs a source.')
      }

      return disconnect(graph, source, target)
    }

    case 'graph_add_arm': {
      const gate = str(args, 'gate')

      if (!gate) {
        return no('graph_add_arm needs a gate.')
      }

      return addArm(graph, gate, args.when as Predicate | undefined, str(args, 'name'))
    }

    case 'graph_remove_arm': {
      const gate = str(args, 'gate')
      const arm = str(args, 'arm')

      if (!gate || !arm) {
        return no('graph_remove_arm needs a gate and an arm.')
      }

      return removeArm(graph, gate, arm)
    }

    case 'graph_set_branch': {
      const gate = str(args, 'gate') ?? str(args, 'source')
      const when = args.when as Predicate | undefined
      const label = str(args, 'name')

      if (!gate) {
        return no('graph_set_branch needs a gate.')
      }

      if (!when && label === undefined) {
        return no('graph_set_branch needs a condition or a name.')
      }

      // Addressable either way: by the output's own id, or by somewhere it
      // goes — which is how you'd say it out loud, and all an agent knows
      // before it has read the gate back.
      const target = str(args, 'target')
      const arm = str(args, 'arm') ?? graph.edges.find(e => e.source === gate && e.target === target)?.sourceHandle

      if (!arm) {
        return no(target ? `There's no wire from ${gate} to ${target}.` : 'graph_set_branch needs an arm or a target.')
      }

      return setBranch(graph, gate, arm, {
        ...(when ? { when } : {}),
        ...(label !== undefined ? { label } : {})
      })
    }

    case 'graph_set_scenario': {
      const scenario = args.scenario as Scenario | undefined

      if (!scenario) {
        return no('graph_set_scenario needs a scenario.')
      }

      return setScenario(graph, scenario)
    }

    case 'graph_validate': {
      const problems = validate(graph)

      return {
        ok: true,
        graph,
        message: problems.length
          ? problems.map(p => `${p.level === 'error' ? '✗' : '!'} ${p.message}`).join(' ')
          : `Looks sound — ${stepNodes(graph).length} steps, nothing unreachable.`
      }
    }

    case 'run_control': {
      switch (str(args, 'action')) {
        case 'start':
          run.start()

          return { ok: true, graph, message: 'Running it.' }

        case 'pause':
          if (!run.running) {
            return no('Nothing is running to pause.')
          }

          run.pause()

          return { ok: true, graph, message: 'Pausing when available.' }

        case 'resume':
          if (!run.paused) {
            return no('Nothing is paused.')
          }

          run.resume()

          return { ok: true, graph, message: 'Resumed.' }

        case 'reset':
          run.reset()

          return { ok: true, graph, message: 'Cleared the run. The scenario is untouched.' }

        default:
          return no('run_control takes start, pause, resume or reset.')
      }
    }

    default:
      return no(`There's no tool called ${name}. Available: ${GRAPH_TOOLS.map(t => t.name).join(', ')}.`)
  }
}
