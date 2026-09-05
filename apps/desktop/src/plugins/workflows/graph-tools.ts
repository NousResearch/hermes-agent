// The scenario as a TOOL SURFACE.
//
// Everything an agent is allowed to do to a workflow, named and typed the way a
// model receives tools. This file is the catalog only — running one is
// `graph-dispatch.ts`.
//
// `parameters` is JSON Schema because that is what every provider's tool-calling
// API takes. Nothing here reads it at runtime; it exists so this file can be
// handed to a model verbatim, and so the built-in planner and a real model are
// describing the same contract.

import { KIND_FIELDS, ON_FAIL_OPTIONS, PROFILES, STEP_KINDS } from './scenario'

export interface ToolDef {
  name: string
  description: string
  parameters: Record<string, unknown>
}

const stepRef = {
  type: 'string',
  description: "A step's id or its title — 'judge' and 'Visual Judge' both work."
}

const armRef = {
  type: 'string',
  description: "An output's id on the gate, e.g. 'pass', 'loop' or 'pass_2'."
}

const armName = {
  type: 'string',
  description: "Short output name, e.g. 'ship it' or 'rework'."
}

const kindRef = {
  type: 'string',
  enum: STEP_KINDS.map(k => k.kind),
  description: STEP_KINDS.map(k => `${k.kind} — ${k.blurb}`).join('; ')
}

const predicateSchema = {
  type: 'object',
  description:
    'When this arm out of a gate is taken. Presets cover the common shapes; ' +
    "'checks' is for checkable rules; 'prose' is read and ruled on by the gate agent itself.",
  properties: {
    mode: { type: 'string', enum: ['all-pass', 'any-fail', 'always', 'checks', 'prose'] },
    join: { type: 'string', enum: ['all', 'any'], description: 'checks only.' },
    checks: {
      type: 'array',
      description: 'checks only.',
      items: {
        type: 'object',
        properties: {
          step: stepRef,
          field: { type: 'string', enum: ['verdict', 'status'] },
          op: { type: 'string', enum: ['is', 'is not'] },
          value: { type: 'string' }
        },
        required: ['step', 'field', 'op', 'value']
      }
    },
    source: { type: 'string', description: 'prose only — what the gate should weigh.' }
  },
  required: ['mode']
} as const

const configSchema = {
  type: 'object',
  description: "Any subset of a step's config. Only the keys you send change.",
  properties: {
    title: { type: 'string' },
    goal: {
      type: 'string',
      description: "The task body — the agent's instruction, or the human's question."
    },
    profile: { type: 'string', enum: [...PROFILES], description: 'The specialist that runs it.' },
    model: { type: 'string', description: "Override this step's model. Empty inherits the profile default." },
    blind: {
      type: 'boolean',
      description: 'Withhold upstream output, so the step judges the artifact and not the reasoning.'
    },
    maxIterations: { type: 'integer', minimum: 1, maximum: 200 },
    maxRetries: { type: 'integer', minimum: 0, maximum: 10 },
    timeoutMins: { type: 'integer', minimum: 0, maximum: 180, description: '0 means no cap.' },
    onFail: { type: 'string', enum: ON_FAIL_OPTIONS.map(o => o.value) },
    maxLoops: { type: 'integer', minimum: 1, maximum: 20, description: 'Gate re-delegation cap.' },
    assignee: { type: 'string', description: 'Human steps: who the run parks on.' },
    until: {
      type: 'object',
      description: 'Wait steps: what the world has to do first.',
      properties: {
        type: { type: 'string', enum: ['timer', 'event', 'poll'] },
        spec: { type: 'string', description: "e.g. '24h', 'github.pull_request.merged', 'every 5m'." }
      },
      required: ['type', 'spec']
    },
    on: {
      type: 'object',
      description: 'Trigger steps: what starts a run.',
      properties: {
        type: { type: 'string', enum: ['manual', 'cron', 'webhook', 'event'] },
        spec: { type: 'string', description: "e.g. 'every 2h', 'github.pull_request.merged'." }
      },
      required: ['type']
    }
  }
} as const

const KIND_ORDER = STEP_KINDS.map(k => k.kind)

export const GRAPH_TOOLS: ToolDef[] = [
  {
    name: 'graph_get',
    description:
      'Read the whole scenario — every step with its config, and every wire with its branch condition. Call this before editing so ids and current values are known.',
    parameters: { type: 'object', properties: {} }
  },
  {
    name: 'graph_add_step',
    description:
      'Add a step. Give it a place in the flow with exactly one of on_edge (splice into a wire), after, or before; omit all three to leave it unwired.',
    parameters: {
      type: 'object',
      properties: {
        kind: kindRef,
        title: { type: 'string' },
        goal: { type: 'string' },
        on_edge: { type: 'string', description: "Wire id, e.g. 'implement->review'." },
        after: stepRef,
        before: stepRef,
        config: configSchema
      },
      required: ['kind', 'title']
    }
  },
  {
    name: 'graph_update_step',
    // Spelled out from KIND_FIELDS rather than prose, so the tool can't promise
    // a knob the op will refuse. Every kind has different ones and a patch is
    // cut to the step's kind before it lands.
    description:
      "Change a step's config. Send only the keys you want changed. Which keys a step has depends on its kind — " +
      KIND_ORDER.map(k => `${k}: ${KIND_FIELDS[k].join(', ')}`).join('; ') +
      '. Anything else is refused.',
    parameters: {
      type: 'object',
      properties: { step: stepRef, patch: configSchema },
      required: ['step', 'patch']
    }
  },
  {
    name: 'graph_rename_step',
    description:
      "Change a step's id, rewriting every wire and gate rule that names it. The id is what conditions refer to, so prefer a readable one.",
    parameters: {
      type: 'object',
      properties: { step: stepRef, new_id: { type: 'string' } },
      required: ['step', 'new_id']
    }
  },
  {
    name: 'graph_set_kind',
    description:
      "Turn a step into a different kind, keeping its name, its instruction and its wiring. Config that only means something to the old kind is dropped; a gate's branch conditions do not survive being turned into anything else.",
    parameters: {
      type: 'object',
      properties: { step: stepRef, kind: kindRef },
      required: ['step', 'kind']
    }
  },
  {
    name: 'graph_remove_step',
    description:
      'Delete a step. Whatever fed it is wired to whatever it fed, so a chain shortens rather than breaking.',
    parameters: { type: 'object', properties: { step: stepRef }, required: ['step'] }
  },
  {
    name: 'graph_connect',
    description:
      'Wire one step into another. A wire that closes a cycle becomes a rework loop automatically. Out of a gate, pass `when` to say which verdicts take this arm.',
    parameters: {
      type: 'object',
      properties: { source: stepRef, target: stepRef, when: predicateSchema },
      required: ['source', 'target']
    }
  },
  {
    name: 'graph_disconnect',
    description: 'Cut the wire between two steps.',
    parameters: {
      type: 'object',
      properties: { source: stepRef, target: stepRef },
      required: ['source', 'target']
    }
  },
  {
    name: 'graph_add_arm',
    description:
      "Add an output to a gate. The output exists whether or not anything is wired to it, so you can lay out a whole routing table and then connect the arms. Outputs are taken in order, so add the catch-all ('always') last.",
    parameters: {
      type: 'object',
      properties: {
        gate: stepRef,
        when: predicateSchema,
        name: armName
      },
      required: ['gate']
    }
  },
  {
    name: 'graph_remove_arm',
    description: 'Drop an output from a gate, along with any wire leaving by it.',
    parameters: {
      type: 'object',
      properties: { gate: stepRef, arm: armRef },
      required: ['gate', 'arm']
    }
  },
  {
    name: 'graph_set_branch',
    description:
      "Set the condition on one of a gate's outputs, and/or name it. Naming is worth doing whenever the condition is longer than a few words — the name is what the canvas prints beside the port. Identify the output by its id, or by a step it routes to.",
    parameters: {
      type: 'object',
      properties: {
        gate: stepRef,
        arm: armRef,
        target: { type: 'string', description: 'Instead of `arm`: a step this output routes to.' },
        when: predicateSchema,
        name: armName
      },
      required: ['gate']
    }
  },
  {
    name: 'graph_set_scenario',
    description:
      "Replace the whole scenario with one you've authored. Use this to build a workflow from scratch; use the surgical tools to change an existing one. Steps without a position are laid out for you.",
    parameters: {
      type: 'object',
      properties: {
        scenario: {
          type: 'object',
          properties: {
            steps: {
              type: 'array',
              items: {
                type: 'object',
                properties: {
                  id: { type: 'string' },
                  kind: { type: 'string', enum: STEP_KINDS.map(k => k.kind) },
                  config: configSchema
                },
                required: ['id', 'kind', 'config']
              }
            },
            edges: {
              type: 'array',
              items: {
                type: 'object',
                properties: {
                  source: { type: 'string' },
                  target: { type: 'string' },
                  when: predicateSchema
                },
                required: ['source', 'target']
              }
            }
          },
          required: ['steps', 'edges']
        }
      },
      required: ['scenario']
    }
  },
  {
    name: 'graph_validate',
    description:
      "Check the scenario for unreachable steps, gates that don't branch, missing defaults and empty goals. Read-only.",
    parameters: { type: 'object', properties: {} }
  },
  {
    name: 'run_control',
    description: 'Drive the run: start it, pause at the next safe point, resume, or clear it.',
    parameters: {
      type: 'object',
      properties: { action: { type: 'string', enum: ['start', 'pause', 'resume', 'reset'] } },
      required: ['action']
    }
  }
]
