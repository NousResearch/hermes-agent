// What's wrong with the scenario as authored.

import { armLabel, armTargets } from './graph-arms'
import { dataOf, type Graph, isLoop, reaches, stepNodes } from './graph-core'
import { hasField, type StepConfig } from './scenario'

export interface Problem {
  level: 'error' | 'warning'
  /** Names the offending step, so the message stands alone in a tool result. */
  message: string
  /** The same thing said under the control it's about, where naming the step
   *  and quoting the arm only repeats what's already on screen. Absent when
   *  the problem has no field to sit under. */
  hint?: string
  /** Which step it's about, for a caller showing one step's problems rather
   *  than the whole list. Absent when it's about the scenario as a whole. */
  step?: string
  /** Which knob it's about, so a form can put it under that one rather than in
   *  a pile at the top. Absent when it's about the step's wiring rather than
   *  anything you could type — those have nowhere to land but a banner. */
  field?: keyof StepConfig
  /** Which gate output, for a problem about one routing rule. */
  arm?: string
}

/** What's wrong with the scenario as authored — the things you'd otherwise only
 *  discover by running it. Deliberately not enforced at edit time: a graph is
 *  allowed to be half-built while you build it. */
export function validate(g: Graph): Problem[] {
  const steps = stepNodes(g)
  const problems: Problem[] = []

  if (!steps.length) {
    return [{ level: 'warning', message: 'The scenario is empty.' }]
  }

  const entries = steps.filter(n => !g.edges.some(e => e.target === n.id && !isLoop(e)))

  if (!entries.length) {
    problems.push({ level: 'error', message: 'Nothing starts the run — every step has an input.' })
  }

  for (const n of steps) {
    const id = n.id
    const { def, config } = dataOf(n)

    if (entries.length && !entries.includes(n) && !entries.some(s => reaches(g, s.id, id))) {
      problems.push({ level: 'error', message: `${id} can't be reached from a start.`, step: id })
    }

    // Deciding when to stop going round is a gate's job — `max takes` is the
    // only place the schema keeps that number. A rework loop drawn straight out
    // of a step has nothing to end it, so the run doesn't follow it, and the
    // wire sits on the canvas looking like it does something.
    if (def.kind !== 'gate' && g.edges.some(e => e.source === id && isLoop(e))) {
      problems.push({
        level: 'warning',
        message: `${id}'s rework loop doesn't leave from a gate, so nothing decides when to stop — the run won't take it.`,
        step: id
      })
    }

    if (def.kind === 'gate') {
      const arms = config.arms ?? []

      if (arms.length < 2) {
        problems.push({
          field: 'arms',
          hint: arms.length === 1 ? 'One output isn\u2019t a branch — add another.' : 'A gate needs outputs to branch.',
          level: 'warning',
          message: `${id} is a gate with ${arms.length === 1 ? 'one output' : 'no outputs'} — it isn't branching.`,
          step: id
        })
      }

      if (arms.length && !arms.some(a => a.when.mode === 'always')) {
        problems.push({
          field: 'arms',
          hint: 'No default rule, so some verdicts route nowhere.',
          level: 'warning',
          message: `${id} has no default arm, so some verdicts route nowhere.`,
          step: id
        })
      }

      arms.forEach(a => {
        const where = `${id}'s "${armLabel(a)}" output`

        // An arm can outlive its wire — that's what lets you write the table
        // before you wire it — but a rule the run can't follow is worth saying.
        if (!armTargets(g, id, a.id).length) {
          problems.push({
            arm: a.id,
            field: 'arms',
            hint: 'Not wired — drag this output on the canvas.',
            level: 'warning',
            message: `${where} isn't wired anywhere.`,
            step: id
          })
        }

        if (a.when.mode === 'checks' && !a.when.checks.length) {
          problems.push({
            arm: a.id,
            field: 'arms',
            hint: 'No conditions yet.',
            level: 'warning',
            message: `${where} has no conditions yet.`,
            step: id
          })
        }

        if (a.when.mode === 'prose' && !a.when.source.trim()) {
          problems.push({
            arm: a.id,
            field: 'arms',
            hint: 'Nothing here for the gate to read.',
            level: 'warning',
            message: `${where} has nothing for the gate to read.`,
            step: id
          })
        }
      })
    }

    // Asked of the schema rather than the kind, so a field that moves between
    // kinds doesn't leave a check behind pointed at a step that no longer has
    // it — or, worse, stop being checked on the kind it moved to.
    if (hasField(def.kind, 'until') && !config.until?.spec.trim()) {
      problems.push({
        field: 'until',
        hint: 'Say what the run is waiting for.',
        level: 'warning',
        message: `${id} doesn't say what it waits for.`,
        step: id
      })
    }

    if (hasField(def.kind, 'on')) {
      const incoming = g.edges.filter(e => e.target === id)
      if (incoming.length) {
        problems.push({
          level: 'warning',
          message: `${id} is a trigger — incoming wires are ignored. A trigger is an entry.`,
          step: id
        })
      }
      // Manual is Play; webhook is the minted URL. Only cron/event need a spec.
      if (config.on && (config.on.type === 'cron' || config.on.type === 'event') && !config.on.spec.trim()) {
        problems.push({
          field: 'on',
          hint: config.on.type === 'cron' ? 'Say how often.' : 'Name the event.',
          level: 'warning',
          message: `${id} doesn't say what starts the run.`,
          step: id
        })
      }
    }

    if (hasField(def.kind, 'goal') && !config.goal?.trim()) {
      problems.push({
        field: 'goal',
        hint: def.kind === 'human' ? 'Nothing to ask yet.' : 'Say what this step should do.',
        level: 'warning',
        message: `${id} has no ${def.kind === 'human' ? 'question' : 'goal'}.`,
        step: id
      })
    }
  }

  return problems
}
