/**
 * How a diagnostic reaches the control it's about, and how loud it is when it
 * gets there.
 *
 * `validate()` is a linter over a document, not a submit check — there is no
 * submit here, every keystroke is already a graph op, and a step you dropped
 * ten seconds ago is incomplete because you haven't finished it yet. So the
 * default voice is the quiet one: the hint sits under its control in the same
 * muted type as help text, and the panel reads as a draft with things left to
 * fill in rather than a form you've failed.
 *
 * Play is the moment that changes. Pressing it says the scenario is finished,
 * and the same unfinished fields are now the reason it won't run — so they go
 * destructive, and the control they belong to goes invalid. Nothing about
 * `validate()` changes; only how the panel says it.
 */

import { atom, type FieldStatus } from '@hermes/plugin-sdk'

import { $currentId } from './documents'
import type { Problem } from './graph'
import type { StepConfig } from './scenario'

/** Set once a run has been asked for, so an unfinished draft stops being a
 *  draft. */
export const $strict = atom(false)

export const demandComplete = () => $strict.set(true)
export const allowDraft = () => $strict.set(false)

// The next document hasn't been asked to run yet, so it goes back to being a
// draft. Here rather than in the page so a canvas that stops remounting per
// document can't quietly leave the previous one's verdict on screen.
$currentId.listen(allowDraft)

/** Everything addressed to one control, in the voice the panel is currently
 *  using. Errors are errors whether or not you've pressed play; warnings only
 *  become errors once you have. */
export function statusesFor(
  problems: Problem[],
  strict: boolean,
  field: keyof StepConfig,
  arm?: string
): FieldStatus[] {
  return problems
    .filter(p => p.field === field && p.arm === arm)
    .map(p => ({
      level: p.level === 'error' || strict ? 'error' : 'notice',
      message: p.hint ?? p.message
    }))
}

/** The worst of them, for the usual case: one control, one line under it. */
export function statusFor(
  problems: Problem[],
  strict: boolean,
  field: keyof StepConfig,
  arm?: string
): FieldStatus | undefined {
  const mine = statusesFor(problems, strict, field, arm)

  return mine.find(s => s.level === 'error') ?? mine[0]
}
