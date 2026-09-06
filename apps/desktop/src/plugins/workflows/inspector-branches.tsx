/**
 * A gate's routing rules — the one part of Config that edits the graph rather
 * than the step. An arm belongs to the gate, so every control here leaves as a
 * graph op; nothing writes node data.
 */

import {
  Button,
  Codicon,
  Field,
  FieldHint,
  Input,
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
  SidePanelSection,
  Textarea
} from '@hermes/plugin-sdk'
import type { Node } from '@xyflow/react'

import { addArm, armsOf, armTargets, type Graph, type OpResult, type Problem, removeArm, setBranch } from './graph'
import type { NodeData } from './nodes'
import {
  type Check,
  CHECK_FIELDS,
  CHECK_OPS,
  defaultPredicate,
  JOIN_OPTIONS,
  type Predicate,
  PREDICATE_MODES,
  type PredicateMode
} from './scenario'
import { statusesFor } from './validation'

/** Radix forbids `""` as an item value; empty here is a real choice (inherit). */
const NONE = '\u0000none'

type Choice = string | { label: string; value: string }

const choiceValue = (o: Choice) => (typeof o === 'string' ? o : o.value)
const choiceLabel = (o: Choice) => (typeof o === 'string' ? o : o.label)

/** The same Select stack Kanban's profile picker uses, for a list of options. */
function Choices({
  onChange,
  options,
  placeholder,
  title,
  value
}: {
  onChange: (v: string) => void
  options: readonly Choice[]
  placeholder?: string
  title?: string
  value: string
}) {
  return (
    <Select onValueChange={v => onChange(v === NONE ? '' : v)} value={value === '' ? NONE : value}>
      <SelectTrigger aria-label={title} className="nodrag" size="sm">
        <SelectValue placeholder={placeholder} />
      </SelectTrigger>
      <SelectContent>
        {placeholder !== undefined && <SelectItem value={NONE}>{placeholder}</SelectItem>}
        {options.map(o => (
          <SelectItem key={choiceValue(o)} value={choiceValue(o)}>
            {choiceLabel(o)}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  )
}

function ConditionRow({
  check,
  onChange,
  onRemove,
  steps
}: {
  check: Check
  onChange: (next: Check) => void
  onRemove: () => void
  steps: Node[]
}) {
  return (
    <div className="flex flex-col gap-1.5">
      <div className="grid grid-cols-2 gap-1.5">
        <Choices onChange={v => onChange({ ...check, step: v })} options={steps.map(n => n.id)} value={check.step} />
        <Choices
          onChange={v => onChange({ ...check, field: v as Check['field'] })}
          options={CHECK_FIELDS}
          value={check.field}
        />
      </div>
      <div className="flex items-center gap-1.5">
        <Choices onChange={v => onChange({ ...check, op: v as Check['op'] })} options={CHECK_OPS} value={check.op} />
        <Input
          className="nodrag min-w-0 flex-1"
          onChange={e => onChange({ ...check, value: e.target.value })}
          placeholder="PASS"
          size="sm"
          value={check.value}
        />
        <Button aria-label="Remove this condition" onClick={onRemove} size="icon-xs" variant="ghost">
          <Codicon name="close" size="0.75rem" />
        </Button>
      </div>
    </div>
  )
}

export function BranchEditor({
  gateId,
  graph,
  onOp,
  problems,
  strict
}: {
  gateId: string
  graph: Graph
  onOp: (op: OpResult) => OpResult
  problems: Problem[]
  strict: boolean
}) {
  const arms = armsOf(graph, gateId)
  const steps = graph.nodes.filter(n => n.id !== gateId && !!(n.data as NodeData)?.def)
  const titleOf = (id: string) => (graph.nodes.find(n => n.id === id)?.data as NodeData)?.config.title ?? id
  const table = statusesFor(problems, strict, 'arms')

  return (
    <SidePanelSection
      action={
        <Button onClick={() => onOp(addArm(graph, gateId))} size="xs" variant="ghost">
          <Codicon name="add" size="0.75rem" />
          Add
        </Button>
      }
      label="Routing rules"
    >
      <FieldHint>Taken in order — the first rule that matches wins.</FieldHint>
      {/* About the table as a whole — too few arms, no default — rather than
          about any one rule below. */}
      {table.map((s, i) => (
        <FieldHint error={s.level === 'error'} key={i}>
          {s.message}
        </FieldHint>
      ))}

      <div className="flex flex-col gap-4">
        {arms.map(arm => {
          const when = arm.when
          const set = (next: Predicate) => onOp(setBranch(graph, gateId, arm.id, { when: next }))
          const goes = armTargets(graph, gateId, arm.id)

          return (
            <div className="flex flex-col gap-3" key={arm.id}>
              <Field label="Output">
                <div className="flex items-center gap-1">
                  <Input
                    className="nodrag"
                    onChange={ev => onOp(setBranch(graph, gateId, arm.id, { label: ev.target.value }))}
                    placeholder={goes.map(e => titleOf(e.target)).join(', ') || 'Unnamed output'}
                    size="sm"
                    title="What the canvas calls this output."
                    value={arm.label ?? ''}
                  />
                  <Button
                    aria-label="Remove this output"
                    onClick={() => onOp(removeArm(graph, gateId, arm.id))}
                    size="icon-xs"
                    variant="ghost"
                  >
                    <Codicon name="close" size="0.75rem" />
                  </Button>
                </div>
              </Field>
              {/* Where it goes, when it goes anywhere. When it doesn't, that's
                  a diagnostic, and it comes through the same slot as the rest
                  of them at the foot of the rule. */}
              {!!goes.length && !!arm.label?.trim() && (
                <p className="text-[0.75rem] text-(--ui-text-tertiary)">
                  → {goes.map(e => titleOf(e.target)).join(', ')}
                </p>
              )}

              <Field label="When" tip={PREDICATE_MODES.find(p => p.value === when.mode)?.hint}>
                <Choices
                  onChange={m => set(defaultPredicate(m as PredicateMode))}
                  options={PREDICATE_MODES.map(p => ({ label: p.label, value: p.value }))}
                  value={when.mode}
                />
              </Field>

              {when.mode === 'prose' && (
                <Textarea
                  className="nodrag nowheel min-h-24 text-[0.75rem]"
                  onChange={e => set({ mode: 'prose', source: e.target.value })}
                  placeholder="What the gate should weigh before taking this arm…"
                  rows={2}
                  value={when.source}
                />
              )}

              {when.mode === 'checks' && (
                <div className="flex flex-col gap-3">
                  {when.checks.map((c, i) => (
                    <div className="flex flex-col gap-1.5" key={i}>
                      {i > 0 && (
                        <Choices
                          onChange={v => set({ ...when, join: v as 'all' | 'any' })}
                          options={JOIN_OPTIONS}
                          value={when.join}
                        />
                      )}
                      <ConditionRow
                        check={c}
                        onChange={next => set({ ...when, checks: when.checks.map((x, j) => (j === i ? next : x)) })}
                        onRemove={() => set({ ...when, checks: when.checks.filter((_, j) => j !== i) })}
                        steps={steps}
                      />
                    </div>
                  ))}
                  <Button
                    onClick={() =>
                      set({
                        ...when,
                        checks: [...when.checks, { field: 'verdict', op: 'is', step: steps[0]?.id ?? '', value: 'PASS' }]
                      })
                    }
                    size="xs"
                    variant="ghost"
                  >
                    <Codicon name="add" size="0.75rem" />
                    Add condition
                  </Button>
                </div>
              )}

              {statusesFor(problems, strict, 'arms', arm.id).map((s, i) => (
                <FieldHint error={s.level === 'error'} key={i}>
                  {s.message}
                </FieldHint>
              ))}
            </div>
          )
        })}
      </div>
    </SidePanelSection>
  )
}
