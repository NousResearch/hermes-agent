/**
 * The step as authored. Which knobs appear is the schema's answer (`hasField`),
 * not the kind's — a kind that grows a field grows the control for free.
 *
 * Every edit leaves as a config patch or a graph op; the tab never reaches into
 * node data itself.
 */

import {
  Callout,
  CopyButton,
  Field,
  FieldStatusSlot,
  Input,
  SegmentedControl,
  SidePanelMeta,
  SidePanelMetaRow,
  SidePanelSection,
  Stepper,
  Switch,
  Textarea,
  useValue
} from '@hermes/plugin-sdk'

import { $currentId, $webhooks } from './documents'
import { type Graph, type OpResult, type Problem, setKind } from './graph'
import { BranchEditor } from './inspector-branches'
import type { StepDef } from './scenario'
import {
  hasField,
  ON_FAIL_OPTIONS,
  type OnFail,
  STEP_KINDS,
  type StepConfig,
  TRIGGER_KIND_OPTIONS,
  type TriggerKind,
  WAIT_KIND_OPTIONS,
  type WaitKind
} from './scenario'
import { statusFor } from './validation'

export function ConfigTab({
  config,
  def,
  graph,
  onChange,
  onOp,
  problems,
  strict
}: {
  config: StepConfig
  def: StepDef
  graph: Graph
  onChange: (patch: Partial<StepConfig>) => void
  onOp: (op: OpResult) => OpResult
  problems: Problem[]
  strict: boolean
}) {
  const has = (f: keyof StepConfig) => hasField(def.kind, f)
  const isHuman = def.kind === 'human'
  const budgets = (['maxIterations', 'maxRetries', 'timeoutMins'] as const).some(has)
  const st = (field: keyof StepConfig) => statusFor(problems, strict, field)
  // What's wrong with the step's wiring rather than with anything you could
  // type here — it has no control to sit under, so it keeps the banner.
  const unplaced = problems.filter(p => !p.field)
  const webhook = useValue($webhooks)[useValue($currentId) ?? '']

  return (
    <div className="flex flex-col gap-4 text-sm">
      {unplaced.map((p, i) => (
        <Callout
          icon={p.level === 'error' ? 'error' : 'warning'}
          key={i}
          title={p.message}
          tone={p.level === 'error' ? 'var(--destructive, #f87171)' : '#fbbf24'}
        />
      ))}

      <SidePanelMeta>
        <SidePanelMetaRow
          control
          label="Type"
          tip="What runs this step. Changing it keeps the name, the instruction and the wiring."
        >
          <SegmentedControl
            className="nodrag w-full"
            onChange={k => onOp(setKind(graph, def.id, k))}
            options={STEP_KINDS.map(k => ({ id: k.kind, label: k.title }))}
            value={def.kind}
          />
        </SidePanelMetaRow>

        {has('model') && (
          <SidePanelMetaRow
            control
            label="Model"
            tip="Overrides the model for this step only. Empty inherits this profile's default."
          >
            <Input
              className="nodrag"
              onChange={e => onChange({ model: e.target.value })}
              placeholder="inherit"
              value={config.model ?? ''}
            />
          </SidePanelMetaRow>
        )}

        {has('onFail') && (
          <SidePanelMetaRow
            control
            label="On failure"
            tip={
              isHuman
                ? 'What the run does if nobody answers in time.'
                : 'What the run does when this step exhausts its retries.'
            }
          >
            <SegmentedControl
              className="nodrag w-full"
              onChange={(v: OnFail) => onChange({ onFail: v })}
              options={ON_FAIL_OPTIONS.map(o => ({ id: o.value, label: o.label }))}
              value={config.onFail ?? 'retry'}
            />
          </SidePanelMetaRow>
        )}

        {has('assignee') && (
          <SidePanelMetaRow control label="Assignee" tip="Who the run parks on. Empty means whoever is watching.">
            <Input
              className="nodrag"
              onChange={e => onChange({ assignee: e.target.value })}
              placeholder="anyone"
              size="sm"
              value={config.assignee ?? ''}
            />
          </SidePanelMetaRow>
        )}

        {has('on') && (
          <>
            <SidePanelMetaRow control label="Starts on" tip="What begins a run of this workflow.">
              <SegmentedControl
                className="nodrag w-full"
                onChange={(v: TriggerKind) => onChange({ on: { spec: config.on?.spec ?? '', type: v } })}
                options={TRIGGER_KIND_OPTIONS.map(o => ({ id: o.value, label: o.label }))}
                value={config.on?.type ?? 'manual'}
              />
            </SidePanelMetaRow>
            {(config.on?.type ?? 'manual') !== 'manual' && (
              <SidePanelMetaRow
                control
                label="When"
                status={st('on')}
                tip={TRIGGER_KIND_OPTIONS.find(o => o.value === (config.on?.type ?? 'manual'))?.hint}
              >
                <Input
                  className="nodrag"
                  onChange={e => onChange({ on: { spec: e.target.value, type: config.on?.type ?? 'cron' } })}
                  placeholder={
                    (config.on?.type ?? 'cron') === 'cron'
                      ? 'every 2h'
                      : (config.on?.type ?? 'cron') === 'webhook'
                        ? 'saved on the gateway as wf:<workflow>'
                        : 'github.pull_request.merged'
                  }
                  size="sm"
                  value={config.on?.spec ?? ''}
                />
              </SidePanelMetaRow>
            )}
            {(config.on?.type ?? 'manual') === 'webhook' && webhook && (
              <>
                <SidePanelMetaRow control label="Route" tip="POST this path on the webhook gateway.">
                  <div className="flex items-center gap-1">
                    <Input className="nodrag min-w-0 flex-1" readOnly size="sm" value={`/webhooks/${webhook.route}`} />
                    <CopyButton appearance="icon" buttonSize="icon-xs" text={`/webhooks/${webhook.route}`} />
                  </div>
                </SidePanelMetaRow>
                <SidePanelMetaRow control label="HMAC" tip="Stored under HERMES_HOME/workflows/secrets.json.">
                  <div className="flex items-center gap-1">
                    <Input className="nodrag min-w-0 flex-1" readOnly size="sm" value={webhook.secret} />
                    <CopyButton appearance="icon" buttonSize="icon-xs" text={webhook.secret} />
                  </div>
                </SidePanelMetaRow>
              </>
            )}
          </>
        )}

        {has('until') && (
          <>
            <SidePanelMetaRow control label="Waiting on" tip="What the world has to do before the run moves on.">
              <SegmentedControl
                className="nodrag w-full"
                onChange={(v: WaitKind) => onChange({ until: { spec: config.until?.spec ?? '', type: v } })}
                options={WAIT_KIND_OPTIONS.map(o => ({ id: o.value, label: o.label }))}
                value={config.until?.type ?? 'timer'}
              />
            </SidePanelMetaRow>
            <SidePanelMetaRow
              control
              label="Condition"
              status={st('until')}
              tip={WAIT_KIND_OPTIONS.find(o => o.value === (config.until?.type ?? 'timer'))?.hint}
            >
              <Input
                className="nodrag"
                onChange={e => onChange({ until: { spec: e.target.value, type: config.until?.type ?? 'timer' } })}
                placeholder={
                  (config.until?.type ?? 'timer') === 'timer'
                    ? '24h'
                    : (config.until?.type ?? 'timer') === 'event'
                      ? 'github.pull_request.merged'
                      : 'every 5m'
                }
                size="sm"
                value={config.until?.spec ?? ''}
              />
            </SidePanelMetaRow>
          </>
        )}

        {has('maxLoops') && (
          <Field label="Max takes" row tip="How many takes the gate may send back before giving up.">
            <Stepper
              className="nodrag"
              max={20}
              min={1}
              onChange={v => onChange({ maxLoops: v })}
              value={config.maxLoops ?? 0}
            />
          </Field>
        )}

        {has('blind') && (
          <label className="flex cursor-pointer items-center gap-2 text-[0.75rem] text-(--ui-text-secondary)">
            <Switch
              aria-label="Blind to upstream output"
              checked={!!config.blind}
              className="nodrag"
              onCheckedChange={v => onChange({ blind: v })}
              size="xs"
            />
            Blind to upstream output
          </label>
        )}
      </SidePanelMeta>

      {has('goal') && (
        <SidePanelSection
          label={isHuman ? 'Ask' : 'Goal'}
          title={
            isHuman
              ? "Shown when the run parks here. Your answer is this step's output."
              : "Sent to delegate_task as the subagent's goal."
          }
        >
          <FieldStatusSlot status={st('goal')}>
            <Textarea
              className="nodrag nowheel min-h-24 text-[0.75rem]"
              onChange={e => onChange({ goal: e.target.value })}
              rows={3}
              value={config.goal ?? ''}
            />
          </FieldStatusSlot>
        </SidePanelSection>
      )}

      {budgets && (
        <SidePanelSection label="Budgets">
          {has('maxIterations') && (
            <Field label="Iterations" row tip="Tool-call budget before the subagent must stop.">
              <Stepper
                className="nodrag"
                max={200}
                min={1}
                onChange={v => onChange({ maxIterations: v })}
                step={5}
                value={config.maxIterations ?? 20}
              />
            </Field>
          )}
          {has('maxRetries') && (
            <Field label="Retries" row tip="Takes before the step reports failed.">
              <Stepper
                className="nodrag"
                max={10}
                min={0}
                onChange={v => onChange({ maxRetries: v })}
                value={config.maxRetries ?? 1}
              />
            </Field>
          )}
          {has('timeoutMins') && (
            <Field
              label="Timeout"
              row
              tip={
                isHuman
                  ? 'How long the run parks here before nobody answering counts as a failure. ∞ = wait forever.'
                  : 'Wall-clock cap on a single take. ∞ = no cap.'
              }
            >
              <Stepper
                className="nodrag"
                max={180}
                min={0}
                onChange={v => onChange({ timeoutMins: v })}
                step={5}
                suffix={(config.timeoutMins ?? 0) > 0 ? 'min' : undefined}
                unboundedAtMin
                value={config.timeoutMins ?? 0}
              />
            </Field>
          )}
        </SidePanelSection>
      )}

      {has('arms') && <BranchEditor gateId={def.id} graph={graph} onOp={onOp} problems={problems} strict={strict} />}
    </div>
  )
}
