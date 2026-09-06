/**
 * The step detail panel. Same sheet as the Kanban task drawer: SidePanel
 * chrome, meta rows for knobs, sections for prose and lists.
 *
 * This file is only the sheet and the tab it shows. Config is the step as
 * authored (inspector-config), Data the step as run (inspector-data); the two
 * share nothing but the step they're pointed at.
 */

import {
  Codicon,
  SidePanelAction,
  SidePanelBody,
  SidePanelClose,
  SidePanelHeader,
  SidePanelTitleInput,
  SidePanelToolbar,
  TextTab,
  useValue
} from '@hermes/plugin-sdk'
import type { Node } from '@xyflow/react'
import { useMemo, useState } from 'react'

import { type Graph, type OpResult, validate } from './graph'
import { ConfigTab } from './inspector-config'
import { DataTab } from './inspector-data'
import type { NodeData } from './nodes'
import type { StepRuntime } from './protocol'
import type { StepConfig } from './scenario'
import { $strict } from './validation'

export function Inspector({
  graph,
  node,
  onChange,
  onClose,
  onDelete,
  onOp,
  rt
}: {
  graph: Graph
  node: Node
  onChange: (patch: Partial<StepConfig>) => void
  onClose: () => void
  onDelete: () => void
  onOp: (op: OpResult) => OpResult
  rt: StepRuntime
}) {
  const { config, def } = node.data as NodeData
  const [tab, setTab] = useState<'config' | 'data'>('config')

  const problems = useMemo(() => validate(graph).filter(p => p.step === def.id), [def.id, graph])
  const strict = useValue($strict)

  return (
    <>
      <SidePanelHeader>
        <SidePanelToolbar>
          <SidePanelTitleInput
            className="min-w-0 flex-1"
            onChange={e => onChange({ title: e.target.value })}
            value={config.title}
          />
          <div className="ml-auto flex shrink-0 items-center gap-0.5">
            <SidePanelAction aria-label="Delete this step" onClick={onDelete}>
              <Codicon name="trash" size="0.8rem" />
            </SidePanelAction>
            <SidePanelClose onClick={onClose} />
          </div>
        </SidePanelToolbar>
        <div className="flex gap-3">
          <TextTab active={tab === 'config'} onClick={() => setTab('config')}>
            Config
          </TextTab>
          <TextTab active={tab === 'data'} onClick={() => setTab('data')}>
            Data
          </TextTab>
        </div>
      </SidePanelHeader>

      <SidePanelBody className="nodrag nowheel" fade>
        {tab === 'config' ? (
          <ConfigTab
            config={config}
            def={def}
            graph={graph}
            onChange={onChange}
            onOp={onOp}
            problems={problems}
            strict={strict}
          />
        ) : (
          <DataTab kind={def.kind} rt={rt} />
        )}
      </SidePanelBody>
    </>
  )
}
