import type { LunarCitySnapshot, LunarEntity } from '../model'
import type { ProjectCompoundReport } from '../world/project-compounds'

export interface ProjectCompoundsPanelProps {
  report: ProjectCompoundReport
  snapshot: LunarCitySnapshot
  onSelectEntity(entity: LunarEntity): void
}

export function ProjectCompoundsPanel({ report, snapshot, onSelectEntity }: ProjectCompoundsPanelProps) {
  return (
    <section aria-label="Project sites">
      <h3>Project sites</h3>
      <p>Sites represent observed projects. Work counts describe source records, not construction progress.</p>
      <p>{report.compounds.length} observed projects · {report.anchors.length} assigned sites · {report.overflowCount} without a physical site</p>
      {report.slotIssues.length ? <p>Some manifest slots are unavailable because their footprints overlap or IDs repeat.</p> : null}
      {report.compounds.length ? <ul>{report.compounds.map(compound => (
        <li key={compound.key}>
          <h4>{compound.projectId}</h4>
          <p>Connection: {compound.connectionId}</p>
          <p>{compound.unplaced ? 'Overflow · no physical site available' : `Site: ${compound.slotId}`}</p>
          <p>{compound.workCount} working records · {compound.authoritativeCount} authoritative records · {compound.totalCount} total observed records</p>
          <ul>{compound.entityKeys.map(key => {
            const entity = snapshot.entities.get(key)

            if (!entity) {return null}
            const identity = entity.identity
            const detail = identity.kind === 'kanban' ? `task ${identity.taskId}` : identity.kind === 'subagent' ? `subagent ${identity.subagentId}` : identity.kind === 'session' ? `session ${identity.sessionId}` : identity.kind

            return <li key={key}><button onClick={() => onSelectEntity(entity)} type="button">
              Inspect {detail} · {identity.profile} · {identity.connectionId}
            </button> · {entity.authority}</li>
          })}</ul>
        </li>
      ))}</ul> : <p>No project identity is present in the current snapshot.</p>}
    </section>
  )
}
