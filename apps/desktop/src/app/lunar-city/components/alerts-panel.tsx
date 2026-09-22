import { useState } from 'react'

import { acknowledgeAlert, EMPTY_ALERT_LEDGER, reconcileAlerts } from '../alerts'
import type { LunarCitySnapshot, LunarEntity } from '../model'

export interface AlertsPanelProps {
  snapshot: LunarCitySnapshot
  onSelectEntity(entity: LunarEntity): void
}

/** The mounted panel owns local acknowledgement/history; it performs no backend writes. */
export function AlertsPanel({ snapshot, onSelectEntity }: AlertsPanelProps) {
  const [state, setState] = useState(() => ({ snapshot, ledger: reconcileAlerts(EMPTY_ALERT_LEDGER, snapshot) }))
  const [showResolved, setShowResolved] = useState(false)
  let ledger = state.ledger

  if (state.snapshot !== snapshot) {
    ledger = reconcileAlerts(state.ledger, snapshot)
    setState({ snapshot, ledger })
  }

  const active = ledger.incidents.filter(row => row.status !== 'resolved')
  const unacknowledged = active.filter(row => row.status !== 'acknowledged').length
  const visible = ledger.incidents.filter(row => showResolved || row.status !== 'resolved')

  return (
    <section aria-label="City attention" className="lunar-city-alerts">
      <h2>City attention</h2>
      <p aria-atomic="true" aria-live="polite" role="status">
        {active.length} attention items · {unacknowledged} unacknowledged
      </p>
      <p>Acknowledgement is local. It does not change tasks or clear blockers.</p>
      <label>
        <input checked={showResolved} onChange={event => setShowResolved(event.target.checked)} type="checkbox" />
        Show resolved history
      </label>
      {ledger.omitted > 0 ? (
        <p>
          {ledger.omitted} additional items exceed the attention limit. Inspect the entity list for all current states.
        </p>
      ) : null}
      {visible.length === 0 ? (
        <p>No observed attention items. Missing data does not establish health.</p>
      ) : (
        <ul>
          {visible.map(incident => {
            const entity = incident.entityKey ? snapshot.entities.get(incident.entityKey) : undefined

            return (
              <li key={incident.id}>
                <strong>{incident.label}</strong>
                <span>
                  {' '}
                  · {incident.kind.replaceAll('-', ' ')} · {incident.status}
                </span>
                <p>{incident.detail}</p>
                {entity ? (
                  <button onClick={() => onSelectEntity(entity)} type="button">
                    Inspect {incident.label}
                  </button>
                ) : null}
                {incident.entityKey && !entity ? <span> Entity currently unavailable</span> : null}
                {incident.status === 'open' || incident.status === 'reopened' ? (
                  <button
                    onClick={() =>
                      setState(current => ({ ...current, ledger: acknowledgeAlert(current.ledger, incident.id) }))
                    }
                    type="button"
                  >
                    Acknowledge {incident.label}
                  </button>
                ) : null}
              </li>
            )
          })}
        </ul>
      )}
    </section>
  )
}
