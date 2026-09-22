import { useCallback, useEffect, useRef, useState } from 'react'

import type { EntityKey, LunarCitySnapshot, LunarCityWorldHandle, LunarEntity } from '../model'
import { socialEligible, type WorkerEncounter, workerRelationships } from '../worker-social'

import { entityFriendlyLabel } from './entity-list'
import { WorkerPersonality } from './worker-personality'

interface WorkerLifeProps {
  demo?: boolean
  snapshot: LunarCitySnapshot
  selectedEntityKey?: EntityKey
  worldRef: { current: LunarCityWorldHandle | undefined }
  onSelect(entity: LunarEntity): void
}

export function WorkerLife({ demo = false, snapshot, selectedEntityKey, worldRef, onSelect }: WorkerLifeProps) {
  const [encounters, setEncounters] = useState<readonly WorkerEncounter[]>([])
  const encounterSignature = useRef('[]')
  const [notice, setNotice] = useState('')
  const noticeValue = useRef('')

  const publishNotice = useCallback((next: string) => {
    if (noticeValue.current !== next) {
      noticeValue.current = next
      setNotice(next)
    }
  }, [])

  const selected = selectedEntityKey ? snapshot.entities.get(selectedEntityKey) : undefined

  const refreshEncounters = useCallback(() => {
    const next = (worldRef.current?.getWorkerEncounters?.() ?? []).map(encounter => ({ ...encounter }))
    const signature = JSON.stringify(next.map(({ id, phase, participants }) => [id, phase, participants]))

    if (signature !== encounterSignature.current) {
      encounterSignature.current = signature
      setEncounters(next)
    }
  }, [worldRef])

  useEffect(() => {
    let timer: ReturnType<typeof setInterval> | undefined

    const visibility = () => {
      if(timer) {clearInterval(timer)}
      timer=undefined

      if(!document.hidden) {refreshEncounters();timer=setInterval(refreshEncounters,1000)}
    }

    visibility()
    document.addEventListener('visibilitychange',visibility)

    return () => {if(timer) {clearInterval(timer);}document.removeEventListener('visibilitychange',visibility)}
  }, [refreshEncounters])
  useEffect(() => publishNotice(''), [selectedEntityKey, publishNotice])

  const request = (key: EntityKey, gesture?: string) => {
    const accepted = worldRef.current?.requestWorkerInteraction?.(key, gesture)
    publishNotice(accepted ? 'Local gesture started. No message was sent.' :
      'Available for nearby resting workers in balanced or detailed mode with motion enabled. A greeting also needs a free nearby partner.')
  }

  return <section aria-label="Worker life" className="lunar-city-worker-life">
    <h3>Worker life</h3>
    <p>Ambient gestures are local presentation. They do not send messages or change work.</p>
    {encounters.length ? <ul>{encounters.map(encounter => <li key={encounter.id}>
      {encounter.participants.map((key,index) => {
        const entity = snapshot.entities.get(key)

        return entity ? <span key={key}>{index ? ' and ' : ''}<button onClick={() => onSelect(entity)}>{entityFriendlyLabel(entity)}</button></span> : null
      })}: {encounter.phase}
    </li>)}</ul> : <p>No local conversations active.</p>}
    {selected ? <>
      <WorkerPersonality allowSoulRead={!demo} entity={selected} onPreviewGesture={request} />
      <button disabled={!socialEligible(selected)} onClick={() => request(selected.key)}>Greet a nearby worker</button>
      <h4>Observed delegation</h4>
      {workerRelationships(selected,snapshot).length ? <ul>{workerRelationships(selected,snapshot).map(entity =>
        <li key={entity.key}><button onClick={() => onSelect(entity)}>{entityFriendlyLabel(entity)}</button> · {entity.authority}</li>)}</ul> :
        <p>No exact parent or delegated worker is present in this snapshot.</p>}
    </> : <p>Select a worker to inspect its personality and observed relationships.</p>}
    {notice ? <p role="status">{notice}</p> : null}
  </section>
}
