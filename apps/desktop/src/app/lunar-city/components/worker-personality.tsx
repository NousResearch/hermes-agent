import { useEffect, useRef, useState } from 'react'

import { getProfileSoul } from '@/api/profiles'
import type { ProfileSoul } from '@/types/hermes'

import { entityKey } from '../identity'
import type { EntityKey, LunarEntity } from '../model'
import { deriveWorkerPersonality, type WorkerGesture } from '../worker-personality'

export interface WorkerPersonalityProps {
  allowSoulRead?: boolean
  entity: LunarEntity
  onPreviewGesture?: (key: EntityKey, gesture: WorkerGesture) => void
}

type SoulState = { status: 'idle' | 'loading' | 'error' } | { status: 'loaded'; soul: ProfileSoul }

function PersonalityDetails({ entity, onPreviewGesture, allowSoulRead = true }: WorkerPersonalityProps) {
  const personality = deriveWorkerPersonality(entity)
  const [soul, setSoul] = useState<SoulState>({ status: 'idle' })
  const request = useRef(0)
  useEffect(() => () => { request.current += 1 }, [])

  async function loadSoul() {
    const generation = ++request.current
    setSoul({ status: 'loading' })

    try {
      const result = await getProfileSoul(entity.identity.profile, {
        connectionId: entity.identity.connectionId,
        profile: entity.identity.profile
      })

      if (generation === request.current) {setSoul({ status: 'loaded', soul: result })}
    } catch {
      if (generation === request.current) {setSoul({ status: 'error' })}
    }
  }

  return (
    <section aria-label="Worker personality">
      <h3>Worker personality</h3>
      <p><strong>{personality.trait}</strong> — {personality.description}</p>
      <p>Authored city presentation. This style does not describe or change the agent’s instructions or work.</p>
      <dl>
        <dt>Configured role</dt>
        <dd>{personality.configuredTitle || 'No fresh configured title'}</dd>
        <dt>Configured groups</dt>
        <dd>{personality.groups.length ? personality.groups.map(group => group.name).join(', ') : 'No fresh group information'}</dd>
        <dt>Role metadata</dt>
        <dd>{personality.metadataState}{personality.metadataSource ? ` · ${personality.metadataSource}` : ''}</dd>
      </dl>
      {onPreviewGesture ? (
        <button onClick={() => onPreviewGesture(entity.key, personality.gesture)} type="button">Preview city gesture</button>
      ) : null}
      <p>Profile SOUL: {entity.identity.profile} on {entity.identity.connectionId}. Loaded only when requested.</p>
      <button disabled={!allowSoulRead || soul.status === 'loading'} onClick={() => void loadSoul()} type="button">
        {soul.status === 'loading' ? 'Loading profile SOUL…' : soul.status === 'error' ? 'Retry profile SOUL' : 'Load profile SOUL'}
      </button>
      <div aria-live="polite">
        {soul.status === 'error' ? <p>Profile SOUL could not be loaded from this owner. Try again.</p> : null}
        {soul.status === 'loaded' ? (
          <section aria-label="Source profile SOUL">
            <h4>Source profile SOUL · read only</h4>
            {soul.soul.exists ? <pre className="whitespace-pre-wrap break-words">{soul.soul.content || 'This SOUL file is empty.'}</pre> : <p>No SOUL file exists for this profile.</p>}
          </section>
        ) : null}
      </div>
    </section>
  )
}

/** Remounting by exact identity isolates both displayed source text and in-flight reads. */
export function WorkerPersonality(props: WorkerPersonalityProps) {
  return <PersonalityDetails key={entityKey(props.entity.identity)} {...props} />
}
