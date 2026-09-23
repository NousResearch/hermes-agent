import { useState } from 'react'

import type { LunarCityWorldHandle } from '../model'

import { InteriorCutawayToggle } from './interior-cutaway-toggle'

export function InteriorCutawayControls({ world, focusedEntityKey }: {
  world?: LunarCityWorldHandle
  focusedEntityKey?: string
}) {
  const building = world?.getInteriorBuildings?.().find(plan => focusedEntityKey === `lunar-city:model:${encodeURIComponent(plan.id)}`)

  if (!building || !world) {return null}

  return <SelectedInterior building={building} key={building.id} world={world} />
}

function SelectedInterior({ building, world }: { building: { id: string; title: string }; world: LunarCityWorldHandle }) {
  const [enabled, setEnabled] = useState(false)

  return <InteriorCutawayToggle buildingId={building.id} enabled={enabled} onChange={next => {
    setEnabled(world.setInteriorBuilding?.(next ? building.id : undefined) ?? false)
  }} title={building.title} />
}
