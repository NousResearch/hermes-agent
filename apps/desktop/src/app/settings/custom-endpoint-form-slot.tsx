import type { FC } from 'react'
import { useMemo } from 'react'

import { useContributions } from '@/contrib'
import { ContribBoundary, ContribRender } from '@/contrib/react/boundary'
import {
  CUSTOM_ENDPOINT_FORM_AREA,
  type CustomEndpointFormContribution,
  type CustomEndpointFormProps
} from '@/lib/custom-endpoint-form'

/**
 * The plugin seam inside the custom-endpoint editor card. Mounts every
 * registration with the live draft/save snapshot and lets each decide — it
 * renders its augmentation, or nothing at all.
 *
 * Mounting all of them (rather than first-wins) keeps composition visible:
 * two augmentations render both, and a throwing one degrades to an inline
 * error inside its boundary without taking the editor down.
 */
const CustomEndpointFormSlotEntry: FC<{
  id: string
  render: CustomEndpointFormContribution['render']
  props: CustomEndpointFormProps
}> = ({ id, render, props }) => {
  // Stable component identity: ContribRender mounts this AS a component, so a
  // fresh closure per render would remount the augmentation on every keystroke.
  const renderSlot = useMemo(() => () => render(props), [render, props])

  return (
    <ContribBoundary id={id} variant="chip">
      <ContribRender render={renderSlot} />
    </ContribBoundary>
  )
}

export const CustomEndpointFormSlot: FC<CustomEndpointFormProps> = props => {
  const contributions = useContributions(CUSTOM_ENDPOINT_FORM_AREA)

  if (contributions.length === 0) {
    return null
  }

  return (
    <>
      {contributions.map(contribution => {
        const render = (contribution.data as CustomEndpointFormContribution | undefined)?.render

        return render ? (
          <CustomEndpointFormSlotEntry
            id={contribution.id}
            key={`${contribution.source ?? 'core'}:${contribution.id}`}
            render={render}
            props={props}
          />
        ) : null
      })}
    </>
  )
}
