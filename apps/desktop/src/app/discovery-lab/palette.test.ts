import { afterEach, describe, expect, it } from 'vitest'

import { PALETTE_AREA, type PaletteContribution } from '@/app/command-palette/contrib'
import { registry } from '@/contrib/registry'
import { $routeRequest } from '@/store/recovery-requests'
import { $shellChoice } from '@/store/shell-variant'

import { $workspaceIsPage, appViewForPath, DISCOVERY_LAB_ROUTE } from '../routes'

import { installDiscoveryLabPaletteRow } from './palette'

const row = () =>
  registry.getArea(PALETTE_AREA).find(c => c.id === 'discovery-lab.open')?.data as PaletteContribution | undefined

let uninstall = () => {}

afterEach(() => {
  uninstall()
  $shellChoice.set(null)
  $workspaceIsPage.set(false)
})

describe('the ⌘K "Discovery lab" row', () => {
  it('is offered in the redesign only, and follows a switch either way', () => {
    $shellChoice.set(false)
    uninstall = installDiscoveryLabPaletteRow()
    expect(row()).toBeUndefined()

    $shellChoice.set(true)
    expect(row()?.label).toBe('Discovery lab')

    $shellChoice.set(false)
    expect(row()).toBeUndefined()
  })

  it('opens the lab as a workspace page, never a chat', () => {
    $shellChoice.set(true)
    uninstall = installDiscoveryLabPaletteRow()

    row()?.run()

    expect($routeRequest.get()?.path).toBe(DISCOVERY_LAB_ROUTE)
    expect($workspaceIsPage.get()).toBe(true)
    expect(appViewForPath(`${DISCOVERY_LAB_ROUTE}?stand-in=sign-up`)).toBe('discovery-lab')
  })
})
