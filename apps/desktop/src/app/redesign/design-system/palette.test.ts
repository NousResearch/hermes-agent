import { afterEach, describe, expect, it } from 'vitest'

import { PALETTE_AREA, type PaletteContribution } from '@/app/command-palette/contrib'
import { registry } from '@/contrib/registry'
import { setRuntimeI18nLocale } from '@/i18n/runtime'
import { $routeRequest } from '@/store/recovery-requests'
import { $shellChoice } from '@/store/shell-variant'

import { $workspaceIsPage, appViewForPath, DESIGN_SYSTEM_ROUTE } from '../../routes'

import { installDesignSystemPaletteRow } from './palette'

const row = () =>
  registry.getArea(PALETTE_AREA).find(c => c.id === 'design-system.open')?.data as PaletteContribution | undefined

let uninstall = () => {}

afterEach(() => {
  uninstall()
  $shellChoice.set(null)
  setRuntimeI18nLocale('en')
  $workspaceIsPage.set(false)
})

describe('the ⌘K "Design system" row', () => {
  it('is offered in the redesign only, and follows a switch either way', () => {
    $shellChoice.set(false)
    uninstall = installDesignSystemPaletteRow()
    expect(row()).toBeUndefined()

    $shellChoice.set(true)
    expect(row()?.label).toBe('Design system')

    $shellChoice.set(false)
    expect(row()).toBeUndefined()
  })

  it('is labelled in the app language', () => {
    $shellChoice.set(true)
    uninstall = installDesignSystemPaletteRow()

    setRuntimeI18nLocale('de')
    expect(row()?.label).toBe('Designsystem')
  })

  it('opens the page as a workspace page', () => {
    $shellChoice.set(true)
    uninstall = installDesignSystemPaletteRow()

    row()?.run()

    expect($routeRequest.get()?.path).toBe(DESIGN_SYSTEM_ROUTE)
    expect($workspaceIsPage.get()).toBe(true)
    expect(appViewForPath(DESIGN_SYSTEM_ROUTE)).toBe('design-system')
  })
})
