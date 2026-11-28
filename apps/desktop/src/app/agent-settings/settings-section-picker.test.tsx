import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { type AgentSettingsLabels, AgentSettingsSectionPicker } from './settings-editor'
import { AGENT_SETTINGS_GROUPS } from './settings-routes'

afterEach(cleanup)

const labels = new Proxy({} as AgentSettingsLabels, {
  get: (_, key) => `L:${String(key)}`
})

const open = () => fireEvent.pointerDown(screen.getByRole('button'), { button: 0, ctrlKey: false })

// A narrow editor once drew the menu as a height-capped grid, so the last groups (Runtime) sat
// below a fold with no scroll cue. The narrow menu must offer every section, and reach it.
it('offers every agent settings section, the Runtime group included, and opens the one picked', async () => {
  const onOpen = vi.fn()
  render(<AgentSettingsSectionPicker counts={{ tools: 27 }} labels={labels} onOpen={onOpen} section="instructions" />)

  expect(screen.getByRole('button').textContent).toContain('L:agent › L:instructions')

  open()
  const items = await screen.findAllByRole('menuitem')
  const every = AGENT_SETTINGS_GROUPS.flatMap(([, sections]) => sections)

  expect(items).toHaveLength(every.length)
  expect(items.map(item => item.textContent)).toContain('L:runtime › L:workspace')
  expect(items.map(item => item.textContent)).toContain('L:runtime › L:limits')
  expect(items.find(item => item.textContent?.startsWith('L:capabilities › L:tools'))?.textContent).toContain('27')

  fireEvent.click(screen.getByRole('menuitem', { name: 'L:runtime › L:limits' }))
  expect(onOpen).toHaveBeenCalledWith('limits')
})

it('names Tools as current while its Advanced page is open', () => {
  render(<AgentSettingsSectionPicker labels={labels} onOpen={vi.fn()} section="tools-advanced" />)

  expect(screen.getByRole('button').textContent).toContain('L:capabilities › L:tools')
})
