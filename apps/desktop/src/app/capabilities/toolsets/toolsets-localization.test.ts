import { expect, it } from 'vitest'

import { ru } from '@/i18n/ru'

import { filteredToolsets } from './toolsets-data'

const browser = {
  name: 'browser',
  label: 'Browser Automation',
  description: 'navigate, click, type, scroll',
  enabled: true,
  configured: true,
  tools: ['browser_click']
}

it('finds a localized built-in by Russian name, English label and tool identifier', () => {
  for (const query of ['браузером', 'Browser Automation', 'browser_click']) {
    const rows = filteredToolsets([browser], query, {}, true, ru.catalog.builtinToolsets)
    expect(rows[0]).toMatchObject({ name: 'browser', label: 'Управление браузером', tools: ['browser_click'] })
  }
})

it('keeps descriptions supplied by plugins and by newer backends', () => {
  for (const row of [
    { ...browser, name: 'plugin_browser' },
    { ...browser, description: 'Custom backend' }
  ]) {
    expect(filteredToolsets([row], '', {}, true, ru.catalog.builtinToolsets)[0].description).toBe(row.description)
  }
})
