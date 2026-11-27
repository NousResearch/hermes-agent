import { describe, expect, it } from 'vitest'

import type { AutomationBlueprint } from '@/hermes'
import { TRANSLATIONS } from '@/i18n/catalog'

import { initialBlueprintValues, localizeBlueprints } from './blueprints'

function blueprint(fields: AutomationBlueprint['fields']): AutomationBlueprint {
  return {
    key: 'test',
    title: 'Test',
    description: '',
    category: 'general',
    tags: [],
    command: '',
    appUrl: '',
    fields
  }
}

describe('initialBlueprintValues', () => {
  it('seeds each field from its default', () => {
    const values = initialBlueprintValues(
      blueprint([
        { name: 'time', type: 'time', label: 'Time', default: '08:00', options: [], optional: false, help: '' },
        {
          name: 'topic',
          type: 'enum',
          label: 'Topic',
          default: 'news',
          options: ['news', 'sports'],
          optional: false,
          help: ''
        }
      ])
    )

    expect(values).toEqual({ time: '08:00', topic: 'news' })
  })

  it('falls back to an empty string when a field has no default', () => {
    const values = initialBlueprintValues(
      blueprint([{ name: 'topic', type: 'text', label: 'Topic', default: null, options: [], optional: true, help: '' }])
    )

    expect(values).toEqual({ topic: '' })
  })

  it('returns an empty object for a blueprint with no fields', () => {
    expect(initialBlueprintValues(blueprint([]))).toEqual({})
  })

  it("seeds the deliver slot to 'local' when its default is the dashboard-only 'origin'", () => {
    const values = initialBlueprintValues(
      blueprint([
        {
          name: 'deliver',
          type: 'enum',
          label: 'Deliver',
          default: 'origin',
          options: ['origin', 'local', 'telegram'],
          optional: false,
          help: ''
        }
      ])
    )

    expect(values).toEqual({ deliver: 'local' })
  })

  it("seeds the deliver slot to 'local' when it has no default", () => {
    const values = initialBlueprintValues(
      blueprint([
        {
          name: 'deliver',
          type: 'enum',
          label: 'Deliver',
          default: null,
          options: ['origin', 'local'],
          optional: false,
          help: ''
        }
      ])
    )

    expect(values).toEqual({ deliver: 'local' })
  })

  it('leaves a non-origin deliver default untouched', () => {
    const values = initialBlueprintValues(
      blueprint([
        {
          name: 'deliver',
          type: 'enum',
          label: 'Deliver',
          default: 'telegram',
          options: ['origin', 'local', 'telegram'],
          optional: false,
          help: ''
        }
      ])
    )

    expect(values).toEqual({ deliver: 'telegram' })
  })
})

describe('localizeBlueprints', () => {
  const ru = TRANSLATIONS.ru.cron.blueprints

  const morning: AutomationBlueprint = {
    ...blueprint([
      { name: 'time', type: 'time', label: 'What time?', default: '08:00', options: [], optional: false, help: 'h' },
      {
        name: 'deliver',
        type: 'enum',
        label: 'Where to deliver?',
        default: 'origin',
        options: ['origin', 'local'],
        optional: false,
        help: 'origin = …'
      }
    ]),
    key: 'morning-brief',
    title: 'Morning briefing'
  }

  it('renders a built-in blueprint in the catalog language without touching what gets scheduled', () => {
    const [localized] = localizeBlueprints([morning], ru)

    expect(localized.title).toBe(ru.catalog['morning-brief'].title)
    expect(localized.description).toBe(ru.catalog['morning-brief'].description)
    expect(localized.fields.map(field => field.label)).toEqual([
      ru.sharedFields.time.label,
      ru.sharedFields.deliver.label
    ])
    expect(localized.key).toBe('morning-brief')
    expect(localized.fields.map(field => [field.name, field.default, field.options])).toEqual(
      morning.fields.map(field => [field.name, field.default, field.options])
    )
  })

  it('leaves plugin and unknown blueprints as the backend sent them', () => {
    const plugin = { ...morning, key: 'acme:morning-brief', title: 'Acme briefing' }

    expect(localizeBlueprints([plugin], ru)).toEqual([plugin])
  })
})
