import { describe, expect, it } from 'vitest'

import type { AutomationBlueprint } from '@/hermes'
import { ptBr } from '@/i18n/pt-br'

import { blueprintDisplayCopy, initialBlueprintValues } from './blueprints'

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

describe('blueprintDisplayCopy', () => {
  it('uses the pt-BR catalog without replacing backend-owned English copy', () => {
    const item = blueprint([])
    item.key = 'morning-brief'
    item.title = 'Morning briefing'
    item.description = 'Backend description'

    expect(blueprintDisplayCopy(item, ptBr.cron.blueprints.catalog)).toEqual({
      title: 'Briefing matinal',
      description: 'Um breve resumo diário: agenda de hoje, clima e itens urgentes aguardando você.'
    })
  })

  it('falls back to backend copy for blueprints added after the locale was shipped', () => {
    const item = blueprint([])
    item.key = 'new-blueprint'
    item.title = 'New blueprint'
    item.description = 'New backend description'

    expect(blueprintDisplayCopy(item, ptBr.cron.blueprints.catalog)).toEqual({
      title: 'New blueprint',
      description: 'New backend description'
    })
  })
})
