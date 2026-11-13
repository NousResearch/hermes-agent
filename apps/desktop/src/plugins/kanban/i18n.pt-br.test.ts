import { describe, expect, it } from 'vitest'

import { en, KANBAN_LOCALES } from './i18n'

function verifyMessages(source: unknown, translated: unknown, path = '') {
  expect(typeof translated, path).toBe(typeof source)

  if (typeof source === 'function') {
    const message = translated as (...args: unknown[]) => string
    expect(message.length, path).toBe(source.length)

    for (const args of [
      [0, 0, ''],
      ['', '', ''],
      [1, 2, 'erro'],
      ['__primeiro__', '__segundo__', '__terceiro__']
    ]) {
      const english = source(...args) as string
      const portuguese = message(...args)
      expect(typeof portuguese, path).toBe('string')
      expect(portuguese, path).not.toMatch(/undefined|NaN/)

      for (const arg of args) {
        if (typeof arg === 'string' && arg && english.includes(arg)) {
          expect(portuguese, path).toContain(arg)
        }
      }
    }

    return
  }

  if (source && typeof source === 'object') {
    expect(Array.isArray(translated), path).toBe(Array.isArray(source))
    expect(Object.keys(translated as object).sort(), path).toEqual(Object.keys(source).sort())

    for (const [key, value] of Object.entries(source)) {
      verifyMessages(value, (translated as Record<string, unknown>)[key], path ? `${path}.${key}` : key)
    }

    return
  }

  expect(translated, path).not.toBe('')

  // Product names, units and runnable identifiers must not be translated.
  if (!['nav', 'title', 'tokUnit', 'skillsPlaceholder', 'projectHintCmd', 'slug'].includes(path)) {
    expect(translated, path).not.toBe(source)
  }
}

describe('Brazilian Portuguese Kanban messages', () => {
  it('ships a complete translated catalog with working task, board and notification messages', () => {
    const pt = KANBAN_LOCALES['pt-br'] as typeof en | undefined
    expect(pt, 'pt-br must be bundled instead of falling back to English').toBeDefined()

    if (!pt) {
      throw new Error('Missing bundled pt-br Kanban messages')
    }

    verifyMessages(en, pt)
    expect(pt.save).toBe('Salvar')
    expect(pt.delete).toBe('Excluir')
    expect(pt.openBoard).toBe('Kanban: Abrir quadro')
    expect(pt.col.ready.label).toBe('Prontas')
    expect(pt.countTip(2, 3)).toBe('Kanban — 2 em execução, 3 prontas')
    expect(pt.moveTo('Revisão')).toBe('Mover para Revisão')
    expect(pt.evtCreated('', '')).toBe('criada')
    expect(pt.evtCreated('Triagem', 'Ana')).toBe('criada em Triagem · atribuída a Ana')
    expect(pt.evtUnblocked('')).toBe('desbloqueada → Prontas')
    expect(pt.evtUnblocked('Revisão')).toBe('desbloqueada → Revisão')
    expect(pt.bulkFailed(2, 3, 'sem conexão')).toContain('2 de 3')
    expect(pt.bulkFailed(2, 3, 'sem conexão')).toContain('sem conexão')
    expect(pt.boardExported('/pasta/quadro.json')).toBe('Quadro exportado para /pasta/quadro.json')
    expect(pt.deleteBoardTitle('Projeto')).toBe('Excluir "Projeto"?')
    expect(pt.notify.completedTitle).toBe('Tarefa concluída')
    expect(pt.notify.artifacts(0)).toBe('0 artefatos')
    expect(pt.notify.artifacts(1)).toBe('1 artefato')
    expect(pt.notify.artifacts(2)).toBe('2 artefatos')
    expect(pt.nSelected(1)).toBe('1 selecionado')
    expect(pt.nSelected(2)).toBe('2 selecionados')
    expect(pt.blockKindTip('needs_input')).toContain('intervenção humana')
    expect(pt.blockKindTip('capability')).toContain('capacidade')
    expect(pt.blockKindTip('transient')).toContain('temporário')
    expect(pt.blockKindTip('dependency')).toContain('dependência')
  })
})
