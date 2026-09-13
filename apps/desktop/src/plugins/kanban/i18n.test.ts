import { describe, expect, it } from 'vitest'

import { KANBAN_LOCALES, ptBr } from './i18n'

describe('Kanban pt-BR locale', () => {
  it('registers the translated board-management labels', () => {
    expect(KANBAN_LOCALES['pt-br']).toBe(ptBr)
    expect(ptBr.newBoard).toBe('Novo quadro')
    expect(ptBr.exportDots).toBe('Exportar…')
    expect(ptBr.deleteBoardTitle('Planejamento')).toBe('Excluir “Planejamento”?')
    expect(ptBr.defaultParen).toBe('(padrão)')
  })
})
