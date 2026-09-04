# Plano de Contribuição: Suporte à Lingua Portuguesa do Brasil (pt-BR)

## Visão Geral

Este plano detalha o processo para adicionar suporte completo à língua Portuguesa do Brasil (pt-BR) ao Hermes Agent Desktop, permitindo que usuários brasileiros utilizem a interface em seu idioma nativo.

## Objetivos

1. **Adicionar pt-BR como linguagem suportada** no core do Hermes Agent
2. **Traduzir completamente** o arquivo `apps/desktop/src/i18n/pt-br.ts` (3.405 linhas)
3. **Integrar a linguagem** no seletor de idiomas do frontend
4. **Testar e validar** todas as traduções
5. **Contribuir de volta** para a comunidade via Pull Request

## Escopo

### Arquivos a Serem Modificados

#### Core (Python)
- ✅ `agent/i18n.py` - Adicionar pt-br a SUPPORTED_LANGUAGES e _LANGUAGE_ALIASES

#### Desktop App (TypeScript)
- ✅ `apps/desktop/src/i18n/types.ts` - Adicionar 'pt-br' ao tipo Locale
- ✅ `apps/desktop/src/i18n/languages.ts` - Adicionar entrada pt-br em LOCALE_OPTIONS
- ✅ `apps/desktop/src/i18n/catalog.ts` - Adicionar import ptBr e mapeamento
- 🔄 `apps/desktop/src/i18n/pt-br.ts` - **TRADUÇÃO COMPLETA NECESSÁRIA** (3.405 linhas)

### Arquivos de Documentação a Serem Criados

1. **CONTRIBUTION-PT-BR.md** - Guia de contribuição em pt-BR
2. **HANDOFF-PT-BR.md** - Documentação de handoff
3. **PLANO-PT-BR.md** - Este plano (✅ Criado)
4. **README-PT-BR.md** - README local do projeto

## Estratégia de Tradução

### Princípios de Tradução

1. **Preservar termos técnicos**: API, Token, GitHub, JSON, YAML, etc. não são traduzidos
2. **Usar verbos no infinitivo**: "Salvar", "Cancelar", "Aplicar" (não "Salve", "Cancele")
3. **Manter formatação**: Preservar `${variables}`, arrow functions (`=>`), template literals (`` `text` ``)
4. **Manter números e booleanos**: `1`, `0`, `true`, `false` não são traduzidos
5. **Consistência**: Usar os mesmos termos para o mesmo conceito em todo o arquivo

### Aliases da Linguagem

- Código principal: `pt-br`
- Aliases: `pt`, `pt_br`, `brazilian`, `brasil`, `portugues`, `portugues-br`, `portuguese`, `brazil`

### Seções do Arquivo pt-br.ts

O arquivo está organizado nas seguintes seções principais:

1. **common** - Termos comuns (100+ linhas)
2. **fileMenu** - Menu de arquivos (20 linhas)
3. **boot** - Mensagens de inicialização (50+ linhas)
4. **notifications** - Notificações (100+ linhas)
5. **remoteDisplayBanner** - Banner de display remoto (5 linhas)
6. **billingBlock** - Bloqueio por créditos (10 linhas)
7. **sendDiagnostics** - Envio de diagnósticos (30 linhas)
8. **titlebar** - Barra de título (20 linhas)
9. **keybinds** - Atalhos de teclado (500+ linhas)
10. **settings** - Configurações (1000+ linhas)
11. **chat** - Chat e mensagens (500+ linhas)
12. **composer** - Composer de mensagens (200+ linhas)
13. **sidebar** - Barra lateral (200+ linhas)
14. **profiles** - Perfis (100+ linhas)
15. **memory** - Memória (100+ linhas)
16. **skills** - Habilidades (100+ linhas)
17. **agents** - Agentes (100+ linhas)
18. **onboarding** - Onboarding (200+ linhas)
19. **updates** - Atualizações (100+ linhas)
20. **errors** - Erros (100+ linhas)

## Timeline

### Fase 1: Preparação (1 dia)
- ✅ Analisar estrutura do projeto
- ✅ Criar plano detalhado
- ✅ Configurar ambiente de desenvolvimento
- ✅ Criar arquivos de documentação inicial

### Fase 2: Tradução (3-5 dias)
- Traduzir arquivo pt-br.ts por seções
- Validar cada seção traduzida
- Testar integração com o desktop app

### Fase 3: Integração e Testes (1-2 dias)
- Verificar todos os arquivos de configuração
- Testar seletor de linguagem
- Validar que não há erros de sintaxe TypeScript
- Testar funcionalidades principais

### Fase 4: Contribuição (1 dia)
- Criar commit com todas as alterações
- Fazer push para branch de contribuição
- Criar Pull Request para o repositório principal
- Solicitar review da comunidade

## Ferramentas Utilizadas

1. **VS Code** - Editor de código com extensões:
   - ESLint
   - Prettier
   - TypeScript
   - GitLens

2. **Git** - Controle de versão

3. **Python 3.12+** - Para execução de scripts de validação

4. **Node.js 20+** - Para build do desktop app

## Padrões de Qualidade

### TypeScript
- ✅ Nunca quebrar a estrutura de tipos
- ✅ Preservar todas as chaves do objeto de tradução
- ✅ Manter formatação consistente
- ✅ Não introduzir erros de sintaxe

### Tradução
- ✅ Todos os textos em português do Brasil
- ✅ Ortografia correta
- ✅ Gramática correta
- ✅ Consistência terminológica
- ✅ Natural para falantes nativos

## Checklist de Aceitação

- [ ] Arquivo pt-br.ts completamente traduzido
- [ ] Todas as chaves do en.ts estão presentes no pt-br.ts
- [ ] Nenhuma string em inglês restante (exceto termos técnicos)
- [ ] Todos os arquivos de configuração atualizados
- [ ] Seletor de linguagem funcionando no desktop app
- [ ] Nenhum erro de TypeScript
- [ ] Documentação completa
- [ ] Commit com mensagem descritiva
- [ ] Push para repositório remoto
- [ ] Pull Request criado

## Próximos Passos

1. **Iniciar tradução** do arquivo pt-br.ts
2. **Criar script de validação** para verificar consistência
3. **Testar integração** no desktop app
4. **Finalizar documentação**
5. **Fazer commit e push**
6. **Criar Pull Request**

---

## Contribuidores

- **LuisCard** - Líder do projeto de tradução
- **Hermes Agent** - Assistente de desenvolvimento

## Data de Início

22 de agosto de 2026

## Data de Término Estimada

28 de agosto de 2026
