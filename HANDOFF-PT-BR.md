# Handoff: Projeto de Localização pt-BR - Hermes Agent

## 📋 Resumo Executivo

**Projeto:** Adição de suporte nativo ao Português do Brasil (pt-BR)  
**Status:** Em desenvolvimento - Tradução em andamento  
**Prioridade:** Alta  
**Stakeholders:** Comunidade Brazilian de IA, Usuários Hermes no Brasil  

---

## 🎯 Objetivos

### Primários
- [x] Configurar backend i18n com pt-BR
- [x] Configurar frontend TypeScript i18n
- [x] Criar estrutura de arquivos
- [ ] **Traduzir completamente pt-br.ts (3405 linhas)** ⭐ **CRÍTICO**
- [ ] Validar integração completa
- [ ] Testar todos os aliases
- [ ] Criar Pull Request

### Secundários
- [ ] Criar documentação de contribuição
- [ ] Criar script de tradução automática
- [ ] Estabelecer processo de manutenção
- [ ] Divulgar para comunidade

---

## 📁 Estrutura do Projeto

```
hermes-agent/
├── CONTRIBUTION-PT-BR.md          # Documentação da contribuição
├── HANDOFF-PT-BR.md               # Este arquivo
├── translate_ptbr.py              # Script de tradução
├── agent/
│   └── i18n.py                    # ✅ Config backend i18n
└── apps/
    └── desktop/
        └── src/
            └── i18n/
                ├── types.ts       # ✅ Tipo Locale + pt-br
                ├── languages.ts   # ✅ LOCALE_OPTIONS + aliases
                ├── catalog.ts      # ✅ Import ptBr
                ├── en.ts          # Arquivo fonte (3405 linhas)
                └── pt-br.ts        # ⚠️ PRECISA SER TRADUZIDO
```

---

## 📊 Métricas

| Métrica | Valor | Status |
|---------|-------|--------|
| Linhas totais pt-br.ts | 3405 | ❌ Em inglês (cópia de en.ts) |
| Linhas traduzidas | 0 | ⚠️ Pendente |
| % Completo | 0% | ⚠️ Pendente |
| Arquivos modificados | 4/5 | ⚠️ 1 pendente |
| Aliases configurados | 8 | ✅ Completo |

---

## 🎨 Padrões de Tradução

### Regras de Ouro

1. **✅ FAÇA:**
   - Use verbos no infinitivo para botões (Salvar, Cancelar, Aplicar)
   - Mantenha termos técnicos em inglês (API, Token, GitHub, Endpoint)
   - Use "…" em vez de "..." para reticências
   - Mantenha a estrutura das strings com variáveis
   - Seja consistente com traduções anteriores

2. **❌ NÃO FAÇA:**
   - Traduzir termos técnicos amplamente usados
   - Alterar a estrutura do TypeScript
   - Remover ou adicionar chaves
   - Usar gírias ou linguagem informal
   - Quebrar strings com variáveis

### Guia Rápido

| Inglês | Português | categoria |
|--------|-----------|-----------|
| Apply | Aplicar | ação |
| Save | Salvar | ação |
| Delete | Excluir | ação |
| Cancel | Cancelar | ação |
| Settings | Configurações | substantivo |
| Error | Erro | substantivo |
| Warning | Aviso | substantivo |
| Loading... | Carregando… | estado |
| Model | Modelo | técnico |
| Provider | Provedor | técnico |
| API key | Chave da API | técnico |

---

## 🔧 Configurações Técnicas

### Backend (agent/i18n.py)

```python
# Linha ~24-28
SUPPORTED_LANGUAGES = {
    "en": "English",
    "pt-br": "Português do Brasil",  # ✅ Adicionado
}

# Linha ~34-42
_LANGUAGE_ALIASES = {
    "pt": "pt-br",
    "pt_br": "pt-br",
    "portugues": "pt-br",
    "portugues-br": "pt-br",
    "portuguese": "pt-br",
    "brazilian": "pt-br",
    "brasil": "pt-br",
    "brazil": "pt-br",
}
```

### Frontend TypeScript

#### types.ts
```typescript
// Linha ~3
export type Locale = 'en' | 'pt-br'  // ✅ Adicionado 'pt-br'
```

#### languages.ts
```typescript
// Linha ~18-25
export const LOCALE_OPTIONS: LocaleOption[] = [
  { value: 'en', label: 'English' },
  { value: 'pt-br', label: 'Português do Brasil', aliases: ['pt-br', 'pt_br', 'brazilian', 'brasil', 'portugues', 'portugues-br'] }, // ✅ Adicionado
]
```

#### catalog.ts
```typescript
// Linha ~3
import { ptBr } from './pt-br'  // ✅ Adicionado

// Linha ~7-12
export const TRANSLATIONS: Record<Locale, any> = {
  en,
  'pt-br': ptBr,  // ✅ Adicionado
}
```

---

## 🚀 Próximos Passos

### Prioridade 1: Tradução do pt-br.ts

**Responsável:** [Seu Nome]  
**Prazo:** Imediato  
**Esforço:** 4-6 horas (tradução manual cuidadosa)

**Ações:**
1. Ler o arquivo `en.ts` completamente
2. Traduzir cada string mantendo:
   - Estrutura TypeScript
   - Chaves exatas
   - Variáveis (${variable})
   - Funções arrow (=>)
3. Verificar consistência das traduções
4. Testar strings com variáveis

### Prioridade 2: Validação

**Responsável:** [Seu Nome]  
**Prazo:** Após tradução  
**Esforço:** 1-2 horas

**Checklist:**
- [ ] `npm run typecheck` passa
- [ ] `npm run build` passa
- [ ] Teste manual da interface
- [ ] Verificar todos os aliases
- [ ] Testar troca de idioma

### Prioridade 3: Commit & Push

**Responsável:** [Seu Nome]  
**Prazo:** Após validação  
**Esforço:** 30 minutos

**Ações:**
1. Criar branch: `git checkout -b feature/pt-br-i18n`
2. Adicionar arquivos: `git add .`
3. Commit: `git commit -m "feat: add Portuguese (Brazil) i18n support"`
4. Push: `git push origin feature/pt-br-i18n`

### Prioridade 4: Pull Request

**Responsável:** [Seu Nome]  
**Prazo:** Após push  
**Esforço:** 30 minutos

**Template PR:**
```markdown
## Description

Add native Portuguese (Brazil) language support to Hermes Agent Desktop.

## Changes

- Added pt-BR to backend i18n configuration (agent/i18n.py)
- Added pt-br locale to frontend TypeScript types
- Added language option with aliases in languages.ts
- Added complete translation file (pt-br.ts) with 3405 translated strings
- Integrated with existing i18n system

## Aliases

Users can set language as: pt-br, pt_br, brazilian, brasil, portugues, portugueses-br, portuguese

## Testing

- TypeScript type checking passes
- Build completes successfully
- Language selection works in settings
- All aliases resolve correctly

## Screenshots

[Adicionar screenshots após implementação]

## Related Issues

Fixes #XXXX (se aplicável)
```

---

## 📞 Contatos

| Papel | Nome | Contato |
|-------|------|---------|
| Autor Principal | [Seu Nome] | [seu-email] |
| Revisor | [Nome Revisor] | [email-revisor] |
| Mantenedor Hermes | Nous Research | [discord] |

---

## 📅 Cronograma

| Data | Atividade | Status |
|------|-----------|--------|
| 2026-08-22 | Início do projeto | ✅ Completo |
| 2026-08-22 | Configuração backend | ✅ Completo |
| 2026-08-22 | Configuração frontend | ✅ Completo |
| 2026-08-22 | **Tradução pt-br.ts** | ⚠️ Em andamento |
| 2026-08-22 | Validação técnica | ⏳ Pendente |
| 2026-08-22 | Commit & Push | ⏳ Pendente |
| 2026-08-22 | Criar Pull Request | ⏳ Pendente |

---

## 🎓 Lições Aprendidas

1. **Backend i18n:** O sistema já está bem estruturado, basta adicionar o novo locale a entrada
2. **Frontend TypeScript:** Precisa manter tipagem forte
3. **Tradução:** Demanda mais tempo do que parece, exigir revisão
4. **Aliases:** Importante para UX, usuário pode usar vários formatos

---

## 🔗 Recursos Úteis

- [Documentação Hermes](https://github.com/NousResearch/hermes-agent)
- [Guia de Contribuição](https://github.com/NousResearch/hermes-agent/blob/main/CONTRIBUTING.md)
- [Discord Hermes](https://discord.gg/nous)
- [Guia de Localização TypeScript](https://www.i18next.com/)

---

## ✅ Checklist Final

- [ ] pt-br.ts completamente traduzido
- [ ] TypeScript typecheck passa
- [ ] Build passa
- [ ] Testes manuais realizados
- [ ] Branch criado
- [ ] Commit com mensagem descritiva
- [ ] Push para repositório remoto
- [ ] Pull Request criado
- [ ] Documentação atualizada
- [ ] Screenshots adicionadas (opcional)

---

**Status:** 🔄 Em andamento  
**Próximo Passo:** Traduzir pt-br.ts  
**Bloqueadores:** Nenhum  
**Riscos:** Tradução pode conter erros - necessita revisão  

---

*Documento gerado: 2026-08-22*  
*Versão: 1.0.0*
