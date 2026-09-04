# Contribuição: Suporte Nativo a Português do Brasil (pt-BR)

## Visão Geral

Esta contribuição adiciona suporte completo ao **Português do Brasil (pt-BR)** ao Hermes Agent Desktop, incluindo:

- ✅ Configuração de i18n no backend (`agent/i18n.py`)
- ✅ Seletor de idioma no frontend com aliases
- ✅ Arquivo de tradução completo (`apps/desktop/src/i18n/pt-br.ts`)
- ✅ Integração com sistema de internacionalização existente

## Arquitetura da Solução

### Backend (Python)

**Localização:** `agent/i18n.py`

```python
SUPPORTED_LANGUAGES = {
    "en": "English",
    "pt-br": "Português do Brasil",  # Adicionado
    # ...
}

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

### Frontend (TypeScript)

**Estrutura:**
```
apps/desktop/src/i18n/
├── types.ts          # Tipo Locale com 'pt-br'
├── languages.ts      # Opções de idioma + aliases
├── catalog.ts        # Catálogo com import ptBr
└── pt-br.ts          # Traduções completas
```

## Arquivos Modificados

### 1. Backend - Configuração i18n
- `agent/i18n.py`
  - Adicionado `"pt-br": "Português do Brasil"` em `SUPPORTED_LANGUAGES`
  - Adicionado 8 aliases em `_LANGUAGE_ALIASES`

### 2. Frontend - Tipos TypeScript
- `apps/desktop/src/i18n/types.ts`
  - Adicionado `'pt-br'` ao tipo `Locale`

### 3. Frontend - Opções de Idioma
- `apps/desktop/src/i18n/languages.ts`
  - Adicionada entrada `pt-br` em `LOCALE_OPTIONS`
  - Adicionados aliases: `pt-br`, `pt_br`, `brazilian`, `brasil`, `portugues`, `portugues-br`

### 4. Frontend - Catálogo de Traduções
- `apps/desktop/src/i18n/catalog.ts`
  - Adicionado import: `import { ptBr } from './pt-br'`
  - Adicionada entrada: `'pt-br': ptBr` em `TRANSLATIONS`

### 5. Frontend - Arquivo de Tradução
- `apps/desktop/src/i18n/pt-br.ts`
  - **Arquivo principal** com 3405 linhas traduzidas
  - Estrutura idêntica ao `en.ts`
  - Todas as strings de interface traduzidas para pt-BR

## Seletor de Idioma

### Aliases Suportados

O usuário pode definir o idioma como:
- `pt-br` (padrão)
- `pt_br`
- `brazilian`
- `brasil`
- `portugues`
- `portugues-br`
- `portuguese` (compatibilidade)

### Como Alterar o Idioma

**Via Configuração:**
```yaml
# ~/.hermes/config.yaml
display:
  language: pt-br
```

**Via Comando:**
```bash
hermes config display.language pt-br
```

**Via Interface Desktop:**
Settings → Appearance → Language → Português do Brasil

## Estrutura do Arquivo pt-br.ts

O arquivo está organizado nas seguintes seções:

```typescript
{
  common: {},           // Termos comuns (Aplicar, Salvar, Cancelar)
  fileMenu: {},         // Menu de arquivo
  boot: {},             // Mensagens de inicialização
  notifications: {},    // Notificações
  remoteDisplayBanner: {}, // Banner de display remoto
  billingBlock: {},     // Bloqueio por faturamento
  sendDiagnostics: {},  // Envio de diagnósticos
  titlebar: {},         // Barra de título
  keybinds: {},         // Atalhos de teclado
  settings: {},         // Configurações
  contextMenu: {},      // Menu de contexto
  chat: {},             // Interface de chat
  composer: {},         // Composer de mensagens
  sidekick: {},         // Assistente lateral
  modelPicker: {},      // Seletor de modelos
  profiles: {},         // Gerenciamento de perfis
  about: {},            // Sobre
  updates: {},          // Atualizações
  onboarding: {},       // Onboarding
  experimental: {},     // Recursos experimentais
  shortcuts: {},        // Atalhos
  tooltips: {},         // Tooltips
  a11y: {},             // Acessibilidade
  fieldLabels: {},      // Rótulos de campos (de constants.ts)
  fieldDescriptions: {} // Descrições de campos (de constants.ts)
}
```

## Padrões de Tradução

### Convenções Adotadas

1. **Termos Técnicos:** Mantidos em inglês quando amplamente utilizados
   - Exemplo: "API", "Token", "Endpoint", "GitHub"

2. **Interface do Usuário:**
   - Botões: Verbos no infinitivo ("Salvar", "Cancelar", "Aplicar")
   - Mensagens: Voz ativa, clara e direta
   - Erros: Descritivas com sugestões de ação

3. **Consistência:**
   - Mesmo termo sempre traduzido da mesma forma
   - Seguir padrões de tradução de projetos populares (VS Code, GitHub, etc.)

4. **Strings com Variáveis:**
   ```typescript
   // CORRETO - manter variáveis
   deleteTitle: name => `Excluir ${name}?`,
   
   // CORRETO - interpolacao
   welcome: name => `Bem-vindo, ${name}!`
   ```

5. **Formatação:**
   - Pontuação: Seguir padrões do Português do Brasil
   - Aspas: Usar aspas duplas (" ") para strings
   - Reticências: Usar "…" (alt+0133) em vez de "..."

### Exemplos de Tradução

| Inglês | Português (pt-BR) |
|--------|-------------------|
| Apply | Aplicar |
| Cancel | Cancelar |
| Save | Salvar |
| Delete | Excluir |
| Settings | Configurações |
| Error | Erro |
| Warning | Aviso |
| Success | Sucesso |
| Loading... | Carregando… |
| Connect | Conectar |
| Disconnect | Desconectar |
| New chat | Nova conversa |
| Send message | Enviar mensagem |
| Model | Modelo |
| Provider | Provedor |
| API key | Chave da API |
| Token | Token |

## Testes Realizados

- [ ] Verificação de importação correta do arquivo pt-br.ts
- [ ] Validação de tipos TypeScript
- [ ] Teste de seleção de idioma via config
- [ ] Teste de seleção de idioma via interface
- [ ] Verificação de aliases funcionando
- [ ] Teste de renderização de strings traduzidas
- [ ] Verificação de strings com variáveis

## Como Contribuir

1. **Fork o repositório**
2. **Crie uma branch:** `git checkout -b feature/pt-br-i18n`
3. **Aplique as alterações** conforme descrito acima
4. **Teste localmente:**
   ```bash
   cd apps/desktop
   npm run dev
   ```
5. **Verifique os tipos:**
   ```bash
   npm run typecheck
   ```
6. **Crie o Pull Request** para a branch `main`

## Manutenção Contínua

Para manter as traduções atualizadas:

```bash
# Verificar novas strings no en.ts
grep -E "^[a-zA-Z_]+:" apps/desktop/src/i18n/en.ts | wc -l
grep -E "^[a-zA-Z_]+:" apps/desktop/src/i18n/pt-br.ts | wc -l

# Script de tradução automática (base)
python translate_ptbr.py
```

## Créditos

- **Autor Principal:** [Seu Nome]
- **Revisão:** Comunidade Hermes Agent
- **Base de Tradução:** Google Translate API + Revisão Manual

## Licença

Esta contribuição segue a licença do projeto Hermes Agent (MIT).

---

**Status:** ✅ Pronto para Revisão  
**Versão:** 1.0.0  
**Data:** 2026-08-22  
**Idioma:** Português do Brasil (pt-BR)
