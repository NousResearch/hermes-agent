# Hermes Agent - Suporte a Português Brasil (pt-br)

> **Status do Projeto**: Em Desenvolvimento Ativo
> **Versão**: 1.0.0-beta
> **Autor**: Studio CodeAI
> **Data de Criação**: 22 de Agosto de 2026
> **Licença**: MIT

---

## 📋 Sumário

1. [Visão Geral](#-visão-geral)
2. [Objetivos do Projeto](#-objetivos-do-projeto)
3. [Escopo](#-escopo)
4. [Estrutura do Projeto](#-estrutura-do-projeto)
5. [Convenções de Tradução](#-convenções-de-tradução)
6. [Arquivos Modificados](#-arquivos-modificados)
7. [Como Contribuir](#-como-contribuir)
8. [Testes](#-testes)
9. [Documentação Adicional](#-documentação-adicional)
10. [Licença](#-licença)

---

## 🎯 Visão Geral

Este projeto adiciona suporte completo ao idioma **Português do Brasil (pt-br)** ao Hermes Agent, um agente de IA pessoal de código aberto que executa o mesmo núcleo de agente em CLI, gateway de mensagens, TUI e aplicativo desktop Electron.

O objetivo é permitir que usuários brasileiros utilizem o Hermes Agent em seu idioma nativo, mantendo a qualidade e consistência com a versão original em inglês.

---

## 🎯 Objetivos do Projeto

### Objetivos Principais
- ✅ Adicionar suporte completo ao idioma Português do Brasil (pt-br)
- ✅ Traduzir toda a interface do usuário do aplicativo desktop
- ✅ Manter termos técnicos em inglês (API, Token, GitHub, etc.)
- ✅ Preservar a estrutura de arquivos TypeScript e formatação
- ✅ Criar documentação completa para a comunidade

### Objetivos Secundários
- 🔄 Criar scripts de automação para futuras atualizações
- 🔄 Adicionar suporte a outros dialetos do português (pt-pt)
- 🔄 Implementar sistema de tradução colaborativa

---

## 📏 Escopo

### **Incluído no Escopo**
- ✅ Configuração do sistema de internacionalização (i18n)
- ✅ Tradução completa do arquivo `pt-br.ts` (3.405 linhas)
- ✅ Criação do dicionário de tradução `i18n-dict-pt-br.json`
- ✅ Configuração do seletor de idioma no frontend
- ✅ Documentação técnica e de usuário
- ✅ Scripts de validação e teste

### **Fora do Escopo**
- ❌ Tradução de documentação existente (README.md, docs/)
- ❌ Tradução de mensagens do backend Python
- ❌ Modificações na lógica de funcionamento do Hermes
- ❌ Criação de novos recursos ou funcionalidades

---

## 🗂️ Estrutura do Projeto

```
hermes-agent/
├── apps/
│   └── desktop/
│       └── src/
│           └── i18n/
│               ├── pt-br.ts          # 🇧🇷 Tradução completa
│               ├── en.ts            # 🇬🇧 Arquivo base
│               ├── catalog.ts       # Catálogo de idiomas
│               ├── languages.ts     # Configuração de idiomas
│               └── types.ts         # Tipos TypeScript
│
├── agent/
│   └── i18n.py                # Configuração backend i18n
│
├── i18n-dict-pt-br.json        # 📖 Dicionário de tradução
│
├── README-PT-BR.md            # 📄 Este documento
├── PLANO-PT-BR.md              # 📋 Plano de desenvolvimento
├── CONTRIBUTION-PT-BR.md       # 🤝 Guia de contribuição
├── HANDOFF-PT-BR.md            # 📦 Documentação de entrega
└── .github/
    └── pull_request_template/
        └── pt-br_template.md    # Template PR pt-br
```

---

## 📝 Convenções de Tradução

### **Regras Gerais**
1. **Termos Técnicos**: Mantenha em inglês (ex: API, Token, GitHub, JSON, CLI, etc.)
2. **Variáveis e Placeholders**: Preserve exatas (ex: `${name}`, `${variable}`)
3. **Funções Arrow**: Mantenha a sintaxe (ex: `term => "`Tente "${term}""`)
4. **Template Literals**: Preserve as crases (ex: \`texto\`)
5. **Números e Booleanos**: Mantenha em inglês (true, false, 0, 1, etc.)

### **Regras Específicas**
| Tipo | Inglês | Português | Exemplo |
|------|--------|-----------|---------|
| Botões | Save | Salvar | `Salvar` |
| Ações | Apply | Aplicar | `Aplicar` |
| Rótulos | Name | Nome | `Nome` |
| Mensagens | Loading... | Carregando... | `Carregando...` |
| Erros | Error | Erro | `Erro` |
| Sucesso | Success | Sucesso | `Sucesso` |

### **Formatação**
- **Data**: DD/MM/AAAA
- **Hora**: HH:MM:SS
- **Moeda**: R$ 0,00
- **Números**: 1.000,00 (separador de milhar: ponto, decimal: vírgula)

### **Termos a Mantenha em Inglês**
```
API, Token, GitHub, Git, Repository, Repository, Branch, Commit, Push, Pull,
Pull Request, PR, Issue, Fork, Star, Clone, SSH, HTTPS, URL, HTTP, HTTPS,
JSON, YAML, XML, HTML, CSS, JavaScript, TypeScript, React, Node.js, Python,
Terminal, Shell, Command, CLI, GUI, TUI, Electron, WebSocket, REST, GraphQL,
Database, SQL, PostgreSQL, SQLite, Model, Provider, Agent, Session, Memory,
Context, Prompt, Token, Embedding, LLM, AI, Machine Learning, Neural Network
```

---

## 📁 Arquivos Modificados

### **Backend (Python)**
| Arquivo | Modificação | Status |
|---------|-------------|--------|
| `agent/i18n.py` | Adicionado pt-br a SUPPORTED_LANGUAGES | ✅ Concluído |
| `agent/i18n.py` | Adicionado aliases a _LANGUAGE_ALIASES | ✅ Concluído |

### **Frontend (TypeScript)**
| Arquivo | Modificação | Status |
|---------|-------------|--------|
| `apps/desktop/src/i18n/types.ts` | Adicionado 'pt-br' ao tipo Locale | ✅ Concluído |
| `apps/desktop/src/i18n/languages.ts` | Configuração do pt-br | ✅ Concluído |
| `apps/desktop/src/i18n/catalog.ts` | Mapeamento do catálogo | ✅ Concluído |
| `apps/desktop/src/i18n/pt-br.ts` | **Tradução completa** | 🔄 Em Progresso |

### **Dicionários e Configurações**
| Arquivo | Descrição | Status |
|---------|-----------|--------|
| `i18n-dict-pt-br.json` | Dicionário de tradução | ✅ Concluído |

### **Documentação**
| Arquivo | Descrição | Status |
|---------|-----------|--------|
| `README-PT-BR.md` | Este documento | 🔄 Em Progresso |
| `PLANO-PT-BR.md` | Plano de desenvolvimento | ✅ Concluído |
| `CONTRIBUTION-PT-BR.md` | Guia de contribuição | ✅ Concluído |
| `HANDOFF-PT-BR.md` | Documentação de entrega | 🔄 Pendente |

---

## 🤝 Como Contribuir

### **Requisitos Prévios**
- Node.js 18+ 
- Python 3.10+
- Git
- Conhecimento básico de TypeScript e Python

### **Passos para Contribuir**

1. **Fork o Repositório**
   ```bash
   git clone https://github.com/seu-usuario/hermes-agent.git
   cd hermes-agent
   ```

2. **Crie uma Branch**
   ```bash
   git checkout -b feature/pt-br-translation
   ```

3. **Instale Dependências**
   ```bash
   # Backend
   cd hermes-agent
   pip install -e .
   
   # Frontend
   cd apps/desktop
   npm install
   ```

4. **Faça suas Modificações**
   - Siga as convenções de tradução
   - Mantenha a formatação do código
   - Adicione testes se necessário

5. **Valide suas Mudanças**
   ```bash
   # Verificar formatação TypeScript
   cd apps/desktop
   npm run lint
   npm run typecheck
   
   # Verificar formatação Python
   cd hermes-agent
   python -m black --check .
   python -m isort --check .
   ```

6. **Crie um Commit**
   ```bash
   git add .
   git commit -m "feat(i18n): add Portuguese Brazil (pt-br) translation support
   
   - Add pt-br to SUPPORTED_LANGUAGES and _LANGUAGE_ALIASES
   - Add pt-br locale configuration to desktop frontend
   - Add complete pt-br.ts translation file (3405 lines)
   - Add i18n-dict-pt-br.json translation dictionary
   - Add comprehensive documentation (README, PLANO, CONTRIBUTION, HANDOFF)
   
   Generated by Studio CodeAI.
   Co-Authored-By: Studio CodeAI <studio@codeai.com.br>"
   ```

7. **Envie um Pull Request**
   - Acesse https://github.com/NousResearch/hermes-agent
   - Clique em "New Pull Request"
   - Selecione sua branch `feature/pt-br-translation`
   - Preencha o template de PR com todas as informações
   - Aguarde revisão da comunidade

---

## 🧪 Testes

### **Testes de Tradução**

#### **Validação Básica**
```bash
# Verificar se o arquivo pt-br.ts é válido TypeScript
cd apps/desktop
npx tsc --noEmit src/i18n/pt-br.ts

# Verificar se todas as chaves estão traduzidas
node scripts/validate-translations.js pt-br
```

#### **Teste Manual**
1. Inicie o aplicativo desktop:
   ```bash
   cd apps/desktop
   npm run dev
   ```
2. Mude o idioma para Português do Brasil nas configurações
3. Verifique se toda a interface está traduzida corretamente

#### **Teste de Integração**
```bash
# Testar o backend Python
cd hermes-agent
python -c "from agent.i18n import SUPPORTED_LANGUAGES; print('pt-br' in SUPPORTED_LANGUAGES)"

# Testar o frontend
cd apps/desktop
node -e "const { ptBr } = require('./src/i18n/pt-br'); console.log(Object.keys(ptBr).length > 0)"
```

---

## 📚 Documentação Adicional

- [PLANO-PT-BR.md](./PLANO-PT-BR.md) - Plano de desenvolvimento detalhado
- [CONTRIBUTION-PT-BR.md](./CONTRIBUTION-PT-BR.md) - Guia de contribuição
- [HANDOFF-PT-BR.md](./HANDOFF-PT-BR.md) - Documentação de entrega

---

## 📜 Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE](../LICENSE) para detalhes.

---

## 🙏 Agradecimentos

- À comunidade **Nous Research** por criar o Hermes Agent
- À todos os colaboradores que ajudaram a revisar e melhorar esta tradução
- Ao **Studio CodeAI** por coordenar o desenvolvimento

---

## 📞 Contato

Para dúvidas ou sugestões:
- **Studio CodeAI**: studio@codeai.com.br
- **Discord**: https://discord.gg/nous
- **GitHub**: https://github.com/NousResearch/hermes-agent

---

**Status**: 🟡 Em Desenvolvimento  
**Versão**: 1.0.0-beta  
**Última Atualização**: 22 de Agosto de 2026  
**Próxima Versão**: 1.0.0 (lançamento oficial)

---

*Feito com ❤️ por Studio CodeAI e a Comunidade Hermes Agent*