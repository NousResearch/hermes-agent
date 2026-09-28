---
name: zettelkasten-luhmann
description: Notas Zettelkasten com o método Luhmann.
version: 2.0.0
author: "Danilo Neto"
platforms: [linux, macos]
tags:
  - zettelkasten
  - luhmann
  - obsidian
  - vault
related_skills:
  - obsidian
---

# Zettelkasten Luhmann Skill

Transforma notas descritivas em teses contestáveis com o método de Niklas
Luhmann. Força o arquivamento consciente — cada nova nota entra *atrás de
uma nota específica*, e o atrito entre afirmações gera originalidade.

Não arquiva automaticamente. Não cria notas órfãs. A decisão de onde conectar
é sempre do usuário.

## When to Use

- Usuário quer criar uma nota, zettel, ou nota permanente
- Processar fleeting note ou daily note em conhecimento reutilizável
- Registrar ideia, insight ou aprendizado no vault Obsidian
- "Quero anotar isso", "vira uma nota", "arquivar essa ideia"
- Montar ou configurar um Zettelkasten do zero

## Prerequisites

- Vault Obsidian em `~/obsidian/ZettelKasten/` com pastas `zettels/`,
  `Daily/`, `templates/`
- `init_vault.py` script na skill (cria estrutura se não existir)
- Notas seguem frontmatter YAML: `type` (permanent | literature | moc | fleeting),
  `date`, `tags` (taxonomia hierárquica)

## How to Run

```bash
# Inicializar vault novo (detecta idioma do usuário automaticamente)
python scripts/init_vault.py ~/obsidian --lang pt

# Vault em inglês
python scripts/init_vault.py ~/obsidian --lang en
```

O script é idempotente — nunca sobrescreve o que já existe. O agente detecta
o idioma da conversa e passa o `--lang` correspondente.

## Quick Reference

| Etapa | O que | Ferramenta |
|-------|-------|------------|
| 0 | Verificar estrutura do vault | `search_files` no `ZettelKasten/` |
| 1 | Transformar ideia em tese | Debate com o usuário |
| 2 | Escrever corpo como argumento | `read_file` + `write_file` |
| 3 | Buscar threads candidatas | `search_files` no vault |
| 4 | Efetivar conexão + backlink | `write_file` + `patch` |
| 5 | Validar no chat | Mostrar antes de sincronizar |
| 6 | Vizinho improvável | Sugerir conexão de outro cluster |
| 7 | Git (opt-in) | `terminal` com `git status` |

## Procedure

### Etapa 0 — Verificar estrutura (sempre)

Antes de criar ou arquivar, confirme que o vault existe:

```
search_files(target="files", pattern="*.md", path="~/obsidian/ZettelKasten")
```

Sem as pastas `zettels/` e `templates/`, o arquivamento não tem onde
acontecer. Se faltar, ofereça rodar `init_vault.py`.

### Etapa 1 — Da ideia à tese

Quando o usuário trouxer uma ideia, rascunho ou fleeting note:

- Se o título **descreve** um assunto, provoque: *"Qual é a afirmação?
  O que alguém poderia contestar aqui?"*
- Proponha 2-3 títulos-tese e deixe ele escolher ou reformular
- Teste do título: se não dá para discordar dele, ainda é tópico

**Provocação disfarçada:** Neto às vezes dropa uma afirmação provocativa
sem sinal de que é um pedido de nota. Sinais: frase absoluta ou
simplificada demais, pede debate em vez de resposta. Engaje a provocação,
depois ofereça a nota — não crie nota prematura (nota rasa).

### Etapa 2 — Corpo como argumento

Ajude a escrever o corpo como resposta a uma pergunta. Estrutura:

- **Permanente:** Ideia Principal, Contexto, Expansão, Conexões
- **Literature:** Resumo (breve), Citações, Comentários Pessoais (o que
  importa), Conexões — a seção de comentários pessoais é a parte que vale

### Etapa 3 — Arquivamento forçado (coração da skill)

Antes de salvar, analise o vault e apresente 2 a 4 threads candidatas:
notas existentes com as quais a nova nota conversa.

**Busca no vault:** Use `search_files` do Hermes:

```python
search_files(target="content", pattern="gestor|raciocínio|simulação",
             path="~/obsidian/ZettelKasten/zettels", file_glob="*.md")
```

`search_files` é ripgrep-backed e lida corretamente com UTF-8 e acentos.

Para cada candidata, nomeie a **relação**:

- **contradiz** — a nova nota tensiona ou refuta a existente
- **completa** — adiciona evidência ou desdobramento
- **exemplifica** — é caso concreto de um princípio existente
- **responde** — resolve pergunta aberta em outra nota

**Pergunte ao usuário qual thread a nota entra.** Nunca escolha por ele
— a decisão é o exercício mental.

#### Sub-etapa — MOC relevante

Após a escolha da thread, identifique qual MOC cobre o tema. Varra MOCs
com `search_files` por tag ou prefixo de tag. Pergunte se quer conectar
ao MOC — se sim, anote para atualizar no passo seguinte.

### Etapa 4 — Efetivar a conexão

1. Crie a nota em `zettels/` com frontmatter completo e `[[link]]` para
   a thread escolhida na seção Conexões
2. Adicione o link reverso na nota escolhida, citando a relação:
   `- [[Nota Nova]] — contradiz a tese central`
3. Se o MOC foi atualizado, adicione a linha no MOC
4. Nunca crie nota órfã (mínimo 1 conexão real) nem notas-stub vazias

### Etapa 5 — Validação no chat

**SEMPRE** exiba o conteúdo da nota e o backlink criado no chat antes de
sincronizar. Espere confirmação explícita do usuário.

### Etapa 6 — Vizinho improvável

Feche com um vizinho improvável: uma nota de um cluster *diferente*
(outro domínio de tag) que poderia se conectar de forma não óbvia. Uma
frase explicando o cruzamento possível. Opcional — mas é o que faz o
arquivo "responder de volta".

### Etapa 7 — Git (opt-in, com segurança)

Após o usuário aprovar a nota e backlinks:

```bash
cd ~/obsidian/ZettelKasten
git status
```

Mostre o status ao usuário. **Pergunte se quer commitar** — só então:

```bash
git add zettels/<nova-nota>.md zettels/<nota-existente>.md
git commit -m "Nota: <título> — <relação> a <thread>"
git pull --rebase && git push
```

Regras:
- **Não use `git add -A`** — stage apenas os arquivos que o workflow
  tocou (a nova nota e as que receberam backlinks)
- **Sempre `git pull --rebase && git push`** — outro dispositivo pode ter
  feito push enquanto você trabalhava

## Idiomas

A skill funciona em qualquer idioma:

1. **Converse no idioma do usuário** — perguntas, opções e explicações
2. **Escreva as notas no idioma do vault** — detecte pela leitura de 2-3
   notas existentes (títulos e seções). Vault é uma conversa de décadas;
   misturar idiomas quebra busca e conexões
3. **Vault novo:** o `init_vault.py` aceita `--lang pt` ou `--lang en`

## Casos especiais

### Nota de marco pessoal

Quando o tema é um **feito pessoal relevante** (PR submetido, skill
publicada, certificação, palestra), **não aplique o fluxo Luhmann padrão**
— o registro de marco não é tese contestável. Use o formato em
`references/milestone-notes.md`.

### Provocação diária (cron)

O sistema pode provocar o usuário com notas antigas usando spaced
repetition. Script em `~/.hermes/scripts/vault-provoke.py` com `--json`
para saída estruturada. Detalhes em `references/daily-vault-provocation.md`.

## Pitfalls

- **Resumir em vez de afirmar** — especialmente em literature notes.
  Comentários pessoais > resumo do autor.
- **Arquivar automaticamente** "para agilizar" — destrói o propósito da
  skill. A decisão é do usuário.
- **Criar nota antes da Etapa 3** — análise de candidatas vem ANTES do
  arquivo existir.
- **Sugerir só candidatas semelhantes** — procure a nota que a nova tese
  incomoda (contradição > semelhança).
- **Notas longas cobrindo várias teses** — sugira dividir em notas
  atômicas conectadas. Dividir proativamente é mais barato que dividir
  retroativamente.
- **Criar notas sem verificar MOC existente** — 3+ notas no mesmo tema
  sem MOC? Ofereça criar um.
- **`git add -A`** — pode commitar mudanças não relacionadas no vault.
  Stage apenas os arquivos tocados pelo workflow.
- **Push sem `--rebase`** — vault editado de múltiplos dispositivos.
  Sempre `git pull --rebase && git push`.

## Verification

- Após criar nota, verifique que o wikilink na nova nota resolve
- Verifique que o backlink na nota existente foi adicionado
- Verifique que nenhuma nota órfã foi criada
- Se MOC foi atualizado, verifique a entrada
- Execute `git status` para confirmar que só os arquivos esperados
  foram modificados
