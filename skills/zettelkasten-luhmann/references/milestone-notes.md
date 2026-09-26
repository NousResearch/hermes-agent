# Nota de Marco Pessoal / Milestone

Formato alternativo para quando o usuário registra um feito pessoal relevante — o fluxo Luhmann padrão (tese contestável → arquivamento forçado → conexão improvável) não se aplica.

## Quando usar

- PR submetido/merged em projeto notável (Anthropic, Microsoft, CNCF, etc.)
- Skill pública publicada
- Certificação conquistada
- Palestra/apresentação aceita
- Open-source repo atingiu milestone
- Primeira vez de algo significativo

Não usar para: updates de rotina, tarefas concluídas, bug fixes merged. Esses vão em daily notes ou zettels técnicos.

## Estrutura

```
---
type: permanent
title: "Nome do Marco"
date: YYYY-MM-DD
tags:
  - carreira
  - reflexao
  - <domain tags>      # ex: ferramentas/claude, aprendizado/zettelkasten
description: "One-liner do marco."
---

# Nome do Marco

## **Ideia Principal**
Parágrafo único: o que aconteceu, data, link para PR/repo/talk.

## **Contexto**
- **Item**: bullets com detalhes (repo, link, data, processo)

## **Seção Específica do Conteúdo**
O que foi construído, decisões de design, números (benchmarks, estrelas, revisores).

## **Por que isso importa**
Bullets articulando a significância. Conectar à trajetória de carreira.

## **Conexões**
- [[Nota Relacionada]] — descrição da relação
```

## Exemplo real

Veja `zettels/Primeira Skill Publica — PR para Anthropic.md` no vault — o milestone que originou este formato.

## Regras

1. **Sem tese contestável** — o marco é um fato, não uma afirmação para debate.
2. **Sem arquivamento forçado** — o valor está em conectar o feito à trajetória, não em escolher uma thread.
3. **Seção "Por que isso importa" obrigatória** — é onde o registro vira reflexão. Sem ela vira log.
4. **Conexões com notas conceituais** — o milestone linka para trás (notas que o viabilizaram), não para frente.
