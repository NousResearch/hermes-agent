#!/usr/bin/env python3
"""Inicializa a estrutura de um vault Zettelkasten (método Luhmann).

Uso: python init_vault.py <caminho-do-vault>

Cria (sem sobrescrever nada que já exista):
  ZettelKasten/
    zettels/     - as notas (teses atômicas conectadas)
    Daily/       - diários e capturas fleeting
    templates/   - templates de nota (permanent, literature, fleeting, moc)
    arquivos/    - imagens e anexos
"""
import os, sys, shutil

TEMPLATES = {
"permanent_notes.md": """---
type: permanent
title:
date:
tags:
  - zettel/permanent
---

# {{title}}

## **Ideia Principal**
(A tese, clara e atômica. Teste: alguém poderia discordar deste título? Se não, ainda é tópico.)

## **Contexto**
(De onde veio: leitura, caso real, conversa. O que provocou a ideia.)

## **Expansão**
(Argumento nas suas palavras. Inclua a melhor objeção contra a própria tese.)

## **Conexões**
- [[Nota existente]] — relação: contradiz | completa | exemplifica | responde
""",

"literature_notes.md": """---
type: literature
title:
source:
author:
date:
tags:
  - zettel/literature
---

# {{title}}

## **Resumo**
(Breve. O resumo é o menos importante desta nota.)

## **Citações Relevantes**
- ""

## **Comentários Pessoais**
(A parte que vale: o que isso significa para as SUAS perguntas — não o que o autor disse.)

## **Conexões**
- [[Nota]] — relação
""",

"fleeting_notes.md": """---
type: fleeting
date:
tags:
  - zettel/fleeting
---

# Fleeting

- **Ideia**: (rápido, sem elaborar — o gancho)
- **Contexto**: (onde surgiu)
- **Processar em**: (qual nota permanente isso pode virar)
""",

"moc_template.md": """---
type: moc
title:
tags:
  - moc
description:
---

# MOC – {{title}}

(Mapa de entrada de um tema. Agrupe notas por subtema e termine com "Pontes a construir":
conexões que o material pede mas ainda não existem.)

## Pontes a construir
-
""",
}

STARTER = """---
type: permanent
title: Uma nota é uma tese, não um tópico
date:
tags:
  - zettel/permanent
---

# Uma nota é uma tese, não um tópico

## **Ideia Principal**
Notas que apenas descrevem um assunto não geram conhecimento novo; notas que afirmam algo contestável criam atrito — e o atrito entre afirmações é de onde a originalidade nasce.

## **Contexto**
Primeira nota deste vault, criada na inicialização. Niklas Luhmann escreveu 70 livros não acumulando conteúdo, mas forçando cada nota nova a entrar numa conversa com as existentes.

## **Expansão**
Teste prático para todo título: dá para discordar dele? "Observabilidade no sistema X" — não dá, é tópico. "Sem telemetria, todo incidente vira arqueologia" — dá, é tese. Objeção honesta: nem tudo precisa ser tese (notas de referência têm valor); o risco é o vault virar SÓ referência.

## **Conexões**
- (esta nota espera sua primeira conexão — crie sua próxima nota e decida a relação entre elas)
"""

def main():
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    base = os.path.join(sys.argv[1], "ZettelKasten")
    created, skipped = [], []
    for d in ("zettels", "Daily", "templates", "arquivos"):
        p = os.path.join(base, d)
        (skipped if os.path.exists(p) else created).append(p)
        os.makedirs(p, exist_ok=True)
    for name, content in TEMPLATES.items():
        p = os.path.join(base, "templates", name)
        if os.path.exists(p): skipped.append(p); continue
        open(p, "w", encoding="utf-8").write(content); created.append(p)
    p = os.path.join(base, "zettels", "Uma nota é uma tese, não um tópico.md")
    if not os.path.exists(p):
        open(p, "w", encoding="utf-8").write(STARTER); created.append(p)
    print(f"Vault: {base}")
    print(f"Criados: {len(created)}")
    for c in created: print(f"  + {c}")
    if skipped: print(f"Já existiam (preservados): {len(skipped)}")

if __name__ == "__main__":
    main()
