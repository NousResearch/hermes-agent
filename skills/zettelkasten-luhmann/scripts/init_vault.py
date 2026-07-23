#!/usr/bin/env python3
"""Initialize a Zettelkasten vault (Luhmann method).

Usage:
    python init_vault.py <vault_path> [--lang pt|en]

Creates (never overwrites existing files):
  ZettelKasten/
    zettels/     - atomic theses and connected notes
    Daily/       - daily journals and fleeting captures
    templates/   - note templates (permanent, literature, fleeting, moc)
    arquivos/    - images and attachments
"""
import argparse, os

TEMPLATES_PT = {
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

TEMPLATES_EN = {
"permanent_notes.md": """---
type: permanent
title:
date:
tags:
  - zettel/permanent
---

# {{title}}

## **Core Idea**
(A clear, atomic thesis. Test: could someone disagree with this title? If not, it's still a topic.)

## **Context**
(Where this came from: reading, real case, conversation. What triggered the idea.)

## **Expansion**
(Argument in your own words. Include the best objection against your own thesis.)

## **Connections**
- [[Existing note]] — relation: contradicts | completes | exemplifies | answers
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

## **Summary**
(Brief. The summary is the least important part of this note.)

## **Key Quotes**
- ""

## **Personal Comments**
(The valuable part: what this means for YOUR questions — not what the author said.)

## **Connections**
- [[Note]] — relation
""",

"fleeting_notes.md": """---
type: fleeting
date:
tags:
  - zettel/fleeting
---

# Fleeting

- **Idea**: (quick, don't elaborate — the hook)
- **Context**: (where it came up)
- **Process into**: (which permanent note this could become)
""",

"moc_template.md": """---
type: moc
title:
tags:
  - moc
description:
---

# MOC – {{title}}

(A topic entry point. Group notes by sub-topic and end with "Bridges to build":
connections the material suggests but doesn't yet have.)

## Bridges to build
-
""",
}

STARTER_PT = """---
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

STARTER_EN = """---
type: permanent
title: A note is a thesis, not a topic
date:
tags:
  - zettel/permanent
---

# A note is a thesis, not a topic

## **Core Idea**
Notes that merely describe a subject generate no new knowledge; notes that assert something contestable create friction — and friction between assertions is where originality is born.

## **Context**
First note of this vault, created on initialization. Niklas Luhmann wrote 70 books not by accumulating content, but by forcing each new note into a conversation with existing ones.

## **Expansion**
Practical test for every title: can someone disagree with it? "Observability in system X" — can't, it's a topic. "Without telemetry, every incident is archaeology" — can, it's a thesis. Honest objection: not everything needs to be a thesis (reference notes have value); the risk is the vault becoming reference-only.

## **Connections**
- (this note awaits its first connection — create your next note and decide the relationship between them)
"""

def main():
    parser = argparse.ArgumentParser(
        description="Initialize a Zettelkasten vault structure (Luhmann method)."
    )
    parser.add_argument("vault_path", help="Directory where the vault will be created")
    parser.add_argument("--lang", choices=["pt", "en"], default="pt",
                        help="Vault language (default: pt)")
    args = parser.parse_args()

    templates = TEMPLATES_PT if args.lang == "pt" else TEMPLATES_EN
    starter = STARTER_PT if args.lang == "pt" else STARTER_EN
    zettels_dir = "zettels"
    daily_dir = "Daily"
    starter_filename = "Uma nota é uma tese, não um tópico.md" if args.lang == "pt" else "A note is a thesis, not a topic.md"

    base = os.path.join(args.vault_path, "ZettelKasten")
    created, skipped = [], []
    for d in (zettels_dir, daily_dir, "templates", "arquivos"):
        p = os.path.join(base, d)
        (skipped if os.path.exists(p) else created).append(p)
        os.makedirs(p, exist_ok=True)

    for name, content in templates.items():
        p = os.path.join(base, "templates", name)
        if os.path.exists(p):
            skipped.append(p)
            continue
        open(p, "w", encoding="utf-8").write(content)
        created.append(p)

    p = os.path.join(base, zettels_dir, starter_filename)
    if not os.path.exists(p):
        open(p, "w", encoding="utf-8").write(starter)
        created.append(p)

    print(f"Vault language: {args.lang}")
    print(f"Vault: {base}")
    print(f"Created: {len(created)}")
    for c in created:
        print(f"  + {c}")
    if skipped:
        print(f"Already existed (preserved): {len(skipped)}")

if __name__ == "__main__":
    main()
