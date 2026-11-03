# ADR-002 — política de upstream e branches

**Status:** aceito (F2-A) · **Data:** 2026-11-02

## Contexto

O fork não tinha commits próprios e estava 26 commits atrás do upstream. Duas opções: `main` como linha
estável de produto (com merges manuais) ou `main` como espelho limpo (só fast-forward).

## Decisão

`main` é **espelho limpo**: recebe apenas fast-forward de `NousResearch/hermes-agent`. Todo trabalho DZ23
(releases, tags, rebranding, schema, orquestração) vive em `b3po-brand` e branches derivadas.

Sincronização documentada: `fetch upstream` → `merge --ff-only` em `main` → `rebase` de `b3po-brand`.

## Consequências

- Sincronizar é mecânico e verificável (`upstream/main..main` deve ser sempre 0).
- A vitrine pública do GitHub mostra o upstream enquanto `main` for a default branch — consequência
  aceita e registrada na ata.
- Renomear em `main` está fora de questão; qualquer commit ali quebra a propriedade.
- Gatilho de revisão: duas sincronizações acima de 30 min ou 3 conflitos.
