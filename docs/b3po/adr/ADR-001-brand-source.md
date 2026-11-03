# ADR-001 — fonte única de marca

**Status:** aceito · **Data:** 2026-11-02

## Contexto

A marca estava em propostas e em um brand kit externo (`b3po-codync/brand`), com hex de cor circulando em
texto (`#00BFFF`) que não corresponde à arte. Sem fonte única, cada cliente (desktop, web, CLI, TUI,
mobile) vira uma cópia da paleta.

## Decisão

`brand/brand.json` + `brand/palette.json` são a **fonte única**. `scripts/brand_sync.py` gera
`brand/generated/{brand.json, brand.py, brand.ts}`; clientes consomem o gerado, nunca o literal.

`scripts/brand_check.py` bloqueia: asset ausente, mascote fora de 512×512, contraste abaixo de AA, cor
provisória como primitiva e habilitação de rename nível 3.

## Consequências

- Trocar uma cor = editar um JSON e rodar o gerador.
- Marca hardcoded fora de `brand/` é detectável (próximo passo: gate de varredura de strings).
- A arte original (2,1 MB) fica fora do repo; o que versiona são exports otimizados.
- `#00BFFF` não é oficial; a paleta vem do pixel da arte.
