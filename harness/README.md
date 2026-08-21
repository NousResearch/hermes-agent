# Harness — Hermes Agent

Documentação e especificação do harness agêntico **local** para operar este checkout.

> **Não upstream.** Não incluir em PRs para NousResearch sem revisão explícita.

## Estrutura

| Pasta | Conteúdo |
|---|---|
| [pesquisa/](pesquisa/) | Mapa, deps, processos, patches locais |
| [planos/](planos/) | Spec, catálogo skills, fluxos |
| [registros/](registros/) | Decisões, status |
| `guias/` | (reservado) regras locais |
| `rascunho/` | Experimentos |
| `entrega/` | Artefatos validados |

## Decisão D-001

Harness vive **dentro deste repo** (`hermes-agent/harness/`).

## Início rápido

1. Ler [registros/STATUS.md](registros/STATUS.md)
2. Patches locais: [pesquisa/PATCHES_LOCAIS.md](pesquisa/PATCHES_LOCAIS.md)
3. Fluxo sync: [planos/FLUXOS_TRABALHO.md](planos/FLUXOS_TRABALHO.md)

## Pendências

- D-007: `.git/info/exclude` para `harness/` (recomendado)
- Próximo: pluginizar patches Hermes One (D-004)

## Boot / retomada

Após reiniciar sessão, ler **[registros/RETOMADA_SESSAO.md](registros/RETOMADA_SESSAO.md)**.

```powershell
python "$env:LOCALAPPDATA\hermes\hermes-agent\harness\scripts\hermes_patch_guard.py"
```
