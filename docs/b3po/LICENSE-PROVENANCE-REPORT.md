# LICENSE-PROVENANCE-REPORT

## 1. Código

| Item | Valor |
|---|---|
| Licença do upstream | **MIT** (`LICENSE`, "Copyright (c) 2025 Nous Research") |
| Fork | DZ23-LTDA/B-3PO |
| Obrigação MIT | preservar copyright + permissão; `NOTICE` criado na Fase B |
| Modificações | Copyright (c) 2026 DZ23-LTDA, declarado no `NOTICE` |

## 2. Marca e assets

| Asset | Origem | Licença declarada |
|---|---|---|
| Logo (`logomarca-b3po.png`) | proprietário (DZ23) | não coberto pelo MIT do upstream; declarado no `NOTICE` |
| 20 mascotes (PNG 512²) | pacote `DZ23-ENGINE-ATUALIZADO-DOWNLOAD.zip`, entregue pelo dono | idem |
| Wordmarks/ícones | gerados do logo pelo brand kit DZ23 | idem |

Nenhum asset de terceiro (fonte, ícone, ilustração) foi importado para `brand/`.

## 3. Dependências

| Ecossistema | Quantidade | Auditoria de licença |
|---|---|---|
| npm (monorepo) | 1222 pacotes instalados | **não executada** (pendente) |
| Python | `uv.lock` pinado | **não executada** (pendente) |

Risco: dependência com licença incompatível (GPL/AGPL) entrando em distribuição. Mitigação prevista na
Fase I do GOAL: gerar SBOM e revisar licenças antes de release.

## 4. Conformidade mínima verificada

- `LICENSE` intacto (não modificado). ✔
- `NOTICE` presente e versionado. ✔
- Atribuição ao upstream visível no README dos forks? **parcial** — o README da `main` é o do upstream; o
  README de produto ainda não foi escrito (Fase J).
- Marca não reivindicada como MIT. ✔ (texto explícito no `NOTICE`)

## 5. Pendências

1. SBOM (`npm sbom`, `uv export`) e revisão de licenças.
2. README de produto com seção de licença e créditos.
3. Verificar se `brand/avatars` tem cadeia de custódia documentada (quem gerou, quando).
