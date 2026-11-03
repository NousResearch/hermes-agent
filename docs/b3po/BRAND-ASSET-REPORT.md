# BRAND-ASSET-REPORT — inventário e proveniência

## Assets versionados em `brand/`

| Grupo | Arquivos | Formato | Tamanho maior | Origem |
|---|---|---|---|---|
| `avatars/` | 20 | PNG 512×512, transparente | 211 KB (`node.png`) | `DZ23-ENGINE-ATUALIZADO-DOWNLOAD.zip` (dono) |
| `logo/` | 3 | webp 1536/768 + png 768 | 383 KB | brand kit `b3po-codync/brand/exports` |
| `icons/` | 6 | PNG 32→512 | 184 KB | idem |
| `generated/` | 3 | json/py/ts | 1 KB | `scripts/brand_sync.py` |

**Arte original** (`logomarca-b3po.png`, 2,1 MB, SHA-256 `a89606e9…`) fica **fora do repositório**, no
brand kit — acima do limite de blob de 500 KiB. Referenciada por hash em `BRAND.md`.

## Paleta e contraste (medidos pelo gate)

| Par | Razão | Mínimo | Resultado |
|---|---|---|---|
| text (`#FCFDFE`) / bg (`#000105`) | 20,49:1 | 4,5 | PASSA |
| textMuted (`#D6DBE6`) / bg | 15,04:1 | 4,5 | PASSA |
| textDim (`#AFB5C4`) / bg | 10,16:1 | 3,0 | PASSA |
| accent (`#73C4EA`) / bg | 10,74:1 | 3,0 | PASSA |

`#00BFFF` é **reprovada por construção** como primitiva (gate `sem cor provisória como primitiva`).

## Gates implementados

`scripts/brand_check.py` — 14 checagens em 6 famílias: estrutura do `brand.json`; existência de assets;
catálogo de mascotes (20, 512×512, ids == arquivos); contraste AA; cor proibida; nível 3 proibido no
`rename-map`. Saída: `brand_check: 14 checagens, 0 falha(s)`.

`scripts/brand_sync.py --check` garante que `brand/generated/*` está atualizado (idempotente).

## Proveniência / licença

- Mascotes e logo: fornecidos pelo proprietário (DZ23-LTDA); declarado no `NOTICE` que **não** estão sob a
  licença MIT do upstream.
- Wordmark/ícones: gerados do logo pelo brand kit (`b3po-codync/tools/brand_kit.py`).
- Nenhum asset de terceiro foi importado.

## Pendências

- Falta `brand/tokens/` (tokens de design além de cor: espaçamento, tipografia, raios).
- Integração dos tokens no desktop/web/TUI ainda não feita (fases D–E).
