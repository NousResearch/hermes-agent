# UPSTREAM-REPORT — política F2-A e sincronização

## Estado

| Item | Valor |
|---|---|
| `main` | `f4cf411f` — espelho do upstream, **0 commits DZ23** |
| `upstream/main` | `6ab0de91` — **26 commits à frente** |
| Convenção | `main` só recebe fast-forward; trabalho em `b3po-brand` |
| Verificação | `git rev-list --count upstream/main..main` → `0` |

## Procedimento de sincronização (documentado)

```bash
git fetch upstream
git checkout main
git merge --ff-only upstream/main      # falha se main tiver commit local = alarme
git push origin main                    # fast-forward
git checkout b3po-brand
git rebase main                         # resolve conflitos na branch de produto
```

## Papel do `rename-map.json`

`brand/rename-map.json` (nível 2) existe como **mapa de compatibilidade**, não como autorização de rename.
O GOAL proíbe rename de nível 3 (módulos/imports) — o mapa declara isso explicitamente em `forbidden`.

Quando o nível 2 for aplicado, o mesmo mapa deverá traduzir o patch do upstream antes do merge
(`scripts/upstream_sync.py`, ainda não implementado — ver bloqueio abaixo).

## Gatilho de revisão (aprovado)

Se duas sincronizações consecutivas passarem de **30 min** de resolução manual ou **3 conflitos
relevantes**, o conselho revisa a estratégia sem abandonar o upstream.

## Situação atual

- Nenhuma sincronização foi executada ainda; o fork está 26 commits atrás.
- A sincronização **não bloqueia** o marco F5 (o código atual funciona para o que o marco exige), mas
  deve ocorrer **antes** do rename de nível 2.
- `scripts/upstream_sync.py` **não existe**: será necessário apenas quando o nível 2 começar. Registrar
  como pendência, não como bloqueio do MVP.
