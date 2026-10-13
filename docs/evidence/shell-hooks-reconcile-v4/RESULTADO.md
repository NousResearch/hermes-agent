# V4 — candidato local; implantação e aceite operacional pendentes

## Entrega

- Base: `db72e58229639b80d68a53cdc1d081f6511457f5`.
- Branch: `fix/shell-hooks-reconcile-v4-resumed`.
- Worktree exclusiva: `/Users/farantes/atlas/wt/hermes-v4-reconcile-writer-20261013`.
- Origem preservada: `/Users/farantes/atlas/wt/hermes-v4-reconcile-20261012`.
- Os dois testes, `contract.md`, `red.log` e `red-confirmed.log` foram transportados byte a byte. Comparação final confirmou igualdade dos cinco arquivos; testes não foram afrouxados nem reescritos.
- Código de produção alterado apenas em `agent/shell_hooks.py`.

A reconciliação registra ownership por identidade de manager e home, com referências fracas ao manager. A chave inclui evento, matcher, comando, timeout e fail_closed. Consentimento é resolvido antes de publicar: erro deixa a geração anterior intacta; recusa de substituição retém a geração anterior inteira. A primeira carga mantém o comportamento de pular entradas não aprovadas.

A publicação usa o lock de discovery e o de registro, sem executar callbacks. Troca listas completas por evento no mesmo dicionário (preserva referências do ledger de plugins); não esvazia a política antes da troca. Uma chamada já iterando a lista antiga termina naquela geração. Callbacks estrangeiros são preservados por identidade, mantendo sua ordem; substitutos ocupam o primeiro antigo slot shell. Registro idêntico não altera listas; reload reconstrói callbacks que não estão mais no manager.

Não foram alterados subprocessos, avaliação de resultados, timeout, exit-2, fail_closed, roteamento de perfis ou consumidor `_make_agent` em produção.

## Provas reais, apenas dois arquivos focais

| Prova | Resultado | Recibo |
|---|---|---|
| Pré-requisito: inflight ajustado sobre base | RC 1, falha funcional `v3` bloqueia `v4`; não timeout espúrio | `inflight-red-resumed.log` |
| RED focal sobre base | RC 1; 52 passaram, 8 falharam, 2 Windows pulados | `red-focal-resumed.{log,json}` |
| Controle negativo: retirar correção somente do snapshot | RC 1; mesmas 8 falhas | `negative-control.{log,json}` |
| GREEN final, ambos os arquivos | RC 0; 60 passaram, 0 falhas, 2 Windows pulados | `green-final.{log,json}` |
| GREEN inflight isolado | RC 0; 1 passou, 0 falhas | `inflight-green.{log,json}` |

Falhas do controle negativo: command, empty, absent, timeout, fail_closed, new_manager, inflight e o consumidor `_make_agent`. O GREEN cobre também consentimento recusado/erro, matchers distintos, reload, perfil B intacto, plugin preservado/bloqueante e chamadas repetidas. No consumidor, V4 válido passa, versão incorreta e V4 incompleto continuam bloqueados. São fixtures locais, não a política Atlas instalada.

`red.log` e `red-confirmed.log` são históricos recuperados, não substituem os recibos novos. `green-focal` registra o primeiro GREEN antes do refinamento de identidade/idempotência; o candidato final é atestado por `green-final` e `inflight-green`, com hashes de produção e testes nos JSONs.

## Execução reproduzível e isolamento

Imagem existente reutilizada, sem rebuild: `hermes-v4-tests:20261012`, ID `sha256:72f85bad31968f2c90fd869e5e23bfcf2160e57dd8610515a1ff8921d2421975`. Receita recuperada em `test-image.Dockerfile`; não garante reconstrução bit a bit porque a base original usa tag mutável. As provas fixam o ID real da imagem.

```sh
python3 docs/evidence/shell-hooks-reconcile-v4/run_proof.py review-negative --base-source
python3 docs/evidence/shell-hooks-reconcile-v4/run_proof.py review-green
python3 docs/evidence/shell-hooks-reconcile-v4/run_proof.py review-inflight --inflight
```

O primeiro comando DEVE retornar 1. Os seguintes DEVEM retornar 0. O script arquiva a base Git e sobrepõe apenas os três arquivos necessários; monta somente esse tar read-only em Docker. `--network none`, 2 CPUs, 2 GiB, HOME isolado; sem checkout vivo, credenciais ou socket Docker no container. Usa `scripts/run_tests.sh`, processos separados por arquivo, 2 workers, sem retries e limite de 120 segundos por arquivo. Não houve suite completa, instalação de dependências ou inferência via API.

Incidente de embalagem: a primeira execução do helper herdou um TMPDIR do macOS diferente do scratch anunciado. O diretório temporário foi removido automaticamente; o helper foi corrigido para scratch explícito. Controle negativo e GREEN finais usam `/Users/farantes/.hermes/cache/scratch`. Não houve bloqueio de dependências; o pré-requisito Docker foi validado antes de editar produção.

Documentação oficial consultada: https://hermes-agent.nousresearch.com/docs/user-guide/features/hooks (consentimento, timeout/fail_closed, composição com plugins e isolamento de perfil). Fetch via curl após `web_extract` informar backend somente de busca.

## Segurança, revisão e freeze

- `git diff --check` antes do stage: RC 0. Check completo do staged diff: RC 2 por trailing whitespace nos logs brutos de pytest (inclusive histórico recuperado); logs preservados sem normalização. Check de código/documentos excluindo apenas `*.log`: RC 0.
- Gitleaks v8.24.0 sobre staged diff: RC 0, `no leaks found` (`gitleaks-staged.log`). Primeira tentativa falhou apenas ao escrever relatório no mount read-only do wrapper (`gitleaks.log`); a execução válida capturou stdout/stderr pelo host, sem alterar o wrapper. Nenhum JSON de achados foi emitido no scan sem vazamentos.
- Revisão independente: **pendente**; autor não se apresenta como revisor.
- Commit local é candidato revisável, não aprovação de merge nem freeze integrado. Registrar o SHA retornado pelo Git no encaminhamento ao Reviewer; qualquer alteração exige novo SHA e nova prova pertinente.
- Nenhum push, merge, instalação, alteração de config/hooks ativos, restart ou ação sobre PID 1407. A única escrita no repositório Git comum é a metadata necessária à nova worktree/branch e ao commit autorizado; arquivos do checkout servido não foram editados.
- **Manager vivo não foi corrigido nem validado.** GREEN em Docker não prova exclusividade V4 no backend já carregado. Implantação/aceite operacional requerem autorização e janela separadas, preservando trabalhos vivos.
- V4 de card não substitui SDDv5. Nenhum contorno de V3, criação/remoção de card, downgrade ou override foi usado.
