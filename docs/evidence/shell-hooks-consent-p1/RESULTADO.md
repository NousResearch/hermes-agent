# P1 — consentimento durante force reload

Candidato de origem: `b68823221a476f5424459881c07aa5ce8a03e516`.
Worktree: `/Users/farantes/atlas/wt/hermes-v4-consent-p1-20261013`.
Escopo: somente P1 do parecer independente em
`/Users/farantes/atlas/evidencia/v4-reconcile-review-20261013/PARECER.md`.

## Correção

`PluginManager.unload()` apagava callbacks vivos, mas preservava o ledger de
propriedade. Ao recusar o comando adicional, a reconciliação retornava
“retaining the previous generation” sem restaurar essa geração no manager.

`re_register_config_hooks()` agora restaura os callbacks anteriormente consentidos
do manager/home corrente antes de ler configuração e resolver novo consentimento.
Usa os mesmos locks de publicação, listas novas e preserva callbacks estrangeiros.
Não executa callbacks no lock, não aprova comandos novos e respeita safe mode.
A reconciliação existente continua responsável pela substituição aprovada ou remoção.

## Prova isolada

Reutilizado e adaptado o reproducer independente `test_review_edges.py`
(`test_force_reload_preserves_approved_policy_with_unapproved_addition`), sem
reinventar o diagnóstico. Acrescentados erro de consentimento, perfil B sob
multiplex e marcador de execução do comando novo. O consumidor `_make_agent`
também é exercitado antes/depois de `discover_and_load(force=True)` real.

| Recibo | Resultado |
|---|---|
| `red-candidate.json` | rc 1: 2 falhas esperadas (recusa/erro deixam manager vazio), 1 aprovado |
| `green-focal.json` | rc 0: 63 passaram, 2 Windows skipped; consumidor original |
| `negative-candidate.json` | rc 1: produção original, mesmos testes da regressão, mesmas 2 falhas |
| `negative-consumer.json` | rc 1: produção original, testes finais idênticos; 60 passaram, 3 falharam, 2 skipped |
| `green-final.json` | rc 0: produção corrigida; 63 passaram, 0 falharam, 2 Windows skipped |

O negativo final também falha no consumidor: payload V3 recebeu `None` em vez de
`v4-denied`. O GREEN mantém o bloqueio V4 e o perfil B, aceita o payload válido,
não executa o comando não consentido e permite o comando quando há aprovação.
Os recibos contêm hashes dos arquivos e dos logs; o negativo substitui somente
`agent/shell_hooks.py` pelo candidato. Logs anteriores não foram editados.

Runner reproduzível: `python3 docs/evidence/shell-hooks-consent-p1/run_proof.py LABEL`
(`--base-source` para negativo, `--regression` para somente o reproducer).
Imagem Docker existente por ID imutável, rede desativada, snapshot tar read-only,
sem mount de checkout/runtime/home/credenciais/socket. Runner canônico
`scripts/run_tests.sh`, apenas três arquivos focais, dois workers, zero retries.
Avisos de pytest-asyncio e indisponibilidade de providers da fixture offline não
foram corrigidos: fora do escopo. Nenhuma chamada de modelo foi executada nos testes.

## Limites

Implementação e evidência, **não aprovação independente**. Rel/QA deve emitir novo
parecer. Runtime ativo, configuração, hooks instalados e V3 operacional não foram
alterados; nenhum restart, instalação, push ou merge. Prova Docker não demonstra
implantação no processo operacional. Windows e suíte ampla não executados.
