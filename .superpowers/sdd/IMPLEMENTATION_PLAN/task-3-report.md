# Task 3 — continuidade, compactação e contabilidade

## Resultado

- Adicionados quatro contratos com `SessionDB(tmp_path / "state.db")`; não houve acesso a DB real, rede, configuração ou runtime vivo.
- **A — prompt após troca de modelo:** passou sem alteração de produção. O `NULL` imediato é a invalidação intencional; a primeira retomada reconstrói e persiste o blob hasheado, e a segunda restaura os mesmos bytes sem novo warning.
- **C — contabilidade main/aux:** passou sem alteração de produção. `sessions` reconcilia apenas com `session_model_usage.task=''`; o total de todas as linhas inclui o uso auxiliar.
- **D — escrita gateway absoluta:** passou sem alteração de produção. Uma atualização absoluta altera somente `sessions`; ela não pode ser atribuída retroativamente às linhas por modelo. Uso auxiliar continua fora do resumo da sessão.
- **B — dupla compactação idêntica:** passou. Não houve duplicação na projeção ativa/modelo nem na projeção de display. Linhas físicas repetidas só existem no histórico arquivado, que é a semântica documentada e foi preservada.

Classificação A/C/D/B: `MUDANCA_INTENCIONAL` / mal-entendido diagnóstico; nenhum patch especulativo de produção foi aplicado.

## Arquivos alterados

- `tests/agent/test_system_prompt_restore.py`
- `tests/hermes_state/test_aux_usage_accounting.py`
- `tests/hermes_state/test_display_projection_parity.py`

A alteração pré-existente em `.superpowers/sdd/IMPLEMENTATION_PLAN/progress.md` foi preservada e não entra no commit desta task.

## Verificação

Contrato novo focado:

```text
pytest -q \
  tests/agent/test_system_prompt_restore.py::TestStoredPromptReuse::test_model_switch_null_is_transitional_and_next_turn_restores \
  tests/hermes_state/test_aux_usage_accounting.py::TestRecordAuxiliaryUsage::test_session_summary_equals_main_task_rows_while_all_rows_include_aux \
  tests/hermes_state/test_aux_usage_accounting.py::TestRecordAuxiliaryUsage::test_absolute_gateway_update_changes_session_only_not_per_model_rows \
  tests/hermes_state/test_display_projection_parity.py::TestDisplayProjectionParity::test_repeated_identical_compaction_does_not_duplicate_live_or_display_projection

4 passed in 0.60s
```

Regressão focada documentada foi executada integralmente: `110 passed, 9 failed in 25.42s`.

As 9 falhas não são causadas pelo delta desta task:

- 8 testes importam `run_agent`, que chega a `tools.environments.file_sync` e falha com `ModuleNotFoundError: No module named 'psutil'`.
- O nono (`test_workspace_snapshot_reprobes_when_cwd_changes`) também é afetado pela mesma dependência ausente: `bounded_git_probe` importa `hermes_cli.local_runtime.processes`, cujo import de `psutil` falha e faz a sonda de `git log` retornar vazio; por isso o snapshot não contém o commit inicial esperado.
- `psutil==7.2.2` já está declarado em `pyproject.toml`; a causa é o ambiente de testes incompleto. Não foi instalado nem contornado, pois a task não autoriza mudar dependências/ambiente.

`git diff --check` passou.

## Concern pendente

A suíte focada inteira só ficará verde quando o runner tiver a dependência declarada `psutil` instalada. Os quatro contratos da Task 3 passam no ambiente atual e não há indício de regressão de continuidade, display ou contabilidade.
