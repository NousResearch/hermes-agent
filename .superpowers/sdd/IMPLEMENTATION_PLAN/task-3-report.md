# Task 3 — continuidade, compactação e contabilidade

## Resultado

- Adicionados quatro contratos com `SessionDB(tmp_path / "state.db")`; não houve acesso a DB real, rede, configuração ou runtime vivo.
- **A — prompt após troca de modelo:** passou sem alteração de produção. O `NULL` imediato é a invalidação intencional; a primeira retomada reconstrói e persiste o blob hasheado, e a segunda restaura os mesmos bytes sem novo warning.
- **C — contabilidade main/aux:** passou sem alteração de produção. `sessions` reconcilia apenas com `session_model_usage.task=''`; o total de todas as linhas inclui o uso auxiliar.
- **D — escrita gateway absoluta:** passou sem alteração de produção. Uma atualização absoluta altera somente `sessions`; ela não pode ser atribuída retroativamente às linhas por modelo. Uso auxiliar continua fora do resumo da sessão.
- **B — dupla compactação idêntica:** passou. A projeção ativa é literalmente o par compacto, e a projeção de display contém, em ordem, `old question`, `old answer` e uma cópia do par compacto. Linhas físicas repetidas só existem no histórico arquivado, que é a semântica documentada e foi preservada.

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

4 passed (após o fix round 1, ver abaixo)
```

Regressão focada documentada foi rerodada pelo controller Hermes em `.venv`: `119 passed in 27.97s`.

`git diff --check` passou.

## Fix round 1/5 — completude da projeção de display

- O check tautológico de IDs físicos foi removido: IDs de uma tabela são únicos por construção e não provavam o contrato público.
- A unicidade isolada de display foi substituída por igualdade literal com o transcript esperado após cada compactação. Assim, tanto um display vazio quanto um display truncado falham; os dois turnos históricos e cada membro do par compacto devem aparecer exatamente uma vez e na ordem correta.
- RED demonstrado sem alterar produção: ao truncar deliberadamente o display em memória, a antiga verificação de unicidade passava, enquanto a nova igualdade literal falhou como esperado.
- A regressão focal e o controller de 119 testes foram rerodados após o ajuste.
