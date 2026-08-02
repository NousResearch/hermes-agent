# Relatório de validação e causa raiz — Issue #76870

## Identificação

- Issue: [#76870 — Model switch mid-session triggers history_version mismatch](https://github.com/NousResearch/hermes-agent/issues/76870)
- Estado consultado em 2026-08-02: aberta, sem responsável, sem comentários.
- Labels: `type/bug`, `comp/tui`, `P1`, `sweeper:risk-session-state`, `area/sessions`.
- Superfície afetada: `tui_gateway`, usada por TUI/Desktop/Webapp via JSON-RPC.
- Branch local analisada: `fix-76574-secret-scope-residuals`.

## Veredito

**Issue verídica. Implementação necessária e realizada.**

Foi reproduzido deterministicamente o defeito causal: uma troca escolhida enquanto um turno está rodando fica em `pending_model_switch`; no início do turno seguinte, o gateway capturava `history` e `history_version` antes de aplicar essa troca. A aplicação da troca adiciona o marcador de modelo ao histórico e incrementa a versão. Ao terminar, o próprio turno é classificado incorretamente como concorrente/stale e seu resultado não é gravado no histórico da sessão.

A evidência local confirma o primeiro turno após a troca. O relato de 75+ mensagens vazias no SQLite é evidência operacional fornecida pelo autor; esse volume completo não foi reproduzido localmente sem o banco e ambiente Docker citados. A correção elimina o desync inicial que inicia a cadeia, sem relaxar o guard defensivo.

## Evidência da issue

O autor relata:

- Docker com s6-overlay, perfil `default`.
- Troca `minimax-m3` → `deepseek-v4-pro`, provider `ollama-cloud`.
- Sessão `20260802_151208_b9d649`.
- Log: `history_version mismatch (expected=1 current=2)`.
- Respostas visíveis pelo stream WebSocket, mas histórico/DB inutilizável para busca, resume, rewind, undo e cron `context_from`.
- Workaround: executar `/new` antes da troca.

## Causa raiz confirmada

Sequência defeituosa anterior:

1. `config.set model` durante turno ativo não troca cliente em uso; salva escolha em `session["pending_model_switch"]`. Isso é correto e evita race entre modelo/client/base URL.
2. `_run_prompt_submit()` do próximo turno copiava `session["history"]` e `history_version` no dispatcher, antes de iniciar worker.
3. Worker chamava `_apply_pending_model_switch()`.
4. `_apply_model_switch()` chamava `_append_model_switch_marker()`.
5. Marcador era anexado e `history_version` incrementava.
6. `agent.run_conversation()` recebia snapshot antigo, sem marcador.
7. Commit final comparava versão antiga com nova, concluía falsamente que histórico mudou externamente e recusava `result["messages"]`.

Defeito nasceu no commit `f27d45e2880b46a2239b184ecc8ab88ecfd2843d` (`feat(tui): ... switch mid-turn`, 2026-07-30). Intenção do commit era válida; erro foi somente ordenação entre aplicação adiada e snapshot.

## Solução implementada

Arquivo: `tui_gateway/server.py`.

Alteração: mover captura de `history` e `history_version` para dentro do worker, imediatamente após `_apply_pending_model_switch()` e `_sync_agent_model_with_config()`, ainda protegida por `history_lock` e antes de qualquer chamada ao modelo.

Nova ordem:

1. Aplicar mutações preparatórias pertencentes ao próprio turno.
2. Capturar histórico e versão consistentes.
3. Executar agente com esse snapshot.
4. Manter comparação final de versão.

Assim, mutação interna esperada entra no baseline. Mutações externas reais ocorridas após snapshot continuam detectadas e rejeitadas pelo guard existente.

### Por que não aceitar qualquer mismatch

A alternativa sugerida pela issue de aceitar resposta mesmo com versão alterada seria insegura: undo, compressão, retry, rollback ou outra escrita concorrente podem invalidar contexto usado para gerar resposta. Sobrescrever histórico nesses casos ressuscitaria mensagens removidas ou perderia estado novo. Guard permanece intacto.

### Por que não remover marcador ou incremento

Marcador informa novo runtime ao modelo e precisa sobreviver em histórico. Incremento representa mutação real. Removê-los esconderia mudança e enfraqueceria invariantes. Problema era snapshot cedo demais.

### Escopo

- 7 linhas produtivas reposicionadas/adicionadas; nenhuma API nova.
- Nenhuma mudança de schema, tool, config ou prompt global.
- Sem alteração em persistência SQLite.
- Sem atualização de Graphify, conforme instrução do projeto.

## Teste de regressão

Adicionado `test_prompt_submit_snapshots_history_after_pending_model_switch` em `tests/test_tui_gateway_server.py`.

O teste simula contrato real:

- sessão começa com versão 0;
- troca adiada adiciona marcador e eleva versão para 1 no início do turno;
- agente deve receber marcador no `conversation_history`;
- resposta deve terminar no histórico;
- `message.complete` não deve carregar warning de mismatch.

### Prova antes da correção

Resultado esperado e observado:

- teste falhou;
- agente recebeu `conversation_history=[]`, não o marcador;
- stderr: `history_version mismatch (expected=0 current=1)`;
- histórico recusou resposta do agente.

### Prova depois da correção

- `pytest ... -k "history_version or pending_model_switch"`: **5 passed**.
- `pytest ... -k "model_switch"`: **7 passed**.
- `ruff check tui_gateway/server.py tests/test_tui_gateway_server.py`: **passou**.
- `git diff --check`: **passou**.

Suite completa de `tests/test_tui_gateway_server.py`: **501 passed, 2 failed** na primeira execução. Falhas foram testes concorrentes não relacionados (`test_write_json_serializes_concurrent_writes` e `test_run_prompt_submit_requeues_all_unstarted_notifications_with_real_threading`). Ambos passaram juntos ao rerodar isoladamente (**2 passed**), classificando-os como flakes/interferência de suite, não regressão desta mudança.

## Invariantes preservados

- Prompt caching: nenhuma mensagem passada é reescrita durante chamada; snapshot continua estável para o turno.
- Alternância/histórico: marcador existente continua sendo mensagem `user`, conforme compatibilidade com providers estritos.
- Concorrência: `history_lock` cobre captura; guard de versão continua protegendo mutações posteriores.
- Escopo por sessão: nenhuma variável global ou ambiente foi alterada.
- Persistência: agente e gateway passam a compartilhar o mesmo baseline que contém marcador.

## Risco residual

Baixo. Mudança só desloca momento do snapshot alguns milissegundos para depois da preparação do turno. Se uma mutação externa legítima ocorrer depois da nova captura, mismatch ainda dispara. Se ocorrer antes, ela é incluída no contexto enviado ao agente, comportamento correto porque chamada ainda não começou.

A alegação de que todo turno posterior necessariamente falha não decorre isoladamente do contador: após consumir `pending_model_switch`, nova versão deveria estabilizar. Isso pode refletir efeito secundário do desync inicial ou particularidade do ambiente/banco do relator. Correção cobre causa confirmada sem alegar reprodução do banco original.

## Arquivos alterados

- `tui_gateway/server.py`: ordem correta do snapshot.
- `tests/test_tui_gateway_server.py`: regressão causal.
- `ISSUE_76870_RELATORIO_CAUSA_RAIZ.md`: este relatório.

## Conclusão

Não cancelar: bug existe no código atual e foi reproduzido. Correção mínima age na fronteira exata: operações preparatórias primeiro, snapshot depois. Não adiciona abstração, não enfraquece proteção anti-stale e preserva intenção da troca adiada.
