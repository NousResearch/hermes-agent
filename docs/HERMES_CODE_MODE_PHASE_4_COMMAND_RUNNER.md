# Hermes Code Mode — Fase 4: CommandRunner Seguro

## 1. Resumo

A Fase 4 introduz o `CommandRunnerService`, que permite que o Hermes Agent execute comandos num workspace (vinculado a uma CodeSession) de maneira segura, rastreável e controlada, com a captura de saída (stdout/stderr) e persistência total no SQLite. A execução utiliza subprocessos simples sem shell injection e define políticas de classificação para barrar comandos perigosos.

## 2. Arquivos alterados

- **`hermes_cli/code/command_runner.py` (NOVO)**: Implementa `CommandRunnerService` com a lógica de classificação, execução (`run_command_sync`), validação de diretórios (CWD) e timeout.
- **`hermes_state.py`**: 
  - Schema version bump para `v11`.
  - Nova tabela `code_commands` e seus índices associados.
  - Implementa a classe `CodeCommandDB` para persistir e buscar os comandos e interagir de forma atômica com o banco SQLite.
- **`hermes_cli/web_server.py`**: Foram adicionados 4 novos endpoints REST para suportar a UI/CLI interagindo com os comandos e a emissão de eventos via WebSocket:
  - `GET /api/code/sessions/{code_session_id}/commands`
  - `POST /api/code/sessions/{code_session_id}/commands/run`
  - `GET /api/code/commands/{command_id}`
  - `POST /api/code/commands/{command_id}/cancel`
- **`tests/hermes_cli/test_command_runner.py` (NOVO)**: Testes completos (unitários e de integração com a API) para o serviço e endpoints.
- **`tests/test_hermes_state.py`, `tests/test_artifacts.py`, `tests/hermes_cli/test_code_session_service.py`, `tests/hermes_cli/test_workspace_service.py`**: Atualizados testes baseados em versão de schema (`test_schema_version_is_10` -> `11`).

## 3. Banco/migration

Foi adicionada a migration para a versão `v11` no `hermes_state.py`, criando a tabela `code_commands` para atrelar cada comando ao banco, com schema:
- `id` (PK), `code_session_id` (FK), `workspace_id` (FK), `command`, `argv_json`, `cwd`, `status`, `safety`, `stdout`, `stderr`, `exit_code`, `pid`, `timeout_seconds`, `started_at`, `completed_at`, `created_at`, `updated_at`.

Os índices criados melhoram as queries por sessão e workspace.

## 4. Serviço criado

O `CommandRunnerService` atua como ponte de orquestração:
- **Responsabilidades**: Cria registros no DB, gerencia a execução e registra resultados no SQLite e em eventos de timeline.
- **Validação de Workspace/CWD**: Assegura que o `cwd` é ou deriva do path absoluto do `workspace_id`. Se a pasta estiver fora, bloqueia com exceção.
- **Execução**: Subprocessos executados via array de strings (`shlex.split`), o que impossibilita sub-shells ou `shell=True`. 
- **Timeout e Outputs**: O tempo máximo (`timeout_seconds`) é respeitado, com subprocessos sendo eliminados se ultrapassarem o limite. A stdout e a stderr são capturadas e guardadas no DB. 

## 5. Política de segurança

Um modelo rigoroso foi empregado:
- **`safe`**: Comandos listados em "SAFE_COMMANDS" ou rodados através de binários seguros (como `python`, `npm`, `cargo`, `go`). Estes são executados imediatamente.
- **`needs_approval`**: Operações que impactam o projeto irreversivelmente ou instalam dependências (`git commit`, `npm install`, etc.). São retidos na Fase 4, bloqueando a execução imediata.
- **`blocked`**: Tentativas destrutivas como `sudo`, `rm -rf`, ou comandos que tenham pipes/redirecionamentos (`;`, `|`, `>`). Bloqueados imediatamente e devolvidos como erro.

## 6. Endpoints adicionados

- **`GET /api/code/sessions/{code_session_id}/commands`**
  - Retorna lista de comandos atrelados a uma sessão.
- **`POST /api/code/sessions/{code_session_id}/commands/run`**
  - Payload: `{"command": "...", "cwd": null, "timeout_seconds": 120}`
  - Executa o comando caso seja *safe*, e emite eventos WebSocket.
- **`GET /api/code/commands/{command_id}`**
  - Busca um comando pelo ID.
- **`POST /api/code/commands/{command_id}/cancel`**
  - Cancela o comando no banco (se ele ainda não tiver executado ou em timeout).

## 7. WebSocket events

Os seguintes eventos foram implementados via `_REALTIME_HUB` do `web_server.py`:
- `command.started`: Quando a execução se inicia.
- `command.output`: Quando stdout/stderr são capturados.
- `command.completed`: Quando `exit_code == 0`.
- `command.failed`: Quando erro de execução ou `exit_code != 0`.
- `command.timeout`: Quando excede `timeout_seconds`.
- `command.cancelled`: Quando cancelado via API.

## 8. Timeline da CodeSession

Qualquer mudança crítica de um comando insere uma atualização de timeline em `code_session_events` (tabela de eventos da code_session):
- `command.completed` ou `command.failed` (com o exit code respectivo) e suas saídas como payload, ficando listado nos logs da UI e visível para o LLM caso necessário.

## 9. Testes executados

Toda a suite de `tests/hermes_cli/test_command_runner.py` foi executada (13 testes no total) validando:
- Classificação correta (safe, needs_approval, blocked).
- Execução de um comando simples em Python.
- Tratamento de falha (`SystemExit`).
- Tratamento de Timeout.
- Tratamento de CWD fora do workspace.
- Rotas API via `TestClient` rodando simulações completas.
- 100% PASS.

## 10. Compatibilidade

Mantém total compatibilidade:
- A tabela `artifacts` continua intacta.
- A tabela `code_workspaces` continua igual, bem como `code_sessions`.
- As interações de chat e UI WebSocket não foram obstruídas.
- Mais de 250 testes críticos (`test_hermes_state`, `test_artifacts`, `test_code_session_service`) continuam passando no CI local sem problema com a migration `v11`.

## 11. Próximos passos

**Recomendação de Avanço: FASE 5 (GitService)**
Na próxima fase, o foco deve ser integrar manipulação local de git com a aprovação em `needs_approval`. A UI ou CLI precisará listar modificações (`diff`), mudar de branch com validação e confirmar commits. A fundação de Comandos já vai suportar o LLM explorando e validando as tasks rodando os comandos de testes criados.
