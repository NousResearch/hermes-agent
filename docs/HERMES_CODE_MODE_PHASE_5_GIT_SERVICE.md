# Hermes Code Mode — Fase 5: GitService Seguro

## 1. Resumo

Esta fase implementa um servico Git seguro e persistente para o Hermes Code Mode. O GitService permite:

- Ler status, branch, remote e diff de workspaces Git.
- Criar snapshots persistentes do estado Git no banco de dados.
- Preparar operacoes de branch e commit com classificacao de seguranca.
- Criar branches de forma segura, apenas quando o workspace estiver limpo.
- Emitir eventos WebSocket e registrar timeline na CodeSession.

Todas as operacoes destrutivas sao bloqueadas ou requerem aprovacao humana.

## 2. Arquivos alterados

| Arquivo | Motivo |
|---------|--------|
| `hermes_state.py` | SCHEMA_VERSION atualizado para 12; migration v12; tabelas e indices adicionados ao SCHEMA_SQL; classes GitSnapshotDB, ApprovalDB, TaskDB, WorkspaceDB, CodeSessionDB, CodeCommandDB criadas/atualizadas; metodos de artifact e session atualizados |
| `hermes_cli/code/git_service.py` | Novo — GitService com operacoes Git seguras |
| `hermes_cli/web_server.py` | 8 endpoints REST para operacoes Git |
| `tests/hermes_cli/test_git_service.py` | Novo — 32 testes unitarios e de integracao |
| `tests/test_artifacts.py` | Atualizado schema version 11 -> 12 |
| `tests/test_hermes_state.py` | Atualizado schema version 11 -> 12 |
| `tests/hermes_cli/test_workspace_service.py` | Atualizado schema version 11 -> 12 |
| `tests/hermes_cli/test_code_session_service.py` | Atualizado schema version 11 -> 12 |

## 3. Banco/migration

### Schema version

`SCHEMA_VERSION = 12`

### Tabela `code_git_snapshots`

```sql
CREATE TABLE IF NOT EXISTS code_git_snapshots (
  id TEXT PRIMARY KEY,
  workspace_id TEXT NOT NULL,
  code_session_id TEXT,
  branch TEXT,
  remote_url TEXT,
  dirty INTEGER DEFAULT 0,
  summary_json TEXT DEFAULT '{}',
  files_json TEXT DEFAULT '[]',
  diff_stat TEXT,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY(workspace_id) REFERENCES code_workspaces(id),
  FOREIGN KEY(code_session_id) REFERENCES code_sessions(id)
);
```

### Indices

- `idx_code_git_snapshots_workspace_id`
- `idx_code_git_snapshots_code_session_id`
- `idx_code_git_snapshots_created_at DESC`

A migration v12 e executada automaticamente em `SessionDB._init_schema()`. A tabela tambem e criada nos `_init_schema()` de `CodeSessionDB` e `GitSnapshotDB` para garantir compatibilidade.

## 4. Servico criado

### GitService (`hermes_cli/code/git_service.py`)

Responsabilidades:
- Executar comandos Git de forma segura (sem `shell=True`, com lista de argumentos).
- Validar se o workspace e um repositorio Git antes de executar comandos.
- Classificar acoes Git em `safe`, `needs_approval` ou `blocked`.
- Persistir snapshots no `GitSnapshotDB`.
- Registrar eventos na timeline da `CodeSession` quando `code_session_id` e fornecido.
- Emitir eventos WebSocket via `realtime_hub`.

#### Validacao de workspace

O metodo `_get_workspace_path` resolve o caminho do workspace via `WorkspaceDB` e verifica se existe e e um diretorio. O metodo `_is_git_repo` executa `git rev-parse --is-inside-work-tree`.

#### Execucao Git

O helper `_run_git` usa:

```python
subprocess.run(
    ["git", *args],
    cwd=str(workspace_path),
    capture_output=True,
    text=True,
    timeout=timeout,
    check=False,
)
```

#### Parsing de status

Usa `git status --porcelain=v1`. Cada linha e parseada conforme:
- `X` = index status
- `Y` = worktree status
- `??` = untracked
- `M/A/D/R` = modified/added/deleted/renamed

#### Diff

Usa `git diff --no-ext-diff -- [path]`. O counting de additions/deletions reutiliza `count_diff_changes` do `hermes_state.py`.

#### Snapshot

O metodo `create_snapshot` captura o status atual, diff stat e persiste no banco via `GitSnapshotDB`.

#### Branch

- `prepare_branch`: valida nome da branch, verifica se ha dirty tree. Retorna `safe`, `needs_approval` ou `blocked`.
- `create_branch`: executa apenas se `prepare_branch` retornar `safe`. Usa `git switch -c` com fallback para `git checkout -b`.

#### Commit

- `prepare_commit`: NAO executa commit. Retorna payload de aprovacao com files, diff_stat e mensagem.

## 5. Politica de seguranca Git

```python
class GitActionSafety:
    SAFE = "safe"
    NEEDS_APPROVAL = "needs_approval"
    BLOCKED = "blocked"
```

### Safe

- `git status --porcelain=v1`
- `git branch --show-current`
- `git rev-parse --is-inside-work-tree`
- `git remote get-url origin`
- `git diff --stat`
- `git diff`
- `git diff -- <file>`
- `git ls-files --others --exclude-standard`

### Needs approval

- `git checkout <branch>`
- `git switch <branch>`
- `git checkout -b <branch>`
- `git switch -c <branch>`
- `git commit`
- `git push`
- `git stash`
- `git add`

Excecao: criar branch nova e permitida apenas se workspace estiver limpo.

### Blocked

- `git reset --hard`
- `git clean -fd`
- `git clean -xfd`
- `git checkout .`
- `git restore .`
- `git restore --source`
- `git reflog expire`
- `git gc --prune=now`

## 6. Endpoints adicionados

### GET /api/code/workspaces/{workspace_id}/git/status

Retorna status Git completo do workspace.

Query opcional: `code_session_id`

Resposta:
```json
{
  "status": {
    "workspace_id": "...",
    "is_git_repo": true,
    "branch": "main",
    "remote_url": "https://github.com/...",
    "dirty": false,
    "files": [],
    "summary": { "modified": 0, "added": 0, ... }
  }
}
```

### GET /api/code/workspaces/{workspace_id}/git/diff

Retorna diff geral ou por arquivo.

Query opcional: `path=src/App.tsx`

Resposta:
```json
{
  "diff": {
    "workspace_id": "...",
    "path": null,
    "diff": "...",
    "additions": 10,
    "deletions": 2
  }
}
```

### GET /api/code/workspaces/{workspace_id}/git/branch

Retorna branch atual.

### GET /api/code/workspaces/{workspace_id}/git/remote

Retorna URL do remote origin.

### POST /api/code/workspaces/{workspace_id}/git/snapshot

Cria snapshot persistente.

Body opcional:
```json
{ "code_session_id": "optional" }
```

### POST /api/code/workspaces/{workspace_id}/git/branch/prepare

Prepara criacao de branch.

Body:
```json
{ "branch_name": "feature/hermes-code-mode" }
```

### POST /api/code/workspaces/{workspace_id}/git/branch

Cria branch se seguro.

Body:
```json
{ "branch_name": "feature/hermes-code-mode", "code_session_id": "optional" }
```

### POST /api/code/workspaces/{workspace_id}/git/commit/prepare

Prepara commit (nao executa).

Body:
```json
{ "message": "feat: add code sessions", "code_session_id": "optional" }
```

## 7. WebSocket events

- `git.status_changed` — emitido apos GET /git/status
- `git.snapshot.created` — emitido apos criar snapshot
- `git.branch.prepared` — emitido apos preparar branch
- `git.branch.created` — emitido apos criar branch com sucesso
- `git.commit.prepared` — emitido apos preparar commit

Formato:
```json
{
  "type": "git.snapshot.created",
  "payload": { "workspace_id": "...", "snapshot": { } },
  "timestamp": "2026-04-25T..."
}
```

## 8. Timeline da CodeSession

Quando `code_session_id` e fornecido, os seguintes eventos sao adicionados a `code_session_events`:

- `git.status_checked`
- `git.snapshot.created`
- `git.branch.prepared`
- `git.branch.created`
- `git.commit.prepared`

Payload minimo inclui `workspace_id`, `branch`, `dirty`, `summary`.

## 9. Testes executados

```bash
python3 -m pytest tests/hermes_cli/test_git_service.py -v --tb=short -o addopts=''
# 32 passed

python3 -m pytest tests/hermes_cli/test_command_runner.py -v --tb=short -o addopts=''
# 13 passed

python3 -m pytest tests/hermes_cli/test_code_session_service.py -v --tb=short -o addopts=''
# 43 passed

python3 -m pytest tests/hermes_cli/test_workspace_service.py -v --tb=short -o addopts=''
# 47 passed

python3 -m pytest tests/test_artifacts.py -v --tb=short -o addopts=''
# 17 passed

python3 -m pytest tests/test_hermes_state.py -v --tb=short -o addopts=''
# 229 passed
```

Total combinado: **166 passed** nos suites criticos.

## 10. Compatibilidade

Confirmado funcionamento de:

- Fase 1 artifacts (create_artifact, get_artifacts_by_session, fallback legacy)
- Fase 2 workspaces (WorkspaceDB, endpoints)
- Fase 3 code sessions (CodeSessionDB, endpoints, timeline)
- Fase 4 commands (CodeCommandDB, endpoints)
- GitService (status, diff, branch, snapshot, prepare)
- WebSocket existente
- Schema migration v6 -> v12

## 11. Proximos passos

Recomenda-se a **Fase 6: ProviderRouter**:

- Registrar provider/model por CodeSession
- Selecao de modelo por tipo de tarefa (fast/strong/cheap/reviewer/planner)
- Fallback entre providers
- Presets de modelo por sessao
- Rastreamento de custo por sessao de codigo
