# HERMES CODE MODE — Phase 7: Code Intelligence / LSP

## Objetivo da Fase
Criar uma camada inicial de inteligência de código para o Hermes Code Mode, permitindo que o Hermes consulte diagnósticos reais do projeto (TypeScript, Go, ESLint) de maneira segura, rápida e pragmática antes ou depois de editar código.

## Arquitetura Implementada
A arquitetura introduz o `CodeIntelligenceService` (LSP Service), que interage de maneira restrita e de apenas leitura com os comandos já expostos pelo repositório.

- **`CodeIntelligenceService` (`hermes_cli/code/lsp_service.py`)**: Serviço central responsável por detectar linguagens, construir comandos seguros (typecheck, lint, vet) e executar processos que geram os diagnósticos sem modificar os arquivos locais.
- **Integração com `CommandRunner`**: Garante que os comandos de lsp/diagnóstico sejam verificados usando a mesma infraestrutura de segurança das fases anteriores.
- **`CodeDiagnosticsDB` (`hermes_state.py`)**: Persistência de resultados em banco de dados SQLite (com schema na versão 14). Salva summary, duração, lista de diagnósticos (`diagnostics_json`) e os vincula tanto ao `workspace_id` quanto ao `code_session_id`.

## Comandos Utilizados por Stack
- **TypeScript/Node**: Tenta `npm run typecheck` (ou `yarn`/`pnpm`/`bun` equivalentes), ou `npx tsc --noEmit` se fallback.
- **ESLint**: Tenta `npm run lint` ou similar que executa o linter.
- **Go**: Usa comandos embutidos e seguros `go vet ./...` e `go test ./...`.

## Endpoints Criados (REST)
Foram adicionados os seguintes endpoints em `hermes_cli/web_server.py`:
- `GET /api/code/workspaces/:workspace_id/diagnostics`: Executa o diagnóstico completo no workspace.
- `GET /api/code/workspaces/:workspace_id/diagnostics/file?path=...`: Executa o diagnóstico, porém retorna filtrado pelo caminho do arquivo.
- `GET /api/code/workspaces/:workspace_id/languages`: Retorna linguagens/frameworks identificadas (ex: `['typescript', 'eslint']`).
- `POST /api/code/workspaces/:workspace_id/lsp/restart`: Stub atual para restart de interface LSP (retorna status `noop`).

## Formato do `Diagnostic`
Os dados de diagnóstico retornam num formato normalizado consistente com o que editores de texto modernos e servidores LSP utilizam:
```json
{
  "file": "src/app.tsx",
  "line": 10,
  "column": 5,
  "severity": "error", // "error" | "warning" | "info" | "hint"
  "source": "typescript",
  "code": "TS2322",
  "message": "Type 'string' is not assignable to type 'number'.",
  "raw": null
}
```

## Limitações Conhecidas
- A Fase 7 utiliza execução pragmática de CLIs (Command Line Interfaces) padrão ao invés de um servidor de protocolo LSP em daemon rodando persistentemente. O tempo de resposta está vinculado diretamente ao tempo de inicialização do CLI local (`tsc`, `eslint`, `go vet`).
- Nenhuma modificação, autofix ou multi-agent root cause analysis foi embutida neste momento (agendado para fases futuras).
- As mensagens de log de comandos que não são interpretáveis são mapeadas usando um parser fallback como raw (especialmente para linters complexos customizados).

## Próximos Passos (LSP Real)
- Introdução de cliente LSP real sob JSON-RPC rodando em daemon.
- Refinamento incremental de autofixes utilizando o `CommandRunner`.
- Integração da inteligência para rodar automaticamente após cada `replace` do agente de code mode.

## Como Testar
Para validar a correta extração, detecção, e gravação no banco, foi implementada uma suíte de testes robusta.
Execute os testes utilizando o Pytest:
```bash
python -m pytest tests/hermes_cli/test_lsp_service.py
```
*(Certifique-se de executar também o restante da suíte `tests/hermes_cli/*` e `tests/test_hermes_state.py` para garantir a integridade dos módulos dependentes).*