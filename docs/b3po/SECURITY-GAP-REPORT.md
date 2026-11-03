# SECURITY-GAP-REPORT — estado real e lacunas

Base: leitura de código + execução controlada. Nenhum teste de penetração foi executado.

## 1. O que existe e foi confirmado

| Controle | Evidência | Avaliação |
|---|---|---|
| Aprovação de ação sensível | `tools/approval.py` — notify → resolve → ack; `file_safety.is_write_approval_required` | **existe** |
| Gate de escrita por caminho | `agent/file_safety.py` | **existe** |
| Segredo fora do agente | ACP recebe `mcp local --connector`; env fica no host (auditoria de projeto anterior) | **existe no desenho** |
| Sanitização de log | `redact_sensitive_text` em `copilot_acp_client` | **existe** |

## 2. Lacunas

| # | Lacuna | Severidade | Observação |
|---|---|---|---|
| S1 | **Não há gancho genérico de política pré-execução** para regras do produto | alta | o v1 usa restrição de toolset + aprovação existente; o resto fica desabilitado |
| S2 | Aprovação é roteada por gateway; desktop precisa de canal próprio | média | mapear antes do marco |
| S3 | `npm audit` do monorepo não foi executado nesta rodada | média | 1222 pacotes; rodar antes de release |
| S4 | `~/.hermes` guarda config e sessões sem criptografia declarada | média | documentar; usar `HERMES_HOME` isolado em teste |
| S5 | Provider local (Ollama) sem autenticação | baixa | binding em `127.0.0.1` |
| S6 | Sem SBOM/assinatura de dependências | média | previsto na Fase I do GOAL |

## 3. Regra aplicada (O1/D3)

> Política no ponto de execução, nunca filtragem de UI.

No v1: `permission_policy` **desabilita ferramentas no toolset** (o modelo nunca as vê) e usa a aprovação
existente onde houver. Capacidade sem esses dois meios é desabilitada e registra `execution.failed`.

## 4. O que NÃO foi feito (e não pode ser alegado)

- Nenhum pentest, SAST completo ou revisão de dependências concluída.
- Nenhuma prova de que o desktop não expõe IPC indevido.
- Nenhuma prova de que logs não vazam segredo em produção.

## 5. Próximos passos obrigatórios antes de release

1. `npm audit` (raiz, web, tui, desktop) + triagem.
2. Mapear superfície IPC do Electron (preload/contextBridge).
3. Teste de path traversal nos endpoints de asset.
4. Verificar redaction de logs com um segredo sintético.
5. Gerar SBOM e checksums dos artefatos.
