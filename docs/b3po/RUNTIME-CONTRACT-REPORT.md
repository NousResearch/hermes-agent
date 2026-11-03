# RUNTIME-CONTRACT-REPORT — o que o Hermes realmente expõe (O1/D3)

Base: código do checkout `b3po-brand @ 21595add`. Método: leitura de código + execução real de um turno.

## 1. Capacidades confirmadas

| Capacidade | Estado | Evidência |
|---|---|---|
| Loop de execução / turno | **CONFIRMADO** | `hermes -z` executou turno real (exit 0, 13,4 s) |
| Sessão persistida | **CONFIRMADO** | `hermes_state_sessions.py`, DB em `HERMES_HOME`; `session.save` |
| Provider configurável | **CONFIRMADO** | `model.provider/base_url/default` aplicados; Ollama respondeu |
| Eventos de turno | **CONFIRMADO** | `turn.start`, `turn.end`, `turn.error` |
| Eventos de ferramenta | **CONFIRMADO** | `tool.start`, `tool.generating`, `tool.complete` |
| Aprovação | **CONFIRMADO** | `tools/approval.py`: `register_gateway_notify`, `resolve_gateway_approval`, `ack` |
| Cancelamento | **IMPLEMENTADO, NÃO VALIDADO** | `_handle_interrupt` em `compute_host.py`; sem prova E2E |
| Política de ferramenta | **PARCIAL** | aprovação existe para ações sensíveis; **não** há gancho de política arbitrária pré-execução documentado |
| `execution_id` | **NÃO EXISTE no runtime** | será criado pela camada B-3PO e correlacionado ao `session_ref` |

## 2. Mapeamento para o contrato B-3PO v1

| Evento B-3PO | Fonte real no Hermes | Observação |
|---|---|---|
| `execution.created` | camada B-3PO | não depende do runtime |
| `execution.started` | `turn.start` | direto |
| `execution.waiting_approval` | `tools/approval.py` (notify) | existe canal |
| `execution.tool_started` | `tool.start` | direto |
| `execution.tool_finished` | `tool.complete` | direto |
| `execution.completed` | `turn.end` | direto |
| `execution.failed` | `turn.error` | direto |
| `execution.cancel_requested` | camada B-3PO | encaminha `interrupt` |
| `execution.cancelled` | resposta de `interrupt` | a confirmar no E2E |

Nenhum evento precisa ser fabricado: os de ferramenta **existem**. Onde não houver (ex.: `cancelled` sem
resposta), o adaptador declara `capabilities.cancellation=false` e emite `execution.failed` com motivo.

## 3. Enforcement de política — o ponto crítico do O1/D3

A ata exige política **no ponto efetivo de execução**. Duas opções reais:

1. **Desabilitar a ferramenta no toolset** (registro) — o modelo nunca vê a ferramenta; é enforcement
   real, não filtragem de UI.
2. **Aprovação pré-execução** via `tools/approval.py` — para ações sensíveis já suportadas.

Não foi encontrado gancho genérico de "política do produto injetada no runtime". Portanto, no v1:
`permission_policy` **restringe toolset** (opção 1) e usa aprovação (opção 2) onde já existe. Capacidade
que não puder ser restringida por esses dois meios fica **desabilitada** e emite `execution.failed`.

## 4. Riscos

| Risco | Impacto | Mitigação |
|---|---|---|
| Evento `cancelled` sem confirmação do runtime | estado travado em "cancelando" | timeout + `execution.failed` com motivo |
| Aprovação só por gateway (Telegram etc.) | desktop sem canal de aprovação | mapear notify→UI antes do marco |
| Provider local com janela < 64K | turno recusado (já ocorreu) | validar modelo na configuração (`ollama_num_ctx`/janela ≥ 64K) |
