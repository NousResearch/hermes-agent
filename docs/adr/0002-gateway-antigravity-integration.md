# 2. Integração do Antigravity no Gateway (Slack, WhatsApp e Mensageria)

## Status
Accepted

## Contexto
O Gateway do Hermes conecta o agente a canais de mensageria como Slack, WhatsApp, Telegram e Discord. Anteriormente, o runtime do Antigravity (`google-antigravity`) não era exposto em `list_picker_providers`, impedindo que os modelos reais do `agy` fossem descobertos ou selecionados interativamente no chat. Além disso, as plataformas não tinham uma forma declarativa de definir o Antigravity como padrão nas configurações.

## Decisão
1. **Descoberta Dinâmica no Chat:** Integrar o `google-antigravity` em `hermes_cli/model_switch_providers.py` com o alias curto `agy`. Ao executar `/model` no Slack ou WhatsApp, o Hermes lista os modelos retornados por `agy models` (ex: `gemini-3.8-flash-high`, `gemini-3.1-pro-high`, `claude-sonnet-4-6`, `claude-opus-4-6-thinking`) e permite a seleção via menu ou comando textual.
2. **Configuração Declarativa por Plataforma:** Suportar a configuração de modelo e provedor padrão diretamente no bloco da plataforma no `config.yaml` (ex: `platforms.slack.provider: google-antigravity` e `platforms.slack.model: gemini-3.8-flash-high`).
3. **Execução Headless com Permissões Automáticas:** Sessões originadas no Gateway herdam `antigravity.dangerously_skip_permissions: true` (ou a configuração explícita de `config.yaml`), evitando que o `agy` pare aguardando aprovações interativas de TTY que não existem no canal de chat.

## Consequências
- Usuários podem usar tanto modelos Gemini quanto Claude através do Antigravity no Slack e WhatsApp com total paridade aos provedores nativos.
- O progresso de ferramentas e raciocínio emitidos pelo `agy` é encaminhado aos adaptadores de mensagem (`tool_progress_callback`, `tool_start_callback`).
