# 3. Herança de Credenciais e Modelo Autônomo de Ferramentas do Antigravity

## Status
Accepted

## Contexto
Por padrão de segurança, o Hermes executa a higienização de variáveis de ambiente para subprocessos filhos através de `hermes_subprocess_env(inherit_credentials=False)`, removendo variáveis como `GOOGLE_APPLICATION_CREDENTIALS`, `GOOGLE_API_KEY` e credenciais associadas. Quando o `AntigravityClient` executava em segundo plano (em especial em jobs de cron e no gateway iniciados pelo systemd), o binário `agy` perdia suas credenciais e abortava com `RuntimeError: agy reported terminal status ERROR: authentication failed or timed out`.

## Decisão
1. **Herança de Credenciais para o Runtime:** O `AntigravityClient` executará com `inherit_credentials=True` (ou preservação explícita de variáveis do ecossistema Google/Antigravity), garantindo que o `agy` acesse os tokens de autenticação da conta em execuções headless e agendadas.
2. **Motor Autônomo de Ferramentas:** O `agy` atua como motor autônomo, executando comandos, navegação e alterações no diretório do projeto (`cwd`).
3. **Ponte de Eventos Unidirecional:** O Hermes atua como ponte de apresentação e telemetria: escuta os eventos NDJSON emitidos pelo `agy` e dispara os callbacks nativos (`_fire_stream_delta`, `_fire_reasoning_delta`, `tool_progress_callback`, `tool_start_callback`), projetando o status das ferramentas em tempo real para Slack, WhatsApp e logs de auditoria.

## Consequências
- Jobs agendados no Cron com modelos do Antigravity rodam com estabilidade e sem quedas por falta de credenciais.
- As ferramentas executadas pelo `agy` são reportadas visualmente nos canais de mensageria com o mesmo nível de detalhe dos provedores nativos.
