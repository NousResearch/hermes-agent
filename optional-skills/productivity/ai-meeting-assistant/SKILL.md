---
name: meeting-assistant
version: 1.0.0
description: "AI Meeting Assistant for Jitsi Meet and Google Meet. Joins calls, transcribes with diarization, interacts via duplex voice, and generates structured minutes through your own self-hosted MCP gateway."
triggers:
  - entrar reunião
  - join meeting
  - transcrever reunião
  - meeting bot
  - jitsi
  - google meet
  - ata de reunião
  - meeting minutes
---

# 🎙️ AI Meeting Assistant — Jitsi Meet & Google Meet

Bot de IA que participa de reuniões no **Jitsi Meet** e no **Google Meet**, transcrevendo em tempo real com diarização de oradores, interagindo via voz duplex (Gemini Live / OpenAI Realtime) e gerando atas estruturadas automaticamente.

> ⚡ Toda a execução é orquestrada pelo **Habilis MCP Gateway** — nenhuma lógica roda localmente.

---

## 🚀 Onboarding & Setup Rápido (Primeiro Uso)

### Checklist de Status:
```
[PENDENTE] Token Habilis MCP
[PENDENTE] Conexão MCP configurada
[PENDENTE] Variáveis de ambiente do anunciante
```

### Passo 1 — Obtenha seu Token
Acesse o portal de desenvolvedores para gerar sua chave de acesso:
👉 [https://<your-domain>](https://<your-domain>)

### Passo 2 — Configure as Variáveis de Ambiente
```bash
export HABILIS_API_KEY="hab_live_..."
export GEMINI_API_KEY="..."       # Para voz Jitsi Meet (Gemini Live)
export OPENAI_API_KEY="..."       # Para voz Google Meet (OpenAI Realtime)
```

### Passo 3 — Conecte o MCP Gateway
Adicione ao `config.yaml` do Hermes / OpenClaw / Claude Desktop:
```yaml
mcp_servers:
  habilis:
    url: "https://<your-domain>/api/mcp"
    headers:
      Authorization: "Bearer ${HABILIS_API_KEY}"
```

Após configurar, o agente receberá automaticamente todas as ferramentas de reunião disponíveis.

---

## 🛠️ Ferramentas MCP Disponíveis (Habilis Gateway)

Esta skill opera **exclusivamente** através das ferramentas fornecidas pelo Habilis MCP Gateway:

### Jitsi Meet
| Ferramenta | Descrição |
|---|---|
| `mcp__habilis__jitsi_join` | Entra em uma sala Jitsi com nome, modo (realtime/listen_only/chat_only) e voz configurável |
| `mcp__habilis__jitsi_status` | Retorna participantes ativos, orador atual e status do bot |
| `mcp__habilis__jitsi_say` | Fala texto na reunião em tempo real via síntese de voz |
| `mcp__habilis__jitsi_transcript` | Obtém transcrição com diarização (últimas N mensagens) |
| `mcp__habilis__jitsi_leave` | Sai da chamada com motivo opcional |

### Google Meet
| Ferramenta | Descrição |
|---|---|
| `mcp__habilis__meet_join` | Entra em uma chamada do Google Meet com nome e modo configurável |
| `mcp__habilis__meet_status` | Verifica participantes e tempo de chamada |
| `mcp__habilis__meet_say` | Fala texto na reunião via síntese de voz |
| `mcp__habilis__meet_transcript` | Obtém legendas transcritas com identificação de oradores |
| `mcp__habilis__meet_leave` | Sai da chamada de forma limpa |

### Utilitários
| Ferramenta | Descrição |
|---|---|
| `mcp__habilis__meeting_summarize` | Gera ata estruturada a partir da transcrição da reunião |

---

## ⚡ Fluxo de Execução Autônoma

### 1. Entrada na Reunião
O agente recebe a URL da sala (Jitsi ou Google Meet) e entra via `mcp__habilis__jitsi_join` ou `mcp__habilis__meet_join`, configurando:
- **Nome do bot** exibido na sala
- **Modo de operação**: `realtime` (voz bidirecional), `listen_only` (apenas transcrição) ou `chat_only`
- **Voz**: Escolha entre vozes disponíveis (Puck, Aoede, Charon, Kore, Fenrir, etc.)

### 2. Transcrição em Tempo Real
Durante a chamada, o gateway processa o áudio dos participantes com:
- 🗣️ **Diarização** — Identificação de quem está falando
- 📝 **Transcrição contínua** — Conversão speech-to-text em tempo real
- 💬 **Captura de chat** — Mensagens do chat da reunião são registradas

### 3. Interação por Voz
O agente pode responder perguntas, fornecer dados e participar ativamente da reunião:
- Em modo `realtime`, responde com voz natural duplex (conversa fluida)
- Suporta **barge-in** (interrupção) — prioriza a fala humana

### 4. Geração de Ata Estruturada
Ao encerrar a chamada, o agente gera automaticamente:
- 📋 **Resumo Executivo** — Síntese dos principais pontos discutidos
- 📌 **Decisões Tomadas** — Lista de deliberações com contexto
- 🎯 **Action Items** — Próximas tarefas com responsáveis e prazos
- ⏱️ **Timeline** — Cronologia dos tópicos abordados

---

## 🔒 Segurança & Privacidade

- **Zero Storage**: Chaves de API (Gemini, OpenAI) residem exclusivamente no ambiente local do cliente e trafegam de forma segura via requisição ao Gateway.
- **Sem scripts locais**: Nenhum código-fonte, endpoint de terceiros ou lógica de bypass é incluído nesta skill.
- **Transcrições efêmeras**: As transcrições são processadas em memória e entregues ao agente — não ficam armazenadas no Gateway.
