# 🐋 Orca Control & Multi-Agent Orchestration Skill

[![Hermes Agent](https://img.shields.io/badge/Hermes_Agent-Skill-blue.svg)](https://hermes-agent.nousresearch.com/)
[![ClawHub](https://img.shields.io/badge/ClawHub-Compatible-orange.svg)](https://clawhub.ai)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Author: Rafa Martins](https://img.shields.io/badge/Author-Rafa_Martins-purple.svg)](mailto:rafacpti@gmail.com)

> 🇬🇧 **English documentation below** | 🇧🇷 **Documentação em Português abaixo**

---

# 🇬🇧 English Documentation

## Overview
**Orca Control** is a production-grade Hermes Agent & ClawHub skill designed to inspect, operate, orchestrate, and automate the **Orca IDE / Multi-Agent Runtime Server**.

It exposes structured access to over 230 CLI capabilities of the Orca runtime, enabling autonomous AI agents to manage git worktrees, supervise subagents, resolve human-in-the-loop decision gates, control background terminal panes, run automations, and manage developer accounts.

### Key Features
- 🚀 **Full Runtime Lifecycle**: Start, stop, restart, and monitor the `orca-serve` background service.
- 🌳 **Git Worktree Isolation**: Create, list, switch, and prune isolated branches for agent task execution.
- 🤖 **Multi-Agent Orchestration**: Dispatch tasks, supervise workers, inspect inter-agent messages, and resolve approval gates.
- 💻 **Live Terminal Multiplexing**: Send background commands, inspect outputs, and manage concurrent terminal tabs.
- ⏰ **Automations & Scheduled Jobs**: Trigger and inspect recurring routines.
- 📱 **Computer Use & Mobile Emulation**: DOM snapshots, navigation, screenshots, and Android emulator interaction.

---

## Installation

### 1. Via Hermes Skills Hub (Recommended)
```bash
hermes skills install orca-control
```

### 2. Manual / Local Installation
Clone into your Hermes skills directory:
```bash
git clone https://github.com/rafacpti23/orca-control-skill.git ~/.hermes/skills/devops/orca-control
```

---

## Quick Start Commands

```bash
# 1. Check Orca server & graph status
orca status

# 2. List active projects and worktrees
orca repo list
orca worktree list

# 3. List active terminals and workers
orca terminal list
orca orchestration worker-list

# 4. Create and dispatch a new orchestration task
orca orchestration task-create --title "Refactor Auth" --spec "Migrate to JWT tokens"
```

---

## 👨💻 Author & Developer Profile

* **Developer:** Rafa Martins
* **Email:** [rafacpti@gmail.com](mailto:rafacpti@gmail.com)
* **GitHub:** [@rafacpti](https://github.com/rafacpti)
* **Ecosystem & Projects:**
  - 🧠 **[Synapse Layer](https://synapselayer.org):** Zero-Knowledge Persistent Memory Layer for AI Agents.
  - 📦 **[PAPI WhatsApp & Cloud API](https://papi.api.br):** Developer-first WhatsApp, SMS & VoIP Cloud Gateway (MCP Server: `papi-mcp` on npm).
  - 🤖 **[Stevo Chat CRM Integration]:** Multi-channel conversational AI & lead qualification engine.

### 🔥 Premium Skill — Meta Ads Autonomous Specialist (Paid · US$ 49)
> A commercial, **paid** Hermes skill (**US$ 49**) — available on request. This section describes *what it does*, not its internal prompts or source.

A fully autonomous Meta Ads (Facebook/Instagram) traffic-management agent that runs the entire lifecycle of a paid-media operation:

- **📊 Full account audit:** Maps the account hierarchy (Campaign → Ad Set → Ad), checks account status, spend cap, available balance and active-ad limits.
- **🚀 End-to-end campaign creation:** Builds complete campaigns via MCP ("Super Prompt"), including objectives, ad sets, targeting and budgets.
- **🎨 AI creative generation:** Produces ad creatives (copy + imagery) with integrated AI.
- **📈 Metrics monitoring:** Reads CPA / ROAS / CTR insights (last 7 days) at campaign and ad level.
- **🛡️ Autonomous Stop-Loss (+30% CPA):** Automatically pauses ads or ad sets that exceed the target CPA by more than 30%.
- **🧑⚖️ Human-in-the-Loop governance:** Requires human confirmation before activating any campaign with a daily budget above R$ 100/day; deletion of data is strictly forbidden without manual validation.
- **🔗 Landing-page validation:** Extracts and HTTP-tests destination URLs (WhatsApp links, YouTube, affiliates, shorteners) and detects redirects.
- **🔁 Cross-skill CRM integration:** Cross-checks unanswered leads in GoHighLevel (GHL) and triages them.
- **📱 Real-time alerts:** Notifies the manager's WhatsApp via the PAPI gateway when action is taken.
- **♻️ API resilience:** Gracefully handles expired tokens (OAuth 190), invalid-media errors, discontinued targeting, and empty insights — never halting the audit.
- **🏢 Multi-account support:** Manages several ad accounts from a single operator.

*Interested in the paid Ads skill? Contact the author on WhatsApp: [wa.me/5527999082624](https://wa.me/5527999082624)*

---

# 🇧🇷 Documentação em Português

## Visão Geral
O **Orca Control** é uma skill para o Hermes Agent e ClawHub projetada para inspecionar, operar, orquestrar e automatizar o **Orca IDE / Multi-Agent Runtime Server**.

Ela expõe controle estruturado sobre mais de 230 comandos do motor Orca, permitindo que agentes de inteligência artificial gerenciem branches isoladas (worktrees), supervisionem subagentes (workers), aprovem decision gates com intervenção humana, controlem terminais interativos em segundo plano, executem automações e configurem contas de IA.

### Principais Recursos
- 🚀 **Ciclo de Vida do Runtime**: Inicializar, parar, reiniciar e monitorar o serviço `orca-serve`.
- 🌳 **Isolamento com Git Worktrees**: Criar, listar, alternar e remover worktrees para execução isolada de tarefas.
- 🤖 **Orquestração Multi-Agente**: Despacho de tarefas, monitoramento de workers, leitura de mensagens inter-agentes e liberação de travas (decision gates).
- 💻 **Multiplexação de Terminais**: Envio de comandos em background, captura de saídas de texto e controle de abas.
- ⏰ **Automações e Rotinas Agendadas**: Disparo e monitoramento de automações.
- 📱 **Automação Web & Emuladores**: Snapshots de DOM, navegação, captura de tela e controle de emulador Android.

---

## Instalação

### 1. Via Hermes Skills Hub (Recomendado)
```bash
hermes skills install orca-control
```

### 2. Instalação Manual
Clone o repositório na pasta de skills do Hermes:
```bash
git clone https://github.com/rafacpti23/orca-control-skill.git ~/.hermes/skills/devops/orca-control
```

---

## Guia Rápido de Uso

```bash
# 1. Verificar saúde do serviço e runtime
orca status

# 2. Listar repositórios e worktrees
orca repo list
orca worktree list

# 3. Listar terminais conectados e workers em execução
orca terminal list
orca orchestration worker-list

# 4. Criar e despachar uma nova tarefa
orca orchestration task-create --title "Refatorar Autenticação" --spec "Migrar para tokens JWT"
```

---

## 👨💻 Dados do Desenvolvedor

* **Autor:** Rafa Martins
* **E-mail:** [rafacpti@gmail.com](mailto:rafacpti@gmail.com)
* **GitHub:** [@rafacpti](https://github.com/rafacpti)
* **Projetos e Skills Relacionadas:**
  - 🧠 **[Synapse Layer](https://synapselayer.org):** Camada de memória persistente Zero-Knowledge para agentes de IA.
  - 📦 **[PAPI API](https://papi.api.br):** Gateway de nuvem para WhatsApp, SMS e VoIP (MCP Server: `papi-mcp` no npm).
  - 🤖 **[Stevo Chat / CRM]:** Integrações de atendimento e qualificação conversacional.

### 🔥 Skill Premium — Especialista Autônomo em Meta Ads (Paga · US$ 49)
> Skill comercial **paga** para Hermes (**US$ 49**) — disponível sob solicitação. Esta seção descreve *o que ela faz*, sem expor prompts internos ou código-fonte.

Um agente autônomo de gestão de tráfego pago no Meta Ads (Facebook/Instagram) que executa todo o ciclo de uma operação de mídia paga:

- **📊 Auditoria completa da conta:** Mapeia a hierarquia (Campanha → Conjunto → Anúncio), verifica status da conta, spend cap, saldo disponível e limite de anúncios ativos.
- **🚀 Criação de campanhas ponta a ponta:** Monta campanhas completas via MCP ("Super Prompt") — objetivos, conjuntos, segmentação e orçamentos.
- **🎨 Geração de criativos com IA:** Produz criativos (copy + imagem) com IA integrada.
- **📈 Monitoramento de métricas:** Lê insights de CPA / ROAS / CTR (últimos 7 dias) em nível de campanha e de anúncio.
- **🛡️ Stop-Loss Autônomo (+30% CPA):** Pausa automaticamente anúncios ou conjuntos que ultrapassam +30% do CPA meta.
- **🧑⚖️ Governança Human-in-the-Loop:** Exige confirmação humana para ativar campanhas com verba acima de R$ 100/dia; exclusão de dados é estritamente proibida sem validação manual.
- **🔗 Validação de landing pages:** Extrai e testa (HTTP) as URLs de destino (links de WhatsApp, YouTube, afiliados, encurtadores) e detecta redirecionamentos.
- **🔁 Integração cruzada com CRM:** Cruza leads sem resposta no GoHighLevel (GHL) e faz a triagem.
- **📱 Alertas em tempo real:** Notifica o WhatsApp do gestor via gateway PAPI quando uma ação é tomada.
- **♻️ Resiliência de API:** Trata com elegância tokens expirados (OAuth 190), erros de mídia inválida, segmentações descontinuadas e insights vazios — sem travar a auditoria.
- **🏢 Suporte multi-contas:** Gerencia várias contas de anúncios a partir de um único operador.

*Interessado na skill paga de Ads? Fale com o autor no WhatsApp: [wa.me/5527999082624](https://wa.me/5527999082624)*

---

## 📄 License
Distribuído sob a licença **MIT**. Consulte `LICENSE` para mais detalhes.
