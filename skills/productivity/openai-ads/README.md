# OpenAI Ads & ChatGPT Marketing Agent

> **Autonomous AI Marketing & Ads Manager for the OpenAI Ads API v1**
> Complete lifecycle management for advertising inside ChatGPT.

[![Author](https://img.shields.io/badge/Author-Rafa%20Martins-blue)](https://portal.sthub.com.br)
[![Website](https://img.shields.io/badge/Website-portal.sthub.com.br-orange)](https://portal.sthub.com.br)
[![License](https://img.shields.io/badge/License-MIT-green)](./LICENSE)

---

## 🇺🇸 English

### Overview
This skill turns your agent into a **full-stack Media Buyer and Traffic Manager** for the brand-new OpenAI Ads platform (advertising natively inside ChatGPT conversations). It covers all **19 functional domains** and **70 endpoints** of the OpenAI Ads API v1 — from campaign structuring and semantic *context hints* to interactive `chat_card` creatives, conversational Business Agents, Lead Forms, e-commerce Product Feeds, CAPI conversion tracking, and granular reporting.

### Key Capabilities
| Area | What the agent does |
| :--- | :--- |
| **Strategy & Targeting** | Validates geo IDs via `/geo_lookup`, builds semantic *context hints*, manages SHA-256 custom audiences and exclusion lists. |
| **Creative Production** | Writes high-converting headlines (≤50 chars) and body copy (≤150 chars), uploads media, and generates live ChatGPT previews. |
| **Bidding & Budget** | Handles micros math (`$1 = 1,000,000`), `cpc` vs `conversions` bidding, and protective spend-limit windows. |
| **Lead Generation** | Publishes native in-chat Lead Forms and wires Webhook `lead_sync_subscriptions` into any CRM (GoHighLevel, Kommo, n8n). |
| **Business Agents** | Creates and publishes conversational agents with custom tools attached to ads. |
| **E-commerce** | Manages dynamic Product Feeds with real-time NDJSON delta ingestion. |
| **Measurement** | Configures Web Pixel + Server-to-Server CAPI with `event_id`/`obref` deduplication. |
| **Optimization** | Pulls insights by day, device, country, and product; scales winners and cuts losers. |

### Installation
```bash
hermes skills install openai-ads
```

### Setup
```bash
export OPENAI_ADS_API_KEY="<your-api-key>"
```

### Quick Start
```bash
SKILL_DIR=~/.hermes/skills/productivity/openai-ads

# 1. Verify account
python3 $SKILL_DIR/scripts/openai_ads_cli.py account

# 2. Create a campaign
python3 $SKILL_DIR/scripts/openai_ads_cli.py campaign-create \
  --name "Q1 Lead Generation" --objective conversions --locations BR,US

# 3. Create an ad group with semantic context hints
python3 $SKILL_DIR/scripts/openai_ads_cli.py adgroup-create \
  --campaign-id camp_123 --name "High Intent Buyers" \
  --bid-amount 2500000 --daily-budget 50000000 \
  --hints "best crm for small business,automate customer service"

# 4. Create the interactive chat_card ad
python3 $SKILL_DIR/scripts/openai_ads_cli.py ad-create \
  --ad-group-id adg_456 --name "Main Creative" \
  --headline "Scale Sales With AI" \
  --body "Automate lead qualification 24/7 and never miss a customer again." \
  --cta LEARN_MORE --url "https://example.com/offer"

# 5. Preview and pull metrics
python3 $SKILL_DIR/scripts/openai_ads_cli.py preview --ad-id ad_789
python3 $SKILL_DIR/scripts/openai_ads_cli.py insights --entity campaigns --id camp_123 --breakdown device
```

### Included Files
- `SKILL.md` — full operational playbook for the agent
- `scripts/openai_ads_cli.py` — executable CLI automation engine
- `scripts/openai_ads_client.py` — reusable Python client library
- `references/api_reference.md` — all 70 endpoints mapped (EN/PT)
- `references/chat_card_best_practices.md` — creative and tracking best practices

---

## 🇧🇷 Português

### Visão Geral
Esta skill transforma seu agente em um **Gestor de Tráfego e Media Buyer completo** para a nova plataforma de anúncios da OpenAI (publicidade nativa dentro das conversas do ChatGPT). Cobre todos os **19 domínios funcionais** e **70 endpoints** da OpenAI Ads API v1 — desde estruturação de campanhas e *context hints* semânticos até criativos interativos `chat_card`, Agentes de Negócio conversacionais, formulários de lead, catálogos de produtos para e-commerce, rastreamento de conversões via CAPI e relatórios granulares.

### Principais Capacidades
| Área | O que o agente faz |
| :--- | :--- |
| **Estratégia & Segmentação** | Valida IDs geográficos via `/geo_lookup`, monta *context hints* semânticos, gerencia públicos personalizados SHA-256 e listas de exclusão. |
| **Produção de Criativos** | Redige títulos de alta conversão (≤50 caracteres) e descrições (≤150 caracteres), faz upload de mídia e gera previews reais no ChatGPT. |
| **Lances & Orçamento** | Converte valores para micros (`$1 = 1.000.000`), define lances `cpc` ou `conversions` e cria travas de gasto (*spend limit windows*). |
| **Geração de Leads** | Publica formulários nativos no chat e conecta Webhooks `lead_sync_subscriptions` a qualquer CRM (GoHighLevel, Kommo, n8n). |
| **Business Agents** | Cria e publica agentes conversacionais com ferramentas personalizadas atreladas aos anúncios. |
| **E-commerce** | Gerencia catálogos dinâmicos com ingestão delta NDJSON em tempo real. |
| **Mensuração** | Configura Pixel Web + CAPI Server-to-Server com deduplicação via `event_id`/`obref`. |
| **Otimização** | Extrai métricas por dia, dispositivo, país e produto; escala vencedores e corta desperdício. |

### Instalação
```bash
hermes skills install openai-ads
```

### Configuração
```bash
export OPENAI_ADS_API_KEY="<sua-api-key>"
```

### Início Rápido
```bash
SKILL_DIR=~/.hermes/skills/productivity/openai-ads

# 1. Verificar a conta de anúncios
python3 $SKILL_DIR/scripts/openai_ads_cli.py account

# 2. Criar uma campanha
python3 $SKILL_DIR/scripts/openai_ads_cli.py campaign-create \
  --name "Geracao de Leads Q1" --objective conversions --locations BR

# 3. Criar grupo de anúncios com context hints semânticos
python3 $SKILL_DIR/scripts/openai_ads_cli.py adgroup-create \
  --campaign-id camp_123 --name "Alta Intencao de Compra" \
  --bid-amount 2500000 --daily-budget 50000000 \
  --hints "melhor crm para pequenas empresas,automatizar atendimento com ia"

# 4. Criar o anúncio interativo chat_card
python3 $SKILL_DIR/scripts/openai_ads_cli.py ad-create \
  --ad-group-id adg_456 --name "Criativo Principal" \
  --headline "Escale Vendas Com IA" \
  --body "Automatize a qualificacao de leads 24/7 e nunca perca um cliente." \
  --cta LEARN_MORE --url "https://exemplo.com/oferta"

# 5. Preview e métricas
python3 $SKILL_DIR/scripts/openai_ads_cli.py preview --ad-id ad_789
python3 $SKILL_DIR/scripts/openai_ads_cli.py insights --entity campaigns --id camp_123 --breakdown device
```

### Arquivos Incluídos
- `SKILL.md` — playbook operacional completo para o agente
- `scripts/openai_ads_cli.py` — motor de automação CLI executável
- `scripts/openai_ads_client.py` — biblioteca cliente Python reutilizável
- `references/api_reference.md` — todos os 70 endpoints mapeados (EN/PT)
- `references/chat_card_best_practices.md` — melhores práticas de criativo e rastreamento

---

## 👤 Author / Autor

**Rafa Martins**

- 🌐 Website: [portal.sthub.com.br](https://portal.sthub.com.br)
- ✉️ Email: rafacpti@gmail.com
- 💬 WhatsApp: [wa.me/5527999082624](https://wa.me/5527999082624)
- 🐙 GitHub: [@rafacpti23](https://github.com/rafacpti23)

## 📄 License / Licença

MIT — see [LICENSE](./LICENSE).
