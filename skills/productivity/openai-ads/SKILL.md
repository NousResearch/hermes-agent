---
name: openai-ads
version: 1.0.0
description: "Autonomous AI Marketing & Ads Manager for OpenAI Ads API v1. End-to-end management for ChatGPT Ads: campaigns, context hints, interactive chat_cards, CAPI, custom audiences, product feeds, and insights."
credits: Rafa Martins (portal.sthub.com.br | rafacpti@gmail.com)
tags: [openai-ads, chatgpt-ads, marketing, advertising, traffic-management, conversions-api, business-agents, capi]
---

# OpenAI Ads & ChatGPT Marketing Manager / Gestor de Anúncios OpenAI & ChatGPT

Bilingual Operational Playbook (English & Português) for autonomous media buying, campaign orchestration, creative generation, conversion tracking, and analytics using the **OpenAI Ads API v1** (`https://api.ads.openai.com/v1`).

---

## 🇺🇸 English: Capabilities & Playbook

### 1. The Role of the Agent
When managing OpenAI Ads, you act as an **Elite Media Buyer & Growth Marketer**. You manage the full lifecycle of advertising in ChatGPT:
1. **Account & Governance:** Inspect balances, set brand assets (`/ad_account/brand`), configure negative keyword protection, and schedule spend limit windows.
2. **Campaign Orchestration:** Structure campaigns with objectives (`traffic`, `conversions`, `lead_generation`, `app_installs`, `brand_awareness`), validate geo targeting via `/geo_lookup`, and allocate daily/lifetime budgets in micros (`$1.00 = 1,000,000 micros`).
3. **Ad Group & Context Hints:** Build semantic relevance matrices instead of old-school keyword lists. Use high-intent phrases that natural ChatGPT conversations evoke.
4. **Interactive Creatives (`chat_card`):** Draft succinct, high-converting copy (Headline ≤ 50 chars, Body ≤ 150 chars, clear CTA, high-resolution media), generate live ChatGPT previews (`/ads/{id}/preview`), and monitor review status.
5. **Business Agents & Lead Capture:** Attach conversational Business Agents (`/business_agents`) to ads or deploy native chat Lead Forms wired via Webhooks to CRM pipelines.
6. **E-commerce & Catalogs:** Manage Product Feeds (`/feeds`), stream NDJSON catalog deltas, and verify item eligibility.
7. **Attribution & CAPI:** Create Web Pixels, generate CAPI access tokens, and send Server-to-Server conversion events with `event_id`/`obref` deduplication.
8. **Optimization:** Analyze metrics across days, countries, devices, and products; pause losing ad groups and scale winning creatives.

---

### 2. Available Scripts & Automation Tools

The skill includes pre-built Python CLI and client tools under `scripts/`:

```bash
SKILL_DIR=~/.hermes/skills/productivity/openai-ads

# Check account status and limits
python3 $SKILL_DIR/scripts/openai_ads_cli.py account

# Create a campaign
python3 $SKILL_DIR/scripts/openai_ads_cli.py campaign-create \
  --name "SaaS Acquisition Q1" \
  --objective conversions \
  --locations US,CA,GB

# Create an ad group with semantic context hints
python3 $SKILL_DIR/scripts/openai_ads_cli.py adgroup-create \
  --campaign-id camp_01JXYZ \
  --name "AI Automation Seekers" \
  --bid-amount 2500000 \
  --daily-budget 50000000 \
  --hints "best ai agent for business,automate lead triage"

# Create a chat_card creative
python3 $SKILL_DIR/scripts/openai_ads_cli.py ad-create \
  --ad-group-id adg_01JABC \
  --name "Value Prop Variant A" \
  --headline "Automate Your Pipeline" \
  --body "Qualify leads 24/7 with autonomous AI workers." \
  --cta LEARN_MORE \
  --url "https://example.com/demo"

# Generate ChatGPT preview link
python3 $SKILL_DIR/scripts/openai_ads_cli.py preview --ad-id ad_01JDEF

# Ingest e-commerce catalog via streaming NDJSON
python3 $SKILL_DIR/scripts/openai_ads_cli.py feed-bulk \
  --feed-id feed_01JGHI \
  --file products_delta.ndjson

# Pull performance metrics
python3 $SKILL_DIR/scripts/openai_ads_cli.py insights \
  --entity campaigns \
  --id camp_01JXYZ \
  --breakdown device
```

---

## 🇧🇷 Português: Capacidades e Playbook Operacional

### 1. O Papel do Agente
Ao gerenciar OpenAI Ads, você atua como um **Gestor de Tráfego e Media Buyer de Alta Performance**. Você domina o ciclo completo de anúncios no ChatGPT:
1. **Governança da Conta:** Consulta saldos e limites, atualiza ativos da marca (`/ad_account/brand`), define palavras-chave negativas globais e programa janelas de teto de gastos (*spend limit windows*).
2. **Estruturação de Campanhas:** Define objetivos estratégicos (`traffic`, `conversions`, `lead_generation`, `app_installs`, `brand_awareness`), valida regiões pelo `/geo_lookup` e aloca orçamentos em micros (`$1,00 = 1.000.000 micros`).
3. **Grupos & Context Hints:** Substitui listas mecânicas de palavras-chave por redes semânticas contextuais de alta intenção alinhadas ao comportamento dos usuários no ChatGPT.
4. **Criativos Interativos (`chat_card`):** Cria textos persuasivos e concisos (Título ≤ 50 caracteres, Texto ≤ 150 caracteres, CTA orientada à ação), gera previews reais no ChatGPT (`/ads/{id}/preview`) e monitora o status de aprovação.
5. **Business Agents & Captação de Leads:** Conecta agentes conversacionais interativos diretamente aos anúncios e cria formulários nativos no chat integrados via Webhook a CRMs.
6. **E-commerce & Catálogos:** Gerencia feeds de produtos (`/feeds`), envia deltas em streaming NDJSON e valida a elegibilidade dos itens.
7. **Atribuição & CAPI:** Cria Pixels Web, gera tokens de CAPI e envia eventos Server-to-Server com deduplicação de `event_id` e token `obref`.
8. **Otimização Contínua:** Analisa métricas por dia, país, dispositivo e produto; pausa criativos ou grupos fracos e escala o que gera ROI positivo.

---

## 📐 Reference Table: Financial Micros / Tabela de Conversão de Micros

| Valor em Dólares ($) | Valor em Micros (API) | Aplicação Comum |
| :--- | :--- | :--- |
| `$0.50` | `500000` | Lance CPC moderado |
| `$1.00` | `1000000` | Lance base CPC |
| `$2.50` | `2500000` | Lance CPC competitivo |
| `$20.00` | `20000000` | Orçamento diário inicial por grupo |
| `$50.00` | `50000000` | Orçamento diário padrão |
| `$500.00` | `500000000` | Teto de gasto mensal / janela |

---

## 👨💻 Author & Contact / Autor e Contato

- **Author / Criador:** Rafa Martins
- **Website:** [portal.sthub.com.br](https://portal.sthub.com.br)
- **Email:** rafacpti@gmail.com
- **WhatsApp:** [+55 27 99908-2624](https://wa.me/5527999082624)
- **GitHub:** [@rafacpti23](https://github.com/rafacpti23)
