# ChatGPT Ads & Chat Card Best Practices
# Melhores Práticas para Anúncios Interativos no ChatGPT

---

## 🇺🇸 English Guide

### 1. The Anatomy of a `chat_card`
A ChatGPT ad creative (`chat_card`) is delivered naturally in conversational streams. It requires succinct, high-value content.

- **Headline (Max 50 chars):** Clear, benefit-driven, concise.
- **Body (Max 150 chars):** Solution-oriented narrative matching conversational context.
- **Call To Action (CTA):** Actionable verbs (`LEARN_MORE`, `SHOP_NOW`, `SIGN_UP`, `GET_OFFER`).
- **Media File (Image/Banner):** High-contrast, clean 1200x628 or square visual without excessive text overlay.

### 2. Context Hints Strategy
Context hints replace legacy keyword auctions by injecting semantic topics into the relevance engine:
- Use **3 to 10 specific semantic phrases** per Ad Group.
- Combine intent descriptors (e.g., `"best crm for real estate"`, `"automate customer service"`).
- Avoid overly generic terms (e.g., `"software"`, `"help"`) to reduce unqualified impressions.

### 3. Conversion Tracking & CAPI Deduplication
Always implement hybrid tracking with browser Pixel and Server-to-Server CAPI:
- Generate unique `event_id` per user action (e.g., checkout/lead submission).
- Send the `event_id` and browser `obref` token to `/conversions/events`.
- SHA-256 hash all customer identifying properties (`email_sha256`, `phone_sha256`).

---

## 🇧🇷 Guia em Português

### 1. Anatomia de um `chat_card`
Os anúncios no ChatGPT (`chat_card`) aparecem no fluxo natural da conversa. Exigem redação direta e de alto valor percebido.

- **Headline / Título (Máx 50 caracteres):** Direto, focado no benefício central da oferta.
- **Body / Descrição (Máx 150 caracteres):** Tom natural, explicativo e convidativo.
- **Call To Action (CTA):** Verbos claros de ação (`LEARN_MORE`, `SHOP_NOW`, `SIGN_UP`, `GET_OFFER`).
- **Imagem de Apoio:** 1200x628 px ou 1:1, limpa e com contraste alto, sem poluição visual.

### 2. Estratégia de Context Hints
Os *context hints* substituem as palavras-chave tradicionais por semântica de intenção:
- Insira de **3 a 10 termos semânticos específicos** por Grupo de Anúncios.
- Foque na dor ou solução desejada (ex: `"como escalar vendas pelo whatsapp"`, `"automação de atendimento ia"`).
- Evite termos excessivamente amplos para não desperdiçar verba com tráfego desqualificado.

### 3. Deduplicação CAPI + Pixel
Implemente sempre a mensuração híbrida com deduplicação:
- Crie um `event_id` único para cada conversão no seu site/aplicação.
- Repasse o `event_id` e o token de clique `obref` no payload da Conversions API.
- Criptografe e-mails e telefones em SHA-256 antes do disparo.
