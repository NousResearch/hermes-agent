# OpenAI Ads API (v1) - Complete Reference & Endpoints Guide
# Guia Completo de Referência da API OpenAI Ads (v1)

This reference catalogs all 19 functional domains, 70 endpoints, and operations available in the **OpenAI Ads API v1** (`https://api.ads.openai.com/v1`).

---

## 🌐 1. Base URL & Authentication / Autenticação

- **Base URL:** `https://api.ads.openai.com/v1`
- **Header:** `Authorization: Bearer <OPENAI_ADS_API_KEY>`
- **Financial Units:** Micros (`1 USD = 1,000,000 micros`). Example: `$25.00 = 25000000`.
- **Rate Limits:** 600 req/min per endpoint; 1,200 req/min account-wide; Bulk API: 10 req/10s.

---

## 📂 2. Endpoint Breakdown by Functional Area / Mapeamento por Domínio

### 1. Ad Account & Spending Limits (`/ad_account`)
| Method | Endpoint | Description (EN) | Descrição (PT-BR) |
| :--- | :--- | :--- | :--- |
| `GET` | `/ad_account` | Retrieve ad account metadata & status | Obter metadados e status da conta de anúncios |
| `POST` | `/ad_account/brand` | Update brand name, website and brand assets | Atualizar nome da marca, site e ativos |
| `POST` | `/ad_account/negative_keywords` | Set global negative keywords for eligibility | Definir palavras-chave negativas globais |
| `POST` | `/ad_account/activate` | Activate the ad account | Ativar a conta de anúncios |
| `POST` | `/ad_account/pause` | Pause the ad account | Pausar a conta de anúncios |
| `GET` | `/ad_account/spend_limit_windows` | List spend limit windows | Listar janelas de limite de gastos |
| `POST` | `/ad_account/spend_limit_windows` | Create scheduled spend limit window | Criar janela com teto de gastos |
| `POST` | `/ad_account/spend_limit_windows/{id}` | Edit spend limit window | Editar janela de limite de gastos |
| `POST` | `/ad_account/spend_limit_windows/{id}/delete` | Delete spend limit window | Deletar janela de teto de gastos |
| `GET` | `/ad_account/insights` | Account-level aggregated metrics | Métricas consolidadas da conta |

### 2. Campaigns (`/campaigns`)
| Method | Endpoint | Description (EN) | Descrição (PT-BR) |
| :--- | :--- | :--- | :--- |
| `GET` | `/campaigns` | List all campaigns | Listar todas as campanhas |
| `POST` | `/campaigns` | Create a new campaign | Criar nova campanha |
| `GET` | `/campaigns/{id}` | Get campaign details | Obter detalhes da campanha |
| `POST` | `/campaigns/{id}` | Update campaign parameters | Atualizar dados da campanha |
| `POST` | `/campaigns/{id}/activate` | Activate campaign | Ativar campanha |
| `POST` | `/campaigns/{id}/pause` | Pause campaign | Pausar campanha |
| `POST` | `/campaigns/{id}/archive` | Archive campaign | Arquivar campanha |
| `GET` | `/campaigns/{id}/insights` | Performance insights for campaign | Relatório de performance da campanha |

### 3. Ad Groups (`/ad_groups`)
| Method | Endpoint | Description (EN) | Descrição (PT-BR) |
| :--- | :--- | :--- | :--- |
| `GET` | `/ad_groups` | List ad groups | Listar grupos de anúncios |
| `POST` | `/ad_groups` | Create ad group with context hints | Criar grupo de anúncios com context hints |
| `GET` | `/ad_groups/{id}` | Get ad group details | Obter detalhes do grupo |
| `POST` | `/ad_groups/{id}` | Update bidding or flight dates | Atualizar lances ou datas |
| `POST` | `/ad_groups/{id}/activate` | Activate ad group | Ativar grupo de anúncios |
| `POST` | `/ad_groups/{id}/pause` | Pause ad group | Pausar grupo de anúncios |
| `POST` | `/ad_groups/{id}/archive` | Archive ad group | Arquivar grupo |
| `GET` | `/ad_groups/{id}/insights` | Ad group performance breakdown | Métricas de performance do grupo |

### 4. Ads & Creatives (`/ads`)
| Method | Endpoint | Description (EN) | Descrição (PT-BR) |
| :--- | :--- | :--- | :--- |
| `GET` | `/ads` | List ads in account | Listar anúncios da conta |
| `POST` | `/ads` | Create native chat_card ad | Criar anúncio interativo chat_card |
| `GET` | `/ads/{id}` | Get ad metadata and review status | Obter dados do anúncio e status de revisão |
| `POST` | `/ads/{id}` | Update ad creative copy/links | Atualizar copy ou links do anúncio |
| `POST` | `/ads/{id}/activate` | Activate ad | Ativar anúncio |
| `POST` | `/ads/{id}/pause` | Pause ad | Pausar anúncio |
| `POST` | `/ads/{id}/archive` | Archive ad | Arquivar anúncio |
| `POST` | `/ads/{id}/preview` | Generate live ChatGPT UI preview | Gerar preview interativo no ChatGPT |
| `GET` | `/ads/{id}/insights` | Creative performance metrics | Métricas de performance do anúncio |

### 5. Custom Audiences (`/custom_audiences`)
| Method | Endpoint | Description (EN) | Descrição (PT-BR) |
| :--- | :--- | :--- | :--- |
| `GET` | `/custom_audiences` | List custom audiences | Listar públicos personalizados |
| `POST` | `/custom_audiences` | Create new SHA-256 audience | Criar público com hashing SHA-256 |
| `GET` | `/custom_audiences/{id}` | Audience status and size | Status e tamanho do público |
| `POST` | `/custom_audiences/{id}/add` | Add SHA-256 hashed users | Adicionar usuários criptografados |
| `POST` | `/custom_audiences/{id}/remove` | Remove hashed users | Remover usuários da lista |
| `POST` | `/custom_audiences/{id}/replace` | Full audience replacement | Substituir base completa de público |
| `POST` | `/custom_audiences/{id}/merge` | Merge another audience | Mesclar com outro público |
| `POST` | `/custom_audiences/{id}/archive` | Archive custom audience | Arquivar público personalizado |

### 6. Product Feeds (`/feeds`)
| Method | Endpoint | Description (EN) | Descrição (PT-BR) |
| :--- | :--- | :--- | :--- |
| `GET` | `/feeds` | List product feeds | Listar feeds de produtos (catálogos) |
| `POST` | `/feeds` | Create product feed | Criar feed de produtos |
| `GET` | `/feeds/{id}` | Feed metadata & ingestion status | Metadados e status de ingestão |
| `POST` | `/feeds/{id}/items/bulk` | Ingest streaming NDJSON delta feed | Ingestão de produtos via NDJSON streaming |
| `GET` | `/feeds/{id}/eligibility` | Check product ad eligibility | Verificar elegibilidade dos itens |

### 7. Conversions API & Pixel (`/conversions`)
| Method | Endpoint | Description (EN) | Descrição (PT-BR) |
| :--- | :--- | :--- | :--- |
| `GET` | `/conversions/pixels` | List web tracking pixels | Listar pixels web configurados |
| `POST` | `/conversions/pixels` | Create web tracking pixel | Criar novo pixel web |
| `POST` | `/conversions/api_keys` | Generate CAPI Server Token | Gerar token para Conversions API (CAPI) |
| `POST` | `/conversions/events` | Send Server-to-Server conversion events | Enviar eventos de conversão CAPI |
| `GET` | `/conversions/insights` | Conversion attribution analytics | Relatório de atribuição de conversões |

### 8. Lead Forms & Subscriptions (`/lead_forms`, `/lead_sync_subscriptions`)
| Method | Endpoint | Description (EN) | Descrição (PT-BR) |
| :--- | :--- | :--- | :--- |
| `GET` | `/lead_forms` | List lead forms | Listar formulários de lead |
| `POST` | `/lead_forms` | Create native chat lead form | Criar formulário nativo no chat |
| `POST` | `/lead_forms/{id}/publish` | Publish lead form | Publicar formulário de captação |
| `POST` | `/lead_sync_subscriptions` | Subscribe Webhook for live CRM sync | Configurar Webhook de envio em tempo real |

### 9. Business Agents (`/business_agents`)
| Method | Endpoint | Description (EN) | Descrição (PT-BR) |
| :--- | :--- | :--- | :--- |
| `GET` | `/business_agents` | List conversational business agents | Listar agentes de negócio |
| `POST` | `/business_agents` | Create draft business agent | Criar agente de negócio para anúncios |
| `POST` | `/business_agents/{id}/publish` | Publish agent to live ads | Publicar agente nos anúncios |
| `GET` | `/business_agent_tools` | List enabled custom toolsets | Listar ferramentas integradas ao agente |
