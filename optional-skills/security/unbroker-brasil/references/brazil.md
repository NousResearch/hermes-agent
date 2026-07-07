# Brazil exposure and removal playbook

## Purpose

Brazil does not have the same people-search broker ecosystem as the United States, but Brazilian personal data is commonly exposed through search-indexed pages, telephone/address directories, legal/court mirrors, CNPJ/MEI mirrors, leaked snippets, social/profile mirrors, and scraping/enrichment sites. This playbook extends the unbroker workflow to those sources.

## Search vectors

Run exact and variant searches. Keep evidence grouped by vector.

### Identity vectors

- Full legal name in quotes.
- Name without accents.
- Common abbreviations and middle-name omissions.
- Aliases/usernames.
- E-mails in quotes.
- Phone numbers in national format, international format, with and without spaces/punctuation.
- Address fragments in quotes.
- City/state combinations.

### Query patterns

Use SearXNG/web search and ordinary search engines when available:

- `"FULL NAME" "CITY"`
- `"FULL NAME" "PHONE"`
- `"FULL NAME" "EMAIL"`
- `"PHONE"`
- `"EMAIL"`
- `"ALIAS" "CITY"`
- `"FULL NAME" "CPF"` only as a search string; never request or disclose CPF automatically.
- `"FULL NAME" site:jusbrasil.com.br`
- `"FULL NAME" site:escavador.com`
- `"FULL NAME" site:diariooficial.*`
- `"FULL NAME" site:telelistas.net`
- `"PHONE" "São Paulo"` or relevant city.
- `"ADDRESS FRAGMENT" "FULL NAME"`

## Source classes

### Search engines and snippets

Record the result URL, snippet, search engine, query, and date. If the page is gone but snippet remains, use search engine removal/de-indexing tools. If the source page is alive, remove at source first, then de-index.

### Directories and phone/address lookup sites

Treat as broker-like. If an opt-out form exists, use it. If not, send LGPD request to privacy/DPO/support/legal/contact e-mail. Disclose only name, city/state, and the URL; add phone/e-mail only if the page already exposes it or it is required to identify the record.

### Jusbrasil, Escavador, Diário Oficial mirrors

These often mirror public legal/court material. Do not claim the official public record can be deleted. Request suppression from the mirror or de-indexing if the page exposes unnecessary personal data. Queue legal/human task if court correction or official removal is required.

### CNPJ/MEI and business mirrors

Many sites mirror Receita Federal public business data. If the page exposes a private residential address, personal phone, stale contact, or extra scraped data, request correction/suppression from the mirror. Official Receita data changes require the user's own business/account/legal process, not autonomous deletion.

### Social/profile mirrors

If the user controls the original account, prefer removing or changing the original profile first. Then request mirror removal and de-indexing. Do not reset accounts or change credentials through this skill.

### Leaks, pastes, and breach mirrors

Do not download or expand leaked datasets. Capture only public page URL/snippet and request removal from the host/platform/search engine. If the site demands payment or identity documents, queue a human task.

## Evidence schema

Use JSON evidence like:

```json
{
  "jurisdiction": "BR",
  "source_class": "search_snippet|directory|legal_mirror|cnpj_mirror|profile_mirror|leak_indicator|other",
  "query": "redacted or exact search query used",
  "listing_urls": ["https://example.com/result"],
  "visible_fields": ["name", "city", "phone"],
  "match_basis": ["exact_name", "phone", "city"],
  "action_channel": "form|email|search_deindex|human",
  "notes": "short factual note, no secrets"
}
```

## State selection

- `found`: source page or snippet clearly identifies the subject.
- `indirect_exposure`: search result or third-party mirror suggests exposure but current page is gated, paid, removed, or partial.
- `not_found`: direct source search or guided form confirms no match.
- `blocked`: hard anti-bot, login wall, payment wall, unavailable JS flow, or source cannot be inspected with available tools.
- `human_task_queued`: source demands CPF/RG/CNH, notarized proof, phone call, manual account action, court/legal filing, or disclosure beyond approved fields.

## LGPD e-mail procedure

1. Locate privacy/LGPD/DPO/Encarregado/contact/legal/abuse address.
2. Use `templates/emails/lgpd-eliminacao-ptbr.txt`.
3. Include only: requester name, contact e-mail, URL, visible fields, and requested action.
4. Do not attach documents unless the user explicitly approves the specific attachment.
5. Record `submitted` or `awaiting_processing` depending on the channel.
6. Re-scan after the legal/operational window.

## Search engine de-indexing

Use after source removal or when the source is unreachable but the search result itself exposes personal data. Record as `indirect_exposure` until the search result disappears.

Typical targets:

- Google outdated content/removal tools.
- Bing content removal.
- Platform-specific privacy/report pages.

## Human digest items

Queue into the final digest instead of interrupting the run when:

- Government ID is demanded.
- CAPTCHA/anti-bot cannot be passed normally.
- A court/tribunal official correction is needed.
- The source requires account login controlled by the user.
- Payment/fax/phone call is required.
- The page is in a legal gray area and should be reviewed before submission.
