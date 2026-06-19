# US 期权 Screener Options

- Category key: `options`

Options are not stock_filter fields. Use option-chain/surface enrichment after candidate discovery.

## Choices

### 期权活跃度/IV/到期日/Put-Call context

- Capability: `option_chain_enrichment`
- Alternate source: `get_option_expiration_date + get_option_chain + option snapshots`
- LLM hint: Use later to assess tradability and income-strategy readiness, not as stock_filter_specs.
