# Google Ads Routing Reference

> **Safety:** All write operations (POST, PUT, PATCH, DELETE) require explicit user confirmation before execution. Verify the target resource and intended effect with the user first. See the main [SKILL.md](../SKILL.md#security--permissions) for full security policy.

**App name:** `google-ads`
**Base URL proxied:** `googleads.googleapis.com`

## API Path Pattern

```
/google-ads/v23/customers/{customerId}/{endpoint}
```

## Common Endpoints

### List Accessible Customers
```bash
GET /google-ads/v23/customers:listAccessibleCustomers
```

Example:

```bash
maton google-ads account list
```

### Search (GAQL Query)
```bash
POST /google-ads/v23/customers/{customerId}/googleAds:search
Content-Type: application/json

{
  "query": "SELECT campaign.id, campaign.name, campaign.status FROM campaign ORDER BY campaign.id"
}
```

Example:

```bash
maton google-ads query -c 1234567890 --resource campaign --fields 'campaign.id, campaign.name, campaign.status' --order-by 'campaign.id'
```

### Search Stream (for large result sets)
```bash
POST /google-ads/v23/customers/{customerId}/googleAds:searchStream
Content-Type: application/json

{
  "query": "SELECT campaign.id, campaign.name FROM campaign"
}
```

Example:

```bash
maton google-ads query-stream -c 1234567890 --resource campaign --fields 'campaign.id, campaign.name'
```

### List Campaigns

Example:

```bash
maton google-ads campaign list -c 1234567890
```

### List Keywords

Example:

```bash
maton google-ads keyword list -c 1234567890 --date-range LAST_7_DAYS -L 25 --campaign-id 99999
```

Note: Keyword queries request metrics, so they cannot be run against a manager (MCC) account directly. Run against the client customer ID under the manager, optionally with `--login-customer-id`.

## Common GAQL Queries

### List Campaigns
```sql
SELECT campaign.id, campaign.name, campaign.status, campaign.advertising_channel_type
FROM campaign
WHERE campaign.status != 'REMOVED'
ORDER BY campaign.name
```

### Campaign Performance
```sql
SELECT campaign.id, campaign.name, metrics.impressions, metrics.clicks, metrics.cost_micros, metrics.conversions
FROM campaign
WHERE segments.date DURING LAST_30_DAYS
ORDER BY metrics.impressions DESC
```

### List Ad Groups
```sql
SELECT ad_group.id, ad_group.name, ad_group.status, campaign.id, campaign.name
FROM ad_group
WHERE ad_group.status != 'REMOVED'
```

### List Keywords with Performance
```sql
SELECT ad_group_criterion.keyword.text, ad_group_criterion.keyword.match_type, metrics.impressions, metrics.clicks, metrics.cost_micros
FROM keyword_view
WHERE segments.date DURING LAST_30_DAYS
  AND ad_group_criterion.status = 'ENABLED'
ORDER BY metrics.cost_micros DESC
LIMIT 50
```

### Search Term Report
```sql
SELECT search_term_view.search_term, campaign.name, ad_group.name, metrics.impressions, metrics.clicks, metrics.conversions
FROM search_term_view
WHERE segments.date DURING LAST_30_DAYS
ORDER BY metrics.clicks DESC
```

### Account-level Performance
```sql
SELECT customer.descriptive_name, segments.date, metrics.impressions, metrics.clicks, metrics.cost_micros, metrics.conversions
FROM customer
WHERE segments.date DURING LAST_7_DAYS
```

## Mutate Operations

### Create Campaign
```bash
POST /google-ads/v23/customers/{customerId}/campaigns:mutate
Content-Type: application/json

{
  "operations": [
    {
      "create": {
        "name": "New Campaign",
        "advertisingChannelType": "SEARCH",
        "status": "PAUSED",
        "manualCpc": {},
        "campaignBudget": "customers/{customerId}/campaignBudgets/{budgetId}"
      }
    }
  ]
}
```

### Update Campaign Status
```bash
POST /google-ads/v23/customers/{customerId}/campaigns:mutate
Content-Type: application/json

{
  "operations": [
    {
      "update": {
        "resourceName": "customers/{customerId}/campaigns/{campaignId}",
        "status": "ENABLED"
      },
      "updateMask": "status"
    }
  ]
}
```

## Manager (MCC) Account Access

When accessing a customer account through a Google Ads manager (MCC) account, pass the manager's customer ID via `--login-customer-id` (CLI) or the `login-customer-id` header (direct API). The customer ID in the path is still the client account being queried.

```bash
# List campaigns in client account 1234567890 via manager 9876543210
maton google-ads campaign list -c 1234567890 --login-customer-id 9876543210
```

## Pagination

Google Ads uses token-based pagination. The CLI handles this automatically with `--paginate`:

```bash
maton google-ads campaign list -c 1234567890 --paginate
```

## Notes

- Authentication is automatic - the router injects OAuth token and developer-token headers
- Use `listAccessibleCustomers` first to get available customer IDs
- Customer IDs are 10-digit numbers (remove dashes if formatted as XXX-XXX-XXXX)
- Monetary values are in micros (divide by 1,000,000)
- Use GAQL (Google Ads Query Language) for querying
- Date ranges: `LAST_7_DAYS`, `LAST_30_DAYS`, `THIS_MONTH`, etc.
- Status values: `ENABLED`, `PAUSED`, `REMOVED`
- API version updates frequently - check release notes for latest (currently v23)

## Resources

- [API Overview](https://developers.google.com/google-ads/api/docs/start)
- [List Accessible Customers](https://developers.google.com/google-ads/api/reference/rpc/v23/CustomerService/ListAccessibleCustomers?transport=rest)
- [Search](https://developers.google.com/google-ads/api/reference/rpc/v23/GoogleAdsService/Search?transport=rest)
- [Search Stream](https://developers.google.com/google-ads/api/reference/rpc/v23/GoogleAdsService/SearchStream?transport=rest)
- [GAQL Reference](https://developers.google.com/google-ads/api/docs/query/overview)
- [Metrics Reference](https://developers.google.com/google-ads/api/fields/v23/metrics)
- [Maton CLI Manual](https://cli.maton.ai/manual)