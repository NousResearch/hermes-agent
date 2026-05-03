---
id: references/operational-auth-and-keys.md
name: 'Auth and API Keys'
description: 'Alchemy uses API keys for most products. Keep keys server-side and scope them to environments.'
tags:
  - alchemy
  - operational
  - operations
related:
  - operational-jwt-and-header-auth.md
  - operational-allowlists.md
updated: 2026-02-05
---
# Auth and API Keys

## Summary
Alchemy uses API keys for most products. Keep keys server-side and scope them to environments.

## Key Practices
- Store keys in environment variables (`ALCHEMY_API_KEY`).
- Use separate keys for dev/staging/prod.
- Rotate keys if compromised.

## Example Usage
```bash
export ALCHEMY_API_KEY="..."
curl -s https://eth-mainnet.g.alchemy.com/v2/$ALCHEMY_API_KEY \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":1,"method":"eth_blockNumber","params":[]}'
```

## Related Files
- `operational-jwt-and-header-auth.md`
- `operational-allowlists.md`

## Official Docs
- [Create an API Key](https://www.alchemy.com/docs/create-an-api-key)
- [Access Keys](https://www.alchemy.com/docs/how-to-create-access-keys)
