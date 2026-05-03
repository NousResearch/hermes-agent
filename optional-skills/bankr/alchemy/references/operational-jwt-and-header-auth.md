---
id: references/operational-jwt-and-header-auth.md
name: 'JWT and Header-Based Auth'
description: 'Some Alchemy APIs use header-based auth (e.g., JWTs or API tokens) instead of API keys in URLs. Use this for server-side integrations where you need stronger access control.'
tags:
  - alchemy
  - operational
  - operations
related:
  - operational-auth-and-keys.md
  - operational-allowlists.md
updated: 2026-02-05
---
# JWT and Header-Based Auth

## Summary
Some Alchemy APIs use header-based auth (e.g., JWTs or API tokens) instead of API keys in URLs. Use this for server-side integrations where you need stronger access control.

## Guidance
- Keep auth tokens server-side only.
- Use the header format specified in the product docs.
- Rotate tokens periodically.

## Example Pattern
```bash
curl -s https://api.g.alchemy.com/your-endpoint \
  -H "Authorization: Bearer $ALCHEMY_JWT" \
  -H "Content-Type: application/json"
```

## Related Files
- `operational-auth-and-keys.md`
- `operational-allowlists.md`

## Official Docs
- [Access Keys](https://www.alchemy.com/docs/how-to-create-access-keys)
