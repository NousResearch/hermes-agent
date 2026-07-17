# Translation API Reference

Use Shopify CLI stored OAuth only. Do not request Admin API tokens, Client IDs, app secrets, or automation tokens.

## Required scopes

```text
read_locales,write_locales,read_markets,write_markets,read_translations,write_translations
```

## Read → translate → review → write

1. Fetch `translatableResources` for the requested type and locale.
2. Translate the eligible fields into a JSON candidate file:

```json
[
  { "resourceId": "gid://shopify/Product/123", "key": "title", "translation": "Example" }
]
```

3. Run `generate-audit --input <fetch.json> --translations <candidates.json>`.
4. Let the merchant review the generated CSV and explicitly approve the write.
5. Run `write`, then `verify-translations`.

## Translation rules

- Preserve meaning and HTML structure; never translate handles, URLs, Liquid, or technical JSON fields.
- A non-CJK translation under 80% of the source length is a warning.
- A Chinese, Japanese, or Korean translation under 60% is a warning.
- A warning requires human review; it does not automatically reject valid concise copy.

## Shopify API constraints

- `translatableResources` is paginated; retain the digest returned with each field.
- `translationsRegister` requires the current digest and returns `userErrors`; treat any user error as a failed resource write.
- Re-fetch before retrying a digest mismatch because the source may have changed.
- Use GraphQL variables for locales, resource IDs, keys, digests, and values.

Official references:

- [Translatable resources](https://shopify.dev/docs/api/admin-graphql/latest/queries/translatableresources)
- [Register translations](https://shopify.dev/docs/api/admin-graphql/latest/mutations/translationsRegister)
