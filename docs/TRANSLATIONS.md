# Documentation translations

This repository uses one canonical English source for project documentation and
localized documentation for readers who prefer another language.

## Canonical sources

The English versions are authoritative for project behavior, contribution
instructions, and security reporting:

- `README.md`
- `CONTRIBUTING.md`
- `SECURITY.md`
- `website/docs/`

A translation must not silently change commands, configuration keys, file paths,
URLs, security requirements, or legal meaning. When a translation differs from
English, the English source and the documented behavior take precedence.

## Where translations belong

Use the documentation site's Docusaurus i18n tree for full documentation
translations:

```text
website/i18n/<locale>/docusaurus-plugin-content-docs/current/
```

This keeps the source pages in `website/docs/`, gives users a language-aware
site navigation experience, and avoids maintaining a second root-level copy of
every documentation page.

Root-level translated files such as `README.<locale>.md` may remain as GitHub
entry points when they are useful for repository discovery. They should be kept
short, link to the translated documentation, and clearly identify the English
source. Do not add a new root-level full translation when the same content can
be served by the website i18n tree.

## Security policy translations

`SECURITY.md` must remain at the repository root so GitHub can discover the
security policy. It is the authoritative version.

A translated security policy may be provided for accessibility, but it must
state that the English `SECURITY.md` controls if the two versions differ. Do not
move or replace the root security policy with a locale-specific file.

## Translation workflow

1. Update the canonical English document first.
2. Update affected translations in the same change when practical.
3. If a translation cannot be updated immediately, mark it as needing review
   rather than presenting it as current.
4. Preserve code blocks, commands, configuration keys, environment variables,
   file paths, and URLs unless a language-specific explanation is required.
5. Run the documentation link checks and the website build before submitting a
   translation change.

Translation PRs should identify:

- the canonical English source pages;
- the locale being added or updated;
- whether the translation is complete or intentionally partial;
- any known differences from the English source.

## Adding a new locale

Before adding a new locale, search open issues and pull requests for existing
work. Prefer one coordinated locale owner and a staged, reviewable scope over a
large unmaintainable copy of the entire documentation tree.

For a new website locale, follow the existing Docusaurus configuration and
include the locale in the site's language navigation. Keep generated skill
catalog pages on their existing generation path instead of maintaining manual
translated copies that will immediately drift.

## Scope of this policy

This policy does not remove existing translations. It defines the direction for
future documentation work and provides a migration path for existing root-level
copies without breaking GitHub discovery or security reporting.
