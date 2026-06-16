# book-to-skill (Hermes plugin)

Thin Hermes wrapper around [virgiliojr94/book-to-skill](https://github.com/virgiliojr94/book-to-skill).

Upstream is an [Agent Skills](https://github.com/agentskills/agentskills) skill that turns PDFs, EPUBs, DOCX, and other documents into structured, on-demand skills (frameworks, chapters, glossary, patterns).

## Enable

Add to `~/.hermes/config.yaml`:

```yaml
plugins:
  enabled:
    - book-to-skill
```

Or run:

```bash
hermes plugins enable book-to-skill
```

## Install upstream + link skill

```bash
hermes book-to-skill install
hermes book-to-skill status
hermes book-to-skill check    # optional extractor dependencies
```

This shallow-clones upstream into `plugins/book-to-skill/vendor/book-to-skill` (gitignored) and links it to `~/.hermes/skills/book-to-skill` so Hermes discovers `/book-to-skill`.

On Windows, if symlinks are unavailable, the plugin copies the skill tree instead.

## Usage in Hermes

Start a **new session** (or reload skills in the CLI), then:

```
/book-to-skill ~/books/my-book.pdf my-book-slug
```

Generated book skills should be written to:

```
~/.hermes/skills/<slug>/
```

(`/profiles/<name>/skills/<slug>/` when using a Hermes profile.)

## Update upstream

```bash
hermes book-to-skill install --force
```
