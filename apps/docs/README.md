# Hermes documentation app

The documentation app uses [Fumadocs](https://fumadocs.dev/) with the Next.js
App Router. English content lives in `content/en`, and Simplified Chinese
translations live in `content/zh-Hans`. Public URLs remain under `/docs/` and
`/docs/zh-Hans/`.

This app has its own npm lockfile and is deliberately excluded from the root
product workspaces. Installing Hermes does not install the docs toolchain.

> **Migration status:** the development preview is available for review. The
> static-export postbuild and output validator are still being completed;
> production publishing is not ready.

## Local development

Use Node.js 26+ and npm 11.17+ with the repository's dependency policy. The
metadata generators also need `python3` and `ruamel.yaml` available on PATH.

From the repository root:

```sh
cd apps/docs
npm ci
npm run dev -- --hostname 127.0.0.1 --port 3031
```

Open <http://127.0.0.1:3031/docs/>. The `predev` hook prepares the catalogs and
LLM-readable documents. It can reuse or download the existing unified skills
index instead of crawling every registry. Missing Python dependencies can leave
empty development catalogs; that is not a valid production build.

To restart only the UI with already-generated data:

```sh
npm start -- --hostname 127.0.0.1 --port 3031
```

## Authoring

- Link between documents using relative `.md` or `.mdx` filenames, including
  fragments: `[Profiles](../user-guide/profiles.md)`. This keeps file links
  navigable on GitHub; the compiler resolves them to site routes.
- Keep stable heading IDs with `## Heading [#stable-id]`, including in translated
  pages. Titled callouts use `:::tip[Title]` followed by their body and `:::`.
- Use ordinary Markdown unless a document needs JSX. Native HTML disclosures
  are supported without converting every file to MDX.
- The dynamic MDX compiler compiles bodies on demand. Register interactive
  components in `src/mdx-components.tsx` rather than importing them inside a
  document. The final deployment is still a static export, not a runtime server.
- Maintain curated navigation in `src/lib/navigation-data.ts`. Its generated
  skills block is owned by `scripts/generate-skill-docs.py`.
- Untranslated documents fall back to English within the Chinese namespace.
- Use logical paths with Next links. The app's `/docs` base path is applied once;
  public assets and canonical URLs use the helpers in `src/lib/urls.ts`.

Generated skill pages and catalogs are rebuilt with:

```sh
python3 scripts/generate-skill-docs.py
```

## Checks

Run from `apps/docs`:

```sh
npm test
node scripts/check-source-links.mjs
python3 scripts/check_doc_links.py
npm run typecheck
```

CI also lints diagrams with `ascii-guard`. Use Mermaid fences or plain lists and
tables instead of ASCII box diagrams.

## Publishing and compatibility

`next.config.mjs` targets a static export in `out/`. The existing repository
workflows own publication to GitHub Pages and retained assets from earlier
deployments; there is no separate `yarn deploy` step.

The `build` command and CI currently reference `scripts/postbuild.mjs` and
`scripts/validate-output.mjs`. Those migration gates must be completed before
this branch is ready to publish. Validation must cover both locales, route and
anchor parity, catalogs, OAuth metadata, LLM documents and generated-file
freshness.

`public/api` also contains machine-consumed catalogs. The repository retains
`website/static/api/model-catalog.json` for released clients whose fallback URL
uses that raw GitHub path. `scripts/build_model_catalog.py` writes both copies;
keep them byte-identical. The remaining `website/` path is a compatibility
artifact, not a second documentation app.
