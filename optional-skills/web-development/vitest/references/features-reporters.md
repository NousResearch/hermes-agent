---
name: reporters
description: Built-in reporters, default selection, and CI/output configuration
---

# Reporters

Select reporters via `--reporter` or `reporters` config. Configuring `reporters` **replaces** the default list — spread `configDefaults.reporters` to keep them.

```ts
import { configDefaults, defineConfig } from 'vitest/config'

export default defineConfig({
  test: {
    reporters: ['verbose', ['junit', { suiteName: 'UI tests' }]],
    // keep defaults and add one:
    // reporters: ['json', ...configDefaults.reporters],
  },
})
```

## Default Selection

When `reporters` is unset, Vitest auto-selects:

- `default` for normal terminal runs
- **`minimal` (alias `agent`)** when it detects an **AI coding agent** — only failed tests + errors, no summary or passing logs, optimized to cut token usage
- `github-actions` is added when `process.env.GITHUB_ACTIONS === 'true'`

## Built-in Reporters

| Reporter | Use |
|----------|-----|
| `default` | Summary + collapses passing files; prints full tree for single/failing file |
| `verbose` | One line per finished test (flat list in v4); only reporter that shows annotations on pass |
| `tree` | Like `default` but always shows each test (the old v3 verbose) |
| `dot` | One dot per test; details only for failures |
| `minimal` / `agent` | Failures only; best for AI/LLM workflows |
| `junit` | JUnit XML (templated, see below) |
| `json` | Jest-compatible JSON; includes `coverageMap` when coverage enabled |
| `tap` / `tap-flat` | TAP (nested / flat) |
| `html` | Interactive UI report (needs `@vitest/ui`) |
| `blob` | Serialized results for `--merge-reports` |
| `github-actions` | Workflow annotations + job summary |
| `hanging-process` | Lists processes preventing exit (debugging) |

> v4 removed the `basic` reporter (equivalent to `['default', { summary: false }]`). The old `verbose` flat behavior moved here; use `tree` for the nested view.

## Output Files

```bash
vitest --reporter=json --outputFile=./test-output.json
```

```ts
defineConfig({
  test: {
    reporters: ['junit', 'json'],
    outputFile: { junit: './junit.xml', json: './report.json' },
  },
})
```

## JUnit Templating

```ts
reporters: [['junit', {
  suiteNameTemplate: '{title}',     // {title} {filename} {basename} {displayName}
  classnameTemplate: '{classname}', // {classname} {title} {suitename} {filename} ...
  titleTemplate: '{title}',
  ancestorSeparator: ' > ',
  addFileAttribute: true,
}]]
```

`{filename}` is the **relative** path (use `{basename}` for the bare name). Templates can also be functions receiving all variables.

## HTML Report (v5 paths)

The HTML reporter writes a directory via `outputDir` (default `.vitest`); the entry is `<outputDir>/index.html`. Use `singleFile: true` for a self-contained shareable file (large; coverage not inlined).

```ts
reporters: [['html', { singleFile: true }]]
```

## Blob & Merge (CI/sharding)

Blobs default to `.vitest/blob/`. Label environments with `VITEST_BLOB_LABEL` or the reporter `label` option:

```bash
vitest run --reporter=blob --outputFile=reports/blob-1.json
vitest --merge-reports=reports --reporter=junit --reporter=default
```

Blob reports don't include file attachments — merge `attachmentsDir` (`.vitest/attachments/`) separately. `--reporter=blob`/`--merge-reports` don't work in watch mode.

## Key Points

- Configuring `reporters` replaces defaults — spread `configDefaults.reporters` to keep them
- The `minimal`/`agent` reporter is auto-selected for AI agents and minimizes token usage
- Use `tree` for the nested per-test view (old v3 `verbose`)
- v5 artifacts (blob, attachments, HTML) live under `.vitest/`

<!--
Source references:
- https://vitest.dev/guide/reporters
- https://vitest.dev/config/reporters
-->
