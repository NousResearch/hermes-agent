# Hermes GPT Plugin

This plugin packages `asimons81/hermes-gpt` as a bundled Hermes Agent plugin.
It runs a local-dev MCP sidecar that exposes selected Hermes Agent capabilities
to MCP clients without adding new Hermes core model tools.

Enable it first:

```bash
hermes plugins enable hermes-gpt
```

Check the tool surface:

```bash
hermes hermes-gpt status
```

Start stdio mode:

```bash
hermes hermes-gpt serve
```

Start local HTTP mode:

```bash
hermes hermes-gpt serve --http --host 127.0.0.1 --port 7677
```

Default visible MCP tools are read-only or local metadata oriented:
`hermes_read_file`, `hermes_search_files`, `hermes_memory` search,
`hermes_skill_list`, and `hermes_skill_view`.

High-risk tools remain opt-in through environment variables:
`HERMES_GPT_ENABLE_WRITE=1`, `HERMES_GPT_ENABLE_MEMORY_WRITE=1`,
`HERMES_GPT_ENABLE_TERMINAL=1`, and
`HERMES_GPT_ENABLE_SESSION_SEARCH=1`.

Do not expose this server publicly without real authentication.
