# Cursor SDK Plugin

This bundled plugin registers `cursor_agent`, a delegation tool that runs a
focused coding task through the official Cursor Python SDK.

Cursor is treated as a separate coding-agent runtime, not as a Hermes model
provider. Hermes keeps its own conversation loop and tools; this plugin hands a
specific task to Cursor and returns the final Cursor run text plus runtime
metadata.

## Setup

Hermes can lazy-install the optional SDK on first use when
`security.allow_lazy_installs` is enabled. To install it manually in the
environment running Hermes:

```bash
pip install "cursor-sdk==0.1.5"
```

Set a Cursor API key:

```bash
export CURSOR_API_KEY=...
```

The plugin is registered by the bundled plugin loader, but the `cursor_sdk`
toolset is opt-in. The tool is only exposed to models after the toolset is
enabled and `CURSOR_API_KEY` is set.

## Usage

Enable the `cursor_sdk` toolset, then call `cursor_agent` with a prompt. Local
runs default to the current Hermes working directory and require
`terminal.backend: local` because Cursor operates on the host workspace:

```json
{
  "prompt": "Review this repository and summarize the risky parts.",
  "runtime": "local",
  "model": "composer-2.5",
  "timeout_seconds": 900
}
```

Cloud runs can target a repository URL or pull request URL through the Cursor
SDK cloud runtime options. This first integration does not request automatic
pull request creation.
