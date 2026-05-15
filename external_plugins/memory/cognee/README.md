# Cognee memory plugin

This directory is a standalone Hermes memory plugin. It is no longer meant to
live under the bundled `plugins/memory/` tree.

Install it by copying this folder to your Hermes home:

```bash
mkdir -p ~/.hermes/plugins
cp -R external_plugins/memory/cognee ~/.hermes/plugins/cognee
hermes config set memory.provider cognee
```

Then add the required environment variables to the active profile `.env` file.
The easiest path is:

```bash
hermes memory setup
```

You can also set the values manually:

```bash
echo "LLM_API_KEY=sk-..." >> ~/.hermes/.env
echo "LLM_PROVIDER=deepseek" >> ~/.hermes/.env
echo "LLM_BASE_URL=https://api.deepseek.com/v1" >> ~/.hermes/.env
echo "GEMINI_API_KEY=..." >> ~/.hermes/.env
echo "COGNEE_SKIP_CONNECTION_TEST=true" >> ~/.hermes/.env
```

What you need installed:

- `pip install cognee>=1.0.9`
- an OpenAI-compatible chat model backend for `LLM_API_KEY`
- a Gemini embedding key, or equivalent embedding setup supported by Cognee
- when using `gemini/gemini-embedding-001`, Hermes sets `EMBEDDING_DIMENSIONS=3072`

What the plugin exposes:

- `cognee_remember` to store durable facts
- `cognee_recall` to retrieve semantic and graph-linked memories
- `cognee_forget` to delete stored memories, guarded by `confirm=true`

Cognee also runs background flows:

- `queue_prefetch` before turns to fetch likely-relevant memories
- `sync_turn` after responses to persist useful conversation state
- `on_session_end` to flush the last part of the conversation

Notes:

- The default dataset name is `hermes_memory`.
- For faster tests, set `COGNEE_SKIP_CONNECTION_TEST=true`.
- For persistent graph storage beyond in-memory NetworkX, configure Cognee to use a durable graph backend such as Neo4j.
