# graphrag-ts MCP

`graphrag-ts` is a TypeScript GraphRAG engine and a self-hosted Streamable HTTP MCP service for Markdown corpora. This Hermes entry connects to a service that the user runs on the same machine; it does not install or host the GraphRAG server.

## Why GraphRAG

Microsoft's GraphRAG paper describes building an entity graph and community summaries from a document corpus, then using those summaries for corpus-wide, query-focused questions that ordinary retrieval-augmented generation can struggle to answer. See [From Local to Global: A Graph RAG Approach to Query-Focused Summarization](https://arxiv.org/abs/2404.16130) and the [Microsoft GraphRAG project](https://www.microsoft.com/en-us/research/project/graphrag/).

This repository implements related, inspectable capabilities rather than claiming to be Microsoft's implementation:

- `query_graph` combines semantic, keyword, and graph/community retrieval and returns source evidence for local questions.
- `query_graph_global` runs Map-Reduce over community summaries for corpus-wide questions.
- The library exposes document-level incremental APIs (`startIncrementalBuild` and `buildIncrementalRAG`). It hashes documents, skips unchanged documents, and reprocesses changed documents. The benchmark also checks that a new fact becomes retrievable after updating one document.

The current MCP `submit_index_job` path builds and atomically publishes a complete workspace snapshot. It does not expose the library's document-level incremental API as an MCP tool.

## Requirements

- Docker Engine and Docker Compose
- Bun and pnpm for corpus generation and the host-side benchmark command
- OpenAI-compatible chat endpoints for slicing and judging
- An embedding endpoint that returns exactly 768 dimensions

The Docker service bundles PostgreSQL with pgvector. Model endpoints must be reachable from the MCP container. Configure credentials in an untracked `.env` file; never commit API keys.

Corpus content is sent to the configured slice, judge, chat, and embedding endpoints during indexing and querying. Use endpoints whose data-handling terms are appropriate for the corpus; a local model endpoint keeps those requests on your own infrastructure.

## Run the service and connect Hermes

From the `graphrag-ts` repository root:

```sh
cp .env.example .env
# Edit .env: set the PostgreSQL password and reachable slice, judge, and embedding endpoints.
docker compose up -d --build --wait
curl --fail http://127.0.0.1:3000/healthz
```

The service binds to host loopback at `http://127.0.0.1:3000/mcp` by default. Keep it local: the endpoint has no authentication. Do not expose it on a public or untrusted network. If Hermes runs in a container, configure a URL reachable from the Hermes backend without disabling the MCP server's localhost Host check.

The MCP service exposes five tools. This catalog entry enables four by default: `submit_index_job`, `get_index_job`, `query_graph`, and `query_graph_global`. `remember_text` remains opt-in because it writes caller-supplied text into the index.

After this catalog entry is merged, install it and start a new Hermes session:

```sh
hermes mcp install graphrag-ts
```

The install prompt defaults to the local URL above and probes the server's tools. `submit_index_job` reads only from the read-only input mount configured for the server. `remember_text` is not enabled by default; enable it explicitly with:

```sh
hermes mcp configure graphrag-ts
```

Keep the same `workspace_id` when submitting a job, polling it, and querying. A workspace ID is an isolation/routing key, not authentication.

`submit_index_job` builds a complete snapshot from the configured input mount and replaces that workspace's active snapshot only after a successful build. A failed build leaves the previous active snapshot available. Use a dedicated workspace for experiments and remember that older snapshots are subject to the server's `MAX_RETAINED_VERSIONS` policy.

## Rebuild and query the benchmark corpus

The benchmark corpus generator downloads selected public-domain chapters from Project Gutenberg. The three-book recipe below assumes a source checkout with CRLF normalization before Sherlock's anchored chapter selection; older revisions can generate an empty Sherlock body. For a smoke test reproducible from the published source tag without that fix, use `pnpm benchmark:docker:quick` and the committed quickstart corpus.

Generate the three Markdown files from a checkout containing the normalization fix:

```sh
pnpm exec bun -e 'import { prepareLongMarkdownCorpus, describeCorpus } from "./examples/benchmark-recall/corpus.ts"; const dir = await prepareLongMarkdownCorpus(); console.log(await describeCorpus(dir));'
```

The recorded corpus contains Alice's Adventures in Wonderland, Frankenstein, and The Adventures of Sherlock Holmes. Mount it read-only and start the service:

```sh
INPUT_HOST_DIR=./examples/generated/long-markdown-corpus docker compose up -d --build --wait
```

In Hermes, submit one full index job for the mounted corpus, then poll the returned job ID until it is `succeeded` before querying:

```text
Use graphrag-ts submit_index_job with workspace_id="benchmark-recall" and path=".".
Poll get_index_job using workspace_id="benchmark-recall" and the returned job_id until status is "succeeded".
Then use query_graph in workspace "benchmark-recall":
What did Alice notice about the White Rabbit before following it down the rabbit-hole?
Include the answer and supporting source excerpts. Do not infer facts absent from the excerpts.
```

For the repeatable MCP recall benchmark, with the long corpus mounted and the MCP server running:

```sh
MCP_BENCHMARK_URL=http://127.0.0.1:3000/mcp pnpm --filter graphrag-mcp-server benchmark
```

That runner submits the mounted corpus through MCP, waits for publication, runs out-of-corpus scope probes and positive-control queries, and prints source-backed recall results. It uses lexical entity and ordered-phrase matching, not human semantic grading. For the smaller Docker smoke profile, run `pnpm benchmark:docker:quick` from the repository root; it selects the bundled quickstart corpus and three queries.

## Recorded verification

On 2026-10-19, Hermes Agent v0.22.2 installed this manifest in a temporary Hermes home, probed the live service, and registered the four default tools above. From a source worktree containing the CRLF normalization fix, a clean MCP index job published a snapshot containing exactly three sources and 156,899 characters. A Hermes `query_graph` call then answered the Alice question with source excerpts identifying the White Rabbit's pink eyes, its watch in the waistcoat pocket, and Alice following it out of curiosity. This is a positive end-to-end smoke test, not an aggregate recall score.

The benchmark-recall tests in that working tree passed 18 tests and 64 assertions, including the CRLF chapter-selection regression. The repository's [benchmark report](https://github.com/sadofriod/graphrag-ts/blob/v0.1.16/examples/output/final-benchmark-report.en.md) records a separate historical 12-query run (10/12 strict hits, 95.8% combined lexical recall) and a one-document incremental-update check (combined recall rose from 50% to 100%). Those report runs used an earlier two-book corpus and different model/runtime configurations; they are not measurements of the three-source smoke run above or a cross-project comparison.
