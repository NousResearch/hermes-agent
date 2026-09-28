# Hindsight memory

This fork connects to a separately deployed Hindsight service. Retention,
prefetch, reflection, and server policy follow the employee specification.
The model receives `recall`; learning from conversations happens automatically.
Automatic recall uses the employee memory-context instruction: supplementary
background information must be verified against live sources.

Configure operational values through `hermes memory setup`, the native memory
settings panel, or the Config editor:

- `hindsight.url`: deployed service URL.
- `hindsight.bank_id`: optional bank override; blank selects a unique bank per profile.
- `HINDSIGHT_API_KEY`: profile secret, saved to `.env` by native secret management.

CLI and both dashboard config surfaces write the same native `config.yaml` and
profile secret store. Settings take effect in new sessions. The old
`hindsight/config.json` is not an authoritative configuration source.

Cloud/embedded mode selectors, alternative inference providers, starter bank
templates, and editable recall/retention policy are not exposed in this fork.
See [deployment instructions](../../../deploy/railway/README.md) for the private
Hindsight service, Codex inference endpoint, embeddings, reranking, and bank
reconciliation. Dependency installation uses native Hermes PM.

Failed conversation batches retry in order with exponential backoff (up to
30 seconds between attempts), including across session switches. Agent shutdown
waits up to 10 seconds and leaves the daemon writer draining afterward. This
queue is in memory: stopping the whole process before the service recovers can
lose pending batches; shutdown logs any outstanding work.
