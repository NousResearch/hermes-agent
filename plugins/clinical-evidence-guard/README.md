# Clinical evidence guard

This opt-in plugin classifies each request, retains protected model output, and evaluates drafts against documentary passages returned by evidence tools in that turn. Direct-care requests stay protected; software, operational reports and technical follow-ups keep ordinary behavior. Evidence is bounded, expires in memory, and is separated by profile, session and turn.

Enable through the normal plugin configuration:

```yaml
plugins:
  enabled:
    - clinical-evidence-guard
```

The guard excludes router-only responses, errors and query echoes. It checks topical overlap, polarity, relation direction, numerical details and citation identifiers. Unsupported recommendations or specifics request repair. Other unsupported general statements receive the original disclosure. These are conservative lexical checks, not a clinical adjudication or live verification of cited sources.

The runtime permits four private regenerations and then fails with a fixed content-free message. Block metadata and repair rows are excluded from user-facing output, durable history and trajectories. Model text, reasoning and tool presentation remain retained until the final approval. A host suffix is checked again against the same turn's evidence.

Optional Jev comparison is disabled by default:

```yaml
jev:
  clinical_classify:
    enabled: true
```

Enabling that option allows the Jev subprocess to classify request text. Its verdict never changes the guard's deterministic decision. Logs contain only verdicts, bounded confidence/action fields, counts and safe error types, under the active profile's logs directory. No live Jev call or external evidence provider was exercised by the port's tests.
