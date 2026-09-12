# Host constraints exercised by this fixture

## Specification provenance

The fixture was checked against the project snapshot `SEP-2640.md` and canonical branch commit `d6b31a03504c15677d49b922b6b6ace0ef65728d`. The saved snapshot labels the SEP Accepted while that branch labels it Draft; the wire fields exercised here agree. This fixture makes no claim about the proposal's governance status.

This fixture is an independent Python MCP SDK server. It exposes only synthetic, public test content and implements the static-manifest portion of SEP-2640 over stdio:

- extension declaration with `directoryRead: true`;
- paginated `skills/list` and point lookup through `skills/get`;
- standard `resources/read` for UTF-8 text and base64 binary content;
- `resources/directory/read` for direct children;
- complete SHA-256/size manifests;
- a no-prefix `skill://portable-demo/SKILL.md` skill whose name occupies the URI authority;
- an alternate-scheme skill, proving that `skill://` is conventional rather than privileged;
- two distinct skill URIs with the same frontmatter name.

The runner checks Hermes host behavior for metadata-only startup, prompt provenance, local/remote and intra-server name collisions, lazy supporting-file reads, and binary materialization. Each remote skill is activated in a separate session so approval state cannot be borrowed across origins or skill identities.

This is not a complete SEP conformance suite. It does not claim coverage of dynamic manifests, unenumerated skills, nested-skill activation, content changes and approval revocation, hostile input scanning, cross-server resource consent, HTTP authentication, newer list-cache attributes, Windows filesystem behavior, or maximum-size limits. The fixture does not patch host scanning or consent code and does not grant execution permission.
