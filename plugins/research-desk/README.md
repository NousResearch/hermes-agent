# Research Desk

Research Desk is the customer-facing product layer for a Hermes Private
Runner. It uses the existing `openmanus` adapter as a bounded worker engine,
while Hermes remains responsible for profile identity, workspace confinement,
public-source collection, approval, synthesis, and redacted receipts.

The first release is deliberately narrow: competitor, pricing, hiring, and
industry-news research from configured public domains. Customer files,
internal credentials, private databases, legal decisions, tax filings,
financing decisions, contracts, invoices, and company-formation actions are
outside this plugin.

## Product boundary

`public_research` is the only initial data classification. Hermes collects
search results and page extracts through its existing web-tool boundary. The
OpenManus worker receives the resulting redacted evidence packet only. It may
connect to the configured LLM endpoint, but browser, web, and MCP tools remain
disabled. Hermes LLM structured synthesis remains the primary report engine,
and unsupported claims are discarded unless they cite an evidence
`source_id`.

The active Hermes profile is the execution principal. Do not pass a customer
or tenant identifier to the plugin. Configure one Private Runner workspace
per profile and make the Research Desk workspace a child of the configured
OpenManus workspace:

```yaml
plugins:
  enabled:
    - openmanus
    - research-desk
  entries:
    openmanus:
      workspace_root: C:/PrivateRunner/workspace
      allow_llm_network: true
      llm:
        model: local-model
        base_url: http://127.0.0.1:9000/v1
        api_key_env: OPENMANUS_API_KEY
    research-desk:
      profile_name: default
      workspace_root: C:/PrivateRunner/workspace
      allowed_domains:
        - example.com
        - another-public-source.jp
      max_workers: 4
      pass_worker_model_secret: false
```

Put API keys, tokens, and passwords in the active profile `.env` only. YAML
contains behaviour settings, never secret values. By default, the OpenManus
worker receives no secret environment variables and therefore requires a
local or otherwise no-auth LLM endpoint. Set
`pass_worker_model_secret: true` only when the worker must use the one model
key named by `api_key_env`; no other environment variables or credentials are
forwarded. The host Hermes LLM remains primary and its credentials are never
copied into receipts.

## Commands

```text
hermes research-desk status
hermes research-desk plan --topic "market movement" --target "Company A" --source-domain example.com
hermes research-desk run --plan-id plan-... --approved --acknowledge-side-effects
hermes research-desk export --run-id run-... --format markdown --approved
```

Planning never performs network access and never starts OpenManus. A run
requires the plan to be approved and requires explicit acknowledgement of
public retrieval and report writes. Export is a separate human approval gate
and remains inside the Private Runner workspace. Recurring operation belongs
to Hermes cron; this plugin does not add another scheduler.

The run workspace contains the evidence packet, report, and export files.
Metadata receipts are kept under the active Hermes home in
`research-desk/receipts/`. Receipts contain hashes, URLs, timestamps, source
classification, worker status, synthesis mode, and approval state. They do
not contain customer raw input, page text, API keys, tokens, or worker stdout.

## Commercial tiers

**Pilot** provides one topic, weekly public research, explicit approval, and
an evidence-linked delivery.

**Standard** provides multiple topics, bounded parallel workers, recurring
execution through Hermes cron, and a monthly review.

**Managed Private Runner** runs on the customer's PC or VM. Source material
stays in that customer-controlled environment; only approved state,
deliverables, and selected evidence metadata are shared.

Pricing, billing, contracts, incorporation filings, tax returns, loan or grant
decisions, and legal advice are intentionally excluded. Business formation
and financing require separate professional confirmation.

## Safety notes

Configured domains are checked before every extraction. URL schemes,
credentials, non-standard ports, private IP literals, loopback, link-local,
reserved, and multicast destinations are rejected. Direct email addresses,
phone numbers, and user-home paths are redacted before evidence reaches a
worker or receipt. Paths are resolved inside the active profile workspace,
symlinks are rejected for configured roots and artifacts, and report export
cannot target a public directory.

The product is designed as a research and evidence workflow, not as an
autonomous decision-maker. Human review remains responsible for whether a
report is sent to a customer or used in a business decision.
