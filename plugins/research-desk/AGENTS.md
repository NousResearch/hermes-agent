# Research Desk plugin rules

Research Desk is a customer-facing product layer around the existing
`openmanus` adapter. It is not a replacement for OpenManus and it must not
modify `vendor/openmanus`.

The initial data classification is `public_research`. Evidence may be
collected only through Hermes web-tool boundaries and only from configured
allowlisted domains. OpenManus workers may connect only to the configured LLM
endpoint; browser, web, and MCP tools remain disabled. The host Hermes LLM is
the primary synthesis path.

The active Hermes profile is the execution principal. Do not accept or infer
tenant identity from a user-supplied customer or tenant field. Each plan,
run, receipt, and export must be tied to the active profile and to the
configured workspace. Reject symlink escapes and paths outside that
workspace.

Receipts are metadata only. Never copy source text, customer input, API keys,
tokens, direct email addresses, phone numbers, or worker stdout into a
receipt. Public evidence packets and reports remain under the configured
workspace; Hermes receipts remain under the active Hermes home.

Keep recurring execution under Hermes cron. Do not add a second scheduler to
this plugin. Product decisions about pricing, billing, contracts, company
formation, tax, financing, and legal advice are out of scope.
