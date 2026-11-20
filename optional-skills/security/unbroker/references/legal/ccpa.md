# CCPA / CPRA (California)

Use for California residents (`residency_jurisdiction` starts with `US-CA`) and, in practice, many US
brokers that honor CCPA-style requests nationwide.

## Rights invoked

- **Delete** personal information (Cal. Civ. Code 1798.105).
- **Opt out** of sale/sharing of personal information (1798.120).

## Request content

Render with `legal.render_request("ccpa", broker, fields)` -> `templates/emails/ccpa-deletion.txt`.
Include only: full legal name, the contact email for correspondence, and the confirmed listing
URL(s). Do **not** include SSN or government IDs.

## Authorized agent

When acting for another consenting subject, use `render_request("ccpa_agent", ...)`
(`templates/emails/ccpa-authorized-agent.txt`) and attach the authorization artifact recorded in the
dossier (`consent.authorization_artifact`). The broker may separately verify the consumer's identity.

## Notes

- Brokers must respond within 45 days (extendable). Track as `awaiting_processing` until confirmed.
- "Hidden from free search" is not deletion - verify the record is actually gone before
  `confirmed_removed`.
