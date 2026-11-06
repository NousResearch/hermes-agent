# GDPR / UK-GDPR (roadmap - Phase 3)

For EU/UK subjects. Not part of the P0 US-first scope; templates and routing land in Phase 3.

## Rights invoked

- **Erasure** ("right to be forgotten") - Article 17.
- **Object** to processing - Article 21.

## Request content

Render with `legal.render_request("gdpr", broker, fields)` ->
`templates/emails/gdpr-erasure.txt`. Address the controller's privacy/DPO contact. Include the data
subject's name, the contact email, and the listing URL(s); cite Article 17.

## Notes

- Controllers must respond within one month (Article 12(3)).
- EU-specific brokers and portals (e.g. Acxiom's EU consumer portals) are added in Phase 3 with
  `jurisdictions: ["EU"]` records and residency-aware routing.
