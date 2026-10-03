# Source Hierarchy and Provenance Standards

All symbolic material must be classified and tagged with full provenance.

## Classification Tiers
- **PRIMARY**: Original texts, artifacts, direct ethnographic records, canonical scriptures, historical documents (e.g., *Rosarium Philosophorum*, ethnographic field notes, ancient inscriptions).
- **EARLY_COMMENTARY**: Near-contemporary interpretations (e.g., medieval alchemical commentaries, early Jung on alchemy).
- **SCHOLARLY**: Peer-reviewed academic work with apparatus (e.g., Mircea Eliade, Victor Turner on liminality, scholarly editions of myths).
- **PRACTITIONER**: Living or recent practitioner accounts with transparent method (must note personal/contextual nature).
- **COMPARATIVE**: Cross-cultural studies that explicitly discuss method and limits.
- **POPULAR**: Accessible books, films, or articles for contemporary usage (document modern reception only).
- **INTERNET**: Forums, blogs, social media (lowest tier; use only for tracking contemporary popular interpretation; never for historical claims).

## Required Metadata per Source Record
- author or tradition
- title
- date
- edition or translation (with translator)
- cultural context
- source_family
- source_tier (from list above)
- reliability_context (e.g., "textual transmission uncertain after 12th c.")
- contested_interpretations (list)
- access_status (open, paywalled, archive only)
- provenance (full citation + how obtained)

## Ingestion Rules
- Primary and Scholarly ground all claims.
- Popular/Internet document *current* usage, not origin.
- Every pattern in SymbolicNarrativePattern must link to at least one PRIMARY or SCHOLARLY source_record.
- Cross-tradition claims require explicit relationship type (see cross-tradition-relationships.md).
- When in doubt, mark confidence low and authority_status as SPECULATIVE or CONTESTED.
