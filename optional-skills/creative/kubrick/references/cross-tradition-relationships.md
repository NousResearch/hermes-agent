# Cross-Tradition Relationship Types

Every claimed correspondence or shared symbol must be tagged with one of the following. Never default to equivalence.

## Allowed Types
- **HISTORICALLY_DERIVED**: Documented historical transmission or influence (e.g., via trade, conquest, scholarly transmission).
- **TEXTUALLY_ATTESTED**: Appears in written sources with explicit linkage.
- **SHARED_FUNCTION**: Structurally similar function in different cultures without claimed origin (e.g., threshold guardians in multiple traditions).
- **FORMAL_RESEMBLANCE**: Visual or structural similarity noted by analyst or practitioner; explicitly not equivalence.
- **MODERN_SYNTHESIS**: 19th-21st century construction (e.g., many "universal" correspondences in popular occultism).
- **PRACTITIONER_ASSOCIATION**: Living practitioner links them for their work (record as such).
- **ANALYST_INFERENCE**: Modern scholarly or artistic inference (not historical claim).
- **AESTHETIC_SIMILARITY**: Chosen for visual or emotional effect in a specific work (no deeper claim).
- **CONTESTED**: Scholars or traditions disagree on the link.
- **UNSUPPORTED**: No credible basis; flag and exclude.

## Usage in Patterns
In SymbolicNarrativePattern.source_records and correspondence_map, tag every link.

Example:
correspondence:
  - source_a: "alchemical nigredo"
    source_b: "certain Japanese yami motifs"
    relationship: FORMAL_RESEMBLANCE | MODERN_SYNTHESIS
    note: "Both involve blackening/darkening as stage of transformation; no historical link attested."
    confidence: 0.4
    authority_status: CONTESTED

Never promote resemblance or modern usage to historical derivation without evidence.
