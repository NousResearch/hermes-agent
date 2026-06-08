# route.bible/skill.md

## Overview
route.bible is a routing and deep-linking layer for Bible passages. This skill allows the agent to generate and resolve app-agnostic Bible links.

## stable URL contract
- `https://route.bible/{reference}` (e.g., `https://route.bible/John.3.16`)
- Supported formats: OSIS identifiers, common abbreviations.

## Local Parsing
Use `grab-bcv` for local parsing of references before routing.

## Integration
To integrate, refer to the hosted documentation at `https://route.bible/docs`.
