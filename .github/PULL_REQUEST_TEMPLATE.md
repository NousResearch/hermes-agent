## Summary

<!-- What changed, and why is this the right approach for Hermes? -->

- 

## Footprint and Architecture

<!-- New capability should live at the narrowest reasonable surface. -->

- [ ] Extends existing code, or explains why a new surface is needed
- [ ] Does not add a core model tool unless terminal/file/CLI/plugin/MCP cannot solve it
- [ ] Keeps plugins inside their plugin directory; no plugin-specific core special cases
- [ ] Preserves byte-stable system prompts and per-conversation prompt caching
- [ ] Preserves strict role alternation; no synthetic mid-loop user messages

## Configuration and Secrets

- [ ] User-facing behavioral settings live in `config.yaml`
- [ ] `.env` additions are credentials only
- [ ] Setup/config UI, docs, and defaults are updated together
- [ ] No telemetry, attribution tags, or third-party identifiers without opt-in gating

## Tests

<!-- Include exact commands and any intentional omissions. -->

- [ ] Reproduces the original bug or validates the new behavior through the real path
- [ ] Covers sibling call paths and provider/config propagation where relevant
- [ ] Avoids change-detector assertions for model lists, versions, or enum counts
- [ ] Commands run:

```bash

```

## Documentation

- [ ] User docs updated, or N/A
- [ ] Tool/schema references updated, or N/A
- [ ] Developer docs/skills updated, or N/A

## Reviewer Notes

<!-- Call out risk, migration notes, follow-ups, screenshots, or logs. -->

