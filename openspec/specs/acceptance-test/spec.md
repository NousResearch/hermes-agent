# Acceptance Test Workflow

### Requirement: Acceptance scenarios generate a deterministic one-to-one test gate
#### Scenario: Generated acceptance test preserves the developer-owned body across regeneration
- Path Code: UT-ACCEPTANCE-001
- GIVEN a Gherkin-style OpenSpec scenario with a valid Path Code
- WHEN the acceptance generator runs more than once
- THEN the generated test remains bound to the same Path Code metadata
- AND the developer-owned body block is preserved verbatim

#### Scenario: Browser-level use cases require an explicit handler
- Path Code: UC-ACCEPTANCE-001
- GIVEN a customer-facing use-case scenario
- WHEN no use-case project handler is configured
- THEN the acceptance workflow emits a handler not configured warning
- AND it does not silently pass the scenario
