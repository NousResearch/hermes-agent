# Pruned-test resurrection manifest

`pruned_test_fingerprints.sha256bin` contains 26,655 sorted, raw SHA-256
records (32 bytes each). Each record binds a test's repository-relative path,
qualified function name, and normalized Python AST.

The source set is the tests removed between the parent of `6b81590c55` and
`39975613b1`, the two July 29, 2026 low-value-test pruning commits. Tests already
present at the current upstream integration point are grandfathered; notably,
that includes required security, redaction, approval, message-role, caching,
issue-regression, and end-to-end coverage. The guard rejects only a dormant
pruned test returning with its exact old identity and behavior. It does not cap
the total number of tests or reject new and materially changed tests.
