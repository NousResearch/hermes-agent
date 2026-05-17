# Blocked-surface PASS formatting pitfall

Context: During crypto_bot S006 branch-local completion evidence, Codex produced a semantically passing sidecar result with:

`- Blocked-surface scan: PASS, with basis: ...`

The completion gate originally rejected it because the parser accepted exactly `pass` or `pass ` but not `pass,`. The branch-local control-plane repair changed the parser to a PASS token-boundary rule:

```python
if normalized == "pass" or re.match(r"^pass\b", normalized):
    return True
```

Durable guidance for future sidecar audits:

- Prefer parser-compatible machine fields over prose-only semantic clarity.
- Emit `Blocked-surface scan: PASS with basis: ...` or `Blocked-surface scan: PASS` plus details elsewhere.
- The punctuation form `PASS, with basis: ...` is now accepted by the repaired completion gate, but the punctuation-free `PASS with basis: ...` remains the safer cross-version format.
- Do not infer blocked surfaces solely because changed files are outside a task allowlist. The allowlist decides whether otherwise sensitive workflow/service/docs paths are approved for the task; ordinary docs, JSON evidence, discovery artifacts, and validation scripts are not blocked merely by absence from the allowlist.
- When a sidecar says final conclusion PASS but completion gate rejects a machine field, inspect the parser before rerunning work. A semantically valid result can fail on field-shape strictness.
- Add targeted regression coverage whenever a sidecar wording shape is accepted into the control-plane contract.

Verification pattern:

1. Add/adjust a narrow test for the exact machine-field shape.
2. Run the single test RED if possible.
3. Patch the parser minimally.
4. Run targeted pytest and ruff.
5. Rerun the completion gate against the same sidecar artifact to prove the blocker was parser-only.
