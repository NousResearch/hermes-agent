# Profile-local file keeping

- Status: active
- Scope: native home initialization, main prompt, `guides/file-keeping`
- Introduced: PR #2

## Downstream intent

Create `documents/` and `repos/` directly under the active Hermes home. The
frozen prompt names both and points to the source-derived filing guide.
Documents hold lasting attachments, data, deliverables and cloud-document
stubs; responsibility packages link to them. The model maintains these files.

## Reconciliation

Keep the additions in native directory initialization and prompt assembly.
Preserve native file operations, permissions, attachment paths, caches and
scratch behavior. No hosted cache/expiry contract or file migration. Keep
profile-specific roots independent of terminal cwd.

## Validation

Check directory creation and preservation in separate homes, resolved prompt
paths, native guide reads, packaging and warm-session prompt stability.
