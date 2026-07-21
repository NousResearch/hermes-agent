# Wave 3 — Review and Correction

Independent review accepted schema, canonicalization, CLI rendering, and the
normal terminal path, but found that timeout and final executor errors returned
before the freshness marker. A partial write could therefore leave old proof
fresh.

The correction centralizes best-effort stale marking after an attempted
foreground `env.execute()` call, including normal return, timeout, and final
error. It does not mark command-guard or workdir-validation rejections.

The reviewer returned PASS after the correction. Added tests exercise return
codes 0 and 1, timeout, final runtime error, stale prior evidence, and later
fresh verification.
