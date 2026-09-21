# Hermes self-evolution policy

Self-evolution may generate candidate changes to skills and prompt text, but it is a review-only development tool. It may not write directly to the active installation, merge a branch, change credentials, alter tool implementations, or modify runtime safety controls.

Every candidate must include the source SHA, old and new content digests, dataset digest, evaluator version, size delta, focused test output, benchmark output where relevant, reviewer, and a tested rollback commit. Scrubbed session data is required; secrets, absolute home paths, account identifiers, and broker data are prohibited.

Promotion is manual and requires passing focused tests plus the relevant compression and tool-performance checks. LunaBot-facing skills are never eligible for automatic promotion into signal, risk, route, broker, promotion, or live-submit paths.
