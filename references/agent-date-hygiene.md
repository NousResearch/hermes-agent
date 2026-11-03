# Agent Date Hygiene

## The Problem
Without explicit date verification, agents may rely on stale cached knowledge, session metadata, or file timestamps — producing memos with wrong dates or incorrect recency judgments.

## Mandatory Procedure
Before any date-sensitive operation:
1. Run `date -u` in terminal to get the actual current date.
2. Compare against any other date signals (file mtimes, session metadata, prior memos).
3. If there is a discrepancy, note both observations and use the terminal date as ground truth.
4. Proceed with date-grounded work.

## Checklist for New Jobs
- [ ] Run `date -u` before any file writes, outputs processing, or memo generation
- [ ] Use the actual date in filenames, frontmatter, and reports
- [ ] If comparing against prior files (e.g., "last week's memo"), verify the date via file content, not assumption