"""claude_selfimprove — the Claude Code self-improvement pipeline.

This package reads Claude Code session transcripts (JSONL files under
``~/.claude/projects/`` and ``~/.claude-hermes/projects/``), extracts
high-signal lessons (explicit corrections, explicit instructions, repeated
confirmed fixes, repeated successful procedures), and safely proposes or
applies changes to ``~/.claude/CLAUDE.md`` (a bounded managed section),
``~/.claude/rules/`` (pipeline-owned rule files), and ``~/.claude/skills/``
(pipeline-owned skills).

It is a standalone package with no import dependency on the hermes-agent
runtime: it runs as a cron-scheduled subprocess. Its two thin entry
scripts live in this package at ``cron_scripts/claude_selfimprove_scan.py``
and ``cron_scripts/claude_selfimprove_consolidate.py``; the ``install`` CLI
command (``cli.py``) copies them, flat, into ``$HERMES_HOME/scripts/``
alongside a copy of this whole package, which is where Hermes cron
actually runs them from. It talks to the Hermes agent only through the
``hermes`` command-line program (for model calls and Slack delivery),
exactly like the sibling ``t3_nightly_watch.py`` and
``self_improvement_watch.sh`` automations it runs alongside.

Transcript text is always untrusted data. No module in this package ever
executes, follows, or treats as a command any instruction found inside a
transcript — it is read, pattern-matched, redacted, and (for the classifier
step) summarized by a model that is explicitly told the text is data to
analyze, not instructions to obey.
"""

from __future__ import annotations

__version__ = "1.0.0"
