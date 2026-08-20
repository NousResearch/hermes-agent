"""GitHub App connector plugin — bundled, auto-loaded.

Registers 7 tools (identity, create issue, comment, list, get, review PR,
merge PR) into the ``github`` toolset. Every tool resolves credentials
bot-first: when GitHub App credentials are configured
(``GITHUB_APP_ID`` + ``GITHUB_APP_PRIVATE_KEY_PATH`` +
``GITHUB_APP_INSTALLATION_ID``), actions are attributed to the bot login
``<slug>[bot]``; otherwise they fall back to a PAT or gh CLI (human
identity). Each result carries an attribution block so agent actions are
always distinguishable from the user's.

Why a plugin instead of a top-level ``tools/`` file?

- ``plugins/`` is where third-party service integrations live (see
  ``plugins/spotify/``). ``tools/`` is reserved for foundational
  capabilities.
- The GitHub App auth itself is shared with ``tools/skills_hub.py`` via
  ``agent/github_auth.py`` — one implementation, two consumers.
- Bundled + ``kind: backend`` auto-loads on startup like spotify; the
  tools' ``check_fn`` keeps them out of the model schema until GitHub
  credentials exist.
"""

from __future__ import annotations

from plugins.github.tools import (
    GITHUB_COMMENT_ISSUE_SCHEMA,
    GITHUB_CREATE_ISSUE_SCHEMA,
    GITHUB_GET_ISSUE_SCHEMA,
    GITHUB_IDENTITY_SCHEMA,
    GITHUB_LIST_ISSUES_SCHEMA,
    GITHUB_MERGE_PR_SCHEMA,
    GITHUB_REVIEW_PR_SCHEMA,
    _HANDLERS,
    _check_github_available,
)

_TOOLS = (
    ("github_identity", GITHUB_IDENTITY_SCHEMA, "🆔"),
    ("github_create_issue", GITHUB_CREATE_ISSUE_SCHEMA, "🐛"),
    ("github_comment_issue", GITHUB_COMMENT_ISSUE_SCHEMA, "💬"),
    ("github_list_issues", GITHUB_LIST_ISSUES_SCHEMA, "📋"),
    ("github_get_issue", GITHUB_GET_ISSUE_SCHEMA, "🔍"),
    ("github_review_pr", GITHUB_REVIEW_PR_SCHEMA, "👁️"),
    ("github_merge_pr", GITHUB_MERGE_PR_SCHEMA, "🔀"),
)


def register(ctx) -> None:
    """Register all GitHub tools. Called once by the plugin loader."""
    for name, schema, emoji in _TOOLS:
        ctx.register_tool(
            name=name,
            toolset="github",
            schema=schema,
            handler=_HANDLERS[name],
            check_fn=_check_github_available,
            emoji=emoji,
        )
