"""GitLab MR Review Plugin — native tools for reviewing GitLab Merge Requests.

Registers 12 tools across 3 categories:

MR Tools (7):
  gitlab_mr_view, gitlab_mr_diff, gitlab_mr_list_files,
  gitlab_mr_comments, gitlab_mr_inline_comment, gitlab_mr_review,
  gitlab_mr_list

Pipeline Tools (3):
  gitlab_mr_pipelines, gitlab_pipeline_jobs, gitlab_pipeline_retry

Context Tools (2):
  gitlab_mr_context, gitlab_mr_discussions

Configuration via environment variables:
  GITLAB_TOKEN   — Personal access token (required). Needs api + read_api scope.
  GITLAB_URL     — Base URL for self-hosted GitLab (default: https://gitlab.com).
"""


def register(ctx) -> None:
    """Register all GitLab MR review tools with the plugin system."""
    from plugins.gitlab_review.tools_mr import register_mr_tools
    from plugins.gitlab_review.tools_pipeline import register_pipeline_tools
    from plugins.gitlab_review.tools_context import register_context_tools

    register_mr_tools(ctx)
    register_pipeline_tools(ctx)
    register_context_tools(ctx)
