"""Regression for #115171: successful downloads must not be called stale index entries."""
import io
from contextlib import ExitStack
from unittest.mock import Mock, patch

import pytest
from rich.console import Console


@pytest.mark.parametrize("adapter", ["url", "github", "skills.sh"])
def test_install_explains_validation_and_resets_on_next_fetch(adapter):
    from hermes_cli.skills_hub import do_install
    from tools.skills_hub_github import GitHubSource
    from tools.skills_hub_skillssh import SkillsShSource
    from tools.skills_hub_sources import UrlSource

    text = "---\nname: revops\ndescription: Example.\n---\nSee [registry](../../tools/REGISTRY.md).\n"
    identifier = "owner/repo/skills/revops"
    with ExitStack() as stack:
        if adapter == "url":
            source = UrlSource()
            identifier = "https://example.com/revops/SKILL.md"
            content = stack.enter_context(patch.object(source, "_fetch_text", return_value=text))
            support = stack.enter_context(patch.object(source, "_fetch_bytes"))
        else:
            github = GitHubSource(Mock())
            stack.enter_context(patch.object(github, "_get_repo_tree", return_value=None))
            content = stack.enter_context(patch.object(github, "_fetch_file_content", return_value=text))
            support = stack.enter_context(patch.object(github, "_fetch_file_bytes"))
            source = github
            if adapter == "skills.sh":
                source = SkillsShSource(Mock())
                source.github = github
                stack.enter_context(patch.object(source, "_fetch_detail_page", return_value={}))
                stack.enter_context(patch.object(source, "_candidate_identifiers", return_value=[identifier]))
                stack.enter_context(patch.object(source, "_discover_identifier", return_value=None))
                stack.enter_context(patch.object(source, "_resolve_github_meta", return_value=github.inspect(identifier)))
                identifier = "skills-sh/owner/repo/revops"
        stack.enter_context(patch("hermes_cli.skills_hub._sources", return_value=[source]))
        quarantine = stack.enter_context(patch("tools.skills_hub_install.quarantine_bundle"))
        output = io.StringIO()
        do_install(identifier, console=Console(file=output, width=160, color_system=None), skip_confirm=True)
        support.assert_not_called()
        quarantine.assert_not_called()
        rendered = output.getvalue()
        assert "outside the skill" in rendered.lower(), rendered
        assert "files no longer exist upstream" not in rendered
        content.return_value = None
        assert source.fetch(identifier) is None
        assert not source.fetch_error
        content.return_value = "---\nname: revops\n---\nSelf-contained instructions.\n"
        assert source.fetch(identifier) is not None
        assert not source.fetch_error
