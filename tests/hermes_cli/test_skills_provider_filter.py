"""Provider filters must run before a source's result limit can discard matches."""

import argparse
import json

import pytest

from hermes_cli import skills_hub as cli_hub
from hermes_cli.subcommands.skills import build_skills_parser
from tools.skills_hub_github import GitHubAuth, GitHubSource, _tap_cache_key, github_provider_for
from tools.skills_hub_models import SkillMeta, _cache_metas, _skill_meta_to_dict
from tools.skills_hub_official import HermesIndexSource
from tools.skills_hub_search import _hermes_index_cache_file, parallel_search_sources


@pytest.fixture(params=["index", "github"])
def catalog(request, monkeypatch):
    def entry(repo, provider, name):
        return SkillMeta(
            name=name, description="GPU utilities", source="github",
            identifier=f"{repo}/skills/{name}", trust_level="trusted",
            extra={"provider": provider},
        )

    # All match the same query, but the requested provider sits beyond the
    # default per-source search window. Neither source's search is mocked.
    others = [entry("openai/skills", "OpenAI", f"gpu-other-{i}") for i in range(60)]
    wanted = [entry("NVIDIA/skills", "NVIDIA", f"gpu-target-{i}") for i in range(3)]
    auth = GitHubAuth()
    if request.param == "index":
        path = _hermes_index_cache_file()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"skills": [_skill_meta_to_dict(m) for m in others + wanted]}), encoding="utf-8")
        source = HermesIndexSource(auth)
    else:
        source = GitHubSource(auth)
        for tap in source.taps:
            key = _tap_cache_key(tap["repo"], tap.get("path", ""), tap.get("bucket"))
            label = github_provider_for(tap["repo"])
            entries = others if label == "OpenAI" else wanted if label == "NVIDIA" else []
            _cache_metas(key, entries)
    monkeypatch.setattr(cli_hub, "_sources", lambda: [source])
    return source, others, wanted


def test_cli_provider_search_finds_matches_beyond_global_window(catalog, capsys):
    source, others, wanted = catalog
    parser = argparse.ArgumentParser(prog="hermes")
    build_skills_parser(parser.add_subparsers(dest="command"), cmd_skills=cli_hub.skills_command)

    for query in ("gpu", ""):
        args = parser.parse_args([
            "skills", "search", query, "--source", "nvidia", "--limit", "2", "--json",
        ])
        args.func(args)
        result = json.loads(capsys.readouterr().out)
        assert [row["identifier"] for row in result] == [m.identifier for m in wanted[:2]]

    # Filtering one call must not mutate the cached catalog for later searches.
    assert [m.identifier for m in source.search("gpu", limit=2)] == [m.identifier for m in others[:2]]

    args = parser.parse_args(["skills", "search", "gpu", "--source", "anthropic", "--json"])
    args.func(args)
    assert json.loads(capsys.readouterr().out) == []


def test_parallel_provider_search_applies_per_source_limit_after_filter(catalog):
    source, _, wanted = catalog
    results, counts, timed_out = parallel_search_sources(
        [source], query="gpu", source_filter=" NVIDIA ",
        per_source_limits={source.source_id(): 2},
    )
    assert [m.identifier for m in results] == [m.identifier for m in wanted[:2]]
    assert counts[source.source_id()] == len(results)
    assert not timed_out
