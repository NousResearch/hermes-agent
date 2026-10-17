"""Prove that a required check was skipped by its own workflow's path filter.

A skipped required check is admitted (``not_applicable``, never ``success``) only when
read-only GitHub evidence shows all of:

- the skip is a completed, exact-head GitHub Actions job pinned to the required app,
  bound by ``details_url`` to a run of this repository and to this check's job id;
- that run is a successful same-repository ``pull_request`` run of the same head, and
  its planner job succeeded;
- the trusted workflow at the PR's base SHA (byte-identical at head, and not in the
  diff) gates the job only by ``needs.<planner>.outputs.<filter> == 'true'`` terms
  (OR'ed, optionally AND'ed with the standard same-repository guard), each output
  wired to a static quoted heredoc of ``name: ERE`` filters fed to an unchanged script;
- the complete changed-path list matches none of those filters.

Anything outside that grammar raises :class:`Unproven`. Nothing here runs repository
code: the workflow is parsed with a safe YAML loader and filters are matched in Python
after restricting them to the ERE subset whose semantics Python ``re`` shares.
"""
from __future__ import annotations

import base64
import re
from urllib.parse import quote

_GUARD = ("(github.event_name != 'pull_request' || "
          "github.event.pull_request.head.repo.full_name == github.repository)")
_NAME = r"[A-Za-z_][A-Za-z0-9_-]*"
_TERM = re.compile(rf"needs\.({_NAME})\.outputs\.({_NAME}) == 'true'")
_STEP_OUTPUT = re.compile(rf"\$\{{\{{ ?steps\.({_NAME})\.outputs\.({_NAME}) ?\}}\}}")
_HEREDOC = re.compile(r"bash ([A-Za-z0-9_.][A-Za-z0-9_./-]*\.sh) <<'([A-Z_]+)'\n(.*)\n\2\n?", re.S)
# grep -E and Python agree on this subset: literals, anchors, groups, alternation,
# */+/? quantifiers, escaped dots and simple bracket classes.
_SHARED_ERE = re.compile(r"(?:[A-Za-z0-9_/^$()|.*+?-]|\\\.|\[\^?[A-Za-z0-9_./-]+\])+")
# Lazy/stacked quantifiers and quantified group openers mean different things (or nothing) in ERE.
_AMBIGUOUS_QUANTIFIER = re.compile(r"(?:^|[(|^*+?])[*+?]")
_WORKFLOW = re.compile(r"\.github/workflows/[A-Za-z0-9_.-]+\.ya?ml")
_STEP_KEYS = {"id", "name", "run", "env"}
_FILES_API_CAP = 3000


class Unproven(ValueError):
    """The skip is not explained by the trusted path policy; the check is not accepted."""


def _require(condition, reason: str) -> None:
    if not condition:
        raise Unproven(reason)


def _same(a, b) -> bool:
    return isinstance(a, str) and isinstance(b, str) and a.lower() == b.lower()


def prove_path_skip(api, repo: str, number: int, sha: str, check: dict,
                    context: str, app_id, cache: dict) -> dict:
    """Return the proof for one skipped required check run, or raise :class:`Unproven`.

    ``api`` is the profile-bound read-only ``gh api`` call; ``cache`` is shared across
    the checks of one acceptance read so each run/file list is fetched once.
    """
    app = check.get("app") or {}
    _require(app_id not in (None, -1) and app.get("id") == app_id and app.get("slug") == "github-actions",
             "skip is not a GitHub Actions job pinned to the required app")
    _require(check.get("status") == "completed" and check.get("conclusion") == "skipped"
             and check.get("head_sha") == sha, "skip is not a completed exact-head check")
    bound = re.fullmatch(rf"https://github\.com/{re.escape(repo)}/actions/runs/([1-9][0-9]*)/job/([1-9][0-9]*)",
                         check.get("details_url") or "", re.I)
    _require(bound and int(bound[2]) == check["id"], "details_url is not this repository's run/job")
    run_id = int(bound[1])

    pr = _cached(cache, "pr", lambda: api(f"repos/{repo}/pulls/{number}"))
    base_sha = pr["base"]["sha"]
    _require(pr["head"]["sha"] == sha and re.fullmatch(r"[0-9a-f]{40}", base_sha or ""),
             "PR head/base is not the evaluated exact head")
    _require(_same((pr["head"].get("repo") or {}).get("full_name"), repo), "PR head is not in this repository")

    run = _cached(cache, ("run", run_id), lambda: api(f"repos/{repo}/actions/runs/{run_id}"))
    _require(run["id"] == run_id and run["head_sha"] == sha and run["event"] == "pull_request"
             and _same(run["repository"]["full_name"], repo) and _same(run["head_repository"]["full_name"], repo),
             "workflow run is not a same-repository pull_request run of this head")
    _require(run["status"] == "completed" and run["conclusion"] == "success", "workflow run did not succeed")
    path = run["path"]
    _require(isinstance(path, str) and _WORKFLOW.fullmatch(path), "workflow path is not a repository workflow")
    jobs = _cached(cache, ("jobs", run_id), lambda: _run_jobs(api, repo, run_id))
    skipped = [j for j in jobs if j["id"] == check["id"]]
    _require(len(skipped) == 1 and skipped[0]["name"] == context and skipped[0]["conclusion"] == "skipped"
             and skipped[0]["run_id"] == run_id, "skipped job is not this run's job")

    workflow = _unchanged(api, cache, repo, path, base_sha, sha)
    from hermes_yaml import safe_load
    try:
        document = safe_load(workflow)
    except Exception:  # any YAML error or unsafe tag: the policy cannot be read
        raise Unproven("base workflow is not plain YAML") from None
    _require(isinstance(document, dict) and isinstance(document.get("jobs"), dict), "workflow has no jobs")
    definitions = document["jobs"]
    owners = [job for key, job in definitions.items()
              if isinstance(job, dict) and (job.get("name") or key) == context]
    _require(len(owners) == 1 and "strategy" not in owners[0], "required check is not one static workflow job")
    owner = owners[0]
    needs = owner.get("needs")
    needs = [needs] if isinstance(needs, str) else needs or []

    changed = _cached(cache, "files", lambda: _changed_paths(api, repo, number, pr))
    _require(path not in changed, "PR changes the gating workflow")
    filters: dict[str, bool] = {}
    for planner, output in _gating_terms(owner.get("if")):
        _require(planner in needs and isinstance(definitions.get(planner), dict), "gate is not a needed planner job")
        script, name, pattern, planner_name = _filter_rule(planner, definitions[planner], output)
        _require(script not in changed, "PR changes the path-filter script")
        _unchanged(api, cache, repo, script, base_sha, sha)
        planners = [j for j in jobs if j["name"] == planner_name]
        _require(len(planners) == 1 and planners[0]["status"] == "completed"
                 and planners[0]["conclusion"] == "success", "planner job did not succeed in this run")
        triggering = next((p for p in changed if pattern.search(p)), None)
        _require(triggering is None, f"changed path {triggering!r} matches filter {name!r}")
        filters[name] = False
    return {"run_id": run_id, "workflow": path, "base_sha": base_sha, "filters": filters,
            "changed_paths": len(changed)}


def _cached(cache: dict, key, load):
    if key not in cache:
        cache[key] = load()
    return cache[key]


def _run_jobs(api, repo: str, run_id: int) -> list[dict]:
    pages = api(f"repos/{repo}/actions/runs/{run_id}/jobs?per_page=100", paginate=True)
    jobs = [job for page in pages for job in page["jobs"]]
    _require(len({j["id"] for j in jobs}) == pages[0]["total_count"], "incomplete job pagination")
    return jobs


def _changed_paths(api, repo: str, number: int, pr: dict) -> frozenset[str]:
    expected = pr["changed_files"]
    _require(isinstance(expected, int) and 0 < expected <= _FILES_API_CAP, "changed-file list is not enumerable")
    entries = [f for page in api(f"repos/{repo}/pulls/{number}/files?per_page=100", paginate=True) for f in page]
    _require(len(entries) == expected, "incomplete changed-file pagination")
    # Renames count both sides: either name may be what the filter keys on.
    paths = {p for f in entries for p in (f["filename"], f.get("previous_filename")) if p is not None}
    _require(all(isinstance(p, str) and p and "\n" not in p for p in paths), "malformed changed path")
    return frozenset(paths)


def _unchanged(api, cache: dict, repo: str, path: str, base_sha: str, sha: str) -> str:
    base, head = (_cached(cache, ("file", path, ref), lambda ref=ref: _content(api, repo, path, ref))
                  for ref in (base_sha, sha))
    _require(base == head, f"{path} differs between the PR base and head")
    return base


def _content(api, repo: str, path: str, ref: str) -> str:
    entry = api(f"repos/{repo}/contents/{quote(path)}?ref={ref}")
    _require(entry.get("type") == "file" and entry.get("encoding") == "base64", f"{path} is not a file")
    return base64.b64decode(entry["content"]).decode("utf-8")


def _gating_terms(condition) -> list[tuple[str, str]]:
    """``needs.P.outputs.F == 'true'`` terms whose disjunction is the job's whole gate."""
    _require(isinstance(condition, str), "required job has no path gate")
    text = " ".join(condition.split())
    if text.startswith("${{") and text.endswith("}}"):
        text = text[3:-2].strip()
    if text.endswith(" && " + _GUARD):
        text = text[:-len(_GUARD) - 4]
        if " || " in text:
            _require(text.startswith("(") and text.endswith(")"), "unsupported job condition")
    if text.startswith("(") and text.endswith(")"):
        text = text[1:-1]
    terms = [_TERM.fullmatch(t) for t in text.split(" || ")]
    _require(all(terms), "unsupported job condition")
    return [(t[1], t[2]) for t in terms]


def _filter_rule(key: str, planner: dict, output: str):
    """Resolve a planner output to its static heredoc filter: (script, name, regex, job name)."""
    _require(not ({"strategy", "continue-on-error", "defaults"} & planner.keys()), "planner job is not static")
    wired = _STEP_OUTPUT.fullmatch(str((planner.get("outputs") or {}).get(output, "")).strip())
    _require(wired, f"planner output {output!r} is not a step output")
    steps = [s for s in planner.get("steps") or [] if isinstance(s, dict) and s.get("id") == wired[1]]
    _require(len(steps) == 1 and steps[0].keys() <= _STEP_KEYS, "filter step is not a plain run step")
    heredoc = _HEREDOC.fullmatch(str(steps[0].get("run", "")).strip() + "\n")
    _require(heredoc and ".." not in heredoc[1].split("/"), "filter step is not a static quoted heredoc")
    rules: dict[str, str] = {}
    for line in heredoc[3].split("\n"):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        name, sep, pattern = line.partition(":")
        name, pattern = "".join(name.split()), pattern.lstrip()
        _require(sep and name and pattern and name not in rules, "malformed filter rule")
        rules[name] = pattern
    pattern = rules.get(wired[2])
    _require(pattern is not None and _SHARED_ERE.fullmatch(pattern) and not _AMBIGUOUS_QUANTIFIER.search(pattern),
             f"filter {wired[2]!r} is undefined or unsupported")
    try:
        compiled = re.compile(pattern)
    except re.error:
        raise Unproven(f"filter {wired[2]!r} is not a valid expression") from None
    return heredoc[1], wired[2], compiled, planner.get("name") or key
