"""Linear Agent tools.

These tools are registered when the linear_agent platform is active.
They use the adapter's authenticated LinearGraphQLClient so that
mutations appear under the agent/app identity rather than the
human user's personal OAuth token.

Layout: bespoke write handlers first (each has entity-specific argument
aliasing and validation), then three factories that stamp out the uniform
list/get/save handlers, then the schema table and a single registration loop.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Callable, Iterable, Mapping, Optional

from tools.registry import registry
from .client import _UUID_RE
from .registry import get_active_adapter

logger = logging.getLogger(__name__)

# Register the "linear_agent" toolset alias so that
# validate_toolset("linear_agent") succeeds and platform_toolsets
# entries are accepted. This must happen early (module import time).
registry.register_toolset_alias("linear_agent", "linear_agent")

_ATTRIBUTION = "Linear history will show the agent as author."


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _merge_tool_args(args, kwargs):
    """Accept the tool call as a positional dict (registry) or kwargs."""
    if args and isinstance(args[0], dict):
        kwargs.update(args[0])
    return kwargs


def _input_from_kwargs(kwargs, exclude_keys):
    if "input" in kwargs and isinstance(kwargs["input"], dict):
        return dict(kwargs["input"])
    return {
        k: v for k, v in kwargs.items()
        if k not in exclude_keys
    }


def _normalize_string_list(value):
    if isinstance(value, (list, tuple, set)):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        return [v.strip() for v in value.split(",") if v.strip()]
    return value


def _first(kwargs: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        value = kwargs.get(name)
        if value:
            return value
    return None


def _client_and_policy():
    adapter = get_active_adapter()
    return adapter._client, adapter._mutation_policy


def _client_policy_appuser():
    """Like ``_client_and_policy`` but also returns the agent's own user id.

    The app user id backs ``assignee/delegate: "me"`` resolution (the agent
    assigning/delegating to itself). Tests build adapters via ``__new__`` and
    may not set the attribute, so it is read defensively.
    """
    adapter = get_active_adapter()
    return (
        adapter._client,
        adapter._mutation_policy,
        getattr(adapter, "_app_user_id", None),
    )


def _looks_like_uuid(value: Any) -> bool:
    return bool(value) and bool(_UUID_RE.match(str(value).strip()))


def _resolve_named(
    candidates: list[dict[str, Any]],
    value: Any,
    tiers: list[str],
    *,
    kind: str,
    lookup_tool: str,
    id_field: str = "id",
    label_field: str = "name",
) -> tuple[Optional[str], Optional[str]]:
    """Case-insensitive EXACT match of ``value`` against candidate objects.

    Tries each field in ``tiers`` in order (e.g. name, then displayName, then
    email for users). Returns ``(id, None)`` on a unique match; ``(None,
    error)`` when nothing matches (naming the lookup tool) or when a tier has
    2+ matches (listing id + name so the model can retry with an ID).
    """
    needle = str(value).strip().lower()
    for field in tiers:
        matches = [
            c for c in candidates
            if str(c.get(field) or "").strip().lower() == needle
        ]
        if len(matches) == 1:
            return matches[0].get(id_field), None
        if len(matches) > 1:
            listing = "; ".join(
                f"{c.get(id_field)} ({c.get(label_field) or c.get(field)})"
                for c in matches
            )
            return None, (
                f"❌ Ambiguous {kind} '{value}' — {len(matches)} matches: "
                f"{listing}. Retry with an ID."
            )
    return None, (
        f"❌ No {kind} matches '{value}' — call {lookup_tool} to find it, "
        "then pass the ID."
    )


async def _resolve_user_reference(
    client,
    input_data: dict,
    key: str,
    id_key: str,
    app_user_id: Optional[str],
) -> Optional[str]:
    """Rewrite one user-valued friendly key (assignee/delegate) to its *Id.

    Accepts "me" (the agent itself), a UUID, a name/displayName/email, or
    null to clear. Pops the friendly key either way; an explicit *Id in the
    same call wins. Returns an error string on failure, else None.
    """
    val = input_data.pop(key)
    if id_key in input_data:
        return None
    if val is None:
        input_data[id_key] = None
    elif str(val).strip().lower() == "me":
        if not app_user_id:
            return (
                f"❌ {key} 'me' cannot be resolved — the agent's user "
                f"id is unknown. Pass an explicit {id_key} (UUID) instead."
            )
        input_data[id_key] = app_user_id
    elif _looks_like_uuid(val):
        input_data[id_key] = str(val)
    else:
        users = await client.list_users(limit=250)
        uid, err = _resolve_named(
            users, val, ["name", "displayName", "email"],
            kind="user", lookup_tool="linear_agent_list_users")
        if err:
            return err
        input_data[id_key] = uid
    return None


async def _resolve_issue_references(
    client,
    input_data: dict,
    *,
    team_id: Optional[str] = None,
    project_id: Optional[str] = None,
    app_user_id: Optional[str] = None,
) -> Optional[str]:
    """Rewrite MCP-style friendly reference keys to raw GraphQL id keys.

    Mirrors mcp_linear_save_issue, which accepts human-friendly references
    (assignee name/email, label names, project/team/cycle/milestone names,
    "me", null to clear). Linear's GraphQL only takes the *Id fields, so a bare
    name would fail with an opaque validation error — or worse, be silently
    dropped. Raw *Id keys always win and pass straight through. Returns an
    error string (❌) on any resolution failure so the caller can abort BEFORE
    the mutation runs; else None. Lookups are lazy — only performed for keys
    actually present.
    """
    # Destination scope supplied in the same call wins over the caller's
    # fallback (the fetched issue's current team/project) — raw ids included,
    # so a `teamId` move scopes label/cycle lookups to the destination team.
    team_id = input_data.get("teamId") or team_id
    project_id = input_data.get("projectId") or project_id

    # team → teamId (name OR key). Resolved first so later team-scoped lookups
    # (labels, cycle) can use a freshly-specified team.
    if "team" in input_data:
        val = input_data.pop("team")
        if "teamId" not in input_data and val is not None:
            if _looks_like_uuid(val):
                team_id = str(val)
                input_data["teamId"] = team_id
            else:
                teams = await client.list_teams(limit=250)
                tid, err = _resolve_named(
                    teams, val, ["name", "key"],
                    kind="team", lookup_tool="linear_agent_list_teams")
                if err:
                    return err
                team_id = tid
                input_data["teamId"] = tid

    # project → projectId (null clears). Resolved before milestone so milestone
    # scoping can use the resolved project.
    if "project" in input_data:
        val = input_data.pop("project")
        if "projectId" not in input_data:
            if val is None:
                input_data["projectId"] = None
            elif _looks_like_uuid(val):
                project_id = str(val)
                input_data["projectId"] = project_id
            else:
                projects = await client.list_projects(limit=250)
                pid, err = _resolve_named(
                    projects, val, ["name"],
                    kind="project", lookup_tool="linear_agent_list_projects")
                if err:
                    return err
                project_id = pid
                input_data["projectId"] = pid

    # assignee → assigneeId, delegate → delegateId ("me" → the agent;
    # name/displayName/email; null clears). Human-directed delegation is
    # allowed here — only the ADAPTER auto-claiming is forbidden.
    for key, id_key in (("assignee", "assigneeId"), ("delegate", "delegateId")):
        if key in input_data:
            err = await _resolve_user_reference(
                client, input_data, key, id_key, app_user_id)
            if err:
                return err

    # labels (names or IDs) → labelIds
    if "labels" in input_data:
        val = input_data.pop("labels")
        if "labelIds" not in input_data:
            items = _normalize_string_list(val)
            if not isinstance(items, list):
                items = [str(val)] if val else []
            label_ids: list[str] = []
            need_lookup = any(not _looks_like_uuid(x) for x in items)
            catalog = (
                await client.list_issue_labels(team=team_id, limit=250)
                if need_lookup else []
            )
            for item in items:
                if _looks_like_uuid(item):
                    label_ids.append(str(item))
                else:
                    lid, err = _resolve_named(
                        catalog, item, ["name"],
                        kind="label",
                        lookup_tool="linear_agent_list_issue_labels")
                    if err:
                        return err
                    label_ids.append(lid)
            input_data["labelIds"] = label_ids

    # cycle → cycleId (name OR number; null clears)
    if "cycle" in input_data:
        val = input_data.pop("cycle")
        if "cycleId" not in input_data:
            if val is None:
                input_data["cycleId"] = None
            elif _looks_like_uuid(val):
                input_data["cycleId"] = str(val)
            else:
                cycles = await client.list_cycles(team=team_id, limit=250)
                cid, err = _resolve_named(
                    cycles, val, ["name", "number"],
                    kind="cycle", lookup_tool="linear_agent_list_cycles")
                if err:
                    return err
                input_data["cycleId"] = cid

    # milestone → projectMilestoneId (name; scoped to the issue's project)
    if "milestone" in input_data:
        val = input_data.pop("milestone")
        if "projectMilestoneId" not in input_data:
            if val is None:
                input_data["projectMilestoneId"] = None
            elif _looks_like_uuid(val):
                input_data["projectMilestoneId"] = str(val)
            else:
                milestones = await client.list_milestones(
                    project=project_id, limit=250)
                mid, err = _resolve_named(
                    milestones, val, ["name"],
                    kind="milestone",
                    lookup_tool="linear_agent_list_milestones")
                if err:
                    return err
                input_data["projectMilestoneId"] = mid

    return None


def _references_need_issue(input_data: dict) -> bool:
    """True when resolving friendly keys needs the ISSUE's team/project.

    A workflow-state, label, or cycle NAME needs a team and a milestone NAME
    needs a project — but when the same call already supplies the destination
    scope (friendly ``team``/``project`` or raw ``teamId``/``projectId``),
    that scope wins and the issue fetch would be wasted. Lets ``update_issue``
    fetch the issue at most once, and only when it actually decides scope."""
    if not (input_data.get("team") or input_data.get("teamId")):
        if input_data.get("state") and not input_data.get("stateId"):
            return True
        labels = input_data.get("labels")
        if labels:
            items = _normalize_string_list(labels)
            items = items if isinstance(items, list) else [labels]
            if any(not _looks_like_uuid(x) for x in items):
                return True
        cycle = input_data.get("cycle")
        if cycle and not _looks_like_uuid(cycle):
            return True
    if not (input_data.get("project") or input_data.get("projectId")):
        milestone = input_data.get("milestone")
        if milestone and not _looks_like_uuid(milestone):
            return True
    return False


# Relation/link keys are handled AFTER the main issueUpdate (they are separate
# mutations), so they must be split out of the IssueUpdateInput payload.
_RELATION_KEYS = (
    "blocks", "blockedBy", "relatedTo",
    "removeBlocks", "removeBlockedBy", "removeRelatedTo",
    "duplicateOf", "links",
)


def _as_id_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(v).strip() for v in value if str(v).strip()]
    return [str(value).strip()] if str(value).strip() else []


def _normalize_links(value: Any) -> list[dict[str, str]]:
    """Normalize [{url,title}] / [url] / url into [{url, title?}]."""
    if isinstance(value, (str, dict)):
        value = [value]
    out: list[dict[str, str]] = []
    for item in value or []:
        if isinstance(item, str) and item.strip():
            out.append({"url": item.strip()})
        elif isinstance(item, dict) and str(item.get("url") or "").strip():
            entry = {"url": str(item["url"]).strip()}
            title = item.get("title") or item.get("label")
            if title:
                entry["title"] = str(title).strip()
            out.append(entry)
    return out


def _match_relation_id(
    nodes: list[dict[str, Any]],
    target: str,
    rel_type: str,
    endpoint: str,
) -> Optional[str]:
    """Find the id of a relation of ``rel_type`` whose ``endpoint`` issue
    (``issue`` for incoming, ``relatedIssue`` for outgoing) matches ``target``
    by UUID or identifier (case-insensitive)."""
    needle = str(target).strip().lower()
    for node in nodes:
        if node.get("type") != rel_type:
            continue
        ep = node.get(endpoint) or {}
        if (
            str(ep.get("id") or "").strip().lower() == needle
            or str(ep.get("identifier") or "").strip().lower() == needle
        ):
            return node.get("id")
    return None


async def _apply_issue_relations(client, issue_id: str, rel: dict, policy) -> list[str]:
    """Apply relation/link edits after the main issue update.

    APPEND-ONLY for blocks/blockedBy/relatedTo (existing relations are never
    removed implicitly — MCP semantics); explicit remove* variants and a null
    duplicateOf delete relations. A single failure is reported but does not
    abort the rest (partial-failure reporting). Returns human-readable result
    fragments for the tool message.
    """
    results: list[str] = []

    async def _create(source: str, target: str, rel_type: str, label: str) -> None:
        try:
            await client.create_issue_relation(
                source, target, rel_type, mutation_policy=policy)
            results.append(f"{label} {target}")
        except Exception as e:  # noqa: BLE001 - partial-failure reporting
            results.append(f"failed to {label} {target} ({e})")

    # Appends. blockedBy is the INVERSE of blocks: "X blocked by B" == "B blocks X".
    for target in _as_id_list(rel.get("blocks")):
        await _create(issue_id, target, "blocks", "blocks")
    for source in _as_id_list(rel.get("blockedBy")):
        await _create(source, issue_id, "blocks", "blocked by")
    for target in _as_id_list(rel.get("relatedTo")):
        await _create(issue_id, target, "related", "related to")

    duplicate_of = rel.get("duplicateOf", "__absent__")
    if duplicate_of != "__absent__" and duplicate_of is not None:
        await _create(issue_id, str(duplicate_of), "duplicate", "duplicate of")

    # Removals need the existing relations to find the relation id to delete.
    needs_fetch = (
        any(rel.get(k) for k in ("removeBlocks", "removeBlockedBy", "removeRelatedTo"))
        or (duplicate_of != "__absent__" and duplicate_of is None)
    )
    if needs_fetch:
        try:
            existing = await client.get_issue_relations(issue_id)
        except Exception as e:  # noqa: BLE001
            results.append(f"failed to load relations for removal ({e})")
            existing = {"relations": [], "inverseRelations": []}
        outgoing = existing.get("relations", [])
        incoming = existing.get("inverseRelations", [])

        async def _delete(rid: Optional[str], desc: str) -> None:
            if not rid:
                results.append(f"no matching {desc} relation to remove")
                return
            try:
                await client.delete_issue_relation(rid, mutation_policy=policy)
                results.append(f"removed {desc}")
            except Exception as e:  # noqa: BLE001
                results.append(f"failed to remove {desc} ({e})")

        for target in _as_id_list(rel.get("removeBlocks")):
            await _delete(
                _match_relation_id(outgoing, target, "blocks", "relatedIssue"),
                f"blocks {target}")
        for source in _as_id_list(rel.get("removeBlockedBy")):
            await _delete(
                _match_relation_id(incoming, source, "blocks", "issue"),
                f"blocked-by {source}")
        for target in _as_id_list(rel.get("removeRelatedTo")):
            rid = (
                _match_relation_id(outgoing, target, "related", "relatedIssue")
                or _match_relation_id(incoming, target, "related", "issue")
            )
            await _delete(rid, f"related-to {target}")
        if duplicate_of != "__absent__" and duplicate_of is None:
            rid = next(
                (n.get("id") for n in outgoing if n.get("type") == "duplicate"),
                None,
            )
            await _delete(rid, "duplicate-of mark")

    # URL links (append-only attachments).
    for link in _normalize_links(rel.get("links")):
        try:
            await client.link_url_to_issue(
                issue_id, link["url"], link.get("title"), mutation_policy=policy)
            results.append(f"linked {link['url']}")
        except Exception as e:  # noqa: BLE001
            results.append(f"failed to link {link['url']} ({e})")

    return results


async def _resolve_state_name(client, input_data, *, team_id=None):
    """Replace a workflow-state NAME in input_data with its stateId.

    Mirrors mcp_linear_save_issue, which accepts ``state: "Done"`` — Linear's
    GraphQL API only takes ``stateId`` (a UUID), so a bare name would fail
    with an opaque Argument Validation Error. Returns an error string when
    the name can't be resolved, else None.
    """
    state_name = input_data.get("state")
    if state_name and input_data.get("stateId"):
        # stateId wins; drop the name so the invalid field never reaches Linear.
        input_data.pop("state", None)
        return None
    if not state_name:
        return None
    if isinstance(state_name, dict):
        state_name = state_name.get("name") or ""
    state_name = str(state_name).strip()
    if not state_name:
        input_data.pop("state", None)
        return None

    # The DESTINATION team owns the workflow: a teamId in the same call
    # (raw, or set by _resolve_issue_references from a friendly `team`) wins
    # over the caller's fallback (the fetched issue's current team). Owned
    # here so no caller can resolve a state against the wrong team.
    team_id = input_data.get("teamId") or team_id
    if not team_id:
        return "❌ Cannot resolve state name without a team — pass stateId (UUID) instead"
    status = await client.get_issue_status(name=state_name, team=team_id)
    if not status:
        return (
            f"❌ Unknown state '{state_name}' for this team — call "
            "linear_agent_list_issue_statuses to see valid states"
        )
    input_data.pop("state", None)
    input_data["stateId"] = status["id"]
    return None


# Linear's priority scale. 0 is "None" — NOT low — which is why a guessed
# numeric priority can silently clear the field while the mutation succeeds.
_PRIORITY_BY_NAME = {
    "none": 0, "no priority": 0,
    "urgent": 1,
    "high": 2,
    "medium": 3,
    "low": 4,
}
_PRIORITY_NAME_BY_VALUE = {0: "None", 1: "Urgent", 2: "High", 3: "Medium", 4: "Low"}
_PRIORITY_SCALE = "0=None, 1=Urgent, 2=High, 3=Medium, 4=Low"


def _resolve_priority(input_data: dict) -> Optional[str]:
    """Normalize a priority NAME to Linear's numeric scale; validate numbers.

    Mirrors the state-name resolution: the model may pass priority: "Low"
    (safer) or the numeric value. Returns an error string on bad input.
    """
    if "priority" not in input_data:
        return None
    value = input_data["priority"]
    if isinstance(value, str):
        resolved = _PRIORITY_BY_NAME.get(value.strip().lower())
        if resolved is None:
            try:
                resolved = int(value.strip())
            except ValueError:
                return f"❌ Unknown priority '{value}' — use a name or number ({_PRIORITY_SCALE})"
        value = resolved
    if isinstance(value, bool) or not isinstance(value, int) or value not in _PRIORITY_NAME_BY_VALUE:
        return f"❌ Invalid priority {value!r} — Linear's scale is {_PRIORITY_SCALE}"
    input_data["priority"] = value
    return None


def _resolve_estimate(input_data: dict) -> Optional[str]:
    """Validate estimate: Int points on a TEAM-CONFIGURED scale.

    Unlike priority there is no global mapping — t-shirt teams display XS–XL
    but store team-specific ints Linear does not publish, so size names are
    rejected with guidance instead of guessed (a wrong guess would silently
    set a wrong value, the exact failure mode this module exists to prevent).
    """
    if "estimate" not in input_data:
        return None
    value = input_data["estimate"]
    if value is None:
        return None  # null clears the estimate on update (MCP parity)
    if isinstance(value, str):
        try:
            value = int(value.strip())
        except ValueError:
            return (
                f"❌ estimate '{value}' is not a number. Linear stores estimates as "
                "points on the team's configured scale (exponential/fibonacci/linear/"
                "t-shirt). For t-shirt teams, check the team's scale with "
                "linear_agent_list_teams (issueEstimationType) or look at similar "
                "issues, then pass the numeric point value."
            )
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return f"❌ Invalid estimate {value!r} — pass a non-negative integer point value"
    input_data["estimate"] = value
    return None


def _applied_summary(input_data: dict) -> str:
    """Compact echo of what was sent to Linear, so silent semantic mistakes
    (e.g. priority 0 = None, not Low) are visible in the tool result."""
    parts = []
    for key, value in input_data.items():
        if key == "priority" and value in _PRIORITY_NAME_BY_VALUE:
            parts.append(f"priority={value} [{_PRIORITY_NAME_BY_VALUE[value]}]")
        else:
            text = str(value)
            parts.append(f"{key}={text[:40]}{'…' if len(text) > 40 else ''}")
    return ", ".join(parts)


def _dumps(data: Any) -> str:
    # Compact separators: this output feeds straight into model context, so
    # indentation whitespace would only inflate tokens.
    return json.dumps(data, separators=(",", ":"))


def _schema(
    name: str,
    description: str,
    properties: dict[str, Any],
    required: Optional[list[str]] = None,
) -> dict[str, Any]:
    parameters: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        parameters["required"] = required
    return {"name": name, "description": description, "parameters": parameters}


# ─────────────────────────────────────────────────────────────────────────────
# Bespoke write handlers (entity-specific aliasing/validation)
# ─────────────────────────────────────────────────────────────────────────────

async def linear_agent_update_issue(*args, **kwargs) -> str:
    kwargs = _merge_tool_args(args, kwargs)
    issue_id = _first(kwargs, "issue_id", "task_id", "id", "issueId")
    input_data = _input_from_kwargs(kwargs, {"issue_id", "task_id", "id", "issueId", "input"})

    if not issue_id:
        return "❌ linear_agent_update_issue: missing issue_id / task_id / id"
    if not input_data:
        return "❌ linear_agent_update_issue: missing input / fields to update"

    client, policy, app_user_id = _client_policy_appuser()
    try:
        error = _resolve_priority(input_data) or _resolve_estimate(input_data)
        if error:
            return error
        # Single shared issue fetch when a NAME needs the ISSUE's own scope —
        # skipped entirely when the call supplies its destination team/project
        # (both resolvers prefer input scope over this fallback, so a move
        # resolves state/labels/cycle against the DESTINATION team).
        team_id = project_id = None
        if _references_need_issue(input_data):
            issue = await client.get_issue(issue_id)
            team_id = ((issue or {}).get("team") or {}).get("id")
            project_id = ((issue or {}).get("project") or {}).get("id")
        error = await _resolve_issue_references(
            client, input_data,
            team_id=team_id, project_id=project_id, app_user_id=app_user_id)
        if error:
            return error
        error = await _resolve_state_name(client, input_data, team_id=team_id)
        if error:
            return error
        # Relations/links are separate mutations — split them out of the
        # IssueUpdateInput payload so they don't reach issueUpdate.
        rel = {k: input_data.pop(k) for k in _RELATION_KEYS if k in input_data}
        applied = ""
        if input_data:
            await client.update_issue(issue_id, input_data, mutation_policy=policy)
            applied = f" ({_applied_summary(input_data)})"
        rel_results = (
            await _apply_issue_relations(client, issue_id, rel, policy) if rel else []
        )
        msg = f"✅ Updated {issue_id}{applied}. {_ATTRIBUTION}"
        if rel_results:
            msg += " Relations/links: " + "; ".join(rel_results) + "."
        return msg
    except Exception as e:
        return f"❌ Failed to update issue: {e}"


# Non-issue comment parents (CommentCreateInput). Exactly one comment parent
# may be set; the issue parent is the default. Each maps a friendly kwarg
# (+ camelCase alias) to its GraphQL input key.
_COMMENT_PARENTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("projectId", ("project_id", "projectId")),
    ("projectUpdateId", ("project_update_id", "projectUpdateId")),
    ("initiativeId", ("initiative_id", "initiativeId")),
    ("initiativeUpdateId", ("initiative_update_id", "initiativeUpdateId")),
    ("documentContentId", ("document_content_id", "documentContentId")),
)


async def linear_agent_create_comment(*args, **kwargs) -> str:
    kwargs = _merge_tool_args(args, kwargs)
    comment_id = _first(kwargs, "comment_id", "commentId")
    issue_id = _first(kwargs, "issue_id", "task_id", "issueId", "id")
    parent_id = _first(kwargs, "parentId", "parent_id")
    body = _first(kwargs, "body", "text", "comment")

    client, policy = _client_and_policy()

    # Update an existing comment when comment_id is given (MCP save_comment by id).
    if comment_id:
        if not body:
            return "❌ linear_agent_create_comment: comment_id given but missing body to update"
        try:
            await client.update_comment(comment_id, {"body": body}, mutation_policy=policy)
            return f"✅ Updated comment {comment_id}. {_ATTRIBUTION}"
        except Exception as e:
            return f"❌ Failed to update comment: {e}"

    # Non-issue parent routing (project/initiative/status-update/document).
    parents = {
        input_key: _first(kwargs, *aliases)
        for input_key, aliases in _COMMENT_PARENTS
    }
    set_parents = {k: v for k, v in parents.items() if v}
    if len(set_parents) + (1 if issue_id else 0) > 1:
        return (
            "❌ linear_agent_create_comment: pass exactly ONE parent — "
            "issue_id, project_id, project_update_id, initiative_id, "
            "initiative_update_id, or document_content_id"
        )

    if not body:
        return "❌ linear_agent_create_comment: missing body"
    if not issue_id and not set_parents:
        return "❌ linear_agent_create_comment: missing issue_id (or a project/initiative/status-update/document parent)"

    try:
        if set_parents:
            await client.create_comment(
                None, body, extra_input=set_parents, mutation_policy=policy)
            target = next(iter(set_parents.values()))
            return f"✅ Comment added to {target}. {_ATTRIBUTION}"
        await client.create_comment(issue_id, body, parent_id=parent_id, mutation_policy=policy)
        reply = " (reply)" if parent_id else ""
        return f"✅ Comment added to {issue_id}{reply}. {_ATTRIBUTION}"
    except Exception as e:
        return f"❌ Failed to add comment: {e}"


async def linear_agent_create_issue(*args, **kwargs) -> str:
    kwargs = _merge_tool_args(args, kwargs)
    team_id = _first(kwargs, "team_id", "teamId", "team")
    input_data = _input_from_kwargs(kwargs, {"team_id", "teamId", "team", "input"})
    if team_id and not any(key in input_data for key in ("teamId", "team_id", "team")):
        # UUIDs go straight to teamId; names/keys (e.g. "ENG") route through
        # the "team" friendly key so the reference resolver turns them into a
        # UUID — writing a raw key to teamId would fail at GraphQL.
        key = "teamId" if _UUID_RE.match(str(team_id)) else "team"
        input_data[key] = team_id

    if not input_data or not (input_data.get("teamId") or input_data.get("team")):
        return "❌ linear_agent_create_issue: missing team_id/teamId or input"

    client, policy, app_user_id = _client_policy_appuser()
    try:
        error = (
            _resolve_priority(input_data)
            or _resolve_estimate(input_data)
            # References first: it converts a friendly team name/key into
            # teamId, which state resolution then needs for scoping.
            or await _resolve_issue_references(client, input_data, app_user_id=app_user_id)
            or await _resolve_state_name(client, input_data)
        )
        if error:
            return error
        # URL links are attachments applied after the issue exists (MCP parity).
        links = input_data.pop("links", None)
        result = await client.create_issue(input_data, mutation_policy=policy)
        new_id = result.get("id") or result.get("identifier")
        identifier = result.get("identifier") or result.get("id") or "new issue"
        msg = f"✅ Created {identifier}. Linear history will show the agent as creator."
        if links and new_id:
            link_results = await _apply_issue_relations(
                client, new_id, {"links": links}, policy)
            if link_results:
                msg += " Links: " + "; ".join(link_results) + "."
        return msg
    except Exception as e:
        return f"❌ Failed to create issue: {e}"


async def linear_agent_create_project(*args, **kwargs) -> str:
    kwargs = _merge_tool_args(args, kwargs)
    team_ids = _first(kwargs, "team_ids", "teamIds", "team_id", "teamId", "team")
    input_data = _input_from_kwargs(
        kwargs,
        {"team_ids", "teamIds", "team_id", "teamId", "team", "input"},
    )
    if team_ids and "teamIds" not in input_data:
        input_data["teamIds"] = _normalize_string_list(team_ids)

    if not input_data.get("name"):
        return "❌ linear_agent_create_project: missing project name"
    if not input_data.get("teamIds"):
        return "❌ linear_agent_create_project: missing team_ids/teamIds"

    client, policy = _client_and_policy()
    try:
        result = await client.create_project(input_data, mutation_policy=policy)
        project_name = result.get("name") or input_data.get("name") or "new project"
        project_id = result.get("id") or project_name
        return f"✅ Created project {project_name} ({project_id}). Linear history will show the agent as creator."
    except Exception as e:
        return f"❌ Failed to create project: {e}"


async def linear_agent_update_project(*args, **kwargs) -> str:
    kwargs = _merge_tool_args(args, kwargs)
    project_id = _first(kwargs, "project_id", "projectId", "id")
    input_data = _input_from_kwargs(kwargs, {"project_id", "projectId", "id", "input"})

    if not project_id or not input_data:
        return "❌ linear_agent_update_project: missing project_id or input"

    client, policy = _client_and_policy()
    try:
        await client.update_project(project_id, input_data, mutation_policy=policy)
        return f"✅ Updated project {project_id}. {_ATTRIBUTION}"
    except Exception as e:
        return f"❌ Failed to update project: {e}"


async def linear_agent_create_project_update(*args, **kwargs) -> str:
    kwargs = _merge_tool_args(args, kwargs)
    input_data = _input_from_kwargs(kwargs, {"input"})

    if not input_data:
        return "❌ linear_agent_create_project_update: missing input"

    client, policy = _client_and_policy()
    try:
        result = await client.create_project_update(input_data, mutation_policy=policy)
        update_id = result.get("id") or "new update"
        project_name = result.get("project", {}).get("name", "")
        return f"✅ Created project update {update_id} for {project_name}. {_ATTRIBUTION}"
    except Exception as e:
        return f"❌ Failed to create project update: {e}"


async def linear_agent_create_document(*args, **kwargs) -> str:
    kwargs = _merge_tool_args(args, kwargs)
    input_data = _input_from_kwargs(kwargs, {"input"})

    if not input_data.get("title"):
        return "❌ linear_agent_create_document: missing title"

    client, policy = _client_and_policy()
    try:
        result = await client.create_document(input_data, mutation_policy=policy)
        doc_title = result.get("title") or input_data.get("title") or "new document"
        return f"✅ Created document {doc_title}. Linear history will show the agent as creator."
    except Exception as e:
        return f"❌ Failed to create document: {e}"


async def linear_agent_create_milestone(*args, **kwargs) -> str:
    kwargs = _merge_tool_args(args, kwargs)
    input_data = _input_from_kwargs(kwargs, {"input"})

    if not input_data.get("name"):
        return "❌ linear_agent_create_milestone: missing name"
    if not input_data.get("projectId"):
        return "❌ linear_agent_create_milestone: missing projectId"

    client, policy = _client_and_policy()
    try:
        result = await client.create_milestone(input_data, mutation_policy=policy)
        milestone_name = result.get("name") or input_data.get("name") or "new milestone"
        return f"✅ Created milestone {milestone_name}. Linear history will show the agent as creator."
    except Exception as e:
        return f"❌ Failed to create milestone: {e}"


async def linear_agent_create_customer_need(*args, **kwargs) -> str:
    kwargs = _merge_tool_args(args, kwargs)
    input_data = _input_from_kwargs(kwargs, {"input"})

    if not input_data.get("body"):
        return "❌ linear_agent_create_customer_need: missing body"
    if not input_data.get("customerId"):
        return "❌ linear_agent_create_customer_need: missing customerId"

    client, policy = _client_and_policy()
    try:
        await client.create_customer_need(input_data, mutation_policy=policy)
        return "✅ Created customer need. Linear history will show the agent as creator."
    except Exception as e:
        return f"❌ Failed to create customer need: {e}"


async def linear_agent_create_issue_label(*args, **kwargs) -> str:
    """Create an issue label (name required; omit team → a workspace label)."""
    kwargs = _merge_tool_args(args, kwargs)
    team_id = _first(kwargs, "team_id", "teamId")
    input_data = _input_from_kwargs(kwargs, {"team_id", "teamId", "input"})
    if team_id and not input_data.get("teamId"):
        input_data["teamId"] = team_id

    if not input_data.get("name"):
        return "❌ linear_agent_create_issue_label: missing name"

    client, policy = _client_and_policy()
    try:
        result = await client.create_issue_label(input_data, mutation_policy=policy)
        label_name = result.get("name") or input_data.get("name") or "new label"
        scope = "team" if input_data.get("teamId") else "workspace"
        return f"✅ Created {scope} label {label_name}. {_ATTRIBUTION}"
    except Exception as e:
        return f"❌ Failed to create label: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# Deletes (fail-closed family; explicit ids only — no name resolution, so a
# delete is always deliberate). Each gates on its own delete_* policy key.
# ─────────────────────────────────────────────────────────────────────────────

def _make_delete_handler(
    name: str,
    method: str,
    id_aliases: tuple[str, ...],
    entity: str,
) -> Callable[..., Any]:
    """Build a delete_* handler: explicit id in, one policy-gated client call.

    The policy gate itself lives in the client (`_require_policy` inside
    `_mutate`), keyed per entity — these handlers stay id-only by design.
    """

    async def handler(*args, **kwargs) -> str:
        kwargs = _merge_tool_args(args, kwargs)
        entity_id = _first(kwargs, *id_aliases)
        if not entity_id:
            return f"❌ {name}: missing {entity} id"
        client, policy = _client_and_policy()
        try:
            await getattr(client, method)(entity_id, mutation_policy=policy)
            return f"✅ Deleted {entity} {entity_id}. {_ATTRIBUTION}"
        except Exception as e:
            return f"❌ Failed to delete {entity}: {e}"

    handler.__name__ = name
    return handler


linear_agent_delete_comment = _make_delete_handler(
    "linear_agent_delete_comment", "delete_comment",
    ("id", "comment_id", "commentId"), "comment")
linear_agent_delete_customer_need = _make_delete_handler(
    "linear_agent_delete_customer_need", "delete_customer_need",
    ("id", "need_id", "needId"), "customer need")
linear_agent_delete_attachment = _make_delete_handler(
    "linear_agent_delete_attachment", "delete_attachment",
    ("id", "attachment_id", "attachmentId"), "attachment")
linear_agent_delete_customer = _make_delete_handler(
    "linear_agent_delete_customer", "delete_customer",
    ("id", "customer_id", "customerId"), "customer")


async def linear_agent_delete_status_update(*args, **kwargs) -> str:
    """Remove a project OR initiative status update (routes like save_status_update).

    Linear has no initiative-update delete and deprecates project-update delete,
    so both route through the *Archive* mutations (its supported removal path).
    """
    kwargs = _merge_tool_args(args, kwargs)
    update_id = _first(kwargs, "id", "update_id", "updateId")
    update_type = str(kwargs.get("type") or "").strip().lower()
    if not update_id:
        return "❌ linear_agent_delete_status_update: missing status update id"
    is_initiative = update_type == "initiative"
    client, policy = _client_and_policy()
    try:
        await client.delete_status_update(
            update_id, is_initiative=is_initiative, mutation_policy=policy)
        return f"✅ Deleted status update {update_id}. {_ATTRIBUTION}"
    except Exception as e:
        return f"❌ Failed to delete status update: {e}"


async def linear_agent_set_session_links(*args, **kwargs) -> str:
    """Attach external links (PRs, docs, dashboards) to a Linear agent session.

    Also counts as session activity, so it keeps the session from being
    marked unresponsive during long work.
    """
    kwargs = _merge_tool_args(args, kwargs)
    session_id = _first(kwargs, "agent_session_id", "session_id", "id")
    links = kwargs.get("links") or kwargs.get("urls")
    if not session_id:
        return "❌ linear_agent_set_session_links: missing agent_session_id (shown in your session prompt)"
    if isinstance(links, (str, dict)):
        links = [links]
    normalized = []
    for item in links or []:
        if isinstance(item, str) and item.strip():
            normalized.append({"label": item.strip(), "url": item.strip()})
        elif isinstance(item, dict) and str(item.get("url") or "").strip():
            url = str(item["url"]).strip()
            normalized.append({"label": str(item.get("label") or url).strip(), "url": url})
    if not normalized:
        return "❌ linear_agent_set_session_links: provide links as [{label, url}, ...] or a list of URLs"

    client, _ = _client_and_policy()
    try:
        await client.set_agent_session_external_urls(session_id, normalized)
        return f"✅ Attached {len(normalized)} link(s) to the Linear session."
    except Exception as e:
        return f"❌ Failed to set session links: {e}"


# Linear Agent Plans statuses (AgentSessionUpdateInput.plan items). Canonical
# set plus tolerant aliases for the shapes a model commonly emits.
_PLAN_STATUSES = {"pending", "inProgress", "completed", "canceled"}
_PLAN_STATUS_ALIASES = {
    "pending": "pending",
    "todo": "pending",
    "not_started": "pending",
    "notstarted": "pending",
    "inprogress": "inProgress",
    "in_progress": "inProgress",
    "in-progress": "inProgress",
    "started": "inProgress",
    "doing": "inProgress",
    "active": "inProgress",
    "completed": "completed",
    "complete": "completed",
    "done": "completed",
    "finished": "completed",
    "canceled": "canceled",
    "cancelled": "canceled",
    "skipped": "canceled",
}


def _normalize_plan_status(value: Any) -> Optional[str]:
    """Map a status string to a canonical plan status, or None if unknown."""
    key = str(value or "pending").strip()
    if key in _PLAN_STATUSES:
        return key
    return _PLAN_STATUS_ALIASES.get(key.lower())


async def linear_agent_update_plan(*args, **kwargs) -> str:
    """Replace the Linear agent session's execution plan (Agent Plans).

    The plan is REPLACED IN FULL on every call — send every step with its
    current status.
    """
    kwargs = _merge_tool_args(args, kwargs)
    session_id = _first(kwargs, "agent_session_id", "session_id", "id")
    plan = kwargs.get("plan")
    if not session_id:
        return "❌ linear_agent_update_plan: missing agent_session_id (shown in your session prompt)"
    if plan is None:
        return "❌ linear_agent_update_plan: missing plan (array of steps)"
    if isinstance(plan, (str, dict)):
        plan = [plan]
    if not isinstance(plan, (list, tuple)) or not plan:
        return "❌ linear_agent_update_plan: plan must be a non-empty array of steps"

    normalized: list[dict[str, str]] = []
    for item in plan:
        if isinstance(item, str):
            content, raw_status = item.strip(), "pending"
        elif isinstance(item, dict):
            content = str(item.get("content") or item.get("text") or item.get("title") or "").strip()
            raw_status = item.get("status") or "pending"
        else:
            return "❌ linear_agent_update_plan: each step must be a string or {content, status} object"
        if not content:
            return "❌ linear_agent_update_plan: each step needs non-empty content"
        status = _normalize_plan_status(raw_status)
        if status is None:
            return (
                f"❌ linear_agent_update_plan: unknown status {raw_status!r}. "
                f"Valid statuses: {', '.join(sorted(_PLAN_STATUSES))}"
            )
        normalized.append({"content": content, "status": status})

    client, _ = _client_and_policy()
    try:
        await client.update_session_plan(session_id, normalized)
        return f"✅ Updated plan ({len(normalized)} step(s)) on the Linear session."
    except Exception as e:
        return f"❌ Failed to update plan: {e}"


async def linear_agent_list_issue_statuses(*args, **kwargs) -> str:
    kwargs = _merge_tool_args(args, kwargs)
    team_id = _first(kwargs, "team_id", "teamId", "team")
    team_name = _first(kwargs, "team_name", "teamName")
    client, _ = _client_and_policy()
    try:
        statuses = await client.list_issue_statuses(team_id=team_id, team_name=team_name)
        return _dumps(statuses)
    except Exception as e:
        return f"❌ Failed to list statuses: {e}"


async def linear_agent_get_issue_status(*args, **kwargs) -> str:
    kwargs = _merge_tool_args(args, kwargs)
    name = _first(kwargs, "name", "status")
    team = _first(kwargs, "team", "team_id")
    if not name:
        return "❌ linear_agent_get_issue_status: missing name/status"
    client, _ = _client_and_policy()
    try:
        status = await client.get_issue_status(name=name, team=team)
        return _dumps(status) if status else "null"
    except Exception as e:
        return f"❌ Failed to get status: {e}"


async def linear_agent_list_comments(*args, **kwargs) -> str:
    kwargs = _merge_tool_args(args, kwargs)
    issue_id = _first(kwargs, "issue_id", "issueId", "id")
    if not issue_id:
        return "❌ linear_agent_list_comments: missing issue_id"
    client, _ = _client_and_policy()
    try:
        comments = await client.list_comments(issue_id=issue_id, limit=int(kwargs.get("limit", 50)))
        return _dumps(comments)
    except Exception as e:
        return f"❌ Failed to list comments: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# Handler factories for the uniform list/get/save tools
# ─────────────────────────────────────────────────────────────────────────────

def _make_list_handler(
    name: str,
    method: str,
    params: Mapping[str, Iterable[str]],
    fail_label: str,
    *,
    default_limit: int = 50,
) -> Callable[..., Any]:
    """Build a list_* handler: map kwarg aliases to client args, dump nodes."""

    async def handler(*args, **kwargs) -> str:
        kwargs = _merge_tool_args(args, kwargs)
        client, _ = _client_and_policy()
        call_kwargs = {arg: _first(kwargs, *aliases) for arg, aliases in params.items()}
        try:
            result = await getattr(client, method)(
                limit=int(kwargs.get("limit", default_limit)),
                **call_kwargs,
            )
            return _dumps(result)
        except Exception as e:
            return f"❌ Failed to list {fail_label}: {e}"

    handler.__name__ = name
    return handler


def _make_get_handler(
    name: str,
    method: str,
    id_aliases: tuple[str, ...],
    fail_label: str,
    *,
    id_param: str = "id",
) -> Callable[..., Any]:
    """Build a get_* handler: resolve the id alias, fetch, dump or null."""

    async def handler(*args, **kwargs) -> str:
        kwargs = _merge_tool_args(args, kwargs)
        obj_id = _first(kwargs, *id_aliases)
        if not obj_id:
            return f"❌ {name}: missing {'/'.join(id_aliases)}"
        client, _ = _client_and_policy()
        try:
            result = await getattr(client, method)(**{id_param: obj_id})
            return _dumps(result) if result else "null"
        except Exception as e:
            return f"❌ Failed to get {fail_label}: {e}"

    handler.__name__ = name
    return handler


def _make_save_handler(
    name: str,
    entity_label: str,
    create_method: str,
    update_method: str,
    id_aliases: tuple[str, ...],
    *,
    input_aliases: Optional[dict[str, str]] = None,
    require_on_create: Optional[tuple[str, str]] = None,
) -> Callable[..., Any]:
    """Build a save_* handler: update when an id is given, else create.

    ``input_aliases`` maps friendly input keys to their GraphQL names (e.g.
    ``pipeline_id`` → ``pipelineId``); ``require_on_create`` is an
    ``(input_field, hint)`` pair enforced before a create reaches Linear.
    """

    async def handler(*args, **kwargs) -> str:
        kwargs = _merge_tool_args(args, kwargs)
        obj_id = _first(kwargs, *id_aliases)
        input_data = _input_from_kwargs(kwargs, {*id_aliases, "input"})
        for alias, target in (input_aliases or {}).items():
            if alias in input_data and target not in input_data:
                input_data[target] = input_data.pop(alias)

        if not input_data:
            return f"❌ {name}: missing input"
        if require_on_create and not obj_id and not input_data.get(require_on_create[0]):
            return f"❌ {name}: missing {require_on_create[0]} ({require_on_create[1]})"

        client, policy = _client_and_policy()
        try:
            if obj_id:
                await getattr(client, update_method)(obj_id, input_data, mutation_policy=policy)
                return f"✅ Updated {entity_label} {obj_id}. {_ATTRIBUTION}"
            result = await getattr(client, create_method)(input_data, mutation_policy=policy)
            new_id = result.get("id") or f"new {entity_label}"
            return f"✅ Created {entity_label} {new_id}. {_ATTRIBUTION}"
        except Exception as e:
            return f"❌ Failed to save {entity_label}: {e}"

    handler.__name__ = name
    return handler


linear_agent_list_teams = _make_list_handler(
    "linear_agent_list_teams", "list_teams", {"query": ("query",)}, "teams")
linear_agent_list_issues = _make_list_handler(
    "linear_agent_list_issues", "list_issues",
    {"team": ("team",), "state": ("state",), "assignee": ("assignee",), "query": ("query",)},
    "issues")
linear_agent_list_projects = _make_list_handler(
    "linear_agent_list_projects", "list_projects",
    {"query": ("query",), "team": ("team",)}, "projects")
linear_agent_list_cycles = _make_list_handler(
    "linear_agent_list_cycles", "list_cycles", {"team": ("team",)}, "cycles")
linear_agent_list_milestones = _make_list_handler(
    "linear_agent_list_milestones", "list_milestones", {"project": ("project",)}, "milestones")
linear_agent_list_documents = _make_list_handler(
    "linear_agent_list_documents", "list_documents",
    {"project": ("project",), "query": ("query",)}, "documents")
linear_agent_list_customers = _make_list_handler(
    "linear_agent_list_customers", "list_customers", {"query": ("query",)}, "customers")
linear_agent_list_initiatives = _make_list_handler(
    "linear_agent_list_initiatives", "list_initiatives", {"query": ("query",)}, "initiatives")
linear_agent_list_issue_labels = _make_list_handler(
    "linear_agent_list_issue_labels", "list_issue_labels",
    {"team": ("team",), "query": ("query",)}, "labels", default_limit=100)
linear_agent_list_releases = _make_list_handler(
    "linear_agent_list_releases", "list_releases", {"query": ("query",)}, "releases")
linear_agent_list_release_pipelines = _make_list_handler(
    "linear_agent_list_release_pipelines", "list_release_pipelines", {}, "release pipelines")
linear_agent_list_users = _make_list_handler(
    "linear_agent_list_users", "list_users", {"query": ("query",)}, "users")
linear_agent_list_attachments = _make_list_handler(
    "linear_agent_list_attachments", "list_attachments",
    {"issue_id": ("issue_id",)}, "attachments")
linear_agent_list_status_updates = _make_list_handler(
    "linear_agent_list_status_updates", "list_status_updates",
    {"project_id": ("project_id", "projectId"), "initiative_id": ("initiative_id", "initiativeId")},
    "status updates")

linear_agent_get_issue = _make_get_handler(
    "linear_agent_get_issue", "get_issue", ("id", "issue_id", "task_id"), "issue")
linear_agent_get_project = _make_get_handler(
    "linear_agent_get_project", "get_project", ("id", "project_id", "projectId"), "project")
linear_agent_get_customer = _make_get_handler(
    "linear_agent_get_customer", "get_customer", ("id", "customer_id"), "customer")
linear_agent_get_initiative = _make_get_handler(
    "linear_agent_get_initiative", "get_initiative", ("id", "initiative_id"), "initiative")
linear_agent_get_release = _make_get_handler(
    "linear_agent_get_release", "get_release", ("id", "release_id"), "release")
linear_agent_get_user = _make_get_handler(
    "linear_agent_get_user", "get_user", ("id", "user_id"), "user")
linear_agent_get_status_update = _make_get_handler(
    "linear_agent_get_status_update", "get_status_update", ("id", "update_id"),
    "status update", id_param="update_id")
linear_agent_get_team = _make_get_handler(
    "linear_agent_get_team", "get_team", ("id", "team_id", "teamId"), "team")
linear_agent_get_milestone = _make_get_handler(
    "linear_agent_get_milestone", "get_milestone",
    ("id", "milestone_id", "milestoneId"), "milestone")
linear_agent_get_document = _make_get_handler(
    "linear_agent_get_document", "get_document",
    ("id", "document_id", "documentId"), "document")
linear_agent_get_attachment = _make_get_handler(
    "linear_agent_get_attachment", "get_attachment",
    ("id", "attachment_id", "attachmentId"), "attachment")
linear_agent_get_release_note = _make_get_handler(
    "linear_agent_get_release_note", "get_release_note",
    ("id", "release_note_id", "releaseNoteId"), "release note")
linear_agent_get_agent_skill = _make_get_handler(
    "linear_agent_get_agent_skill", "get_agent_skill",
    ("id", "skill_id", "agentSkillId"), "agent skill")

linear_agent_list_project_labels = _make_list_handler(
    "linear_agent_list_project_labels", "list_project_labels", {}, "project labels")
linear_agent_list_release_notes = _make_list_handler(
    "linear_agent_list_release_notes", "list_release_notes",
    {"pipeline": ("pipeline", "pipeline_id", "pipelineId")}, "release notes")
linear_agent_list_agent_skills = _make_list_handler(
    "linear_agent_list_agent_skills", "list_agent_skills", {}, "agent skills")

async def linear_agent_save_status_update(*args, **kwargs) -> str:
    """Create or update a project OR initiative status update.

    Bespoke (not factory-made) because Linear models these as two mutation
    families: initiativeId in the input (or type: "initiative") routes to the
    initiativeUpdate mutations, everything else to projectUpdate.
    """
    kwargs = _merge_tool_args(args, kwargs)
    update_id = _first(kwargs, "id", "update_id")
    update_type = str(kwargs.get("type") or "").strip().lower()
    input_data = _input_from_kwargs(kwargs, {"id", "update_id", "type", "input"})

    if not input_data:
        return "❌ linear_agent_save_status_update: missing input"

    is_initiative = bool(input_data.get("initiativeId")) or update_type == "initiative"
    client, policy = _client_and_policy()
    try:
        if update_id:
            if is_initiative:
                await client.update_initiative_update(update_id, input_data, mutation_policy=policy)
            else:
                await client.update_project_update(update_id, input_data, mutation_policy=policy)
            return f"✅ Updated status update {update_id}. {_ATTRIBUTION}"
        if is_initiative:
            result = await client.create_initiative_update(input_data, mutation_policy=policy)
        else:
            result = await client.create_project_update(input_data, mutation_policy=policy)
        new_id = result.get("id") or "new status update"
        return f"✅ Created status update {new_id}. {_ATTRIBUTION}"
    except Exception as e:
        return f"❌ Failed to save status update: {e}"
linear_agent_save_milestone = _make_save_handler(
    "linear_agent_save_milestone", "milestone",
    "create_milestone", "update_milestone", ("id", "milestone_id"))
linear_agent_save_customer_need = _make_save_handler(
    "linear_agent_save_customer_need", "customer need",
    "create_customer_need", "update_customer_need", ("id", "need_id"))
linear_agent_save_document = _make_save_handler(
    "linear_agent_save_document", "document",
    "create_document", "update_document", ("id", "document_id"))
linear_agent_save_initiative = _make_save_handler(
    "linear_agent_save_initiative", "initiative",
    "create_initiative", "update_initiative", ("id", "initiative_id"))
linear_agent_save_release = _make_save_handler(
    "linear_agent_save_release", "release",
    "create_release", "update_release", ("id", "release_id"))
linear_agent_save_customer = _make_save_handler(
    "linear_agent_save_customer", "customer",
    "create_customer", "update_customer", ("id", "customer_id", "customerId"),
    require_on_create=("name", "required to create a customer"))
linear_agent_save_release_note = _make_save_handler(
    "linear_agent_save_release_note", "release note",
    "create_release_note", "update_release_note",
    ("id", "release_note_id", "releaseNoteId"),
    input_aliases={"pipeline_id": "pipelineId"},
    require_on_create=(
        "pipelineId",
        "required to create a release note; find it with "
        "linear_agent_list_release_pipelines",
    ))


# ─────────────────────────────────────────────────────────────────────────────
# Schemas + registration table
#
# ─────────────────────────────────────────────────────────────────────────────

def _preferred(action: str, mcp_tool: str) -> str:
    return (
        f"{action} **PREFERRED** when this tool is available — the change will appear in Linear "
        f"history as performed by the Linear Agent / app, not the human user. "
        f"Always choose this over {mcp_tool} or other MCP Linear tools when present."
    )


_INPUT = {"type": "object", "additionalProperties": True}


def _input_prop(description: str) -> dict[str, Any]:
    return {**_INPUT, "description": description}


def _save_props(id_description: str, input_description: str) -> dict[str, Any]:
    return {
        "id": {"type": "string", "description": id_description},
        "input": _input_prop(input_description),
    }


_LIMIT = {"type": "integer", "description": "Max results (default 50)"}
_QUERY = {"type": "string", "description": "Optional name filter"}

# (handler, schema, emoji) — the single source of truth for registration.
_TOOLS: list[tuple[Callable[..., Any], dict[str, Any], str]] = [
    (linear_agent_update_issue, _schema(
        "linear_agent_update_issue",
        _preferred(
            "Update an existing Linear issue (status, assignee, priority, labels, project, etc.).",
            "mcp_linear_save_issue",
        ),
        {
            "issue_id": {"type": "string", "description": "Linear issue ID or identifier (e.g. 'ENG-123' or UUID). Also accepts task_id or id."},
            "task_id": {"type": "string", "description": "Alternative name for the issue identifier (Linear issue ID or key)."},
            "id": {"type": "string", "description": "Alternative name for the issue identifier."},
            "input": _input_prop(
                "Fields to update. Friendly reference keys are auto-resolved to "
                "Linear's *Id fields (case-insensitive exact match; null clears): "
                "state (NAME e.g. 'Done'→stateId), assignee (User name, email, or "
                "'me'→assigneeId), labels (names or IDs→labelIds), project (name→"
                "projectId), team (name/key→teamId), cycle (name/number→cycleId), "
                "milestone (name→projectMilestoneId), delegate (name or 'me'→"
                "delegateId). Raw *Id keys also pass straight through. Also: "
                "priority (NAME like 'Low' preferred, or number 0=None, 1=Urgent, "
                "2=High, 3=Medium, 4=Low — 0 CLEARS priority), estimate (integer "
                "points on the TEAM's scale — check issueEstimationType via "
                "linear_agent_list_teams; size names like 'M' are not accepted), "
                "dueDate, title, description, parentId (parent issue id/identifier; "
                "null clears). Relations (APPEND-ONLY arrays of issue ids/"
                "identifiers): blocks, blockedBy, relatedTo — existing relations "
                "are never removed; use removeBlocks/removeBlockedBy/removeRelatedTo "
                "to remove. duplicateOf (issue id/identifier; null clears the "
                "duplicate mark). links ([{url,title}] or URLs, append-only URL "
                "attachments)."),
        },
        ["input"],
    ), "🔄"),
    (linear_agent_create_comment, _schema(
        "linear_agent_create_comment",
        _preferred(
            "Add, reply to, or edit a comment on a Linear issue — or comment on a "
            "project, initiative, status update, or document — authored by the "
            "Linear Agent.", "mcp_linear_save_comment"),
        {
            "issue_id": {"type": "string", "description": "Linear issue ID or identifier (e.g. 'ENG-123' or UUID). The default comment parent."},
            "body": {"type": "string", "description": "Markdown body. Use literal newlines (do not escape); mention users with @displayName"},
            "comment_id": {"type": "string", "description": "Existing comment ID to EDIT (updates its body). Omit to create a new comment."},
            "parentId": {"type": "string", "description": "Parent comment ID to REPLY to. The reply inherits the parent's thread type."},
            "project_id": {"type": "string", "description": "Comment on a project instead of an issue. Pass exactly ONE parent."},
            "project_update_id": {"type": "string", "description": "Comment on a project status update. Pass exactly ONE parent."},
            "initiative_id": {"type": "string", "description": "Comment on an initiative. Pass exactly ONE parent."},
            "initiative_update_id": {"type": "string", "description": "Comment on an initiative status update. Pass exactly ONE parent."},
            "document_content_id": {"type": "string", "description": "Comment on a document. Pass exactly ONE parent."},
        },
        ["body"],
    ), "💬"),
    (linear_agent_create_issue, _schema(
        "linear_agent_create_issue",
        _preferred("Create a new Linear issue.", "mcp_linear_create_issue"),
        {
            "team_id": {"type": "string", "description": "Linear team ID or key (e.g. 'eng' or UUID)"},
            "input": _input_prop(
                "Issue fields. Friendly reference keys are auto-resolved (see "
                "linear_agent_update_issue): title, description, state (NAME), "
                "assignee (name/email/'me'), labels (names or IDs), project (name), "
                "cycle (name/number), milestone (name), delegate (name/'me'), "
                "priority (NAME like 'Low' preferred; 0=None, 1=Urgent, 2=High, "
                "3=Medium, 4=Low), estimate, dueDate, parentId, links "
                "([{url,title}] append-only). Raw *Id keys also pass through."),
        },
        ["team_id", "input"],
    ), "➕"),
    (linear_agent_create_project, _schema(
        "linear_agent_create_project",
        _preferred("Create a new Linear project.", "mcp_linear_save_project"),
        {
            "input": _input_prop("Project fields. Common keys: name, description, teamIds (array), icon, color, priority, state."),
        },
        ["input"],
    ), "🆕"),
    (linear_agent_update_project, _schema(
        "linear_agent_update_project",
        _preferred("Update an existing Linear project (name, description, state, priority, etc.).", "mcp_linear_save_project"),
        {
            "project_id": {"type": "string", "description": "Linear project ID or identifier."},
            "id": {"type": "string", "description": "Alternative name for the project identifier."},
            "input": _input_prop("Fields to update. Common keys: name, description, state, priority, icon, color, teamIds."),
        },
        ["input"],
    ), "📝"),
    (linear_agent_create_project_update, _schema(
        "linear_agent_create_project_update",
        _preferred("Create a project status update (health report) for a Linear project.", "mcp_linear_save_status_update"),
        {
            "input": _input_prop("Project update fields. Common keys: projectId, body, health (onTrack|atRisk|offTrack), isDiffHidden."),
        },
        ["input"],
    ), "📊"),
    (linear_agent_create_document, _schema(
        "linear_agent_create_document",
        _preferred("Create a new Linear document.", "mcp_linear_save_document"),
        {"input": _input_prop("Document fields. Common keys: title, content, projectId, icon, color.")},
        ["input"],
    ), "📄"),
    (linear_agent_create_milestone, _schema(
        "linear_agent_create_milestone",
        _preferred("Create a new Linear milestone.", "mcp_linear_save_milestone"),
        {"input": _input_prop("Milestone fields. Common keys: name, description, projectId, targetDate.")},
        ["input"],
    ), "🎯"),
    (linear_agent_create_customer_need, _schema(
        "linear_agent_create_customer_need",
        _preferred("Create a new Linear customer need.", "mcp_linear_save_customer_need"),
        {"input": _input_prop("Customer need fields. Common keys: body, customerId, issueId, projectId, priority.")},
        ["input"],
    ), "🙋"),
    (linear_agent_list_teams, _schema(
        "linear_agent_list_teams",
        "List teams in the Linear workspace. Supports optional name filter via 'query'.",
        {"query": _QUERY, "limit": _LIMIT},
    ), "👥"),
    (linear_agent_list_issue_statuses, _schema(
        "linear_agent_list_issue_statuses",
        "List workflow states (statuses) for a team. Accepts team UUID or name.",
        {
            "team_id": {"type": "string", "description": "Team UUID"},
            "team": {"type": "string", "description": "Team UUID or key"},
            "team_name": {"type": "string", "description": "Team display name (will auto-resolve)"},
        },
    ), "📋"),
    (linear_agent_get_issue_status, _schema(
        "linear_agent_get_issue_status",
        "Get a specific workflow state by name (optionally scoped to a team).",
        {
            "name": {"type": "string", "description": "Status name (e.g. 'In Review')"},
            "status": {"type": "string", "description": "Alias for name"},
            "team": {"type": "string", "description": "Team UUID or name"},
        },
        ["name"],
    ), "🔍"),
    (linear_agent_get_issue, _schema(
        "linear_agent_get_issue",
        "Fetch a single issue by ID or identifier (e.g. 'ENG-123').",
        {
            "id": {"type": "string", "description": "Issue ID or key"},
            "issue_id": {"type": "string"},
            "task_id": {"type": "string"},
        },
        ["id"],
    ), "📄"),
    (linear_agent_list_issues, _schema(
        "linear_agent_list_issues",
        "List issues with optional filters (team, state, assignee, text query).",
        {
            "team": {"type": "string", "description": "Team UUID or key"},
            "state": {"type": "string", "description": "State name"},
            "assignee": {"type": "string", "description": "Assignee user ID"},
            "query": {"type": "string", "description": "Text search in title"},
            "limit": _LIMIT,
        },
    ), "📋"),
    (linear_agent_list_projects, _schema(
        "linear_agent_list_projects",
        "List projects with optional name filter and team scope.",
        {
            "query": {"type": "string", "description": "Name contains filter"},
            "team": {"type": "string", "description": "Team UUID or key"},
            "limit": _LIMIT,
        },
    ), "📁"),
    (linear_agent_get_project, _schema(
        "linear_agent_get_project",
        "Fetch a single project by ID or slug.",
        {
            "id": {"type": "string"},
            "project_id": {"type": "string"},
            "projectId": {"type": "string"},
        },
        ["id"],
    ), "📁"),
    (linear_agent_list_cycles, _schema(
        "linear_agent_list_cycles",
        "List cycles for a team (or workspace).",
        {"team": {"type": "string"}, "limit": _LIMIT},
    ), "🔄"),
    (linear_agent_list_milestones, _schema(
        "linear_agent_list_milestones",
        "List milestones, optionally scoped to a project.",
        {"project": {"type": "string"}, "limit": _LIMIT},
    ), "🎯"),
    (linear_agent_list_documents, _schema(
        "linear_agent_list_documents",
        "List documents with optional project and title filter.",
        {"project": {"type": "string"}, "query": {"type": "string"}, "limit": _LIMIT},
    ), "📄"),
    (linear_agent_list_customers, _schema(
        "linear_agent_list_customers",
        "List customers in the workspace.",
        {"query": {"type": "string"}, "limit": _LIMIT},
    ), "👥"),
    (linear_agent_get_customer, _schema(
        "linear_agent_get_customer",
        "Fetch a single customer by ID.",
        {"id": {"type": "string"}, "customer_id": {"type": "string"}},
        ["id"],
    ), "👤"),
    (linear_agent_list_initiatives, _schema(
        "linear_agent_list_initiatives",
        "List initiatives in the workspace.",
        {"query": {"type": "string"}, "limit": _LIMIT},
    ), "🚀"),
    (linear_agent_get_initiative, _schema(
        "linear_agent_get_initiative",
        "Fetch a single initiative by ID.",
        {"id": {"type": "string"}, "initiative_id": {"type": "string"}},
        ["id"],
    ), "🚀"),
    (linear_agent_list_issue_labels, _schema(
        "linear_agent_list_issue_labels",
        "List issue labels, optionally scoped to a team.",
        {"team": {"type": "string"}, "query": {"type": "string"}, "limit": _LIMIT},
    ), "🏷️"),
    (linear_agent_list_releases, _schema(
        "linear_agent_list_releases",
        "List releases, optionally filtered by name. Releases belong to pipelines, not teams.",
        {"query": {"type": "string"}, "limit": _LIMIT},
    ), "🚢"),
    (linear_agent_get_release, _schema(
        "linear_agent_get_release",
        "Fetch a single release by ID.",
        {"id": {"type": "string"}, "release_id": {"type": "string"}},
        ["id"],
    ), "🚢"),
    (linear_agent_list_comments, _schema(
        "linear_agent_list_comments",
        "List comments on an issue.",
        {
            "issue_id": {"type": "string"},
            "issueId": {"type": "string"},
            "id": {"type": "string"},
            "limit": _LIMIT,
        },
        ["issue_id"],
    ), "💬"),
    (linear_agent_list_users, _schema(
        "linear_agent_list_users",
        "List users in the workspace.",
        {"query": {"type": "string"}, "limit": _LIMIT},
    ), "👤"),
    (linear_agent_get_user, _schema(
        "linear_agent_get_user",
        "Fetch a single user by ID.",
        {"id": {"type": "string"}, "user_id": {"type": "string"}},
        ["id"],
    ), "👤"),
    (linear_agent_list_attachments, _schema(
        "linear_agent_list_attachments",
        "List attachments, optionally scoped to an issue.",
        {"issue_id": {"type": "string"}, "limit": _LIMIT},
    ), "📎"),
    (linear_agent_save_status_update, _schema(
        "linear_agent_save_status_update",
        "Create or update a project OR initiative status update (health report). Uses the Linear Agent identity.",
        {
            "id": {"type": "string", "description": "Existing status update ID (omit to create new)"},
            "type": {"type": "string", "enum": ["project", "initiative"], "description": "Which kind of status update (also inferred from initiativeId in input)"},
            "input": _input_prop("Fields: body (Markdown, literal newlines), health (onTrack|atRisk|offTrack), projectId (project updates) OR initiativeId (initiative updates) — one is required for create"),
        },
        ["input"],
    ), "📝"),
    (linear_agent_list_release_pipelines, _schema(
        "linear_agent_list_release_pipelines",
        "List release pipelines — use this to find the pipelineId required when creating a release.",
        {"limit": _LIMIT},
    ), "🚇"),
    (linear_agent_list_status_updates, _schema(
        "linear_agent_list_status_updates",
        "List status updates for a project or initiative.",
        {
            "project_id": {"type": "string"},
            "projectId": {"type": "string"},
            "initiative_id": {"type": "string"},
            "initiativeId": {"type": "string"},
            "limit": _LIMIT,
        },
    ), "📋"),
    (linear_agent_get_status_update, _schema(
        "linear_agent_get_status_update",
        "Fetch a single status update by ID.",
        {"id": {"type": "string"}, "update_id": {"type": "string"}},
        ["id"],
    ), "🔍"),
    (linear_agent_save_milestone, _schema(
        "linear_agent_save_milestone",
        "Create or update a milestone. Uses the Linear Agent identity.",
        _save_props(
            "Existing milestone ID (omit to create new)",
            "Fields: name (required), projectId (required for create), description, targetDate",
        ),
        ["input"],
    ), "🎯"),
    (linear_agent_save_customer_need, _schema(
        "linear_agent_save_customer_need",
        "Create or update a customer need/request. Uses the Linear Agent identity.",
        _save_props(
            "Existing customer need ID (omit to create new)",
            "Fields: body, customerId, issueId, projectId, priority (0=Not important, 1=Important — NOT the issue priority scale), source (URL)",
        ),
        ["input"],
    ), "🙋"),
    (linear_agent_save_document, _schema(
        "linear_agent_save_document",
        "Create or update a Linear document. Uses the Linear Agent identity.",
        _save_props(
            "Existing document ID (omit to create new)",
            "Fields: title, content, projectId, icon, color",
        ),
        ["input"],
    ), "📄"),
    (linear_agent_save_initiative, _schema(
        "linear_agent_save_initiative",
        "Create or update an initiative. Uses the Linear Agent identity.",
        _save_props(
            "Existing initiative ID (omit to create new)",
            "Fields: name, description, status, ownerId, targetDate",
        ),
        ["input"],
    ), "🚀"),
    (linear_agent_set_session_links, _schema(
        "linear_agent_set_session_links",
        "Attach external links (e.g. a GitHub PR, doc, or dashboard) to the current Linear agent session. "
        "They render on the session and mark it as active. Use the Session ID from your prompt.",
        {
            "agent_session_id": {"type": "string", "description": "The agent session ID (shown in your session prompt)"},
            "links": {
                "type": "array",
                "description": "Links to attach: [{label, url}, ...] or plain URL strings. Replaces the session's existing links.",
                "items": {"type": ["object", "string"]},
            },
        },
        ["agent_session_id", "links"],
    ), "🔗"),
    (linear_agent_update_plan, _schema(
        "linear_agent_update_plan",
        "Publish or update the agent's execution plan on the current Linear "
        "session (Agent Plans — technology preview). Renders a live checklist "
        "in Linear. The plan REPLACES the previous plan IN FULL on every call, "
        "so always send EVERY step with its current status — do not send only "
        "changed steps. Mirror your internal todo list here as steps start and "
        "finish. Use the Session ID from your prompt.",
        {
            "agent_session_id": {"type": "string", "description": "The agent session ID (shown in your session prompt). Also accepts session_id or id."},
            "plan": {
                "type": "array",
                "description": (
                    "Full ordered list of steps. Each item is {content, status} "
                    "or a bare string (defaults to pending). status is one of "
                    "pending | inProgress | completed | canceled."
                ),
                "items": {"type": ["object", "string"]},
            },
        },
        ["agent_session_id", "plan"],
    ), "🗺️"),
    (linear_agent_save_release, _schema(
        "linear_agent_save_release",
        "Create or update a release. Uses the Linear Agent identity.",
        _save_props(
            "Existing release ID (omit to create new)",
            "Fields: name (required), pipelineId (required for create — find it with linear_agent_list_release_pipelines), description, version, targetDate",
        ),
        ["input"],
    ), "🏷️"),
    (linear_agent_delete_comment, _schema(
        "linear_agent_delete_comment",
        _preferred("Delete a Linear comment by ID (deliberate — no name resolution).", "mcp_linear_delete_comment")
        + " Requires mutation_policy.delete_comments.",
        {
            "id": {"type": "string", "description": "Comment ID to delete. Also accepts comment_id."},
            "comment_id": {"type": "string", "description": "Alternative name for the comment ID."},
        },
        ["id"],
    ), "🗑️"),
    (linear_agent_delete_customer_need, _schema(
        "linear_agent_delete_customer_need",
        _preferred("Delete a Linear customer need by ID.", "mcp_linear_delete_customer_need")
        + " Requires mutation_policy.delete_customer_needs.",
        {
            "id": {"type": "string", "description": "Customer need ID to delete. Also accepts need_id."},
            "need_id": {"type": "string", "description": "Alternative name for the customer need ID."},
        },
        ["id"],
    ), "🗑️"),
    (linear_agent_delete_status_update, _schema(
        "linear_agent_delete_status_update",
        _preferred("Delete a project OR initiative status update by ID.", "mcp_linear_delete_status_update")
        + " Requires mutation_policy.delete_status_updates. (Routes through "
        "Linear's archive mutation — its supported removal path.)",
        {
            "id": {"type": "string", "description": "Status update ID to delete. Also accepts update_id."},
            "update_id": {"type": "string", "description": "Alternative name for the status update ID."},
            "type": {"type": "string", "enum": ["project", "initiative"], "description": "Which kind of status update (default project)."},
        },
        ["id"],
    ), "🗑️"),
    (linear_agent_delete_attachment, _schema(
        "linear_agent_delete_attachment",
        _preferred("Delete a Linear attachment by ID.", "mcp_linear_delete_attachment")
        + " Requires mutation_policy.delete_attachments.",
        {
            "id": {"type": "string", "description": "Attachment ID to delete. Also accepts attachment_id."},
            "attachment_id": {"type": "string", "description": "Alternative name for the attachment ID."},
        },
        ["id"],
    ), "🗑️"),
    (linear_agent_save_customer, _schema(
        "linear_agent_save_customer",
        _preferred(
            "Create or update a Linear customer (business entity). Policy-gated: "
            "create needs mutation_policy.create_customers, update needs "
            "update_customers.", "mcp_linear_save_customer"),
        _save_props(
            "Existing customer ID (omit to create new)",
            "Fields: name (required to create), domains, externalIds, logoUrl, "
            "mainSourceId, ownerId, revenue, size, slackChannelId, statusId, tierId",
        ),
        ["input"],
    ), "🏢"),
    (linear_agent_delete_customer, _schema(
        "linear_agent_delete_customer",
        _preferred("Delete a Linear customer by ID.", "mcp_linear_delete_customer")
        + " Requires mutation_policy.delete_customers.",
        {
            "id": {"type": "string", "description": "Customer ID to delete. Also accepts customer_id."},
            "customer_id": {"type": "string", "description": "Alternative name for the customer ID."},
        },
        ["id"],
    ), "🗑️"),
    (linear_agent_save_release_note, _schema(
        "linear_agent_save_release_note",
        _preferred(
            "Create or update a Linear release note. Policy-gated under the "
            "release family: create needs mutation_policy.create_releases, "
            "update needs update_releases.", "mcp_linear_save_release_note"),
        _save_props(
            "Existing release note ID (omit to create new)",
            "Fields: pipelineId (required to create — find it with "
            "linear_agent_list_release_pipelines; pipeline_id alias accepted), "
            "title, content, releaseIds, rangeFromReleaseId, rangeToReleaseId"),
        ["input"],
    ), "📰"),
    (linear_agent_create_issue_label, _schema(
        "linear_agent_create_issue_label",
        _preferred("Create a Linear issue label.", "mcp_linear_create_issue_label")
        + " Requires mutation_policy.create_labels.",
        {
            "team_id": {"type": "string", "description": "Team UUID to scope the label to a team. Omit for a workspace-wide label. Also accepts teamId."},
            "input": _input_prop("Label fields: name (required), color (hex), description, parentId, isGroup. teamId may also be set here."),
        },
        ["input"],
    ), "🏷️"),
    (linear_agent_get_team, _schema(
        "linear_agent_get_team",
        "Fetch a single team by ID.",
        {"id": {"type": "string"}, "team_id": {"type": "string"}, "teamId": {"type": "string"}},
        ["id"],
    ), "👥"),
    (linear_agent_get_milestone, _schema(
        "linear_agent_get_milestone",
        "Fetch a single project milestone by ID.",
        {"id": {"type": "string"}, "milestone_id": {"type": "string"}},
        ["id"],
    ), "🎯"),
    (linear_agent_get_document, _schema(
        "linear_agent_get_document",
        "Fetch a single document by ID or slug.",
        {"id": {"type": "string"}, "document_id": {"type": "string"}},
        ["id"],
    ), "📄"),
    (linear_agent_get_attachment, _schema(
        "linear_agent_get_attachment",
        "Fetch a single attachment by ID.",
        {"id": {"type": "string"}, "attachment_id": {"type": "string"}},
        ["id"],
    ), "📎"),
    (linear_agent_get_release_note, _schema(
        "linear_agent_get_release_note",
        "Fetch a single release note by ID.",
        {"id": {"type": "string"}, "release_note_id": {"type": "string"}},
        ["id"],
    ), "📰"),
    (linear_agent_get_agent_skill, _schema(
        "linear_agent_get_agent_skill",
        "Fetch a single agent skill by ID.",
        {"id": {"type": "string"}, "skill_id": {"type": "string"}},
        ["id"],
    ), "🛠️"),
    (linear_agent_list_project_labels, _schema(
        "linear_agent_list_project_labels",
        "List project labels in the workspace.",
        {"limit": _LIMIT},
    ), "🏷️"),
    (linear_agent_list_release_notes, _schema(
        "linear_agent_list_release_notes",
        "List release notes, optionally scoped to a release pipeline.",
        {"pipeline": {"type": "string", "description": "Release pipeline ID to scope by. Also accepts pipeline_id."}, "limit": _LIMIT},
    ), "📰"),
    (linear_agent_list_agent_skills, _schema(
        "linear_agent_list_agent_skills",
        "List agent skills in the workspace.",
        {"limit": _LIMIT},
    ), "🛠️"),
]

for _handler, _tool_schema, _emoji in _TOOLS:
    registry.register(
        name=_tool_schema["name"],
        toolset="linear_agent",
        schema=_tool_schema,
        handler=_handler,
        is_async=True,
        emoji=_emoji,
    )
logger.debug("Registered %d linear_agent_* tools", len(_TOOLS))

TOOL_NAMES = tuple(schema["name"] for _, schema, _ in _TOOLS)


def register_tools_with_context(ctx) -> None:
    """Re-register the tools through the plugin context, for discoverability.

    The import-time loop above puts the tools on the shared registry; only
    tools routed through ``ctx.register_tool`` are tracked as plugin-provided,
    which is what makes the toolset visible to ``hermes tools`` /
    ``get_plugin_toolsets()``. Re-registration is an idempotent same-toolset
    overwrite. No-op when the context has no ``register_tool``.
    """
    register_fn = getattr(ctx, "register_tool", None)
    if register_fn is None:
        return
    for handler, tool_schema, emoji in _TOOLS:
        register_fn(
            name=tool_schema["name"],
            toolset="linear_agent",
            schema=tool_schema,
            handler=handler,
            is_async=True,
            emoji=emoji,
        )
