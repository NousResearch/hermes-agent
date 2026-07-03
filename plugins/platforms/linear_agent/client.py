"""Small Linear GraphQL client for the Linear Agent platform adapter."""

from __future__ import annotations

import asyncio
import logging
import os
import re
from typing import Any, Callable, Optional

try:
    import aiohttp

    AIOHTTP_AVAILABLE = True
except ImportError:  # pragma: no cover - dependency checked by adapter
    aiohttp = None  # type: ignore[assignment]
    AIOHTTP_AVAILABLE = False

from .activity import AGENT_ACTIVITY_CREATE_MUTATION, build_activity_input
from .oauth import LinearOAuthError, LinearOAuthTokenManager

logger = logging.getLogger(__name__)

DEFAULT_LINEAR_GRAPHQL_URL = "https://api.linear.app/graphql"
_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)


def _looks_like_auth_error(message: str) -> bool:
    lowered = message.lower()
    return "401" in lowered or "unauthorized" in lowered or "not authenticated" in lowered


class LinearApiError(RuntimeError):
    """Raised when Linear returns an unsuccessful GraphQL response."""


class LinearRateLimitError(LinearApiError):
    """Raised on HTTP 429; carries the server-suggested retry delay."""

    def __init__(self, message: str, retry_after: float = 1.0) -> None:
        super().__init__(message)
        self.retry_after = retry_after


def _require_policy(
    mutation_policy: Optional[dict[str, Any]],
    key: str,
    what: str,
) -> None:
    """Raise unless the mutation policy explicitly enables `key`.

    Every write method must call this before executing its mutation so that
    all writes fail closed when the operator has not opted in.
    """
    if not (mutation_policy or {}).get(key, False):
        raise LinearApiError(f"{what} is disabled by mutation_policy (enable '{key}')")


class LinearGraphQLClient:
    """Minimal async GraphQL client for Linear's API."""

    def __init__(
        self,
        access_token: Optional[str] = None,
        *,
        api_url: str = DEFAULT_LINEAR_GRAPHQL_URL,
        timeout_seconds: float = 30.0,
        session_factory: Optional[Callable[..., Any]] = None,
        token_manager: Optional[LinearOAuthTokenManager] = None,
        proxy_url: Optional[str] = None,
    ) -> None:
        self.access_token = (access_token or os.getenv("LINEAR_AGENT_ACCESS_TOKEN", "")).strip()
        self.api_url = api_url or DEFAULT_LINEAR_GRAPHQL_URL
        self.timeout_seconds = timeout_seconds
        self._session_factory = session_factory
        self._token_manager = token_manager
        self._proxy_url = (proxy_url or "").strip() or None
        self._proxy_request_kwargs: dict[str, Any] = {}
        self._session: Optional[Any] = None  # cached aiohttp session (lazy)
        self._pooled_loop: Optional[asyncio.AbstractEventLoop] = None

    @property
    def configured(self) -> bool:
        return bool(self.access_token or (self._token_manager and self._token_manager.configured))

    def bind_pooled_loop(self, loop: Optional[asyncio.AbstractEventLoop] = None) -> None:
        """Allow connection pooling on one explicit, long-lived event loop.

        The Linear gateway adapter calls this from its stable aiohttp loop.
        Hermes tool handlers run on sync->async bridge loops, so unbound or
        non-owner loops intentionally use one-shot sessions.
        """
        loop = loop or asyncio.get_running_loop()
        if self._pooled_loop is not None and self._pooled_loop is not loop:
            # Rebinding (reconnect without a clean disconnect, e.g. a gateway
            # revive): the cached session belongs to the old loop and must
            # not be reused across loops.
            self._discard_session()
        self._pooled_loop = loop

    async def execute(
        self,
        query: str,
        variables: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Execute a GraphQL operation against Linear."""
        if not AIOHTTP_AVAILABLE:
            raise LinearApiError("aiohttp is not installed")

        token = await self._resolve_access_token()
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        payload = {"query": query, "variables": variables or {}}

        try:
            return await self._post_graphql(payload, headers)
        except LinearApiError as exc:
            if not _looks_like_auth_error(str(exc)) or not self._token_manager:
                raise
            logger.info("Linear GraphQL auth failed; refreshing OAuth token and retrying once")
            self._token_manager.force_refresh_after_auth_error()
            token = await self._resolve_access_token()
            headers["Authorization"] = f"Bearer {token}"
            return await self._post_graphql(payload, headers)

    async def _resolve_access_token(self) -> str:
        if self._token_manager:
            try:
                token = await self._token_manager.get_access_token()
            except LinearOAuthError as exc:
                raise LinearApiError(str(exc)) from exc
            self.access_token = token
            return token
        if self.access_token:
            return self.access_token
        raise LinearApiError(
            "LINEAR_AGENT_ACCESS_TOKEN is not configured and no refreshable Linear OAuth credentials are available"
        )

    def _new_session(self) -> Any:
        factory = self._session_factory or aiohttp.ClientSession
        session_kwargs: dict[str, Any] = {
            "timeout": aiohttp.ClientTimeout(total=self.timeout_seconds),
        }
        if self._proxy_url:
            # Fresh kwargs per session: the SOCKS path returns a connector
            # instance that is owned (and closed) by the session it joins.
            from gateway.platforms.base import proxy_kwargs_for_aiohttp

            proxy_session_kwargs, self._proxy_request_kwargs = proxy_kwargs_for_aiohttp(
                self._proxy_url
            )
            session_kwargs.update(proxy_session_kwargs)
        return factory(**session_kwargs)

    def _get_session(self) -> Any:
        """Pooled session for the bound adapter loop; close via aclose()."""
        if self._session is None or self._session.closed:
            self._session = self._new_session()
        return self._session

    def _discard_session(self) -> None:
        """Drop the cached session, closing it on its owning loop when possible."""
        session, owner = self._session, self._pooled_loop
        self._session = None
        if session is None or session.closed:
            return
        try:
            if owner is not None and owner.is_running():
                owner.call_soon_threadsafe(lambda: owner.create_task(session.close()))
            # else: the owning loop is gone; dropping the reference lets GC
            # reclaim the connector (aiohttp logs an unclosed-session warning
            # at worst, which beats reusing a loop-bound session).
        except Exception:  # noqa: BLE001 - discard is best-effort
            pass

    async def aclose(self) -> None:
        """Close the cached HTTP session, if any."""
        session = self._session
        self._session = None
        self._pooled_loop = None
        if session is not None and not session.closed:
            await session.close()

    async def _post_graphql(
        self,
        payload: dict[str, Any],
        headers: dict[str, str],
    ) -> dict[str, Any]:
        # Linear tool calls are sync->async bridged from the gateway through
        # worker event loops. A pooled aiohttp.ClientSession is bound to the
        # loop that created it, so reusing the adapter's gateway loop session
        # inside those worker loops raises:
        # "Timeout context manager should be used inside a task".
        #
        # Pool only on the explicitly-bound stable adapter loop. Every other
        # loop gets a one-shot session that closes before the loop can vanish.
        assert aiohttp is not None
        # One retry on 429, honoring Retry-After (capped so webhook handling
        # can't stall past Linear's own response deadline).
        for attempt in (0, 1):
            try:
                if self._pooled_loop is asyncio.get_running_loop():
                    return await self._post_graphql_with_session(
                        self._get_session(), payload, headers
                    )
                async with self._new_session() as session:
                    return await self._post_graphql_with_session(session, payload, headers)
            except LinearRateLimitError as exc:
                if attempt:
                    raise
                delay = min(max(exc.retry_after, 0.0), 10.0)
                logger.info("Linear rate limited (429); retrying once in %.1fs", delay)
                await asyncio.sleep(delay)
        raise LinearApiError("unreachable")  # pragma: no cover - loop always returns/raises

    async def _post_graphql_with_session(
        self,
        session: Any,
        payload: dict[str, Any],
        headers: dict[str, str],
    ) -> dict[str, Any]:
        async with session.post(
            self.api_url, json=payload, headers=headers, **self._proxy_request_kwargs
        ) as resp:
            text = await resp.text()
            if resp.status == 429:
                try:
                    retry_after = float(getattr(resp, "headers", {}).get("Retry-After", 1.0))
                except (TypeError, ValueError):
                    retry_after = 1.0
                raise LinearRateLimitError(
                    f"Linear GraphQL HTTP 429: {text[:200]}", retry_after=retry_after
                )
            if resp.status != 200:
                raise LinearApiError(f"Linear GraphQL HTTP {resp.status}: {text[:200]}")
            try:
                data = await resp.json()
            except Exception:
                data = {"raw": text}
            if "errors" in data:
                msgs = "; ".join(e.get("message", str(e)) for e in data["errors"])
                raise LinearApiError(f"Linear GraphQL errors: {msgs}")
            return data.get("data", {})

    async def _mutate(
        self,
        mutation: str,
        variables: dict[str, Any],
        *,
        op: str,
        entity: Optional[str],
        policy_key: Optional[str],
        what: str,
        mutation_policy: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Run a policy-gated mutation and unwrap Linear's success envelope.

        ``policy_key=None`` marks agent-session protocol mutations that are
        intentionally exempt from mutation_policy (they mutate only the
        agent's own session, not workspace content).

        ``entity=None`` marks mutations whose payload carries no entity object
        (Linear's ``DeletePayload``/``ArchivePayload`` return only
        ``{ success, entityId }``); success is still verified and the deleted
        entity's id is returned when present.
        """
        if policy_key is not None:
            _require_policy(mutation_policy, policy_key, what)
        data = await self.execute(mutation, variables)
        result = data.get(op) or {}
        if not result.get("success", False):
            raise LinearApiError(f"Linear {op} returned success=false")
        if entity is None:
            entity_id = result.get("entityId") if isinstance(result, dict) else None
            return {"entityId": entity_id} if entity_id else {}
        return result.get(entity) or {}

    # --- Mutation helpers ---
    async def update_issue(
        self,
        issue_id: str,
        input_payload: dict[str, Any],
        *,
        mutation_policy: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        return await self._mutate(
            """
mutation IssueUpdate($id: String!, $input: IssueUpdateInput!) {
  issueUpdate(id: $id, input: $input) {
    success
    issue { id identifier title }
  }
}
""",
            {"id": issue_id, "input": input_payload},
            op="issueUpdate", entity="issue",
            policy_key="update_issues", what="Issue update",
            mutation_policy=mutation_policy,
        )

    async def create_comment(
        self,
        issue_id: Optional[str],
        body: str,
        *,
        parent_id: Optional[str] = None,
        extra_input: Optional[dict[str, Any]] = None,
        mutation_policy: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        # issue_id is the default parent; ``extra_input`` carries an alternate
        # CommentCreateInput parent (projectId/projectUpdateId/initiativeId/
        # initiativeUpdateId/documentContentId) when the comment targets a
        # non-issue entity. Exactly one parent is enforced by the tool handler.
        comment_input: dict[str, Any] = {"body": body}
        if issue_id:
            comment_input["issueId"] = issue_id
        if parent_id:
            # A reply. Linear ties the reply to the parent's thread and
            # inherits the parent's thread type (per MCP save_comment).
            comment_input["parentId"] = parent_id
        if extra_input:
            comment_input.update(extra_input)
        return await self._mutate(
            """
mutation CommentCreate($input: CommentCreateInput!) {
  commentCreate(input: $input) {
    success
    comment { id body }
  }
}
""",
            {"input": comment_input},
            op="commentCreate", entity="comment",
            policy_key="create_comments", what="Comment creation",
            mutation_policy=mutation_policy,
        )

    async def update_comment(
        self,
        comment_id: str,
        input_payload: dict[str, Any],
        *,
        mutation_policy: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        return await self._mutate(
            """
mutation CommentUpdate($id: String!, $input: CommentUpdateInput!) {
  commentUpdate(id: $id, input: $input) {
    success
    comment { id body }
  }
}
""",
            {"id": comment_id, "input": input_payload},
            op="commentUpdate", entity="comment",
            policy_key="update_comments", what="Comment update",
            mutation_policy=mutation_policy,
        )

    async def create_issue(
        self,
        input_payload: dict[str, Any],
        *,
        mutation_policy: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        return await self._mutate(
            """
mutation IssueCreate($input: IssueCreateInput!) {
  issueCreate(input: $input) {
    success
    issue { id identifier title }
  }
}
""",
            {"input": input_payload},
            op="issueCreate", entity="issue",
            policy_key="create_issues", what="Issue creation",
            mutation_policy=mutation_policy,
        )

    async def create_project(
        self,
        input_payload: dict[str, Any],
        *,
        mutation_policy: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        return await self._mutate(
            """
mutation ProjectCreate($input: ProjectCreateInput!) {
  projectCreate(input: $input) {
    success
    project { id name }
  }
}
""",
            {"input": input_payload},
            op="projectCreate", entity="project",
            policy_key="update_projects", what="Project mutation",
            mutation_policy=mutation_policy,
        )

    async def update_project(
        self,
        project_id: str,
        input_payload: dict[str, Any],
        *,
        mutation_policy: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        return await self._mutate(
            """
mutation ProjectUpdate($id: String!, $input: ProjectUpdateInput!) {
  projectUpdate(id: $id, input: $input) {
    success
    project { id name }
  }
}
""",
            {"id": project_id, "input": input_payload},
            op="projectUpdate", entity="project",
            policy_key="update_projects", what="Project mutation",
            mutation_policy=mutation_policy,
        )

    async def create_project_update(
        self,
        input_payload: dict[str, Any],
        *,
        mutation_policy: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        return await self._mutate(
            """
mutation ProjectUpdateCreate($input: ProjectUpdateCreateInput!) {
  projectUpdateCreate(input: $input) {
    success
    projectUpdate { id }
  }
}
""",
            {"input": input_payload},
            op="projectUpdateCreate", entity="projectUpdate",
            policy_key="update_projects", what="Project mutation",
            mutation_policy=mutation_policy,
        )

    async def update_project_update(
        self,
        update_id: str,
        input_payload: dict[str, Any],
        *,
        mutation_policy: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        return await self._mutate(
            """
mutation ProjectUpdateUpdate($id: String!, $input: ProjectUpdateUpdateInput!) {
  projectUpdateUpdate(id: $id, input: $input) {
    success
    projectUpdate { id }
  }
}
""",
            {"id": update_id, "input": input_payload},
            op="projectUpdateUpdate", entity="projectUpdate",
            policy_key="update_projects", what="Project mutation",
            mutation_policy=mutation_policy,
        )

    async def create_initiative_update(
        self,
        input_payload: dict[str, Any],
        *,
        mutation_policy: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        return await self._mutate(
            """
mutation InitiativeUpdateCreate($input: InitiativeUpdateCreateInput!) {
  initiativeUpdateCreate(input: $input) {
    success
    initiativeUpdate { id }
  }
}
""",
            {"input": input_payload},
            op="initiativeUpdateCreate", entity="initiativeUpdate",
            policy_key="update_projects", what="Project mutation",
            mutation_policy=mutation_policy,
        )

    async def update_initiative_update(
        self,
        update_id: str,
        input_payload: dict[str, Any],
        *,
        mutation_policy: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        return await self._mutate(
            """
mutation InitiativeUpdateUpdate($id: String!, $input: InitiativeUpdateUpdateInput!) {
  initiativeUpdateUpdate(id: $id, input: $input) {
    success
    initiativeUpdate { id }
  }
}
""",
            {"id": update_id, "input": input_payload},
            op="initiativeUpdateUpdate", entity="initiativeUpdate",
            policy_key="update_projects", what="Project mutation",
            mutation_policy=mutation_policy,
        )

    async def list_status_updates(
        self,
        *,
        project_id: Optional[str] = None,
        initiative_id: Optional[str] = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """List project status updates, or initiative updates when initiative_id is given.

        Linear models these as two separate types (ProjectUpdate / InitiativeUpdate),
        so an initiative_id routes to the initiativeUpdates query.
        """
        if initiative_id:
            query = """
query InitiativeUpdates($first: Int, $filter: InitiativeUpdateFilter) {
  initiativeUpdates(first: $first, filter: $filter) {
    nodes {
      id
      body
      health
      createdAt
      initiative { id name }
    }
  }
}
"""
            variables: dict[str, Any] = {
                "first": limit,
                "filter": {"initiative": {"id": {"eq": initiative_id}}},
            }
            data = await self.execute(query, variables)
            return data.get("initiativeUpdates", {}).get("nodes", [])

        query = """
query ProjectUpdates($first: Int, $filter: ProjectUpdateFilter) {
  projectUpdates(first: $first, filter: $filter) {
    nodes {
      id
      body
      health
      createdAt
      project { id name }
    }
  }
}
"""
        variables = {"first": limit}
        if project_id:
            variables["filter"] = {"project": {"id": {"eq": project_id}}}
        return await self._list_nodes(query, "projectUpdates", variables)

    async def create_document(
        self,
        input_payload: dict[str, Any],
        *,
        mutation_policy: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        return await self._mutate(
            """
mutation DocumentCreate($input: DocumentCreateInput!) {
  documentCreate(input: $input) {
    success
    document { id title project { id name } }
  }
}
""",
            {"input": input_payload},
            op="documentCreate", entity="document",
            policy_key="create_documents", what="Document creation",
            mutation_policy=mutation_policy,
        )

    async def create_milestone(
        self,
        input_payload: dict[str, Any],
        *,
        mutation_policy: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        return await self._mutate(
            """
mutation ProjectMilestoneCreate($input: ProjectMilestoneCreateInput!) {
  projectMilestoneCreate(input: $input) {
    success
    projectMilestone { id name project { id name } }
  }
}
""",
            {"input": input_payload},
            op="projectMilestoneCreate", entity="projectMilestone",
            policy_key="update_projects", what="Project mutation",
            mutation_policy=mutation_policy,
        )

    async def update_milestone(
        self,
        milestone_id: str,
        input_payload: dict[str, Any],
        *,
        mutation_policy: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        return await self._mutate(
            """
mutation ProjectMilestoneUpdate($id: String!, $input: ProjectMilestoneUpdateInput!) {
  projectMilestoneUpdate(id: $id, input: $input) {
    success
    projectMilestone { id name }
  }
}
""",
            {"id": milestone_id, "input": input_payload},
            op="projectMilestoneUpdate", entity="projectMilestone",
            policy_key="update_projects", what="Project mutation",
            mutation_policy=mutation_policy,
        )

    async def get_status_update(self, update_id: str) -> dict[str, Any]:
        query = """
query ProjectUpdate($id: String!) {
  projectUpdate(id: $id) {
    id
    body
    health
    createdAt
    url
    project { id name }
  }
}
"""
        data = await self.execute(query, {"id": update_id})
        return data.get("projectUpdate") or {}

    async def update_customer_need(
        self,
        need_id: str,
        input_payload: dict[str, Any],
        *,
        mutation_policy: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        return await self._mutate(
            """
mutation CustomerNeedUpdate($id: String!, $input: CustomerNeedUpdateInput!) {
  customerNeedUpdate(id: $id, input: $input) {
    success
    customerNeed { id body }
  }
}
""",
            {"id": need_id, "input": input_payload},
            op="customerNeedUpdate", entity="customerNeed",
            policy_key="update_customer_needs", what="Customer need update",
            mutation_policy=mutation_policy,
        )

    async def create_customer_need(
        self,
        input_payload: dict[str, Any],
        *,
        mutation_policy: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        return await self._mutate(
            """
mutation CustomerNeedCreate($input: CustomerNeedCreateInput!) {
  customerNeedCreate(input: $input) {
    success
    customerNeed { id body customer { id name } }
  }
}
""",
            {"input": input_payload},
            op="customerNeedCreate", entity="customerNeed",
            policy_key="create_customer_needs", what="Customer need creation",
            mutation_policy=mutation_policy,
        )

    async def set_agent_session_external_urls(
        self,
        agent_session_id: str,
        urls: list[str | dict[str, str]],
    ) -> dict[str, Any]:
        return await self._mutate(
            """
mutation AgentSessionUpdate($id: String!, $input: AgentSessionUpdateInput!) {
  agentSessionUpdate(id: $id, input: $input) {
    success
    agentSession { id }
  }
}
""",
            {"id": agent_session_id, "input": {"externalUrls": urls}},
            op="agentSessionUpdate", entity="agentSession",
            policy_key=None, what="Agent session update",
        )

    async def update_session_plan(
        self,
        agent_session_id: str,
        plan: list[dict[str, str]],
    ) -> dict[str, Any]:
        """Replace the agent session's execution plan (Agent Plans, tech preview).

        ``AgentSessionUpdateInput.plan`` (JSONObject) is replaced IN FULL on
        every call — Linear renders live progress from the latest update.
        """
        return await self._mutate(
            """
mutation AgentSessionUpdate($id: String!, $input: AgentSessionUpdateInput!) {
  agentSessionUpdate(id: $id, input: $input) {
    success
    agentSession { id }
  }
}
""",
            {"id": agent_session_id, "input": {"plan": plan}},
            op="agentSessionUpdate", entity="agentSession",
            policy_key=None, what="Agent session update",
        )

    # --- Read helpers ---
    @staticmethod
    def _team_filter(team: str) -> dict[str, Any]:
        """UUID-ish values filter by team id; short values are team keys."""
        value = str(team or "").strip()
        if not value:
            return {}
        if _UUID_RE.match(value):
            return {"id": {"eq": value}}
        return {"key": {"eq": value}}

    async def _list_nodes(
        self,
        query: str,
        root: str,
        variables: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Execute a connection query and unwrap ``{root: {nodes: [...]}}``."""
        data = await self.execute(query, variables)
        return data.get(root, {}).get("nodes", [])

    async def _get_node(
        self,
        query: str,
        root: str,
        variables: dict[str, Any],
    ) -> Optional[dict[str, Any]]:
        """Execute a single-node query and unwrap ``{root: {...}}``."""
        data = await self.execute(query, variables)
        return data.get(root)

    async def list_teams(self, query: Optional[str] = None, limit: int = 50) -> list[dict[str, Any]]:
        """List teams in the workspace."""
        q = """
        query Teams($first: Int, $filter: TeamFilter) {
          teams(first: $first, filter: $filter) {
            nodes {
              id key name description
              issueEstimationType issueEstimationAllowZero defaultIssueEstimate
            }
          }
        }
        """
        variables: dict[str, Any] = {"first": limit}
        if query:
            variables["filter"] = {"name": {"contains": query}}
        return await self._list_nodes(q, "teams", variables)

    async def list_issue_statuses(self, team_id: Optional[str] = None, team_name: Optional[str] = None) -> list[dict[str, Any]]:
        """List workflow states (statuses) for a team. Accepts team UUID or name."""
        if not team_id and not team_name:
            raise LinearApiError("team_id or team_name is required")

        if team_name and not team_id:
            teams = await self.list_teams(query=team_name, limit=5)
            if not teams:
                raise LinearApiError(f"No team found matching '{team_name}'")
            team_id = teams[0]["id"]

        q = """
        query WorkflowStates($first: Int, $filter: WorkflowStateFilter) {
          workflowStates(first: $first, filter: $filter) {
            nodes { id name type position team { id key name } }
          }
        }
        """
        variables = {"first": 100, "filter": {"team": {"id": {"eq": team_id}}}}
        return await self._list_nodes(q, "workflowStates", variables)

    async def first_started_state(self, team_id: str) -> Optional[dict[str, Any]]:
        """Lowest-position workflow state of type 'started' for a team.

        Linear displays states in ascending ``position`` within their type
        group, so the lowest-position ``started`` state is the natural "first
        in progress" target for auto-starting a delegated issue.
        """
        states = await self.list_issue_statuses(team_id=team_id)
        started = [s for s in states if s.get("type") == "started"]
        return min(started, key=lambda s: s.get("position") or 0) if started else None

    async def get_issue_status(self, name: str, team: Optional[str] = None) -> Optional[dict[str, Any]]:
        """Get a specific workflow state by name (optionally scoped to a team)."""
        statuses = await self.list_issue_statuses(team_id=team, team_name=team)
        for s in statuses:
            if s.get("name", "").lower() == name.lower():
                return s
        return None

    async def get_issue(self, id: str) -> Optional[dict[str, Any]]:
        """Fetch a single issue by ID or identifier."""
        q = """
        query Issue($id: String!) {
          issue(id: $id) {
            id identifier title description state { id name type } assignee { id name } team { id key name }
            delegate { id } priority project { id name } dueDate createdAt updatedAt
          }
        }
        """
        return await self._get_node(q, "issue", {"id": id})

    async def list_issues(self, team: Optional[str] = None, state: Optional[str] = None,
                          assignee: Optional[str] = None, query: Optional[str] = None,
                          limit: int = 50) -> list[dict[str, Any]]:
        """List issues with optional filters."""
        q = """
        query Issues($first: Int, $filter: IssueFilter) {
          issues(first: $first, filter: $filter) {
            nodes { id identifier title state { id name } assignee { id name } team { id key name } }
          }
        }
        """
        variables: dict[str, Any] = {"first": limit}
        f: dict[str, Any] = {}
        if team:
            f["team"] = self._team_filter(team)
        if state:
            f["state"] = {"name": {"eq": state}}
        if assignee:
            f["assignee"] = {"id": {"eq": assignee}}
        if query:
            f["title"] = {"contains": query}
        if f:
            variables["filter"] = f
        return await self._list_nodes(q, "issues", variables)
    async def list_projects(self, query: Optional[str] = None, team: Optional[str] = None, limit: int = 50) -> list[dict[str, Any]]:
        """List projects with optional name filter and team scope."""
        q = """
        query Projects($first: Int, $filter: ProjectFilter) {
          projects(first: $first, filter: $filter) {
            nodes { id name state teams(first: 5) { nodes { id key name } } }
          }
        }
        """
        variables: dict[str, Any] = {"first": limit}
        f: dict[str, Any] = {}
        if query:
            f["name"] = {"contains": query}
        if team:
            # Projects belong to MANY teams; ProjectFilter exposes
            # accessibleTeams (TeamCollectionFilter), not a singular team.
            f["accessibleTeams"] = {"some": self._team_filter(team)}
        if f:
            variables["filter"] = f
        return await self._list_nodes(q, "projects", variables)

    async def get_project(self, id: str) -> Optional[dict[str, Any]]:
        """Fetch a single project by ID or slug."""
        q = """
        query Project($id: String!) {
          project(id: $id) {
            id name description state teams(first: 5) { nodes { id key name } } lead { id name }
            startDate targetDate priority createdAt updatedAt
          }
        }
        """
        return await self._get_node(q, "project", {"id": id})

    async def list_cycles(self, team: Optional[str] = None, limit: int = 50) -> list[dict[str, Any]]:
        """List cycles for a team (or workspace if no team)."""
        q = """
        query Cycles($first: Int, $filter: CycleFilter) {
          cycles(first: $first, filter: $filter) {
            nodes { id name number startsAt endsAt team { id key name } }
          }
        }
        """
        variables: dict[str, Any] = {"first": limit}
        if team:
            variables["filter"] = {"team": self._team_filter(team)}
        return await self._list_nodes(q, "cycles", variables)

    async def list_milestones(self, project: Optional[str] = None, limit: int = 50) -> list[dict[str, Any]]:
        """List project milestones, optionally scoped to a project."""
        q = """
        query ProjectMilestones($first: Int, $filter: ProjectMilestoneFilter) {
          projectMilestones(first: $first, filter: $filter) {
            nodes { id name project { id name } targetDate }
          }
        }
        """
        variables: dict[str, Any] = {"first": limit}
        if project:
            variables["filter"] = {"project": {"id": {"eq": project}}}
        return await self._list_nodes(q, "projectMilestones", variables)

    async def list_documents(self, project: Optional[str] = None, query: Optional[str] = None, limit: int = 50) -> list[dict[str, Any]]:
        """List documents with optional project and title filter."""
        q = """
        query Documents($first: Int, $filter: DocumentFilter) {
          documents(first: $first, filter: $filter) {
            nodes { id title project { id name } createdAt updatedAt }
          }
        }
        """
        variables: dict[str, Any] = {"first": limit}
        f: dict[str, Any] = {}
        if project:
            f["project"] = {"id": {"eq": project}}
        if query:
            f["title"] = {"contains": query}
        if f:
            variables["filter"] = f
        return await self._list_nodes(q, "documents", variables)
    async def list_customers(self, query: Optional[str] = None, limit: int = 50) -> list[dict[str, Any]]:
        q = """
        query Customers($first: Int, $filter: CustomerFilter) {
          customers(first: $first, filter: $filter) {
            nodes { id name slugId tier status owner { id name } }
          }
        }
        """
        variables: dict[str, Any] = {"first": limit}
        if query:
            variables["filter"] = {"name": {"contains": query}}
        return await self._list_nodes(q, "customers", variables)

    async def get_customer(self, id: str) -> Optional[dict[str, Any]]:
        q = """
        query Customer($id: String!) {
          customer(id: $id) {
            id name slugId tier status owner { id name } createdAt updatedAt
          }
        }
        """
        return await self._get_node(q, "customer", {"id": id})

    async def list_initiatives(self, query: Optional[str] = None, limit: int = 50) -> list[dict[str, Any]]:
        q = """
        query Initiatives($first: Int, $filter: InitiativeFilter) {
          initiatives(first: $first, filter: $filter) {
            nodes { id name status health targetDate owner { id name } }
          }
        }
        """
        variables: dict[str, Any] = {"first": limit}
        if query:
            variables["filter"] = {"name": {"contains": query}}
        return await self._list_nodes(q, "initiatives", variables)

    async def get_initiative(self, id: str) -> Optional[dict[str, Any]]:
        q = """
        query Initiative($id: String!) {
          initiative(id: $id) {
            id name description status health targetDate url owner { id name } createdAt updatedAt
          }
        }
        """
        return await self._get_node(q, "initiative", {"id": id})

    async def list_issue_labels(self, team: Optional[str] = None, query: Optional[str] = None, limit: int = 100) -> list[dict[str, Any]]:
        q = """
        query IssueLabels($first: Int, $filter: IssueLabelFilter) {
          issueLabels(first: $first, filter: $filter) {
            nodes { id name color team { id key name } }
          }
        }
        """
        variables: dict[str, Any] = {"first": limit}
        f: dict[str, Any] = {}
        if team:
            f["team"] = self._team_filter(team)
        if query:
            f["name"] = {"contains": query}
        if f:
            variables["filter"] = f
        return await self._list_nodes(q, "issueLabels", variables)
    async def list_release_pipelines(self, limit: int = 50) -> list[dict[str, Any]]:
        """List release pipelines — needed to resolve pipelineId for releases."""
        q = """
        query ReleasePipelines($first: Int) {
          releasePipelines(first: $first) {
            nodes { id name slugId }
          }
        }
        """
        return await self._list_nodes(q, "releasePipelines", {"first": limit})

    async def list_releases(self, query: Optional[str] = None, limit: int = 50) -> list[dict[str, Any]]:
        """List releases. Releases belong to pipelines (not teams) in Linear's schema."""
        q = """
        query Releases($first: Int, $filter: ReleaseFilter) {
          releases(first: $first, filter: $filter) {
            nodes { id name description version stage targetDate createdAt url }
          }
        }
        """
        variables: dict[str, Any] = {"first": limit}
        if query:
            variables["filter"] = {"name": {"contains": query}}
        return await self._list_nodes(q, "releases", variables)

    async def get_release(self, id: str) -> Optional[dict[str, Any]]:
        q = """
        query Release($id: String!) {
          release(id: $id) {
            id name description version stage startDate targetDate completedAt createdAt url
          }
        }
        """
        return await self._get_node(q, "release", {"id": id})

    async def list_comments(self, issue_id: str, limit: int = 50) -> list[dict[str, Any]]:
        q = """
        query Comments($issueId: String!, $first: Int) {
          issue(id: $issueId) {
            comments(first: $first) {
              nodes { id body user { id name } createdAt }
            }
          }
        }
        """
        data = await self.execute(q, {"issueId": issue_id, "first": limit})
        return data.get("issue", {}).get("comments", {}).get("nodes", [])

    async def list_users(self, query: Optional[str] = None, limit: int = 50) -> list[dict[str, Any]]:
        q = """
        query Users($first: Int, $filter: UserFilter) {
          users(first: $first, filter: $filter) {
            nodes { id name email displayName avatarUrl }
          }
        }
        """
        variables: dict[str, Any] = {"first": limit}
        if query:
            variables["filter"] = {"name": {"contains": query}}
        return await self._list_nodes(q, "users", variables)

    async def get_user(self, id: str) -> Optional[dict[str, Any]]:
        q = """
        query User($id: String!) {
          user(id: $id) {
            id name email displayName avatarUrl createdAt
          }
        }
        """
        return await self._get_node(q, "user", {"id": id})

    async def list_attachments(self, issue_id: Optional[str] = None, limit: int = 50) -> list[dict[str, Any]]:
        """List attachments. AttachmentFilter cannot filter by issue, so an
        issue_id is resolved through the issue's own attachments connection."""
        if issue_id:
            q = """
            query IssueAttachments($issueId: String!, $first: Int) {
              issue(id: $issueId) {
                attachments(first: $first) {
                  nodes { id title subtitle url sourceType createdAt creator { id name } }
                }
              }
            }
            """
            data = await self.execute(q, {"issueId": issue_id, "first": limit})
            return (data.get("issue") or {}).get("attachments", {}).get("nodes", [])

        q = """
        query Attachments($first: Int) {
          attachments(first: $first) {
            nodes { id title subtitle url sourceType createdAt creator { id name } }
          }
        }
        """
        data = await self.execute(q, {"first": limit})
        return data.get("attachments", {}).get("nodes", [])

    async def get_team(self, id: str) -> Optional[dict[str, Any]]:
        q = """
        query Team($id: String!) {
          team(id: $id) {
            id key name description
            issueEstimationType issueEstimationAllowZero defaultIssueEstimate
          }
        }
        """
        return await self._get_node(q, "team", {"id": id})

    async def get_milestone(self, id: str) -> Optional[dict[str, Any]]:
        q = """
        query ProjectMilestone($id: String!) {
          projectMilestone(id: $id) {
            id name description targetDate project { id name }
          }
        }
        """
        return await self._get_node(q, "projectMilestone", {"id": id})

    async def get_document(self, id: str) -> Optional[dict[str, Any]]:
        # ``document(id:)`` accepts a UUID or a slug.
        q = """
        query Document($id: String!) {
          document(id: $id) {
            id title content url icon color project { id name } createdAt updatedAt
          }
        }
        """
        return await self._get_node(q, "document", {"id": id})

    async def get_attachment(self, id: str) -> Optional[dict[str, Any]]:
        q = """
        query Attachment($id: String!) {
          attachment(id: $id) {
            id title subtitle url sourceType createdAt creator { id name } issue { id identifier }
          }
        }
        """
        return await self._get_node(q, "attachment", {"id": id})

    async def get_release_note(self, id: str) -> Optional[dict[str, Any]]:
        q = """
        query ReleaseNote($id: String!) {
          releaseNote(id: $id) {
            id title url content createdAt updatedAt
          }
        }
        """
        return await self._get_node(q, "releaseNote", {"id": id})

    async def list_release_notes(self, pipeline: Optional[str] = None, limit: int = 50) -> list[dict[str, Any]]:
        """List release notes, optionally scoped to a release pipeline.

        ReleaseNoteFilter exposes ``pipeline`` (a ReleasePipelineFilter with an
        ``id`` comparator), so a pipeline id filters by that endpoint.
        """
        q = """
        query ReleaseNotes($first: Int, $filter: ReleaseNoteFilter) {
          releaseNotes(first: $first, filter: $filter) {
            nodes { id title url createdAt }
          }
        }
        """
        variables: dict[str, Any] = {"first": limit}
        if pipeline:
            variables["filter"] = {"pipeline": {"id": {"eq": pipeline}}}
        return await self._list_nodes(q, "releaseNotes", variables)

    async def list_project_labels(self, limit: int = 50) -> list[dict[str, Any]]:
        q = """
        query ProjectLabels($first: Int) {
          projectLabels(first: $first) {
            nodes { id name color description }
          }
        }
        """
        return await self._list_nodes(q, "projectLabels", {"first": limit})

    async def list_agent_skills(self, limit: int = 50) -> list[dict[str, Any]]:
        q = """
        query AgentSkills($first: Int) {
          agentSkills(first: $first) {
            nodes { id title description slugId }
          }
        }
        """
        return await self._list_nodes(q, "agentSkills", {"first": limit})

    async def get_agent_skill(self, id: str) -> Optional[dict[str, Any]]:
        q = """
        query AgentSkill($id: String!) {
          agentSkill(id: $id) {
            id title description body slugId
          }
        }
        """
        return await self._get_node(q, "agentSkill", {"id": id})

    async def create_response(
        self,
        agent_session_id: str,
        content: str,
        *,
        response_type: str = "response",
        ephemeral: bool = False,
        signal: Optional[str] = None,
    ) -> dict[str, Any]:
        """Create an agent activity response for a Linear Agent session.

        Not gated by mutation_policy (policy_key=None): activities into the
        agent's own session are the protocol, not a workspace write.

        `ephemeral=True` makes the activity disappear once the next activity
        arrives (useful for acks/progress thoughts); `signal` passes a Linear
        agent signal (e.g. "stop") alongside the activity.
        """
        input_payload = build_activity_input(agent_session_id, response_type, content)
        # ephemeral/signal live at the input level, not inside content.
        if ephemeral:
            input_payload["ephemeral"] = True
        if signal:
            input_payload["signal"] = signal
        return await self._mutate(
            AGENT_ACTIVITY_CREATE_MUTATION,
            {"input": input_payload},
            op="agentActivityCreate", entity="agentActivity",
            policy_key=None, what="Agent activity",
        )

    async def create_thought(
        self,
        agent_session_id: str,
        body: str,
        *,
        ephemeral: bool = False,
    ) -> dict[str, Any]:
        """Create a `thought` agent activity (progress/acknowledgement)."""
        return await self.create_response(
            agent_session_id,
            body,
            response_type="thought",
            ephemeral=ephemeral,
        )

    async def create_error(self, agent_session_id: str, body: str) -> dict[str, Any]:
        """Create an `error` agent activity."""
        return await self.create_response(agent_session_id, body, response_type="error")

    # --- Document, Initiative, Release update/create methods (Batch 6) ---

    async def update_document(
        self,
        document_id: str,
        input_payload: dict[str, Any],
        *,
        mutation_policy: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        return await self._mutate(
            """
mutation DocumentUpdate($id: String!, $input: DocumentUpdateInput!) {
  documentUpdate(id: $id, input: $input) {
    success
    document { id title }
  }
}
""",
            {"id": document_id, "input": input_payload},
            op="documentUpdate", entity="document",
            policy_key="update_documents", what="Document update",
            mutation_policy=mutation_policy,
        )

    async def create_initiative(
        self,
        input_payload: dict[str, Any],
        *,
        mutation_policy: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        return await self._mutate(
            """
mutation InitiativeCreate($input: InitiativeCreateInput!) {
  initiativeCreate(input: $input) {
    success
    initiative { id name }
  }
}
""",
            {"input": input_payload},
            op="initiativeCreate", entity="initiative",
            policy_key="update_projects", what="Project mutation",
            mutation_policy=mutation_policy,
        )

    async def update_initiative(
        self,
        initiative_id: str,
        input_payload: dict[str, Any],
        *,
        mutation_policy: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        return await self._mutate(
            """
mutation InitiativeUpdate($id: String!, $input: InitiativeUpdateInput!) {
  initiativeUpdate(id: $id, input: $input) {
    success
    initiative { id name }
  }
}
""",
            {"id": initiative_id, "input": input_payload},
            op="initiativeUpdate", entity="initiative",
            policy_key="update_projects", what="Project mutation",
            mutation_policy=mutation_policy,
        )

    async def create_release(
        self,
        input_payload: dict[str, Any],
        *,
        mutation_policy: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        return await self._mutate(
            """
mutation ReleaseCreate($input: ReleaseCreateInput!) {
  releaseCreate(input: $input) {
    success
    release { id name }
  }
}
""",
            {"input": input_payload},
            op="releaseCreate", entity="release",
            policy_key="create_releases", what="Release creation",
            mutation_policy=mutation_policy,
        )

    async def update_release(
        self,
        release_id: str,
        input_payload: dict[str, Any],
        *,
        mutation_policy: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        return await self._mutate(
            """
mutation ReleaseUpdate($id: String!, $input: ReleaseUpdateInput!) {
  releaseUpdate(id: $id, input: $input) {
    success
    release { id name }
  }
}
""",
            {"id": release_id, "input": input_payload},
            op="releaseUpdate", entity="release",
            policy_key="update_releases", what="Release update",
            mutation_policy=mutation_policy,
        )

    # --- Customers, release notes, labels (MCP parity round 2) ---

    async def create_customer(
        self,
        input_payload: dict[str, Any],
        *,
        mutation_policy: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        return await self._mutate(
            """
mutation CustomerCreate($input: CustomerCreateInput!) {
  customerCreate(input: $input) {
    success
    customer { id name }
  }
}
""",
            {"input": input_payload},
            op="customerCreate", entity="customer",
            policy_key="create_customers", what="Customer creation",
            mutation_policy=mutation_policy,
        )

    async def update_customer(
        self,
        customer_id: str,
        input_payload: dict[str, Any],
        *,
        mutation_policy: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        return await self._mutate(
            """
mutation CustomerUpdate($id: String!, $input: CustomerUpdateInput!) {
  customerUpdate(id: $id, input: $input) {
    success
    customer { id name }
  }
}
""",
            {"id": customer_id, "input": input_payload},
            op="customerUpdate", entity="customer",
            policy_key="update_customers", what="Customer update",
            mutation_policy=mutation_policy,
        )

    async def create_release_note(
        self,
        input_payload: dict[str, Any],
        *,
        mutation_policy: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        # Release notes gate under the release-family umbrella (create_releases),
        return await self._mutate(
            """
mutation ReleaseNoteCreate($input: ReleaseNoteCreateInput!) {
  releaseNoteCreate(input: $input) {
    success
    releaseNote { id title url }
  }
}
""",
            {"input": input_payload},
            op="releaseNoteCreate", entity="releaseNote",
            policy_key="create_releases", what="Release note creation",
            mutation_policy=mutation_policy,
        )

    async def update_release_note(
        self,
        release_note_id: str,
        input_payload: dict[str, Any],
        *,
        mutation_policy: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        return await self._mutate(
            """
mutation ReleaseNoteUpdate($id: String!, $input: ReleaseNoteUpdateInput!) {
  releaseNoteUpdate(id: $id, input: $input) {
    success
    releaseNote { id title url }
  }
}
""",
            {"id": release_note_id, "input": input_payload},
            op="releaseNoteUpdate", entity="releaseNote",
            policy_key="update_releases", what="Release note update",
            mutation_policy=mutation_policy,
        )

    async def create_issue_label(
        self,
        input_payload: dict[str, Any],
        *,
        mutation_policy: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        # Omit teamId in the input → a workspace-wide label.
        return await self._mutate(
            """
mutation IssueLabelCreate($input: IssueLabelCreateInput!) {
  issueLabelCreate(input: $input) {
    success
    issueLabel { id name color }
  }
}
""",
            {"input": input_payload},
            op="issueLabelCreate", entity="issueLabel",
            policy_key="create_labels", what="Issue label creation",
            mutation_policy=mutation_policy,
        )

    # --- Issue relations + URL links (MCP parity) ---
    #
    # Direction is load-bearing. In Linear's schema, IssueRelation.issue is the
    # SOURCE and IssueRelation.relatedIssue is the TARGET; the ``type`` field
    # describes the source's relationship to the target. So a ``blocks``
    # relation means ``issueId`` blocks ``relatedIssueId``. "A is blocked by B"
    # is therefore the SAME relation with the operands swapped: B blocks A.
    # There is no ``blockedBy`` enum value — it is expressed as ``blocks``
    # inverted (see IssueRelationType: blocks | duplicate | related | similar).
    async def create_issue_relation(
        self,
        issue_id: str,
        related_issue_id: str,
        relation_type: str,
        *,
        mutation_policy: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        return await self._mutate(
            """
mutation IssueRelationCreate($input: IssueRelationCreateInput!) {
  issueRelationCreate(input: $input) {
    success
    issueRelation { id type }
  }
}
""",
            {"input": {
                "issueId": issue_id,
                "relatedIssueId": related_issue_id,
                "type": relation_type,
            }},
            op="issueRelationCreate", entity="issueRelation",
            policy_key="update_issues", what="Issue relation create",
            mutation_policy=mutation_policy,
        )

    async def delete_issue_relation(
        self,
        relation_id: str,
        *,
        mutation_policy: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        return await self._mutate(
            """
mutation IssueRelationDelete($id: String!) {
  issueRelationDelete(id: $id) {
    success
    entityId
  }
}
""",
            {"id": relation_id},
            op="issueRelationDelete", entity=None,
            policy_key="update_issues", what="Issue relation delete",
            mutation_policy=mutation_policy,
        )

    async def link_url_to_issue(
        self,
        issue_id: str,
        url: str,
        title: Optional[str] = None,
        *,
        mutation_policy: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        variables: dict[str, Any] = {"issueId": issue_id, "url": url}
        if title:
            variables["title"] = title
        return await self._mutate(
            """
mutation AttachmentLinkURL($issueId: String!, $url: String!, $title: String) {
  attachmentLinkURL(issueId: $issueId, url: $url, title: $title) {
    success
    attachment { id title url }
  }
}
""",
            variables,
            op="attachmentLinkURL", entity="attachment",
            policy_key="update_issues", what="Issue URL link",
            mutation_policy=mutation_policy,
        )

    async def get_issue_relations(self, issue_id: str) -> dict[str, list[dict[str, Any]]]:
        """Return an issue's outgoing (``relations``) and incoming
        (``inverseRelations``) relations, each carrying the relation id, type,
        and both endpoints' id + identifier — enough to find the exact
        relation to delete for the remove* variants."""
        q = """
        query IssueRelations($id: String!) {
          issue(id: $id) {
            relations(first: 100) {
              nodes { id type issue { id identifier } relatedIssue { id identifier } }
            }
            inverseRelations(first: 100) {
              nodes { id type issue { id identifier } relatedIssue { id identifier } }
            }
          }
        }
        """
        data = await self.execute(q, {"id": issue_id})
        issue = data.get("issue") or {}
        return {
            "relations": (issue.get("relations") or {}).get("nodes", []),
            "inverseRelations": (issue.get("inverseRelations") or {}).get("nodes", []),
        }

    # --- Deletes (fail-closed; explicit ids only, no name resolution) ---
    async def delete_comment(
        self,
        comment_id: str,
        *,
        mutation_policy: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        return await self._mutate(
            """
mutation CommentDelete($id: String!) {
  commentDelete(id: $id) {
    success
    entityId
  }
}
""",
            {"id": comment_id},
            op="commentDelete", entity=None,
            policy_key="delete_comments", what="Comment deletion",
            mutation_policy=mutation_policy,
        )

    async def delete_customer(
        self,
        customer_id: str,
        *,
        mutation_policy: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        return await self._mutate(
            """
mutation CustomerDelete($id: String!) {
  customerDelete(id: $id) {
    success
    entityId
  }
}
""",
            {"id": customer_id},
            op="customerDelete", entity=None,
            policy_key="delete_customers", what="Customer deletion",
            mutation_policy=mutation_policy,
        )

    async def delete_customer_need(
        self,
        need_id: str,
        *,
        mutation_policy: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        return await self._mutate(
            """
mutation CustomerNeedDelete($id: String!) {
  customerNeedDelete(id: $id) {
    success
    entityId
  }
}
""",
            {"id": need_id},
            op="customerNeedDelete", entity=None,
            policy_key="delete_customer_needs", what="Customer need deletion",
            mutation_policy=mutation_policy,
        )

    async def delete_attachment(
        self,
        attachment_id: str,
        *,
        mutation_policy: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        return await self._mutate(
            """
mutation AttachmentDelete($id: String!) {
  attachmentDelete(id: $id) {
    success
    entityId
  }
}
""",
            {"id": attachment_id},
            op="attachmentDelete", entity=None,
            policy_key="delete_attachments", what="Attachment deletion",
            mutation_policy=mutation_policy,
        )

    async def delete_status_update(
        self,
        update_id: str,
        *,
        is_initiative: bool = False,
        mutation_policy: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Remove a project OR initiative status update.

        Linear has no ``initiativeUpdateDelete`` mutation, and
        ``projectUpdateDelete`` is deprecated in favour of the archive
        mutations, so both kinds route through *Archive* (Linear's supported
        "remove from view" path and what its own UI uses).
        """
        op = "initiativeUpdateArchive" if is_initiative else "projectUpdateArchive"
        return await self._mutate(
            f"""
mutation StatusUpdateArchive($id: String!) {{
  {op}(id: $id) {{
    success
    entityId
  }}
}}
""",
            {"id": update_id},
            op=op, entity=None,
            policy_key="delete_status_updates", what="Status update deletion",
            mutation_policy=mutation_policy,
        )
