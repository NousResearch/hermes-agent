"""Fail-closed admission policy for GitHub feedback receipts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import re
import subprocess
from typing import Mapping, Sequence

_REPOSITORY = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_SHA = re.compile(r"^[0-9a-fA-F]{40}$")
_FEEDBACK_KINDS = frozenset(
    {
        "issue_comment",
        "review_comment",
        "review",
        "pr_local_ci",
        "pr_repair",
        "pr_actions_needed",
    }
)
# NousResearch/hermes-agent is a foreign upstream repository (its issue
# tracker is the hermes-white-knight flow's own "assistant-brand-neutral"
# public surface); every other configured repository is ours, so a completed
# repair reply or audit comment there must self-identify as automated instead
# of reading like an ordinary human comment.
_BRAND_NEUTRAL_REPOSITORIES = frozenset({"NousResearch/hermes-agent"})
HERMES_ATTRIBUTION_PREFIX = "Hermes automated"
PR_REPAIR_ATTRIBUTION_PREFIX = f"{HERMES_ATTRIBUTION_PREFIX} repair"


def pr_repair_attribution_required(repository: str) -> bool:
    return repository not in _BRAND_NEUTRAL_REPOSITORIES


def pr_repair_attribution_line(assignee: str) -> str:
    return f"{PR_REPAIR_ATTRIBUTION_PREFIX} ({assignee})"


def hermes_attribution_line(assignee: str, *, action: str) -> str:
    """A generic 'Hermes automated <action> (<assignee>)' line for non-repair comments."""

    return f"{HERMES_ATTRIBUTION_PREFIX} {action} ({assignee})"


# Codex's GitHub App only re-reviews on PR-opened, marked-ready, or an
# explicit "@codex review" mention -- never on an ordinary push. Every path
# that pushes a new commit to an already-open PR (a worker's repair push, or
# the deterministic base-refresh merge-forward) must mention this after
# pushing, or the merge maintainer's codex_review_pending gate would wait
# forever for a re-review nothing ever asks for.
CODEX_REVIEW_TRIGGER = "@codex review"


MAX_ASSIGNEE_RULES = 32
MAX_MATCH_TERMS_PER_RULE = 32
MAX_COMMAND_ARGUMENTS = 32
MAX_COMMAND_ARGUMENT_LENGTH = 4096
_SHELL_COMMANDS = frozenset({"bash", "dash", "fish", "sh", "zsh"})
_MERGE_METHODS = frozenset({"squash", "rebase", "merge"})
_ROUTING_PRIORITIES = frozenset({"P0", "P1", "P2", "P3", "P4"})
_BLAST_RADII = frozenset({"contained", "moderate", "broad", "massive"})
_MAINTENANCE_LANE = re.compile(r"^[a-z0-9][a-z0-9-]{0,47}$")
MAX_MAINTENANCE_LANES = 8


def _nonempty_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _repository(value: object, field: str) -> str:
    repository = _nonempty_string(value, field)
    if not _REPOSITORY.fullmatch(repository):
        raise ValueError(f"{field} must be an exact owner/repository name")
    return repository


def _sha(value: object, field: str) -> str:
    candidate = _nonempty_string(value, field)
    if not _SHA.fullmatch(candidate):
        raise ValueError(f"{field} must be a full Git object ID")
    return candidate.casefold()


def _string_list(value: object, field: str, *, normalize=str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{field} must be a non-empty list of strings")
    items = tuple(normalize(_nonempty_string(item, field)) for item in value)
    if not items:
        raise ValueError(f"{field} must not be empty")
    return items


def _is_git_worktree(path: Path) -> bool:
    """Accept only a Git worktree root, including a linked-worktree `.git` file."""

    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(path),
                "rev-parse",
                "--is-inside-work-tree",
                "--show-toplevel",
            ],
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    lines = result.stdout.splitlines()
    return result.returncode == 0 and lines == ["true", str(path.resolve())]


@dataclass(frozen=True, slots=True)
class FeedbackReceipt:
    """The immutable, head-scoped identity of one item of review feedback."""

    repository: str
    pr_number: int
    feedback_kind: str
    feedback_id: str
    head_sha: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "repository", _repository(self.repository, "repository")
        )
        if (
            not isinstance(self.pr_number, int)
            or isinstance(self.pr_number, bool)
            or self.pr_number < 1
        ):
            raise ValueError("pr_number must be a positive integer")
        if self.feedback_kind not in _FEEDBACK_KINDS:
            raise ValueError("feedback_kind is not supported")
        object.__setattr__(
            self, "feedback_id", _nonempty_string(self.feedback_id, "feedback_id")
        )
        object.__setattr__(
            self, "head_sha", _nonempty_string(self.head_sha, "head_sha")
        )

    @property
    def key(self) -> tuple[str, int, str, str, str]:
        return (
            self.repository,
            self.pr_number,
            self.feedback_kind,
            self.feedback_id,
            self.head_sha,
        )


@dataclass(frozen=True, slots=True)
class PullRequest:
    """Canonical PR fields required to make an admission decision."""

    number: int
    state: str
    base_repository: str
    head_repository: str
    author_login: str
    head_ref_name: str
    head_sha: str
    labels: tuple[str, ...] = ()
    base_branch: str | None = None
    base_sha: str | None = None
    updated_at: datetime | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.number, int)
            or isinstance(self.number, bool)
            or self.number < 1
        ):
            raise ValueError("number must be a positive integer")
        object.__setattr__(self, "state", _nonempty_string(self.state, "state").upper())
        object.__setattr__(
            self,
            "base_repository",
            _repository(self.base_repository, "base_repository"),
        )
        object.__setattr__(
            self,
            "head_repository",
            _repository(self.head_repository, "head_repository"),
        )
        if self.updated_at is not None:
            if not isinstance(self.updated_at, datetime) or self.updated_at.tzinfo is None:
                raise ValueError("updated_at must be a timezone-aware datetime")
            object.__setattr__(
                self, "updated_at", self.updated_at.astimezone(timezone.utc)
            )
        object.__setattr__(
            self, "author_login", _nonempty_string(self.author_login, "author_login")
        )
        object.__setattr__(
            self, "head_ref_name", _nonempty_string(self.head_ref_name, "head_ref_name")
        )
        object.__setattr__(
            self, "head_sha", _nonempty_string(self.head_sha, "head_sha")
        )
        object.__setattr__(
            self,
            "labels",
            tuple(_nonempty_string(label, "pull request label") for label in self.labels),
        )
        if (self.base_branch is None) != (self.base_sha is None):
            raise ValueError("base_branch and base_sha must be provided together")
        if self.base_branch is not None:
            object.__setattr__(
                self, "base_branch", _nonempty_string(self.base_branch, "base_branch")
            )
            object.__setattr__(self, "base_sha", _sha(self.base_sha, "base_sha"))


@dataclass(frozen=True, slots=True)
class Reviewer:
    login: str
    association: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "login", _nonempty_string(self.login, "reviewer login")
        )
        if self.association is not None:
            object.__setattr__(
                self,
                "association",
                _nonempty_string(self.association, "reviewer association").upper(),
            )


@dataclass(frozen=True, slots=True)
class RepositoryTarget:
    base_repository: str
    head_repository: str
    local_path: Path
    owner_login: str
    branch_prefixes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Admission:
    admitted: bool
    reason: str | None = None
    target: RepositoryTarget | None = None


@dataclass(frozen=True, slots=True)
class AssigneeRule:
    """A bounded, deterministic content route with no model invocation."""

    assignee: str
    match_any: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RoutingRule:
    """A typed, deterministic Kanban route derived from canonical labels and bounded text."""

    assignee: str
    precedence: int
    match_any: tuple[str, ...]
    match_labels_any: tuple[str, ...]
    tags: tuple[str, ...]
    priority: str
    blast_radius: str
    risks: tuple[str, ...]
    requires_review: bool


@dataclass(frozen=True, slots=True)
class RoutingDecision:
    assignee: str
    tags: tuple[str, ...] = ()
    priority: str | None = None
    blast_radius: str | None = None
    risks: tuple[str, ...] = ()
    requires_review: bool = False
    ambiguous: bool = False


@dataclass(frozen=True, slots=True)
class LocalCIAuditPolicy:
    """Opt-in, read-only local CI coverage for repositories without Actions."""

    assignee: str
    post_results: bool
    repositories: frozenset[str] = frozenset()

    def applies_to(self, repository: str) -> bool:
        return not self.repositories or repository in self.repositories


@dataclass(frozen=True, slots=True)
class PostMergePolicy:
    """A fixed, local-only deployment action after a confirmed merge."""

    deployment_path: Path
    protected_runtime_entry: str
    package_argv: tuple[str, ...]
    bundle_path: str
    bundle_identifier: str
    relaunch_argv: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MergeMaintainerPolicy:
    """Strict scope and freshness limits for the deterministic merge owner."""

    assignee: str
    repository: str
    author_login: str
    base_branch: str
    merge_methods: tuple[str, ...]
    receipt_max_age_seconds: int
    report_only: bool
    post_merge: PostMergePolicy | None


@dataclass(frozen=True, slots=True)
class RepairStewardPolicy:
    """Exact-head, non-merging authority for bounded PR repairs."""

    assignee: str
    repositories: frozenset[str]
    report_only: bool
    max_base_refresh_in_flight: int = 1


@dataclass(frozen=True, slots=True)
class ReleaseMaintenanceLane:
    """One bounded, deterministic command lane and its specialist owner."""

    name: str
    assignee: str
    command: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ReleaseMaintenancePolicy:
    """Quiescence-gated, exact-head repository maintenance policy."""

    assignee: str
    repository: str
    base_branch: str
    quiet_period_seconds: int
    max_runtime_seconds: int
    lanes: tuple[ReleaseMaintenanceLane, ...]
    # Default preserves the original design: never run while any PR remains
    # open repository-wide. A continuously-active burndown repository can
    # legitimately carry dozens of open PRs indefinitely, so that condition
    # alone would mean maintenance never runs at all. quiet_period_seconds
    # already re-arms per new base SHA and is the gate that actually matters
    # for "don't run mid-churn" -- set this false to rely on it alone.
    require_zero_open_prs: bool = True


@dataclass(frozen=True, slots=True)
class PluginPolicy:
    enabled: bool
    targets: Mapping[str, RepositoryTarget]
    reviewer_logins: frozenset[str]
    reviewer_associations: frozenset[str]
    include_self_feedback: bool
    include_bot_feedback: bool
    auto_dispatch: bool
    not_before: datetime | None
    assignee: str | None
    board: str | None
    assignee_rules: tuple[AssigneeRule, ...] = ()
    routing_rules: tuple[RoutingRule, ...] = ()
    local_ci_audit: LocalCIAuditPolicy | None = None
    merge_maintainer: MergeMaintainerPolicy | None = None
    repair_steward: RepairStewardPolicy | None = None
    release_maintenance: ReleaseMaintenancePolicy | None = None

    def assignee_for(self, body: str) -> str:
        """Choose the unique highest-scoring specialist, otherwise the fallback."""

        if self.routing_rules:
            return self.route(body).assignee

        return self._legacy_assignee_for(body)

    def _legacy_assignee_for(self, body: str) -> str:

        normalized = " ".join(body.casefold().split())
        scores = tuple(
            (
                sum(_term_matches(normalized, term) for term in rule.match_any),
                rule.assignee,
            )
            for rule in self.assignee_rules
        )
        best_score = max((score for score, _assignee in scores), default=0)
        winners = {
            assignee for score, assignee in scores if score == best_score and score > 0
        }
        if len(winners) == 1:
            return winners.pop()
        return self.assignee or ""

    def route(self, body: str, *, labels: Sequence[str] = ()) -> RoutingDecision:
        """Select one typed route, failing equal-precedence ambiguity to the fallback."""

        normalized = " ".join(body.casefold().split())
        normalized_labels = {label.casefold() for label in labels}
        matches: list[tuple[int, int, RoutingRule]] = []
        for rule in self.routing_rules:
            text_hits = sum(_term_matches(normalized, term) for term in rule.match_any)
            label_hits = sum(label in normalized_labels for label in rule.match_labels_any)
            score = text_hits + label_hits
            if score:
                matches.append((rule.precedence, score, rule))
        if not matches:
            return RoutingDecision(self._legacy_assignee_for(body))
        best_rank = max((precedence, score) for precedence, score, _rule in matches)
        winners = [
            rule for precedence, score, rule in matches if (precedence, score) == best_rank
        ]
        if len(winners) != 1:
            return RoutingDecision(
                self.assignee or "",
                tags=("routing/ambiguous",),
                requires_review=True,
                ambiguous=True,
            )
        winner = winners[0]
        return RoutingDecision(
            assignee=winner.assignee,
            tags=winner.tags,
            priority=winner.priority,
            blast_radius=winner.blast_radius,
            risks=winner.risks,
            requires_review=winner.requires_review,
        )

    def admit_pull_request(self, pull_request: PullRequest) -> Admission:
        """Admit only an exact configured PR before reading any feedback bodies."""

        if not self.enabled:
            return Admission(False, "disabled")
        target = self.targets.get(pull_request.base_repository)
        if target is None:
            return Admission(False, "base_repository_not_allowed")
        if pull_request.head_repository != target.head_repository:
            return Admission(False, "head_repository_not_allowed")
        if pull_request.state != "OPEN":
            return Admission(False, "pull_request_not_open")
        if pull_request.author_login.casefold() != target.owner_login.casefold():
            return Admission(False, "author_not_allowed")
        if not any(
            pull_request.head_ref_name.startswith(prefix)
            for prefix in target.branch_prefixes
        ):
            return Admission(False, "branch_not_allowed")
        return Admission(True, target=target)

    def admit(
        self,
        pull_request: PullRequest,
        reviewer: Reviewer,
        receipt: FeedbackReceipt,
        *,
        is_bot: bool = False,
    ) -> Admission:
        pull_request_admission = self.admit_pull_request(pull_request)
        target = pull_request_admission.target
        if not pull_request_admission.admitted or target is None:
            return pull_request_admission
        if receipt.repository != pull_request.base_repository:
            return Admission(False, "base_repository_not_allowed")
        trusted_login = reviewer.login.casefold() in self.reviewer_logins
        trusted_association = (reviewer.association or "") in self.reviewer_associations
        trusted_self = (
            self.include_self_feedback
            and reviewer.login.casefold() == target.owner_login.casefold()
        )
        trusted_bot = self.include_bot_feedback and is_bot
        if not (trusted_login or trusted_association or trusted_self or trusted_bot):
            return Admission(False, "reviewer_not_allowed")
        if (
            receipt.pr_number != pull_request.number
            or receipt.head_sha != pull_request.head_sha
        ):
            return Admission(False, "head_changed")
        return Admission(True, target=target)


def _parse_target(raw: object) -> RepositoryTarget:
    if not isinstance(raw, Mapping):
        raise ValueError("repositories entries must be mappings")
    expected = {
        "base_repository",
        "head_repository",
        "local_path",
        "owner_login",
        "branch_prefixes",
    }
    if set(raw) != expected:
        raise ValueError("repository target has missing or unknown fields")
    path = Path(_nonempty_string(raw["local_path"], "local_path"))
    if not path.is_absolute() or not path.is_dir() or not _is_git_worktree(path):
        raise ValueError("local_path must be an existing local Git repository")
    prefixes = _string_list(raw["branch_prefixes"], "branch_prefixes")
    if any(
        prefix.startswith("refs/") or any(char.isspace() for char in prefix)
        for prefix in prefixes
    ):
        raise ValueError("branch_prefixes must be literal branch prefixes")
    return RepositoryTarget(
        base_repository=_repository(raw["base_repository"], "base_repository"),
        head_repository=_repository(raw["head_repository"], "head_repository"),
        local_path=path.resolve(),
        owner_login=_nonempty_string(raw["owner_login"], "owner_login"),
        branch_prefixes=prefixes,
    )


def _not_before(value: object) -> datetime:
    text = _nonempty_string(value, "not_before")
    try:
        boundary = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("not_before must be ISO-8601") from error
    if boundary.tzinfo is None:
        raise ValueError("not_before must include a timezone")
    return boundary.astimezone(timezone.utc)


def _term_matches(body: str, term: str) -> int:
    return int(re.search(rf"(?<!\w){re.escape(term)}(?!\w)", body) is not None)


def _parse_assignee_rules(raw: object) -> tuple[AssigneeRule, ...]:
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence) or not raw:
        raise ValueError("assignee_rules must be a non-empty list")
    if len(raw) > MAX_ASSIGNEE_RULES:
        raise ValueError("assignee_rules exceeds its bounded limit")
    rules: list[AssigneeRule] = []
    for item in raw:
        if not isinstance(item, Mapping) or set(item) != {"assignee", "match_any"}:
            raise ValueError("assignee_rules entries have missing or unknown fields")
        terms = _string_list(item["match_any"], "match_any", normalize=str.casefold)
        if len(terms) > MAX_MATCH_TERMS_PER_RULE:
            raise ValueError("match_any exceeds its bounded limit")
        rules.append(
            AssigneeRule(_nonempty_string(item["assignee"], "assignee"), terms)
        )
    return tuple(rules)


def _optional_string_list(value: object, field: str, *, normalize=str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{field} must be a list of strings")
    return tuple(normalize(_nonempty_string(item, field)) for item in value)


def _parse_routing_rules(raw: object) -> tuple[RoutingRule, ...]:
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence) or not raw:
        raise ValueError("routing_rules must be a non-empty list")
    if len(raw) > MAX_ASSIGNEE_RULES:
        raise ValueError("routing_rules exceeds its bounded limit")
    expected = {
        "assignee",
        "precedence",
        "match_any",
        "match_labels_any",
        "tags",
        "priority",
        "blast_radius",
        "risks",
        "requires_review",
    }
    rules: list[RoutingRule] = []
    for item in raw:
        if not isinstance(item, Mapping) or set(item) != expected:
            raise ValueError("routing_rules entries have missing or unknown fields")
        precedence = item["precedence"]
        requires_review = item["requires_review"]
        if (
            not isinstance(precedence, int)
            or isinstance(precedence, bool)
            or not 0 <= precedence <= 1000
        ):
            raise ValueError("routing precedence must be an integer from 0 to 1000")
        if not isinstance(requires_review, bool):
            raise ValueError("routing requires_review must be a boolean")
        terms = _optional_string_list(item["match_any"], "routing match_any", normalize=str.casefold)
        labels = _optional_string_list(
            item["match_labels_any"], "routing match_labels_any", normalize=str.casefold
        )
        if not terms and not labels:
            raise ValueError("routing rule must include a text term or label")
        if len(terms) > MAX_MATCH_TERMS_PER_RULE or len(labels) > MAX_MATCH_TERMS_PER_RULE:
            raise ValueError("routing match terms exceed their bounded limit")
        tags = _optional_string_list(item["tags"], "routing tags")
        risks = _optional_string_list(item["risks"], "routing risks", normalize=str.casefold)
        if len(tags) > MAX_MATCH_TERMS_PER_RULE or len(risks) > MAX_MATCH_TERMS_PER_RULE:
            raise ValueError("routing metadata exceeds its bounded limit")
        priority = _nonempty_string(item["priority"], "routing priority").upper()
        blast_radius = _nonempty_string(
            item["blast_radius"], "routing blast_radius"
        ).casefold()
        if priority not in _ROUTING_PRIORITIES:
            raise ValueError("routing priority must be P0 through P4")
        if blast_radius not in _BLAST_RADII:
            raise ValueError("routing blast_radius is invalid")
        rules.append(
            RoutingRule(
                assignee=_nonempty_string(item["assignee"], "routing assignee"),
                precedence=precedence,
                match_any=terms,
                match_labels_any=labels,
                tags=tags,
                priority=priority,
                blast_radius=blast_radius,
                risks=risks,
                requires_review=requires_review,
            )
        )
    return tuple(rules)


def _parse_local_ci_audit(raw: object) -> LocalCIAuditPolicy | None:
    if not isinstance(raw, Mapping):
        raise ValueError("local_ci_audit must be a mapping")
    required = {"enabled", "assignee", "post_results"}
    if not required.issubset(raw) or set(raw).difference(required | {"repositories"}):
        raise ValueError("local_ci_audit has missing or unknown fields")
    enabled = raw["enabled"]
    post_results = raw["post_results"]
    if not isinstance(enabled, bool) or not isinstance(post_results, bool):
        raise ValueError("local_ci_audit booleans are invalid")
    assignee = _nonempty_string(raw["assignee"], "local_ci_audit assignee")
    repositories = (
        frozenset(_string_list(raw["repositories"], "local_ci_audit repositories"))
        if "repositories" in raw
        else frozenset()
    )
    if not enabled:
        return None
    return LocalCIAuditPolicy(
        assignee=assignee, post_results=post_results, repositories=repositories
    )


def _parse_repair_steward(
    raw: object, *, targets: Mapping[str, RepositoryTarget]
) -> RepairStewardPolicy | None:
    if not isinstance(raw, Mapping):
        raise ValueError("repair_steward must be a mapping")
    required = {"enabled", "assignee", "repositories", "report_only"}
    if not required.issubset(raw) or set(raw) - (
        required | {"max_base_refresh_in_flight"}
    ):
        raise ValueError("repair_steward has missing or unknown fields")
    if not isinstance(raw["enabled"], bool) or not isinstance(raw["report_only"], bool):
        raise ValueError("repair_steward booleans are invalid")
    repositories = frozenset(
        _string_list(raw["repositories"], "repair_steward repositories")
    )
    if not repositories or not repositories.issubset(targets):
        raise ValueError("repair_steward repositories must be configured targets")
    max_base_refresh_in_flight = raw.get("max_base_refresh_in_flight", 1)
    if (
        not isinstance(max_base_refresh_in_flight, int)
        or isinstance(max_base_refresh_in_flight, bool)
        or not 1 <= max_base_refresh_in_flight <= 8
    ):
        raise ValueError(
            "repair_steward max_base_refresh_in_flight must be an integer from 1 to 8"
        )
    if not raw["enabled"]:
        return None
    return RepairStewardPolicy(
        assignee=_nonempty_string(raw["assignee"], "repair_steward assignee"),
        repositories=repositories,
        report_only=raw["report_only"],
        max_base_refresh_in_flight=max_base_refresh_in_flight,
    )


def _relative_path(value: object, field: str) -> str:
    text = _nonempty_string(value, field)
    path = Path(text)
    if (
        path.is_absolute()
        or "\x00" in text
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"{field} must be a normalized relative path")
    return text


def _command_argv(value: object, field: str) -> tuple[str, ...]:
    argv = _string_list(value, field)
    if len(argv) > MAX_COMMAND_ARGUMENTS:
        raise ValueError(f"{field} exceeds its bounded limit")
    if any(
        len(argument) > MAX_COMMAND_ARGUMENT_LENGTH
        or "\x00" in argument
        or "\n" in argument
        or "\r" in argument
        for argument in argv
    ):
        raise ValueError(f"{field} contains an invalid argument")
    if Path(argv[0]).name.casefold() in _SHELL_COMMANDS:
        raise ValueError(f"{field} must not invoke a command shell")
    return argv


def _parse_post_merge(
    raw: object, *, target: RepositoryTarget
) -> PostMergePolicy | None:
    if not isinstance(raw, Mapping):
        raise ValueError("post_merge must be a mapping")
    enabled = raw.get("enabled")
    if not isinstance(enabled, bool):
        raise ValueError("post_merge enabled must be a boolean")
    if not enabled:
        if set(raw) != {"enabled"}:
            raise ValueError("disabled post_merge has unknown fields")
        return None
    expected = {
        "enabled",
        "deployment_path",
        "protected_runtime_entry",
        "package_argv",
        "bundle_path",
        "bundle_identifier",
        "relaunch_argv",
    }
    if set(raw) != expected:
        raise ValueError("post_merge has missing or unknown fields")
    deployment_path = Path(_nonempty_string(raw["deployment_path"], "deployment_path"))
    if (
        not deployment_path.is_absolute()
        or not deployment_path.is_dir()
        or not _is_git_worktree(deployment_path)
    ):
        raise ValueError("deployment_path must be an existing Git worktree")
    deployment_path = deployment_path.resolve()
    if deployment_path == target.local_path:
        raise ValueError("deployment_path must be distinct from the audit worktree")
    return PostMergePolicy(
        deployment_path=deployment_path,
        protected_runtime_entry=_relative_path(
            raw["protected_runtime_entry"], "protected_runtime_entry"
        ),
        package_argv=_command_argv(raw["package_argv"], "package_argv"),
        bundle_path=_relative_path(raw["bundle_path"], "bundle_path"),
        bundle_identifier=_nonempty_string(
            raw["bundle_identifier"], "bundle_identifier"
        ),
        relaunch_argv=_command_argv(raw["relaunch_argv"], "relaunch_argv"),
    )


def _parse_merge_maintainer(
    raw: object, *, targets: Mapping[str, RepositoryTarget]
) -> MergeMaintainerPolicy | None:
    if not isinstance(raw, Mapping):
        raise ValueError("merge_maintainer must be a mapping")
    enabled = raw.get("enabled")
    if not isinstance(enabled, bool):
        raise ValueError("merge_maintainer enabled must be a boolean")
    if not enabled:
        if set(raw) != {"enabled"}:
            raise ValueError("disabled merge_maintainer has unknown fields")
        return None
    expected = {
        "enabled",
        "assignee",
        "repository",
        "author_login",
        "base_branch",
        "merge_methods",
        "receipt_max_age_seconds",
        "report_only",
        "post_merge",
    }
    if set(raw) != expected:
        raise ValueError("merge_maintainer has missing or unknown fields")
    repository = _repository(raw["repository"], "merge_maintainer repository")
    target = targets.get(repository)
    if target is None or target.head_repository != repository:
        raise ValueError(
            "merge_maintainer repository must be an exact same-repository target"
        )
    author_login = _nonempty_string(
        raw["author_login"], "merge_maintainer author_login"
    )
    if author_login.casefold() != target.owner_login.casefold():
        raise ValueError("merge_maintainer author_login must match the target owner")
    base_branch = _nonempty_string(raw["base_branch"], "merge_maintainer base_branch")
    if base_branch.startswith("refs/") or any(
        character.isspace() for character in base_branch
    ):
        raise ValueError("merge_maintainer base_branch must be a literal branch name")
    merge_methods = _string_list(
        raw["merge_methods"], "merge_maintainer merge_methods", normalize=str.casefold
    )
    if len(set(merge_methods)) != len(merge_methods) or any(
        method not in _MERGE_METHODS for method in merge_methods
    ):
        raise ValueError("merge_methods must be unique squash, rebase, or merge values")
    receipt_max_age_seconds = raw["receipt_max_age_seconds"]
    if (
        not isinstance(receipt_max_age_seconds, int)
        or isinstance(receipt_max_age_seconds, bool)
        or receipt_max_age_seconds < 1
    ):
        raise ValueError("receipt_max_age_seconds must be a positive integer")
    report_only = raw["report_only"]
    if not isinstance(report_only, bool):
        raise ValueError("report_only must be a boolean")
    return MergeMaintainerPolicy(
        assignee=_nonempty_string(raw["assignee"], "merge_maintainer assignee"),
        repository=repository,
        author_login=author_login,
        base_branch=base_branch,
        merge_methods=merge_methods,
        receipt_max_age_seconds=receipt_max_age_seconds,
        report_only=report_only,
        post_merge=_parse_post_merge(raw["post_merge"], target=target),
    )


def _parse_release_maintenance(
    raw: object, *, targets: Mapping[str, RepositoryTarget]
) -> ReleaseMaintenancePolicy | None:
    if not isinstance(raw, Mapping):
        raise ValueError("release_maintenance must be a mapping")
    enabled = raw.get("enabled")
    if not isinstance(enabled, bool):
        raise ValueError("release_maintenance enabled must be a boolean")
    if not enabled:
        if set(raw) != {"enabled"}:
            raise ValueError("disabled release_maintenance has unknown fields")
        return None
    required = {
        "enabled",
        "assignee",
        "repository",
        "base_branch",
        "quiet_period_seconds",
        "max_runtime_seconds",
        "lanes",
    }
    if not required.issubset(raw) or set(raw) - (
        required | {"require_zero_open_prs"}
    ):
        raise ValueError("release_maintenance has missing or unknown fields")
    require_zero_open_prs = raw.get("require_zero_open_prs", True)
    if not isinstance(require_zero_open_prs, bool):
        raise ValueError("release_maintenance require_zero_open_prs must be a boolean")
    repository = _repository(raw["repository"], "release_maintenance repository")
    target = targets.get(repository)
    if target is None or target.head_repository != repository:
        raise ValueError(
            "release_maintenance repository must be an exact same-repository target"
        )
    base_branch = _nonempty_string(
        raw["base_branch"], "release_maintenance base_branch"
    )
    if base_branch.startswith("refs/") or any(
        character.isspace() for character in base_branch
    ):
        raise ValueError(
            "release_maintenance base_branch must be a literal branch name"
        )
    quiet_period_seconds = raw["quiet_period_seconds"]
    max_runtime_seconds = raw["max_runtime_seconds"]
    if (
        not isinstance(quiet_period_seconds, int)
        or isinstance(quiet_period_seconds, bool)
        or quiet_period_seconds < 60
    ):
        raise ValueError("quiet_period_seconds must be an integer of at least 60")
    if (
        not isinstance(max_runtime_seconds, int)
        or isinstance(max_runtime_seconds, bool)
        or max_runtime_seconds < 60
    ):
        raise ValueError("max_runtime_seconds must be an integer of at least 60")
    raw_lanes = raw["lanes"]
    if isinstance(raw_lanes, (str, bytes)) or not isinstance(raw_lanes, Sequence):
        raise ValueError("release_maintenance lanes must be a non-empty list")
    if not 1 <= len(raw_lanes) <= MAX_MAINTENANCE_LANES:
        raise ValueError(
            "release_maintenance must configure between one and eight lanes"
        )
    lanes: list[ReleaseMaintenanceLane] = []
    for item in raw_lanes:
        if not isinstance(item, Mapping) or set(item) != {
            "name",
            "assignee",
            "command",
        }:
            raise ValueError("release_maintenance lane has missing or unknown fields")
        name = _nonempty_string(
            item["name"], "release_maintenance lane name"
        ).casefold()
        if not _MAINTENANCE_LANE.fullmatch(name):
            raise ValueError("release_maintenance lane name is invalid")
        command = _command_argv(item["command"], "release_maintenance lane command")
        if any(Path(argument).name.casefold() == "main.py" for argument in command):
            raise ValueError("release_maintenance cannot launch the protected runtime")
        lanes.append(
            ReleaseMaintenanceLane(
                name=name,
                assignee=_nonempty_string(
                    item["assignee"], "release_maintenance lane assignee"
                ),
                command=command,
            )
        )
    if len({lane.name for lane in lanes}) != len(lanes):
        raise ValueError("release_maintenance lane names must be unique")
    return ReleaseMaintenancePolicy(
        assignee=_nonempty_string(raw["assignee"], "release_maintenance assignee"),
        repository=repository,
        base_branch=base_branch,
        quiet_period_seconds=quiet_period_seconds,
        max_runtime_seconds=max_runtime_seconds,
        lanes=tuple(lanes),
        require_zero_open_prs=require_zero_open_prs,
    )


def load_policy(raw: object) -> PluginPolicy:
    """Parse plugin configuration, retaining no enabled behavior on any omission."""

    if not isinstance(raw, Mapping):
        raise ValueError("plugin configuration must be a mapping")
    enabled = raw.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("enabled must be a boolean")
    if not enabled:
        return PluginPolicy(
            False, {}, frozenset(), frozenset(), False, False, False, None, None, None
        )
    required = {
        "enabled",
        "repositories",
        "reviewer_logins",
        "reviewer_associations",
        "not_before",
        "assignee",
        "board",
    }
    optional = {
        "include_self_feedback",
        "include_bot_feedback",
        "auto_dispatch",
        "assignee_rules",
        "routing_rules",
        "local_ci_audit",
        "merge_maintainer",
        "repair_steward",
        "release_maintenance",
    }
    if not required.issubset(raw) or set(raw) - required - optional:
        raise ValueError("enabled configuration has missing or unknown fields")
    include_self_feedback = raw.get("include_self_feedback", False)
    include_bot_feedback = raw.get("include_bot_feedback", False)
    auto_dispatch = raw.get("auto_dispatch", False)
    if (
        not isinstance(include_self_feedback, bool)
        or not isinstance(include_bot_feedback, bool)
        or not isinstance(auto_dispatch, bool)
    ):
        raise ValueError("feedback inclusion settings must be booleans")
    repositories = raw["repositories"]
    if isinstance(repositories, (str, bytes)) or not isinstance(repositories, Sequence):
        raise ValueError("repositories must be a non-empty list")
    parsed_targets = tuple(_parse_target(target) for target in repositories)
    if not parsed_targets:
        raise ValueError("repositories must not be empty")
    targets = {target.base_repository: target for target in parsed_targets}
    if len(targets) != len(parsed_targets):
        raise ValueError(
            "each base_repository may have only one configured head_repository"
        )
    reviewer_logins = (
        frozenset()
        if raw["reviewer_logins"] == []
        else frozenset(
            _string_list(
                raw["reviewer_logins"], "reviewer_logins", normalize=str.casefold
            )
        )
    )
    reviewer_associations = (
        frozenset()
        if raw["reviewer_associations"] == []
        else frozenset(
            _string_list(
                raw["reviewer_associations"],
                "reviewer_associations",
                normalize=str.upper,
            )
        )
    )
    if not reviewer_logins and not reviewer_associations:
        raise ValueError("at least one reviewer login or association is required")
    return PluginPolicy(
        enabled=True,
        targets=targets,
        reviewer_logins=reviewer_logins,
        reviewer_associations=reviewer_associations,
        include_self_feedback=include_self_feedback,
        include_bot_feedback=include_bot_feedback,
        auto_dispatch=auto_dispatch,
        not_before=_not_before(raw["not_before"]),
        assignee=_nonempty_string(raw["assignee"], "assignee"),
        board=_nonempty_string(raw["board"], "board"),
        assignee_rules=(
            _parse_assignee_rules(raw["assignee_rules"])
            if "assignee_rules" in raw
            else ()
        ),
        routing_rules=(
            _parse_routing_rules(raw["routing_rules"]) if "routing_rules" in raw else ()
        ),
        local_ci_audit=_validated_local_ci_audit(raw, targets),
        merge_maintainer=(
            _parse_merge_maintainer(raw["merge_maintainer"], targets=targets)
            if "merge_maintainer" in raw
            else None
        ),
        repair_steward=(
            _parse_repair_steward(raw["repair_steward"], targets=targets)
            if "repair_steward" in raw
            else None
        ),
        release_maintenance=(
            _parse_release_maintenance(raw["release_maintenance"], targets=targets)
            if "release_maintenance" in raw
            else None
        ),
    )


def _validated_local_ci_audit(
    raw: Mapping[str, object], targets: Mapping[str, RepositoryTarget]
) -> LocalCIAuditPolicy | None:
    if "local_ci_audit" not in raw:
        return None
    policy = _parse_local_ci_audit(raw["local_ci_audit"])
    if (
        policy is not None
        and policy.repositories
        and not policy.repositories.issubset(targets)
    ):
        raise ValueError("local_ci_audit repositories must be configured targets")
    return policy
