"""Who is calling the control plane, and what they may see.

Before this, the control plane had one shared bearer token and no concept of a caller. That
is enough to keep strangers out and not enough to run a deployment: there is no per-user
audit, no read/approve separation, and no way to revoke one person's access without
rotating everyone's.

It matters more than it looks, because the control plane is where a human will eventually
*approve* an escalated business action. An approval gate with no identity cannot record who
approved, and an approval nobody can be held to is not a control.

**This is deliberately not OIDC.** NOVA loads with the standard library and PyYAML, and an
identity provider is a dependency, a deployment, and a failure mode. What is here is the
abstraction OIDC would slot into — a :class:`Principal` resolved from a credential, with
routes gated on its role — so adding a real IdP later replaces :func:`authenticate` and
touches nothing else.

**Tokens are stored hashed.** The file NOVA reads holds SHA-256 digests, never the tokens
themselves. A leaked principals file is then an inconvenience rather than an incident, and
it means NOVA never holds a credential in the clear — the same rule the deployment seam is
built on, applied to NOVA's own front door.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

from nova.errors import NovaError

#: Roles, least to most privileged. Two, because two is the number of genuinely different
#: answers to "what may this caller see?" — not because a third could not be invented.
#:
#: ``viewer``  operational state: is it healthy, what agents exist, what work is running.
#: ``admin``   the above plus the governance surfaces — what each agent is permitted to do,
#:             what was refused, and what it cost.
ROLES = ("viewer", "admin")

#: Minimum role per route. A route absent from here requires ``admin``: a new endpoint
#: should have to be *declared* readable by a viewer, rather than becoming readable by
#: everyone because someone forgot this file existed.
ROUTE_ROLES: Mapping[str, str] = {
    "/health": "viewer",
    "/identity": "viewer",
    "/agents": "viewer",
    "/tasks": "viewer",
    "/objectives": "viewer",
    "/knowledge": "viewer",
    # What is connected and which agents it may reach. Readable by a viewer: it is
    # operational state, and it contains no credential — only variable names.
    "/channels": "viewer",
    # What runs on a schedule is operational state, like the work board. The prompt an
    # automation carries is NOT returned, so a viewer sees what runs and when, never the
    # instruction text.
    "/automations": "viewer",
    # Who the deployment serves and what the product is called. Readable by a viewer: it is
    # the branding already on every screen they are looking at.
    "/settings": "viewer",
    # The logo, served as an image. Readable by anyone who can see the dashboard it is on.
    "/branding/logo": "viewer",
    "/branding/favicon": "viewer",
    # Governance surfaces. What an agent may do, what it was refused, and what it spent are
    # the questions an attacker asks first and an auditor asks legitimately — same data,
    # different principal.
    "/policy": "admin",
    "/policy/simulate": "admin",
    "/decisions": "admin",
    "/budget": "admin",
}

#: Minimum role per *write* route, kept separate from :data:`ROUTE_ROLES` on purpose.
#: Reading a route and changing what it describes are different permissions, and a single
#: table would make them the same one by accident the first time somebody added an endpoint.
#:
#: A route absent from here cannot be written **at all** — not "requires admin", but
#: unroutable. The default for a write has to be refusal: a read nobody meant to expose
#: leaks, and a write nobody meant to expose acts.
WRITE_ROUTES: Mapping[str, str] = {
    # Acting on work already on the board: release it, send it back for changes, resume it,
    # or leave a note a worker will read.
    "/work/decide": "admin",
    # Agent lifecycle. Every one is admin: an agent's persona is its standing instruction
    # on every turn, and its permissions are what it may do — neither is operational state
    # a viewer should be able to change.
    "/agents/create": "admin",
    "/agents/update": "admin",
    "/agents/soul": "admin",
    "/agents/duplicate": "admin",
    "/agents/archive": "admin",
    "/agents/restore": "admin",
    "/agents/delete": "admin",
    # The only route in NOVA that writes a secret. Admin, like every other agent write —
    # and the value never comes back out, so there is no matching read to gate.
    "/agents/credentials": "admin",
    # The tenant's own identity. Admin: these are what every branded surface shows, and the
    # audit log's subject is the tenant itself rather than an agent.
    "/settings/organization": "admin",
    "/settings/identity": "admin",
    "/settings/logo": "admin",
    "/settings/agent-name": "admin",
    # Putting a declared objective's steps on the board.
    "/objectives/submit": "admin",
    # Making the runtime deliver declared conversations to granted agents.
    "/channels/apply": "admin",
    # Pausing, resuming or deleting a schedule the runtime already holds. Admin:
    # stopping a nightly reconciliation is as consequential as releasing held work.
    "/automations/decide": "admin",
    # Declaring a new automation. Admin, and it is the heaviest write here: it gives an
    # agent a standing instruction on a timer. It is only reachable through the
    # compiler, which refuses anything the owning agent was not already granted.
    "/automations/create": "admin",
}

#: Read routes that address ONE member of a collection, e.g. ``/tasks/t_ab12``. Only the
#: prefixes listed here resolve to their collection's role; everything else still falls
#: through to admin.
#:
#: Needed because ``ROUTE_ROLES`` is an exact-match table, so ``/tasks`` was readable by a
#: viewer while ``/tasks/<id>`` silently required admin — a viewer could see that work
#: existed but never open it. Declared as its own table rather than by loosening the
#: default: the fail-closed default is the thing protecting every route nobody has
#: thought about yet.
ROUTE_COLLECTION_PREFIXES: Mapping[str, str] = {
    "/tasks/": "viewer",
}


#: Reads of the shape ``/<collection>/<id>/<leaf>``, keyed on collection and leaf.
#:
#: Kept separate from :data:`ROUTE_COLLECTION_PREFIXES` and enumerated rather than matched
#: by prefix, because the id sits in the middle: a rule that accepted any leaf under
#: ``/agents/<id>/`` would hand out every future sub-read as soon as somebody added one.
#:
#: ``soul`` is **absent on purpose** and therefore admin, like every undeclared route. An
#: agent's persona is its standing instruction on every turn — the same class of text as an
#: automation's objective, which a viewer already cannot read.
ROUTE_MEMBER_LEAVES: Mapping[tuple[str, str], str] = {
    # What runs and when is operational state, exactly as the /automations collection is.
    ("agents", "automations"): "viewer",
}


def _collection_role(route: str) -> Optional[str]:
    """The role for a member read like ``/tasks/<id>`` or ``/agents/<id>/automations``.

    A bare member takes the collection's role; a leaf under a member takes one only if it
    is declared in :data:`ROUTE_MEMBER_LEAVES`. Anything else returns None and falls
    through to the admin default, which is the safe direction for a route nobody has
    thought about yet.
    """
    for prefix, role in ROUTE_COLLECTION_PREFIXES.items():
        if route.startswith(prefix):
            member = route[len(prefix):]
            if member and "/" not in member:
                return role

    parts = [part for part in route.split("/") if part]
    if len(parts) == 3:
        return ROUTE_MEMBER_LEAVES.get((parts[0], parts[2]))
    return None


#: The file, inside the runtime home, that lists who may call the control plane.
PRINCIPALS_FILENAME = "control-principals.yaml"

#: Length of the API prefix, so the server can map a request path to a ROUTE_ROLES key
#: without importing the API module and creating a cycle.
API_PREFIX_LEN = len("/platform/v1")

#: Tokens NOVA generates. 32 bytes of urandom, URL-safe — long enough that rate limiting is
#: a defence in depth rather than the thing standing between a deployment and a guess.
TOKEN_BYTES = 32


@dataclass(frozen=True)
class Principal:
    """An authenticated caller."""

    name: str
    role: str
    #: How this caller was authenticated, for the access log.
    via: str = "token"

    def may(self, route: str) -> bool:
        """Whether this principal may read ``route``. Unknown routes require admin."""
        required = ROUTE_ROLES.get(route) or _collection_role(route) or "admin"
        return ROLES.index(self.role) >= ROLES.index(required)

    def may_write(self, route: str) -> bool:
        """Whether this principal may call the *write* route ``route``.

        An undeclared route is refused outright rather than defaulted to admin. The
        asymmetry with :meth:`may` is deliberate: forgetting to declare a read exposes
        data, and forgetting to declare a write hands out an action.
        """
        required = WRITE_ROUTES.get(route)
        if required is None:
            return False
        return ROLES.index(self.role) >= ROLES.index(required)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "role": self.role, "via": self.via}


#: The caller on a loopback bind with no principals file. Named rather than anonymous so
#: the access log says how a request was authorised, and so an operator reading the log can
#: tell "nobody configured auth" from "someone authenticated".
LOCAL_ADMIN = Principal(name="local", role="admin", via="loopback")


def hash_token(token: str) -> str:
    """The digest stored for a token. Never reversible, and never the token."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def new_token() -> str:
    """A fresh bearer token. Returned once; NOVA stores only its digest."""
    return secrets.token_urlsafe(TOKEN_BYTES)


@dataclass(frozen=True)
class PrincipalStore:
    """Everyone permitted to call this deployment's control plane."""

    principals: tuple[tuple[str, Principal], ...] = ()
    source: Optional[Path] = None

    @property
    def configured(self) -> bool:
        return bool(self.principals)

    def authenticate(self, token: str) -> Optional[Principal]:
        """The principal holding ``token``, or None.

        Every candidate is compared even after a match, and the comparison is
        constant-time. A store that returned early would leak which prefix was right
        through response timing — a small leak, but the whole point of a bearer token is
        that guessing it is the only attack, so nothing should make guessing cheaper.
        """
        if not token:
            return None
        presented = hash_token(token)
        found: Optional[Principal] = None
        for digest, principal in self.principals:
            if hmac.compare_digest(digest, presented):
                found = principal
        return found

    @classmethod
    def load(cls, path: Path) -> "PrincipalStore":
        """Read a principals file. An absent file is an empty store, not an error."""
        import yaml

        if not path.is_file():
            return cls(source=path)
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError) as exc:
            raise NovaError(f"{path}: could not be read as YAML: {exc}") from exc

        raw = data.get("principals")
        if raw is None:
            return cls(source=path)
        if not isinstance(raw, list):
            raise NovaError(f"{path}: 'principals' must be a list")

        entries: list[tuple[str, Principal]] = []
        seen: set[str] = set()
        for index, item in enumerate(raw):
            if not isinstance(item, Mapping):
                raise NovaError(f"{path}: principals[{index}] must be a mapping")
            name = str(item.get("name") or "").strip()
            role = str(item.get("role") or "").strip()
            digest = str(item.get("token_sha256") or "").strip().lower()

            if not name:
                raise NovaError(f"{path}: principals[{index}] has no name")
            if role not in ROLES:
                raise NovaError(
                    f"{path}: principals[{index}] role {role!r} is not one of "
                    f"{', '.join(ROLES)}"
                )
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                # Almost always someone pasting the token where the digest goes. Saying so
                # is the difference between a fixable message and a silent 401 loop — and
                # it stops the token itself being stored by accident.
                raise NovaError(
                    f"{path}: principals[{index}] ({name}) needs token_sha256 to be a "
                    "64-character SHA-256 hex digest. If you pasted the token itself, "
                    "generate a new one with `nova token new` — NOVA stores only digests"
                )
            if name in seen:
                raise NovaError(f"{path}: duplicate principal name {name!r}")
            seen.add(name)
            entries.append((digest, Principal(name=name, role=role)))

        return cls(principals=tuple(entries), source=path)

    def describe(self) -> list[dict[str, str]]:
        """Who is configured, for an operator. Never includes a digest."""
        return [{"name": p.name, "role": p.role} for _, p in self.principals]
