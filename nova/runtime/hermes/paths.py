"""Where the Hermes runtime keeps its state, resolved without importing Hermes.

Resolution order is the alias chain from the boundary document: a NOVA-specific location
wins when set, and the Hermes locations remain fully supported so an existing
installation keeps working untouched.

    $NOVA_HOME  ->  ~/.nova  ->  $HERMES_HOME  ->  ~/.hermes

Deliberately re-implemented rather than importing ``hermes_constants.get_hermes_home``:
importing it would pull the runtime into NOVA's import graph and invert the dependency
arrow the architecture depends on. The cost is this function; the benefit is that
``nova`` imports with only PyYAML present.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional

#: Marker files that identify a directory as a runtime home. Mirrors the runtime's own
#: markers; a mismatch means we create a fresh home rather than adopt a wrong directory.
HOME_MARKERS = ("config.yaml", ".env", "state.db")


def resolve_home(env: Optional[Mapping[str, str]] = None) -> Path:
    """The runtime home directory, by the alias chain above.

    Returns a path that may not exist yet; callers that need it present create it.
    """
    environ = os.environ if env is None else env

    for var in ("NOVA_HOME", "HERMES_HOME"):
        raw = (environ.get(var) or "").strip()
        if raw:
            return Path(raw).expanduser()

    home = Path(environ.get("HOME") or Path.home()).expanduser()
    nova_home = home / ".nova"
    if _looks_like_home(nova_home):
        return nova_home
    hermes_home = home / ".hermes"
    if _looks_like_home(hermes_home):
        return hermes_home
    # Neither exists yet: prefer the NOVA location for a fresh deployment, which keeps a
    # new install branded without disturbing any Hermes install that appears later.
    return nova_home


def _looks_like_home(path: Path) -> bool:
    return path.is_dir() and any((path / marker).exists() for marker in HOME_MARKERS)


@dataclass(frozen=True)
class HermesPaths:
    """Paths inside one runtime home that the adapter is allowed to touch.

    Anything not named here is out of bounds — in particular credentials, session
    history, memories and databases, which belong to the customer and to the runtime.
    """

    home: Path

    @classmethod
    def resolve(cls, env: Optional[Mapping[str, str]] = None) -> "HermesPaths":
        return cls(home=resolve_home(env))

    @property
    def profiles_dir(self) -> Path:
        """Where per-agent runtime configuration lives."""
        return self.home / "profiles"

    @property
    def skins_dir(self) -> Path:
        """User skins. The runtime loads these ahead of its built-ins."""
        return self.home / "skins"

    def profile_dir(self, agent_id: str) -> Path:
        return self.profiles_dir / agent_id

    def config_path(self, agent_id: str) -> Path:
        return self.profile_dir(agent_id) / "config.yaml"

    def persona_path(self, agent_id: str) -> Path:
        return self.profile_dir(agent_id) / "SOUL.md"

    def provenance_path(self, agent_id: str) -> Path:
        """NOVA's ownership marker — see :mod:`nova.runtime.hermes.materialize`."""
        return self.profile_dir(agent_id) / "nova-agent.json"

    def policy_path(self, agent_id: str) -> Path:
        """The agent's compiled policy, read by the enforcement plugin."""
        return self.profile_dir(agent_id) / "nova-policy.json"

    def policy_plugin_dir(self, agent_id: str) -> Path:
        """The enforcement plugin.

        A worker runs with its profile as the runtime home, and plugin discovery scans
        ``<home>/plugins``. Installing here therefore scopes the plugin to exactly one
        agent, with no shared state between agents on the same host.
        """
        return self.profile_dir(agent_id) / "plugins" / "nova-policy"

    def knowledge_plugin_dir(self, agent_id: str) -> Path:
        """The knowledge tool, scoped to one agent by the same mechanism as the policy plugin."""
        return self.profile_dir(agent_id) / "plugins" / "nova-knowledge"

    def knowledge_config_path(self, agent_id: str) -> Path:
        """Which corpora this agent may search, read by the knowledge plugin."""
        return self.profile_dir(agent_id) / "nova-knowledge.json"

    @property
    def knowledge_index(self) -> Path:
        """The corpus index, shared by every agent in this deployment.

        One index rather than one per agent, because the documents are the same documents
        and chunking them once per agent would multiply both the ingest cost and the number
        of places a deleted document has to be removed from. Agents are separated by the
        scope filter applied at query time, which is enforced in SQL and cannot be widened
        from the model's side.
        """
        return self.home / "nova-knowledge.db"
