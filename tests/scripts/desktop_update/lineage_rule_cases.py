"""The launcher-lineage rule shared by marker.sh (marker_launcher_rule) and
marker-claim.ps1 (Test-MarkerLauncherRule): one table, every consumer runs it.

An OLD packaged Desktop overwrites the marker with a v1 bridge naming the launcher it
spawned. Adopted iff the bridge is v1, does not name the Desktop, and
  named pid alive: it is our parent AND (its parent is the Desktop OR env == line 2);
  named pid gone:  env == line 2.
"""
from __future__ import annotations

FACTS = ("v1", "names_desktop", "named_alive", "named_is_our_parent", "named_parent_is_desktop", "env_started_matches")


def _case(case_id: str, expect: bool, **facts: bool) -> dict:
    assert set(facts) == set(FACTS), case_id
    return {"id": case_id, "facts": facts, "expect": expect}


RULE_CASES = [
    # the old Desktop's own shapes
    _case("alive_launcher_child_of_desktop", True, v1=True, names_desktop=False, named_alive=True,
          named_is_our_parent=True, named_parent_is_desktop=True, env_started_matches=False),
    _case("gone_launcher_env_matches", True, v1=True, names_desktop=False, named_alive=False,
          named_is_our_parent=False, named_parent_is_desktop=False, env_started_matches=True),
    # D11 (round 5): bash adopted, PowerShell refused -- the Desktop already quit, so its live
    # launcher was re-parented, but it carries the hand-off's own startedAt.
    _case("alive_launcher_reparented_env_matches", True, v1=True, names_desktop=False, named_alive=True,
          named_is_our_parent=True, named_parent_is_desktop=False, env_started_matches=True),
    # PowerShell also required the gone launcher to be our recorded parent; POSIX cannot know it.
    _case("gone_named_pid_not_our_recorded_parent_env_matches", True, v1=True, names_desktop=False,
          named_alive=False, named_is_our_parent=False, named_parent_is_desktop=False, env_started_matches=True),
    _case("alive_launcher_reparented_env_differs", False, v1=True, names_desktop=False, named_alive=True,
          named_is_our_parent=True, named_parent_is_desktop=False, env_started_matches=False),
    _case("gone_launcher_env_differs", False, v1=True, names_desktop=False, named_alive=False,
          named_is_our_parent=True, named_parent_is_desktop=True, env_started_matches=False),
    # an unrelated live process is never adopted, whatever the env says
    _case("alive_not_our_parent_env_matches", False, v1=True, names_desktop=False, named_alive=True,
          named_is_our_parent=False, named_parent_is_desktop=True, env_started_matches=True),
    _case("alive_not_our_parent_child_of_desktop", False, v1=True, names_desktop=False, named_alive=True,
          named_is_our_parent=False, named_parent_is_desktop=True, env_started_matches=False),
    # only a v1 bridge: a v2 claim (ct line) or one with a delegate is somebody's real claim.
    # (PowerShell used to accept these when they named its parent.)
    _case("v2_claim_naming_our_parent", False, v1=False, names_desktop=False, named_alive=True,
          named_is_our_parent=True, named_parent_is_desktop=True, env_started_matches=True),
    _case("v2_claim_gone_env_matches", False, v1=False, names_desktop=False, named_alive=False,
          named_is_our_parent=False, named_parent_is_desktop=False, env_started_matches=True),
    # the Desktop's own bridge is not a launcher bridge (it has its own adoption path)
    _case("names_the_desktop", False, v1=True, names_desktop=True, named_alive=True,
          named_is_our_parent=True, named_parent_is_desktop=False, env_started_matches=True),
]

# HERMES_UPDATE_STARTED_AT vs line 2 (its digits as parsed): plain ASCII digits of the same value.
# (PowerShell used to Trim() and Int64.TryParse the env value: spaces and a sign matched.)
ENV_CASES = [
    {"id": "equal", "env": "1791079348", "line2": "1791079348", "expect": True},
    {"id": "leading_zeros_env", "env": "01791079348", "line2": "1791079348", "expect": True},
    {"id": "leading_zeros_line2", "env": "1791079348", "line2": "0001791079348", "expect": True},
    {"id": "zero", "env": "00", "line2": "0", "expect": True},
    {"id": "different", "env": "1791079349", "line2": "1791079348", "expect": False},
    {"id": "padded_env", "env": " 1791079348", "line2": "1791079348", "expect": False},
    {"id": "trailing_space_env", "env": "1791079348 ", "line2": "1791079348", "expect": False},
    {"id": "signed_env", "env": "+1791079348", "line2": "1791079348", "expect": False},
    {"id": "empty_env", "env": "", "line2": "1791079348", "expect": False},
    {"id": "garbage_env", "env": "1791079348x", "line2": "1791079348", "expect": False},
]
