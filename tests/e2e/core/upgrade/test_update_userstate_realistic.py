"""`hermes update` on a realistic user home: everything the user owns survives, sessions resume.

One real install (HEAD's ``scripts/install.sh`` into an empty sandbox HOME, git redirected to a
local bare origin), then a home built the way users build it, mostly through the real CLI:

* three profiles (default + two made with ``hermes profile create``), each with a hand-edited
  config.yaml carrying comments and a key HEAD does not know, its own .env, SOUL.md and
  MEMORY.md, and sessions in its own state.db from real one-shot turns. The default profile's
  config sits at the oldest ``_config_version`` HEAD still migrates and ``work``'s one version
  behind, so the update's config migration (active profile + siblings) runs; ``research`` is
  current and must come through byte-for-byte;
* a custom skill, a user plugin under ``~/.hermes/plugins/``, a cron job made by ``hermes cron``;
* two broken profiles next to them (a dangling symlink, a profile whose config.yaml is not YAML);
* local files the user dropped inside the checkout (an untracked notes file and an extension dir).

Upstream then publishes a new commit and the user runs ``hermes update --yes``. A second leg
publishes a commit that touches a file the user edited in the checkout (the autostash cannot be
restored cleanly) while an untracked extension dir sits in the tree (#120179).

Assertions read the persisted state directly (bytes, sqlite rows, the cron store) and the next
wire request a resumed session sends to the fake provider.
"""

from __future__ import annotations

import json
import os
import shutil

import pytest
import hermes_yaml as yaml

from tests.e2e.core.upgrade import _helpers as H
from tests.e2e.core.upgrade import _install_helpers as I
from tests.fakes.fake_llm_provider import FakeLLMServer

pytestmark = [
    pytest.mark.platforms("linux"),
    # The real updater runs against a throwaway install inside the sandbox, never this checkout.
    pytest.mark.live_system_guard_bypass,
    pytest.mark.skipif(H.sandbox_required_reason() is not None, reason=str(H.sandbox_required_reason())),
    pytest.mark.skipif(shutil.which("git") is None, reason="git required"),
    pytest.mark.skipif(I.real_uv() is None, reason="uv required"),
]

UPDATE_TIMEOUT = 1500
PROFILES = ("default", "work", "research")
_VERSIONS_PY = ("from hermes_cli.config_defaults import DEFAULT_CONFIG as D; "
                "from hermes_cli.config_migrations import SUPPORT_FLOOR_VERSION as F; "
                "print(D['_config_version'], F)")


def _profile_home(sb: I.Sandbox, name: str):
    return sb.hermes_home if name == "default" else sb.hermes_home / "profiles" / name


def _pargs(name: str) -> list[str]:
    return [] if name == "default" else ["-p", name]


def _ok(cp):
    assert cp.returncode == 0 and I.TRACEBACK not in cp.stdout + cp.stderr, I.describe(cp)
    return cp


def _config_versions(sb: I.Sandbox) -> tuple[int, int]:
    """(latest, support floor) from the installed PM-selected interpreter."""
    cp = _ok(sb.run([sb.python, "-c", _VERSIONS_PY]))
    latest, floor = cp.stdout.strip().splitlines()[-1].split()
    return int(latest), int(floor)


def _leaves(node, path: tuple = ()):
    if isinstance(node, dict) and node:
        for k, v in node.items():
            yield from _leaves(v, (*path, k))
    else:
        yield path, node


def _at(node, path: tuple):
    for k in path:
        if not isinstance(node, dict) or k not in node:
            return "<missing>"
        node = node[k]
    return node


def _snapshot(sb: I.Sandbox) -> dict:
    snap: dict = {}
    for name in PROFILES:
        home = _profile_home(sb, name)
        snap[name] = {
            "config.yaml": (home / "config.yaml").read_bytes(),
            ".env": (home / ".env").read_bytes(),
            "SOUL.md": (home / "SOUL.md").read_bytes(),
            "MEMORY.md": (home / "memories" / "MEMORY.md").read_bytes(),
            "db": I.db_state(home / "state.db"),
        }
    hh = sb.hermes_home
    snap["custom_skill"] = I.tree_digest(hh / "skills" / "my-own-skill")
    snap["plugin"] = I.tree_digest(hh / "plugins" / "my-plugin")
    snap["cron"] = sorted((j.get("id"), j.get("name"), j.get("prompt"), j.get("schedule_display") or str(j.get("schedule")))
                          for j in I.cron_jobs(hh))
    snap["badyaml"] = (hh / "profiles" / "badyaml" / "config.yaml").read_bytes()
    snap["ghost"] = os.readlink(hh / "profiles" / "ghost")
    return snap


def _seed_home(sb: I.Sandbox, provider: FakeLLMServer) -> tuple[dict[str, str], dict[str, int]]:
    """Build the realistic home; returns ({profile: marker of its pre-update session},
    {profile: the _config_version its config.yaml was written at})."""
    latest, floor = _config_versions(sb)
    assert floor < latest, f"harness: no older config version to migrate from (floor {floor}, latest {latest})"
    versions = {"default": floor, "work": max(floor, latest - 1), "research": latest}
    for name in PROFILES[1:]:
        _ok(sb.cli("profile", "create", name, "--no-alias"))
    markers = {}
    for name in PROFILES:
        home = _profile_home(sb, name)
        # The per-profile marker lives in a key of the user's own: known keys the ladder resets
        # on purpose (display.personality, stale defaults) are not user state it must keep.
        extra = f"my_profile_note: \"{name}'s own value\"  # {name}'s choice\n"
        (home / "config.yaml").write_text(I.provider_config(provider.base_url, versions[name], extra), encoding="utf-8")
        (home / ".env").write_text(f"OPENAI_API_KEY={I.FAKE_KEY}-{name}\n# {name} keys\n", encoding="utf-8")
        (home / "SOUL.md").write_text(f"You are the {name} agent. Keep it short.\n", encoding="utf-8")
        (home / "memories").mkdir(exist_ok=True)
        (home / "memories" / "MEMORY.md").write_text(f"- {name}: user prefers tabs\n", encoding="utf-8")
        markers[name] = f"pre-update session in the {name} profile"
        _ok(sb.cli(*_pargs(name), "-z", markers[name]))
        # A normal CLI turn migrates old configs on launch. Model the user's older
        # hand-edited config as the state immediately before the update, not before
        # the turns that seed the sessions.
        (home / "config.yaml").write_text(I.provider_config(provider.base_url, versions[name], extra), encoding="utf-8")
    hh = sb.hermes_home
    skill = hh / "skills" / "my-own-skill"
    skill.mkdir(parents=True, exist_ok=True)
    (skill / "SKILL.md").write_text("---\nname: my-own-skill\ndescription: mine\n---\nDo my thing.\n", encoding="utf-8")
    (skill / "scripts").mkdir(exist_ok=True)
    (skill / "scripts" / "helper.py").write_text("print('helper')\n", encoding="utf-8")
    plugin = hh / "plugins" / "my-plugin"
    plugin.mkdir(parents=True, exist_ok=True)
    (plugin / "plugin.yaml").write_text("name: my-plugin\nversion: 0.0.1\ndescription: user plugin\n", encoding="utf-8")
    (plugin / "__init__.py").write_text("def register(ctx):\n    pass\n", encoding="utf-8")
    _ok(sb.cli("cron", "create", "--name", "nightly", "0 3 * * *", "summarize the day"))
    os.symlink(str(sb.root / "moved-away-home"), hh / "profiles" / "ghost")
    (hh / "profiles" / "badyaml").mkdir()
    (hh / "profiles" / "badyaml" / "config.yaml").write_text("model: [unclosed\n  : :\n", encoding="utf-8")
    return markers, versions


@pytest.fixture(scope="module")
def provider():
    with FakeLLMServer(default_text="fake reply for the user-state suite") as srv:
        yield srv


@pytest.fixture(scope="module")
def world(tmp_path_factory, provider):
    root = tmp_path_factory.mktemp("userstate")
    origin = I.make_origin(root, I.head_sha())
    sb = I.new_sandbox(root / "sb", origin)
    _ok(I.run_installer(sb))
    markers, versions = _seed_home(sb, provider)
    (sb.checkout / "my_local_notes.txt").write_text("untracked notes in the checkout\n", encoding="utf-8")
    snap = _snapshot(sb)
    target = I.publish_commit(origin, root, "release: e2e bump 1", {"docs/e2e-update-marker.txt": "release 1\n"})
    update = sb.cli("update", "--yes", "--branch", "main", timeout=UPDATE_TIMEOUT)
    return {"sb": sb, "origin": origin, "root": root, "markers": markers, "versions": versions,
            "snap": snap, "target": target, "update": update}


def test_update_on_a_realistic_home_exits_clean_at_the_new_commit(world):
    sb, up = world["sb"], world["update"]
    assert up.returncode == 0, "hermes update failed on a realistic home:\n" + I.describe(up)
    assert I.TRACEBACK not in up.stdout + up.stderr, I.describe(up)
    assert I.git("rev-parse", "HEAD", cwd=sb.checkout) == world["target"], "update exited 0 but HEAD is not the target"
    assert not (sb.checkout / ".git" / "index.lock").exists()
    assert (sb.checkout / "my_local_notes.txt").read_text(encoding="utf-8") == "untracked notes in the checkout\n"


def test_user_state_survives_byte_for_byte(world):
    sb, before = world["sb"], world["snap"]
    assert world["update"].returncode == 0, I.describe(world["update"])
    after = _snapshot(sb)
    latest = max(world["versions"].values())
    for name in PROFILES:
        # An outdated config.yaml is migrated on purpose (see the migration test below).
        keys = ("config.yaml", ".env", "SOUL.md", "MEMORY.md") if world["versions"][name] == latest else (
            ".env", "SOUL.md", "MEMORY.md")
        for key in keys:
            assert after[name][key] == before[name][key], (
                f"{name}/{key} changed by the update\n--- before ---\n{before[name][key].decode()}"
                f"\n--- after ---\n{after[name][key].decode()}")
        assert after[name]["db"]["integrity"] == [("ok",)], f"{name} state.db: {after[name]['db']['integrity']}"
        assert after[name]["db"] == before[name]["db"], f"{name} state.db sessions/messages changed by the update"
    for key in ("custom_skill", "plugin", "cron", "badyaml", "ghost"):
        assert after[key] == before[key], f"{key} changed by the update: {before[key]!r} -> {after[key]!r}"
    assert before["cron"], "harness: no cron job was seeded"


def test_outdated_configs_migrate_to_the_current_version_keeping_user_values(world):
    """Active profile (``_check_and_apply_config_migration``) and sibling
    (``_migrate_sibling_profile_configs``) alike: the version reaches the running code's, and
    every value the user wrote (known keys, unknown keys, nested and unicode) is still there."""
    sb, up = world["sb"], world["update"]
    assert up.returncode == 0, I.describe(up)
    latest, _floor = _config_versions(sb)
    outdated = [n for n in PROFILES if world["versions"][n] < latest]
    assert {"default", "work"} <= set(outdated), f"harness: {world['versions']} vs latest {latest}"
    for name in outdated:
        before = yaml.safe_load(world["snap"][name]["config.yaml"])
        text = (_profile_home(sb, name) / "config.yaml").read_text(encoding="utf-8")
        after = yaml.safe_load(text)
        assert before["_config_version"] == world["versions"][name], (
            f"harness: {name}'s config was migrated before the update ran")
        assert isinstance(after, dict) and after.get("_config_version") == latest, (
            f"{name}: config.yaml left at v{before['_config_version']} (code is v{latest}) by the update\n"
            f"{text}\n{I.describe(up)}")
        lost = {".".join(p): (v, _at(after, p)) for p, v in _leaves(before)
                if p != ("_config_version",) and _at(after, p) != v}
        assert not lost, f"{name}: config migration lost or changed user values {{key: (before, after)}}: {lost}\n{text}"


def test_every_profile_session_resumes_after_update(world, provider):
    sb = world["sb"]
    assert world["update"].returncode == 0, I.describe(world["update"])
    for name in PROFILES:
        sid = world["snap"][name]["db"]["sessions"][0]
        n = len(provider.main_requests())
        follow = f"follow-up after the update in {name}"
        cp = _ok(sb.cli(*_pargs(name), "-z", follow, "--resume", sid))
        assert provider.default_text in cp.stdout, I.describe(cp)
        new = provider.main_requests()[n:]
        assert len(new) == 1, f"{name}: resumed turn sent {len(new)} provider requests"
        wire = json.dumps(new[0]["messages"])
        assert world["markers"][name] in wire and follow in wire, (
            f"{name}: resumed session did not carry its pre-update history to the provider")
        others = [world["markers"][o] for o in PROFILES if o != name]
        assert not any(m in wire for m in others), f"{name}: another profile's history leaked into the resumed turn"
        db = I.db_state(_profile_home(sb, name) / "state.db")
        assert db["integrity"] == [("ok",)] and db["sessions"] == world["snap"][name]["db"]["sessions"], (
            f"{name}: resume forked a new session instead of continuing {sid}: {db['sessions']}")


def test_broken_profiles_warn_but_do_not_fail_the_update(world):
    sb, up = world["sb"], world["update"]
    out = up.stdout + up.stderr
    assert up.returncode == 0, "a broken profile under ~/.hermes/profiles/ failed the whole update:\n" + I.describe(up)
    bad = str(sb.hermes_home / "profiles" / "badyaml" / "config.yaml")
    assert bad in out, "the update did not warn about the profile whose config.yaml is broken:\n" + I.describe(up)
    assert (sb.hermes_home / "profiles" / "badyaml" / "config.yaml").read_bytes() == world["snap"]["badyaml"], (
        "the update rewrote the user's broken config instead of leaving it for them to fix")


@pytest.fixture(scope="module")
def conflicting_leg(world):
    """Second release touches a file the user edited in the checkout; an untracked extension dir
    (installed into the tree by a third-party tool) sits next to it."""
    sb = world["sb"]
    assert world["update"].returncode == 0, I.describe(world["update"])
    ext = sb.checkout / "plugins" / "my-inplace-ext"
    ext.mkdir(parents=True, exist_ok=True)
    (ext / "__init__.py").write_text("NAME = 'my-inplace-ext'\n", encoding="utf-8")
    (ext / "server.py").write_text("PORT = 8754\n", encoding="utf-8")
    (sb.checkout / "docs" / "e2e-update-marker.txt").write_text("release 1\nuser's local hook line\n", encoding="utf-8")
    before = I.tree_digest(ext)
    target = I.publish_commit(world["origin"], world["root"], "release: e2e bump 2",
                              {"docs/e2e-update-marker.txt": "release 2 rewrote this file\n"})
    update = sb.cli("update", "--yes", "--branch", "main", timeout=UPDATE_TIMEOUT)
    return {"ext": ext, "before": before, "target": target, "update": update}


def test_conflicting_update_keeps_the_users_edit_recoverable(conflicting_leg, world):
    sb, up = world["sb"], conflicting_leg["update"]
    assert I.TRACEBACK not in up.stdout + up.stderr, I.describe(up)
    assert I.git("rev-parse", "HEAD", cwd=sb.checkout) == conflicting_leg["target"], I.describe(up)
    edit = "user's local hook line"
    in_tree = edit in (sb.checkout / "docs" / "e2e-update-marker.txt").read_text(encoding="utf-8")
    stashes = I.git("stash", "list", "--format=%H", cwd=sb.checkout, check=False).split()
    parked = any(edit in I.git("stash", "show", "-p", s, cwd=sb.checkout, check=False) for s in stashes)
    assert in_tree or parked, "the user's tracked edit was lost by the update"


def test_conflicting_update_leaves_untracked_extension_in_the_tree(conflicting_leg):
    up = conflicting_leg["update"]
    assert I.TRACEBACK not in up.stdout + up.stderr, I.describe(up)
    after = I.tree_digest(conflicting_leg["ext"]) if conflicting_leg["ext"].exists() else {}
    assert after == conflicting_leg["before"], (
        f"untracked extension files left the working tree: {sorted(conflicting_leg['before'])} -> {sorted(after)}\n"
        + I.describe(up))
