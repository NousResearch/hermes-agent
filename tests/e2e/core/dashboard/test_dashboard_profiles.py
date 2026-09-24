"""Profile switching through the dashboard API never crosses profiles, and a Config-page save
round-trips the user's file.

Users hit the first as "I switched the dashboard to profile B and saw A's sessions / saved B's key
into A's .env" (one dashboard process serves every profile; the SPA's switcher appends
``?profile=<name>`` to each call, ``web/src/lib/api.ts``). Each profile owns random canaries (model
id, provider key, a config marker, session ids); the invariant after every step of an A -> B -> A ->
default -> B walk, and under concurrent interleaved requests, is that a response for profile X
carries only X's canaries and a write addressed to X changes only X's files.

The second is the Config page's save: it GETs the defaulted record, edits one field and PUTs the
whole record back. The file on disk must change in exactly that field: unknown keys (a newer
version's settings, hand-added blocks) and explicit ``null``s survive, and defaults are not
materialised into the user's sparse YAML.
"""

from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest
import yaml

from . import _helpers as H

pytestmark = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="POSIX dashboard process")

NAMES = ("default", "alpha", "beta")
WALK = ("alpha", "beta", "alpha", "default", "beta")
SEEDS = {"default": 3, "alpha": 5, "beta": 7}  # distinct counts: a wrong-profile list is visible


@pytest.fixture(scope="module")
def fleet(tmp_path_factory: pytest.TempPathFactory):
    sb = H.make_sandbox(tmp_path_factory.mktemp("dash-profiles"), NAMES)
    for name, p in sb.profiles.items():
        H.seed_sessions(sb, p, f"{name}-{p.tag}", SEEDS[name])
    d = H.Dashboard(sb, sb.root / "dashboard.log")
    try:
        yield sb, d
    finally:
        d.close()
        sb.stop()


def _files(sb: H.Sandbox) -> dict[str, bytes]:
    return {f"{p.name}/{n}": (p.home / n).read_bytes() for p in sb.profiles.values() for n in ("config.yaml", ".env")}


def _foreign(sb: H.Sandbox, owner: str, blob: str) -> list[str]:
    return [f"{u.name}'s {kind}" for u in sb.profiles.values() if u.name != owner
            for kind, value in {**u.canaries(), "session prefix": f"{u.name}-{u.tag}-"}.items() if value in blob]


def _check_reads(sb: H.Sandbox, d: H.Dashboard, name: str) -> list[str]:
    p = sb.profiles[name]
    out: list[str] = []
    listing = d.ok("GET", "/api/sessions", name, params={"limit": 100})
    ids = sorted(s["id"] for s in listing["sessions"])
    want = sorted(f"{name}-{p.tag}-{i:04d}" for i in range(SEEDS[name]))
    if ids != want or listing["total"] != SEEDS[name]:
        out.append(f"{name}: session list {ids} (total {listing['total']}), expected {want}")
    if any(s.get("profile") != name for s in listing["sessions"]):
        out.append(f"{name}: rows stamped {sorted({s.get('profile') for s in listing['sessions']})}")
    for other in sb.profiles.values():
        if other.name != name:
            r = d.request("GET", f"/api/sessions/{other.name}-{other.tag}-0000", name)
            if r.status_code != 404:
                out.append(f"{name}: detail of {other.name}'s session -> {r.status_code}")
    cfg = d.ok("GET", "/api/config", name)
    model = cfg.get("model")  # the web shape flattens model to its id string
    model = model.get("default") if isinstance(model, dict) else model
    if cfg.get("dash_canary", {}).get("marker") != p.marker or model != p.model:
        out.append(f"{name}: /api/config served marker={cfg.get('dash_canary')} model={model}")
    env = d.ok("GET", "/api/env", name)
    if not env.get(H.PROVIDER_KEY_ENV, {}).get("is_set"):
        out.append(f"{name}: /api/env lost its own {H.PROVIDER_KEY_ENV}")
    blob = json.dumps(listing) + json.dumps(cfg) + json.dumps(env)
    if p.provider_key in json.dumps(env):
        out.append(f"{name}: /api/env returned the raw provider key")
    return out + [f"{name} read carries {leak}" for leak in _foreign(sb, name, blob)]


def test_profile_walk_reads_and_writes_only_the_selected_profile(fleet) -> None:
    sb, d = fleet
    problems: list[str] = []
    for step, name in enumerate(WALK):
        problems += _check_reads(sb, d, name)
        p = sb.profiles[name]
        before = _files(sb)
        d.ok("PUT", "/api/config", name, json={"config": {"dash_canary": {"marker": p.marker, "step": f"{name}-{step}"}}})
        d.ok("PUT", "/api/env", name, json={"key": "DASH_EXTRA", "value": f"extra-{name}-{p.tag}-{step}"})
        changed = {k for k, v in _files(sb).items() if before[k] != v}
        if changed != {f"{name}/config.yaml", f"{name}/.env"}:
            problems.append(f"step {step} ({name}): writes changed {sorted(changed)}")
        if p.config().get("dash_canary", {}).get("step") != f"{name}-{step}":
            problems.append(f"step {step} ({name}): config write did not land in {name}/config.yaml")
        if f"DASH_EXTRA=extra-{name}-{p.tag}-{step}" not in (p.home / ".env").read_text():
            problems.append(f"step {step} ({name}): env write did not land in {name}/.env")
    assert not problems, "profile crossing during the A->B->A walk:\n  " + "\n  ".join(problems)


def test_interleaved_profile_requests_never_cross(fleet) -> None:
    """Two tabs on different profiles, requests racing: each response is the asked profile's."""
    sb, d = fleet

    def hammer(name: str) -> list[str]:
        p = sb.profiles[name]
        out = []
        for _ in range(25):
            cfg = d.ok("GET", "/api/config", name)
            if cfg.get("dash_canary", {}).get("marker") != p.marker:
                out.append(f"{name}: got marker {cfg.get('dash_canary', {}).get('marker')}")
            total = d.ok("GET", "/api/sessions", name, params={"limit": 1})["total"]
            if total != SEEDS[name]:
                out.append(f"{name}: session total {total} != {SEEDS[name]}")
        return out

    with ThreadPoolExecutor(max_workers=6) as pool:
        problems = [x for f in [pool.submit(hammer, n) for n in NAMES * 2] for x in f.result(timeout=240)]
    assert not problems, "interleaved profile requests crossed:\n  " + "\n  ".join(problems[:30])
    leaks = [f"{p.name}: {path.relative_to(p.home)} carries {leak}"
             for p in sb.profiles.values() for path in p.home.rglob("*")
             if path.is_file() and not (p.name == "default" and path.relative_to(p.home).parts[:1] == ("profiles",))
             for leak in _foreign(sb, p.name, path.read_bytes().decode("utf-8", "replace"))]
    assert not leaks, "profile files carry another profile's canaries:\n  " + "\n  ".join(leaks)


def _flatten(d: Any, prefix: str = "") -> dict[str, Any]:
    if not isinstance(d, dict) or not d:
        return {prefix: d}
    out: dict[str, Any] = {}
    for k, v in d.items():
        out.update(_flatten(v, f"{prefix}.{k}" if prefix else str(k)))
    return out


def test_config_page_save_changes_only_the_edited_field(fleet) -> None:
    sb, d = fleet
    p = sb.profiles["alpha"]
    path: Path = p.home / "config.yaml"
    cfg = p.config()
    cfg["zz_future_block"] = {"nested": [1, {"deep": "x"}], "flag": True}  # a newer version's setting
    cfg["zz_cleared"] = None
    cfg.setdefault("display", {})["zz_future_leaf"] = "keep-me"
    cfg["display"]["skin"] = None  # explicit null on a key that HAS a default
    cfg["timezone"] = None
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    before = _flatten(yaml.safe_load(path.read_text(encoding="utf-8")))

    record = d.ok("GET", "/api/config", "alpha")  # defaulted record, what the Config page renders
    assert record["zz_future_block"] == cfg["zz_future_block"] and "zz_cleared" in record, record.keys()
    record.setdefault("agent", {})["max_turns"] = 77
    d.ok("PUT", "/api/config", "alpha", json={"config": record})

    after = _flatten(yaml.safe_load(path.read_text(encoding="utf-8")))
    changed = {k for k in before.keys() | after.keys() if before.get(k, "<absent>") != after.get(k, "<absent>")}
    assert changed == {"agent.max_turns"}, (
        f"a one-field Config-page save changed {sorted(changed)}:\n"
        + "\n".join(f"  {k}: {before.get(k, '<absent>')!r} -> {after.get(k, '<absent>')!r}" for k in sorted(changed)))
    assert after["agent.max_turns"] == 77
    again = d.ok("GET", "/api/config", "alpha")
    assert again["zz_future_block"] == cfg["zz_future_block"] and again["display"]["skin"] is None
