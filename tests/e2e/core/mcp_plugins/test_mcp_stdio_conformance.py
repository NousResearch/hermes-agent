"""MCP conformance over stdio against a REAL ``mcp`` 2.x server (``mcp_fixture_server.py``).

One ``hermes chat -q`` turn per scenario group; the fake model calls the MCP tools and every
assertion reads either what the MCP server RECEIVED on its stdin (teed to a JSONL log) or what the
next provider request carried back to the model (the tool result the model saw):

* trust tiers: on a ``trust: untrusted`` server a tool annotated ``readOnlyHint: true`` runs without
  approval, while a ``destructiveHint`` tool is refused BEFORE the RPC (the server never sees it);
  a default-trust server runs both (#121042 / #120483 for the read-only half);
* argument shape: a tool with no required params receives an ``arguments`` JSON object — ``{}`` —
  for every way a model spells "no arguments", and an optional object param keeps its ``{}``
  (#120269), on the direct path and through the Tool Search ``tool_call`` bridge;
* image results: a cacheable PNG reaches the model as a ``MEDIA:`` file that exists; formats the
  cache cannot store (SVG, AVIF, TIFF, HEIC, malformed base64) must degrade visibly, never vanish
  (#120227).

``tests/e2e/core/parity/test_mcp_lifecycle.py`` owns process lifecycle (death mid-call, reaping);
this file owns what crosses the wire.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import pytest

from tests.e2e.core.mcp_plugins._helpers import (
    FINAL,
    build_home,
    calls_received,
    provider,
    run_chat_q,
    script,
    stdio_server,
    tool_name,
    tool_results,
)
from tests.e2e.core.mcp_plugins._plugin_helpers import reap_tagged

pytestmark = [
    pytest.mark.skipif(not sys.platform.startswith("linux"), reason="orphan sweep uses /proc"),
    pytest.mark.live_system_guard_bypass,  # reap_tagged signals only this run's tagged tree
]

SERVER = "e2e"
UNSUPPORTED_IMAGES = ("svg", "avif", "tiff", "heic", "badb64")

# Open bugs on origin/main. Strict: the test FAILS the moment the bug is fixed, forcing the entry out.
KNOWN: dict[str, str] = {
    "test_untrusted_server_runs_read_only_tool_without_approval":
        "#121042 readOnlyHint read by camelCase attribute under mcp 2.x; every tool needs approval",
    **{f"test_uncacheable_image_is_reported_to_the_model[{fmt}]":
       "#120227 an MCP image the cache cannot store vanishes from the tool result" for fmt in UNSUPPORTED_IMAGES},
}


@pytest.fixture(autouse=True)
def _known(request: pytest.FixtureRequest) -> None:
    reason = KNOWN.get(request.node.name)
    if reason:
        request.applymarker(pytest.mark.xfail(strict=True, raises=AssertionError, reason=reason))


def _run(root: Path, calls: list[tuple[str, dict[str, Any] | str]], *, extra: dict | None = None,
         **server_cfg: Any) -> dict[str, Any]:
    """One ``chat -q`` turn issuing ``calls`` (bare MCP tool names); returns the observations."""
    log = root / "inbound.jsonl"
    with provider(script(*[(tool_name(SERVER, n), a) for n, a in calls])) as srv:
        eh = build_home(root, srv.base_url, extra=extra)
        cfg = {**stdio_server(SERVER, log, eh.tag, MCPE2E_CANARY=f"CANARY-{root.name}"), **server_cfg}
        eh.update_config(lambda c: c["mcp_servers"].__setitem__(SERVER, cfg))
        try:
            proc = run_chat_q(eh, "Use the e2e tools, then report.")
        finally:
            reap_tagged(eh)
        detail = f"exit {proc.returncode}\nstdout: {proc.stdout[-1500:]}\nstderr: {proc.stderr[-1500:]}"
        assert proc.returncode == 0 and FINAL in proc.stdout, f"turn did not complete:\n{detail}"
        offered = {(t.get("function") or {}).get("name") for t in srv.main_requests()[0].get("tools") or []}
        return {"log": log, "results": tool_results(srv), "canary": f"CANARY-{root.name}", "offered": offered,
                "stdout": proc.stdout, "detail": detail, "home": eh}


def _payload(result: str) -> dict[str, Any]:
    """The JSON object inside the untrusted-tool-result wrapper."""
    match = re.search(r"^\{.*\}$", result, re.M | re.S)
    assert match, f"no JSON payload in tool result: {result!r}"
    return json.loads(match.group(0))


# Trust tiers ---------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def untrusted(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    return _run(tmp_path_factory.mktemp("untrusted"), [("ro_probe", {"nonce": "r"}), ("rw_probe", {"nonce": "w"})],
                trust="untrusted")


def test_untrusted_server_refuses_destructive_tool_before_the_rpc(untrusted: dict[str, Any]) -> None:
    assert not calls_received(untrusted["log"], "rw_probe"), (
        "a destructiveHint tool on a trust: untrusted server reached the server without approval")
    rw = _payload(untrusted["results"][1])
    assert "error" in rw and untrusted["canary"] not in json.dumps(rw), rw


def test_untrusted_server_runs_read_only_tool_without_approval(untrusted: dict[str, Any]) -> None:
    received = calls_received(untrusted["log"], "ro_probe")
    ro = _payload(untrusted["results"][0])
    assert received and f"RO:{untrusted['canary']}:r" in json.dumps(ro), (
        f"readOnlyHint=true tool was gated on an untrusted server: server got {received}, model got {ro}")


def test_default_trust_server_runs_destructive_tool(tmp_path: Path) -> None:
    obs = _run(tmp_path, [("rw_probe", {"nonce": "w"})])
    assert calls_received(obs["log"], "rw_probe"), f"default-trust server never received the call\n{obs['detail']}"
    assert f"RW:{obs['canary']}:w" in obs["results"][0], obs["results"]


# Argument shape ---------------------------------------------------------------------------------------

# How a model says "no arguments" on the chat-completions wire, and what the server must receive.
NO_ARG_SPELLINGS: list[tuple[str, dict[str, Any] | str, dict[str, Any]]] = [
    ("noargs_probe", {}, {}),
    ("noargs_probe", "", {}),
    ("noargs_probe", "{}", {}),
    ("optional_obj_probe", {"parameters": {}}, {"parameters": {}}),
    ("optional_obj_probe", {}, {}),
]


@pytest.mark.parametrize("bridge", ["direct", "tool_call bridge"])
def test_no_required_param_tools_receive_an_arguments_object(tmp_path: Path, bridge: str) -> None:
    search = "on" if bridge == "tool_call bridge" else "off"
    calls = [(name, args) for name, args, _ in NO_ARG_SPELLINGS]
    obs = _run(tmp_path, calls, extra={"tools": {"tool_search": {"enabled": search}}})
    direct = tool_name(SERVER, "noargs_probe") in obs["offered"]
    assert direct == (bridge == "direct"), f"{bridge}: tools offered {sorted(obs['offered'])}"
    received = [p for name in dict.fromkeys(n for n, _, _ in NO_ARG_SPELLINGS) for p in calls_received(obs["log"], name)]
    got = sorted((p["name"], json.dumps(p.get("arguments"), sort_keys=True)) for p in received)
    want = sorted((name, json.dumps(expect, sort_keys=True)) for name, _, expect in NO_ARG_SPELLINGS)
    assert got == want, f"server received {got}, want {want}"
    for p in received:
        meta = p.get("_meta", {})
        assert isinstance(meta, dict), f"_meta must be an object when present: {p}"
    assert all(obs["canary"] in r for r in obs["results"]), obs["results"]


# Image results -----------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def images(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    fmts = ("png", *UNSUPPORTED_IMAGES)
    obs = _run(tmp_path_factory.mktemp("images"), [("image_probe", {"fmt": f}) for f in fmts])
    assert len(obs["results"]) == len(fmts), obs["results"]
    obs["by_fmt"] = dict(zip(fmts, (_payload(r) for r in obs["results"])))
    return obs


def test_cacheable_image_reaches_the_model_as_a_media_file(images: dict[str, Any]) -> None:
    text = str(images["by_fmt"]["png"].get("result", ""))
    assert f"IMG-STATUS:{images['canary']}:png" in text, text
    paths = re.findall(r"MEDIA:(\S+)", text)
    assert paths and all(Path(p).is_file() and Path(p).stat().st_size > 0 for p in paths), (
        f"PNG image block did not reach the model as a readable MEDIA file: {text!r}")


@pytest.mark.parametrize("fmt", UNSUPPORTED_IMAGES)
def test_uncacheable_image_is_reported_to_the_model(images: dict[str, Any], fmt: str) -> None:
    payload = images["by_fmt"][fmt]
    text = json.dumps(payload)
    status = f"IMG-STATUS:{images['canary']}:{fmt}"
    assert status in text, f"the text block next to the image was lost: {payload}"
    rest = text.replace(status, "")
    assert "image" in rest.lower(), (
        f"{fmt} image block vanished: the model got no sign the tool returned an image: {payload}")
