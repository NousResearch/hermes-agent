"""Tests for visual companion publish behavior."""

import concurrent.futures
import http.cookiejar
import json
import os
import runpy
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "skills/creative/claude-design/scripts/visual_companion.py"
)
SKILL_PATH = SCRIPT_PATH.parents[1] / "SKILL.md"
ALTERNATIVE_CHOICE = '<button data-choice="alternate">Alternate</button>'


def _publish(tmp_path, fragment, *, round_id="layout-directions", session_dir=None):
    round_html = tmp_path / f"{round_id}.html"
    round_html.write_text(fragment, encoding="utf-8")
    session_dir = session_dir or tmp_path / "session"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "publish",
            "--session-dir",
            str(session_dir),
            "--file",
            str(round_html),
            "--round-id",
            round_id,
        ],
        capture_output=True,
        text=True,
    )

    return result, session_dir


def _wait_for_state(session_dir, process):
    state_path = session_dir / "state.json"
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        if state_path.exists():
            return json.loads(state_path.read_text(encoding="utf-8"))
        if process.poll() is not None:
            _, stderr = process.communicate()
            raise AssertionError(f"server exited before readiness: {stderr}")
        time.sleep(0.02)
    process.terminate()
    _, stderr = process.communicate(timeout=2)
    raise AssertionError(f"server did not write state.json: {stderr}")


def _start_server(session_dir):
    process = subprocess.Popen(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "serve",
            "--session-dir",
            str(session_dir),
            "--port",
            "0",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return process, _wait_for_state(session_dir, process)


def _authenticated_opener(state):
    cookie_jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cookie_jar))
    request = urllib.request.Request(
        state["bootstrap_url"],
        data=urllib.parse.urlencode({"key": state["bootstrap_token"]}).encode(),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    with opener.open(request, timeout=2) as response:
        assert response.status == 200
    with opener.open(f'{state["base_url"]}/', timeout=2) as response:
        assert response.status == 200
    return opener, cookie_jar


def _post_choice(opener, base_url, choice_id, *, page_version=1, feedback=None):
    payload = {"choice_id": choice_id, "page_version": page_version}
    if feedback is not None:
        payload["feedback"] = feedback
    request = urllib.request.Request(
        f"{base_url}/__choice",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "X-Visual-Companion": "choice"},
        method="POST",
    )
    with opener.open(request, timeout=2) as response:
        return json.loads(response.read())


def test_publish_creates_versioned_round(tmp_path):
    result, session_dir = _publish(
        tmp_path,
        '<section data-choice="council"><h1>Council layout</h1></section>\n'
        + ALTERNATIVE_CHOICE,
    )
    assert result.returncode == 0, result.stderr

    current_html = session_dir / "current.html"
    assert current_html.exists()
    assert '<section data-choice="council"' in current_html.read_text(encoding="utf-8")

    round_json = session_dir / "round.json"
    data = json.loads(round_json.read_text(encoding="utf-8"))
    assert data["round_id"] == "layout-directions"
    assert isinstance(data["page_version"], int)
    assert data["page_version"] == 1

    stdout_data = json.loads(result.stdout.strip())
    assert stdout_data["round_id"] == "layout-directions"
    assert stdout_data["page_version"] == 1


def test_publish_increments_the_page_version(tmp_path):
    first, session_dir = _publish(
        tmp_path,
        '<button data-choice="council">Council</button>' + ALTERNATIVE_CHOICE,
    )
    assert first.returncode == 0, first.stderr

    second, _ = _publish(
        tmp_path,
        '<button data-choice="focused">Focused</button>' + ALTERNATIVE_CHOICE,
        round_id="focused-refinement",
        session_dir=session_dir,
    )
    assert second.returncode == 0, second.stderr

    metadata = json.loads((session_dir / "round.json").read_text(encoding="utf-8"))
    assert metadata == {"round_id": "focused-refinement", "page_version": 2}
    assert 'data-choice="focused"' in (session_dir / "current.html").read_text(encoding="utf-8")


def test_concurrent_publish_allocates_each_page_version_once(tmp_path):
    session_dir = tmp_path / "session"
    publish_count = 8
    sources = []
    for index in range(publish_count):
        source = tmp_path / f"round-{index}.html"
        source.write_text(
            f'<button data-choice="choice-{index}">Choice {index}</button>'
            f'<button data-choice="alternate-{index}">Alternate {index}</button>',
            encoding="utf-8",
        )
        sources.append(source)

    barrier = threading.Barrier(publish_count)

    def run_publish(index):
        barrier.wait()
        return subprocess.run(
            [
                sys.executable,
                str(SCRIPT_PATH),
                "publish",
                "--session-dir",
                str(session_dir),
                "--file",
                str(sources[index]),
                "--round-id",
                f"round-{index}",
            ],
            capture_output=True,
            text=True,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=publish_count) as executor:
        results = list(executor.map(run_publish, range(publish_count)))

    assert all(result.returncode == 0 for result in results), [result.stderr for result in results]
    versions = sorted(json.loads(result.stdout)["page_version"] for result in results)
    assert versions == list(range(1, publish_count + 1))
    assert json.loads((session_dir / "round.json").read_text())["page_version"] == publish_count


def test_publish_reclaims_a_session_lease_from_an_exited_process(tmp_path):
    first, session_dir = _publish(
        tmp_path,
        '<button data-choice="council">Council</button>' + ALTERNATIVE_CHOICE,
    )
    assert first.returncode == 0, first.stderr

    exited = subprocess.Popen([sys.executable, "-c", "pass"])
    exited.wait(timeout=2)
    (session_dir / ".session.lock").write_text(
        json.dumps({"lease_id": "abandoned", "pid": exited.pid}),
        encoding="utf-8",
    )

    second, _ = _publish(
        tmp_path,
        '<button data-choice="focused">Focused</button>' + ALTERNATIVE_CHOICE,
        round_id="focused-refinement",
        session_dir=session_dir,
    )

    assert second.returncode == 0, second.stderr
    metadata = json.loads((session_dir / "round.json").read_text(encoding="utf-8"))
    assert metadata["page_version"] == 2


def test_publish_requires_a_selectable_choice(tmp_path):
    result, session_dir = _publish(tmp_path, "<section>No choices here</section>")

    assert result.returncode != 0
    assert "data-choice" in result.stderr
    assert not (session_dir / "current.html").exists()
    assert not (session_dir / "round.json").exists()


@pytest.mark.parametrize("choice_count", [1, 5])
def test_publish_requires_two_to_four_comparable_choices(tmp_path, choice_count):
    fragment = "".join(
        f'<button data-choice="choice-{index}">Choice {index}</button>'
        for index in range(choice_count)
    )
    result, session_dir = _publish(tmp_path, fragment)

    assert result.returncode != 0
    assert "two to four" in result.stderr
    assert not (session_dir / "current.html").exists()


@pytest.mark.parametrize(
    "fragment",
    [
        f'<button data-choice="{"x" * 129}">Long ID</button>' + ALTERNATIVE_CHOICE,
        f'<button data-choice="choice" data-label="{"界" * 257}">Choice</button>'
        + ALTERNATIVE_CHOICE,
    ],
)
def test_publish_bounds_choice_ids_and_labels(tmp_path, fragment):
    result, session_dir = _publish(tmp_path, fragment)

    assert result.returncode != 0
    assert not (session_dir / "round.json").exists()


def test_publish_bounds_round_ids_before_persisting_or_printing_them(tmp_path):
    result, session_dir = _publish(
        tmp_path,
        '<button data-choice="one">One</button>' + ALTERNATIVE_CHOICE,
        round_id="r" * 129,
    )

    assert result.returncode != 0
    assert "128 characters or fewer" in result.stderr
    assert result.stdout == ""
    assert not (session_dir / "round.json").exists()


@pytest.mark.parametrize(
    "fragment",
    [
        '<button data-choice="one" data-choice="two">Ambiguous</button>'
        + ALTERNATIVE_CHOICE,
        '<button data-choice="one" data-label="One" data-label="Two">Ambiguous</button>'
        + ALTERNATIVE_CHOICE,
    ],
)
def test_publish_rejects_duplicate_choice_metadata_attributes(tmp_path, fragment):
    result, session_dir = _publish(tmp_path, fragment)

    assert result.returncode != 0
    assert "duplicate choice metadata attribute" in result.stderr
    assert not (session_dir / "round.json").exists()


def test_publish_rejects_a_fragment_larger_than_one_mib(tmp_path):
    fragment = (
        '<button data-choice="one">One</button>'
        '<button data-choice="two">Two</button>'
        + ("x" * 1_048_576)
    )
    result, session_dir = _publish(tmp_path, fragment)

    assert result.returncode != 0
    assert "1 MiB" in result.stderr
    assert not (session_dir / "current.html").exists()


def test_publish_rejects_executable_or_remote_fragment_content(tmp_path):
    unsafe_fragments = [
        '<script>alert("x")</script><button data-choice="x">X</button>'
        + ALTERNATIVE_CHOICE,
        '<button data-choice="x" onclick="alert(1)">X</button>' + ALTERNATIVE_CHOICE,
        '<button/data-choice="x"/onclick="alert(1)">X</button>' + ALTERNATIVE_CHOICE,
        '<a data-choice="x" href="javascript:alert(1)">X</a>' + ALTERNATIVE_CHOICE,
        '<a data-choice="x" href="java&#x73;cript:alert(1)">X</a>' + ALTERNATIVE_CHOICE,
        '<a data-choice="x" href="https:example.com">X</a>' + ALTERNATIVE_CHOICE,
        '<meta http-equiv="refresh" content="0;url=https://example.com">'
        '<button data-choice="x">X</button>'
        + ALTERNATIVE_CHOICE,
        '<img data-choice="x" src="https://example.com/track.png">' + ALTERNATIVE_CHOICE,
        '<style>.x { background: url(//example.com/track.png) }</style>'
        '<button data-choice="x">X</button>'
        + ALTERNATIVE_CHOICE,
        '<style>.choice{background:url(https:example.com/image.png)}</style>'
        '<button class="choice" data-choice="x">X</button>'
        + ALTERNATIVE_CHOICE,
    ]

    for index, fragment in enumerate(unsafe_fragments):
        session_dir = tmp_path / f"unsafe-{index}"
        result, _ = _publish(tmp_path, fragment, session_dir=session_dir)

        assert result.returncode != 0, fragment
        assert not (session_dir / "current.html").exists()
        assert not (session_dir / "round.json").exists()


def test_round_manifest_points_to_an_immutable_published_page(tmp_path):
    result, session_dir = _publish(
        tmp_path,
        '<button data-choice="council">Council</button>' + ALTERNATIVE_CHOICE,
    )
    assert result.returncode == 0, result.stderr

    (session_dir / "current.html").write_text(
        '<button data-choice="partial">Partial next round</button>' + ALTERNATIVE_CHOICE,
        encoding="utf-8",
    )
    server, state = _start_server(session_dir)
    try:
        opener, _ = _authenticated_opener(state)
        with opener.open(f'{state["base_url"]}/', timeout=2) as response:
            page = response.read().decode()
        assert "Council" in page
        assert "Partial next round" not in page
        with pytest.raises(urllib.error.HTTPError) as invalid:
            _post_choice(opener, state["base_url"], "partial")
        assert invalid.value.code == 400
    finally:
        server.terminate()
        server.communicate(timeout=2)


def test_active_round_fails_closed_when_its_versioned_page_is_missing(tmp_path):
    result, session_dir = _publish(
        tmp_path,
        '<button data-choice="council">Council</button>' + ALTERNATIVE_CHOICE,
    )
    assert result.returncode == 0, result.stderr
    (session_dir / ".pages" / "page-1.html").unlink()

    companion_module = runpy.run_path(str(SCRIPT_PATH))
    error_type = companion_module["VisualCompanionError"]
    with pytest.raises(error_type, match="published page is missing"):
        companion_module["_active_round"](session_dir)


def test_generated_fragment_is_sandboxed_from_trusted_host_controls(tmp_path):
    fragment = (
        '<style>:host{position:fixed;inset:0;z-index:20000}'
        '.forged{position:fixed;inset:0;z-index:20000}</style>'
        '<input id="companion-feedback" value="forged fragment feedback">'
        '<div id="companion-status">forged fragment status</div>'
        '<button data-choice="one">One</button>'
        '<button data-choice="two">Two</button>'
    )
    result, session_dir = _publish(tmp_path, fragment)
    assert result.returncode == 0, result.stderr

    server, state = _start_server(session_dir)
    try:
        opener, _ = _authenticated_opener(state)
        with opener.open(f'{state["base_url"]}/', timeout=2) as response:
            page = response.read().decode()
            csp = response.headers["Content-Security-Policy"]
        assert page.count('id="companion-feedback"') == 1
        assert page.count('id="companion-status"') == 1
        assert '<input id="companion-feedback"' not in page
        assert 'sandbox="allow-scripts"' in page
        assert "boardFrame.srcdoc =" in page
        assert "event.source !== boardFrame.contentWindow" in page
        assert "attachShadow" not in page
        assert "frame-src 'self'" in csp
    finally:
        server.terminate()
        server.communicate(timeout=2)


def test_generated_client_fences_rapid_clicks_before_sending_the_choice(tmp_path):
    result, session_dir = _publish(
        tmp_path,
        '<button data-choice="one">One</button>' + ALTERNATIVE_CHOICE,
    )
    assert result.returncode == 0, result.stderr

    companion_module = runpy.run_path(str(SCRIPT_PATH))
    page, _ = companion_module["_render_page"](session_dir)
    source = page.decode()
    pending_index = source.index("selectionPending = true;")
    disable_index = source.index("setChoicesDisabled(true);")
    request_index = source.index("const response = await fetch")

    assert "if (selectionRecorded || selectionPending) return;" in source
    assert pending_index < request_index
    assert disable_index < request_index


def test_server_authenticates_preview_and_returns_a_valid_choice(tmp_path):
    result, session_dir = _publish(
        tmp_path,
        '<button data-choice="council" data-label="Council Chamber">Council</button>'
        + ALTERNATIVE_CHOICE,
    )
    assert result.returncode == 0, result.stderr

    server, state = _start_server(session_dir)
    try:
        with pytest.raises(urllib.error.HTTPError) as unauthenticated:
            urllib.request.urlopen(f'{state["base_url"]}/', timeout=2)
        assert unauthenticated.value.code == 403

        opener, cookie_jar = _authenticated_opener(state)
        cookie = next(iter(cookie_jar))
        assert cookie.has_nonstandard_attr("HttpOnly")
        assert cookie.get_nonstandard_attr("SameSite") == "Lax"

        with pytest.raises(urllib.error.HTTPError) as reused_capability:
            _authenticated_opener(state)
        assert reused_capability.value.code == 403

        with opener.open(f'{state["base_url"]}/', timeout=2) as response:
            page = response.read().decode()
            assert "default-src 'none'" in response.headers["Content-Security-Policy"]
        assert "Council Chamber" in page
        assert "event.preventDefault();" in page

        with pytest.raises(urllib.error.HTTPError) as forged:
            _post_choice(opener, state["base_url"], "not-an-option")
        assert forged.value.code == 400

        with pytest.raises(urllib.error.HTTPError) as boolean_version:
            _post_choice(opener, state["base_url"], "council", page_version=True)
        assert boolean_version.value.code == 400

        with pytest.raises(urllib.error.HTTPError) as long_feedback:
            _post_choice(opener, state["base_url"], "council", feedback="x" * 2001)
        assert long_feedback.value.code == 400

        with pytest.raises(urllib.error.HTTPError) as oversized_body:
            _post_choice(opener, state["base_url"], "council", feedback="x" * 40_000)
        assert oversized_body.value.code == 413

        recorded = _post_choice(opener, state["base_url"], "council")
        assert recorded == {
            "choice_id": "council",
            "cursor": 1,
            "feedback": "",
            "label": "Council Chamber",
            "page_version": 1,
            "round_id": "layout-directions",
        }
        if os.name != "nt":
            assert (session_dir / "events.jsonl").stat().st_mode & 0o777 == 0o600

        waited = subprocess.run(
            [
                sys.executable,
                str(SCRIPT_PATH),
                "wait",
                "--session-dir",
                str(session_dir),
                "--after",
                "0",
                "--timeout",
                "1",
            ],
            capture_output=True,
            text=True,
        )
        assert waited.returncode == 0, waited.stderr
        assert json.loads(waited.stdout) == recorded
    finally:
        server.terminate()
        server.communicate(timeout=2)


def test_parallel_sessions_use_isolated_cookie_paths_and_tokens(tmp_path):
    first_result, first_dir = _publish(
        tmp_path,
        '<button data-choice="council">Council</button>' + ALTERNATIVE_CHOICE,
        session_dir=tmp_path / "session-a",
    )
    second_result, second_dir = _publish(
        tmp_path,
        '<button data-choice="focused">Focused</button>' + ALTERNATIVE_CHOICE,
        session_dir=tmp_path / "session-b",
    )
    assert first_result.returncode == 0, first_result.stderr
    assert second_result.returncode == 0, second_result.stderr

    first_server, first_state = _start_server(first_dir)
    second_server, second_state = _start_server(second_dir)
    cookie_jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cookie_jar))
    try:
        for state in (first_state, second_state):
            request = urllib.request.Request(
                state["bootstrap_url"],
                data=urllib.parse.urlencode({"key": state["bootstrap_token"]}).encode(),
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                method="POST",
            )
            with opener.open(request, timeout=2) as response:
                assert response.status == 200

        with opener.open(f'{first_state["base_url"]}/', timeout=2) as response:
            assert response.status == 200

        cookies = list(cookie_jar)
        assert len(cookies) == 2
        assert len({cookie.name for cookie in cookies}) == 2
        assert all(cookie.path != "/" for cookie in cookies)
        session_tokens = {cookie.value for cookie in cookies}
        for state in (first_state, second_state):
            assert state["bootstrap_token"] not in session_tokens
    finally:
        first_server.terminate()
        first_server.communicate(timeout=2)
        second_server.terminate()
        second_server.communicate(timeout=2)


def test_wait_ignores_old_events_and_reports_timeout(tmp_path):
    result, session_dir = _publish(
        tmp_path,
        '<button data-choice="council">Council</button>' + ALTERNATIVE_CHOICE,
    )
    assert result.returncode == 0, result.stderr

    server, state = _start_server(session_dir)
    try:
        opener, _ = _authenticated_opener(state)
        first = _post_choice(opener, state["base_url"], "council")

        waited = subprocess.run(
            [
                sys.executable,
                str(SCRIPT_PATH),
                "wait",
                "--session-dir",
                str(session_dir),
                "--after",
                str(first["cursor"]),
                "--timeout",
                "0.05",
            ],
            capture_output=True,
            text=True,
        )
        assert waited.returncode == 3
        assert json.loads(waited.stdout) == {
            "after": 1,
            "status": "timeout",
        }
    finally:
        server.terminate()
        server.communicate(timeout=2)


def test_wait_reads_events_under_the_session_transaction_lock(tmp_path):
    result, session_dir = _publish(
        tmp_path,
        '<button data-choice="council">Council</button>' + ALTERNATIVE_CHOICE,
    )
    assert result.returncode == 0, result.stderr

    companion_module = runpy.run_path(str(SCRIPT_PATH))
    events_path = session_dir / "events.jsonl"
    observed = []
    errors = []

    def wait_for_event():
        try:
            observed.append(companion_module["wait_for_choice"](session_dir, 0, 1))
        except Exception as exc:  # noqa: BLE001 - inspect cross-thread failure below
            errors.append(exc)

    with companion_module["_session_lock"](session_dir):
        events_path.write_text('{"cursor":', encoding="utf-8")
        waiter = threading.Thread(target=wait_for_event)
        waiter.start()
        time.sleep(0.1)
        assert waiter.is_alive(), "wait read a partially written event outside the session lock"
        events_path.write_text('{"cursor":1}\n', encoding="utf-8")

    waiter.join(timeout=2)
    assert not waiter.is_alive()
    assert errors == []
    assert observed == [{"cursor": 1}]


def test_server_rejects_a_choice_from_a_stale_page_version(tmp_path):
    result, session_dir = _publish(
        tmp_path,
        '<button data-choice="council">Council</button>' + ALTERNATIVE_CHOICE,
    )
    assert result.returncode == 0, result.stderr

    server, state = _start_server(session_dir)
    try:
        opener, _ = _authenticated_opener(state)
        second, _ = _publish(
            tmp_path,
            '<button data-choice="saffron">Graphite &amp; Saffron</button>'
            + ALTERNATIVE_CHOICE,
            round_id="accent-directions",
            session_dir=session_dir,
        )
        assert second.returncode == 0, second.stderr

        with opener.open(f'{state["base_url"]}/__version', timeout=2) as response:
            assert json.loads(response.read()) == {
                "page_version": 2,
                "round_id": "accent-directions",
            }

        with pytest.raises(urllib.error.HTTPError) as stale:
            _post_choice(opener, state["base_url"], "saffron", page_version=1)
        assert stale.value.code == 409

        fresh = _post_choice(opener, state["base_url"], "saffron", page_version=2)
        assert fresh["choice_id"] == "saffron"
        assert fresh["page_version"] == 2
    finally:
        server.terminate()
        server.communicate(timeout=2)


def test_choice_transaction_uses_the_cross_process_session_lock(tmp_path):
    result, session_dir = _publish(
        tmp_path,
        '<button data-choice="council">Council</button>' + ALTERNATIVE_CHOICE,
    )
    assert result.returncode == 0, result.stderr

    server, state = _start_server(session_dir)
    companion_module = runpy.run_path(str(SCRIPT_PATH))
    outcome = {}

    def submit_choice():
        try:
            opener, _ = _authenticated_opener(state)
            outcome["event"] = _post_choice(opener, state["base_url"], "council")
        except Exception as exc:  # pragma: no cover - asserted through outcome below
            outcome["error"] = exc

    submitter = threading.Thread(target=submit_choice)
    try:
        with companion_module["_session_lock"](session_dir):
            submitter.start()
            time.sleep(0.2)
            assert submitter.is_alive(), "choice transaction ignored the held session lock"

        submitter.join(timeout=2)
        assert not submitter.is_alive()
        assert "error" not in outcome
        assert outcome["event"]["choice_id"] == "council"
    finally:
        submitter.join(timeout=2)
        server.terminate()
        server.communicate(timeout=2)


def test_status_and_stop_commands_manage_the_server_lifecycle(tmp_path):
    result, session_dir = _publish(
        tmp_path,
        '<button data-choice="council">Council</button>' + ALTERNATIVE_CHOICE,
    )
    assert result.returncode == 0, result.stderr

    server, _ = _start_server(session_dir)
    status = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "status", "--session-dir", str(session_dir)],
        capture_output=True,
        text=True,
    )
    assert status.returncode == 0, status.stderr
    assert json.loads(status.stdout) == {"status": "ok"}

    stop = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "stop", "--session-dir", str(session_dir)],
        capture_output=True,
        text=True,
    )
    assert stop.returncode == 0, stop.stderr
    assert json.loads(stop.stdout) == {"status": "stopping"}
    assert server.wait(timeout=2) == 0
    assert not (session_dir / "state.json").exists()


def test_serve_stdout_contains_only_non_secret_readiness_data(tmp_path):
    result, session_dir = _publish(
        tmp_path,
        '<button data-choice="council">Council</button>' + ALTERNATIVE_CHOICE,
    )
    assert result.returncode == 0, result.stderr

    server, state = _start_server(session_dir)
    try:
        assert server.stdout is not None
        readiness_line = server.stdout.readline()
        assert json.loads(readiness_line) == {
            "pid": server.pid,
            "port": state["port"],
            "status": "ready",
        }
        assert "session_token" not in readiness_line
        assert "preview_url" not in readiness_line
        assert "cookie_name" not in readiness_line
        assert state["session_token"] not in readiness_line
        assert state["bootstrap_token"] not in readiness_line
        assert state["bootstrap_url"] not in readiness_line
    finally:
        server.terminate()
        server.communicate(timeout=2)


def test_serve_writes_a_private_preview_launcher_for_opaque_desktop_handoff(tmp_path):
    result, session_dir = _publish(
        tmp_path,
        '<button data-choice="council">Council</button>' + ALTERNATIVE_CHOICE,
    )
    assert result.returncode == 0, result.stderr

    server, state = _start_server(session_dir)
    try:
        launcher_path = session_dir / "open-preview.html"
        launcher = launcher_path.read_text(encoding="utf-8")

        assert "preview_url" not in state
        assert state["bootstrap_url"] in launcher
        assert state["bootstrap_token"] in launcher
        assert state["session_token"] not in launcher
        assert 'method="post"' in launcher
        assert "?key=" not in launcher
        if os.name != "nt":
            assert launcher_path.stat().st_mode & 0o777 == 0o600
    finally:
        server.terminate()
        server.communicate(timeout=2)


def test_bootstrap_cookie_allows_the_local_launcher_top_level_navigation(tmp_path):
    result, session_dir = _publish(
        tmp_path,
        '<button data-choice="council">Council</button>' + ALTERNATIVE_CHOICE,
    )
    assert result.returncode == 0, result.stderr

    server, state = _start_server(session_dir)
    try:
        cookie_jar = http.cookiejar.CookieJar()
        opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cookie_jar))
        legacy_query_url = (
            f'{state["base_url"]}/?key='
            f'{urllib.parse.quote(state["bootstrap_token"], safe="")}'
        )
        with pytest.raises(urllib.error.HTTPError) as query_bootstrap:
            opener.open(legacy_query_url, timeout=2)
        assert query_bootstrap.value.code == 403

        request = urllib.request.Request(
            state["bootstrap_url"],
            data=urllib.parse.urlencode({"key": state["bootstrap_token"]}).encode(),
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            method="POST",
        )
        with opener.open(request, timeout=2) as response:
            transition = response.read().decode()
            set_cookie = response.headers["Set-Cookie"]

        assert "location.replace" in transition
        assert state["bootstrap_token"] not in transition
        assert "HttpOnly" in set_cookie
        assert "SameSite=Lax" in set_cookie
        with opener.open(f'{state["base_url"]}/', timeout=2) as response:
            assert response.status == 200
    finally:
        server.terminate()
        server.communicate(timeout=2)


def test_only_one_server_may_serve_a_session(tmp_path):
    result, session_dir = _publish(
        tmp_path,
        '<button data-choice="council">Council</button>' + ALTERNATIVE_CHOICE,
    )
    assert result.returncode == 0, result.stderr

    first, first_state = _start_server(session_dir)
    second = subprocess.Popen(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "serve",
            "--session-dir",
            str(session_dir),
            "--port",
            "0",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 1.5
        while second.poll() is None and time.monotonic() < deadline:
            time.sleep(0.02)

        assert second.poll() is not None, "second server remained active for the same session"
        _, second_stderr = second.communicate(timeout=1)
        assert second.returncode != 0
        assert "already has a running server" in second_stderr

        current_state = json.loads((session_dir / "state.json").read_text(encoding="utf-8"))
        assert current_state == first_state
    finally:
        if second.poll() is None:
            second.terminate()
            second.communicate(timeout=2)
        first.terminate()
        first.communicate(timeout=2)


@pytest.mark.parametrize("unicode_character", ["界", "🙂"])
def test_feedback_accepts_two_thousand_unicode_characters(tmp_path, unicode_character):
    result, session_dir = _publish(
        tmp_path,
        '<button data-choice="council">Council</button>' + ALTERNATIVE_CHOICE,
    )
    assert result.returncode == 0, result.stderr

    server, state = _start_server(session_dir)
    try:
        opener, _ = _authenticated_opener(state)
        feedback = unicode_character * 2_000
        recorded = _post_choice(
            opener,
            state["base_url"],
            "council",
            feedback=feedback,
        )
        assert recorded["feedback"] == feedback
    finally:
        server.terminate()
        server.communicate(timeout=2)


def test_choice_preserves_optional_feedback_and_deduplicates_a_round(tmp_path):
    result, session_dir = _publish(
        tmp_path,
        (
            '<button data-choice="council" data-label="Council Chamber">Council</button>'
            '<button data-choice="focused" data-label="Focused Correspondence">Focused</button>'
        ),
    )
    assert result.returncode == 0, result.stderr

    server, state = _start_server(session_dir)
    try:
        opener, _ = _authenticated_opener(state)
        first = _post_choice(
            opener,
            state["base_url"],
            "council",
            feedback="Keep the structure, but make the accent less orange.",
        )
        assert first["feedback"] == "Keep the structure, but make the accent less orange."

        duplicate = _post_choice(
            opener,
            state["base_url"],
            "council",
            feedback="Keep the structure, but make the accent less orange.",
        )
        assert duplicate == first

        with pytest.raises(urllib.error.HTTPError) as conflicting:
            _post_choice(opener, state["base_url"], "focused")
        assert conflicting.value.code == 409
    finally:
        server.terminate()
        server.communicate(timeout=2)


def test_skill_frontmatter_links_and_visual_workflow_contract_are_valid(monkeypatch):
    from tools import skills_tool
    from tools.skill_manager_tool import _validate_frontmatter

    skill_text = SKILL_PATH.read_text(encoding="utf-8")
    reference_text = (SKILL_PATH.parent / "references/visual-companion.md").read_text(
        encoding="utf-8"
    )
    assert _validate_frontmatter(skill_text) is None
    assert "references/visual-companion.md" in skill_text

    monkeypatch.setattr(skills_tool, "SKILLS_DIR", SKILL_PATH.parents[2])
    viewed = json.loads(skills_tool.skill_view("claude-design", preprocess=False))
    assert viewed["success"] is True
    assert "references/visual-companion.md" in viewed["linked_files"]["references"]
    assert "scripts/visual_companion.py" in viewed["linked_files"]["scripts"]

    assert "Maintain `decision-ledger.json`" in reference_text
    assert "single dimension explored next" in reference_text
    assert "ask the same options through `clarify`" in reference_text
    assert "state the accepted direction in ordinary assistant text" in reference_text
    assert "open-preview.html" in reference_text
    assert "`open_preview`" in reference_text
    assert "Do not read `state.json`" in reference_text
    assert "complete representative product surface" in reference_text
    assert "not a thumbnail, moodboard, or palette swatch" in reference_text
    assert "one-sentence composition thesis" in reference_text
    assert "numbered anatomy callouts" in reference_text
    assert "wide and narrow behavior" in reference_text
    assert "actual product vocabulary" in reference_text
    assert "consolidated presentation artifact" in reference_text
    assert "Run the Slop Diagnostic before publication" in reference_text
    assert "preserve the same viewport, crop, content, and zoom" in reference_text
    assert "inspect the primary wide viewport and the named narrow viewport" in reference_text
    assert "changing only the ledger's `varying` dimension" in reference_text
    assert "This artifact is the design handoff, not another vote" in reference_text
    assert "high-fidelity presentation contract is mandatory" in skill_text
