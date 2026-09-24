"""Google Meet browser behavior contracts."""

from __future__ import annotations
import json
import pytest


@pytest.fixture(autouse=True)
def _isolate_home(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    yield hermes_home


def test_detect_admission_returns_false_on_error():
    from plugins.google_meet.meet_bot import _ADMISSION_PROBE_JS, _probe

    class _FakePage:
        def evaluate(self, _js): raise RuntimeError("boom")

    assert _probe(_FakePage(), _ADMISSION_PROBE_JS) is False


def test_classify_meet_ui_blocks_getting_ready_page():
    from plugins.google_meet.meet_bot import _classify_meet_ui

    result = _classify_meet_ui(
        "Getting ready... You'll be able to join in just a moment",
        in_call_control=True,
        url="https://meet.google.com/abc-defg-hij",
    )

    assert result["inCall"] is False
    assert result["preJoin"] is True


def test_classify_meet_ui_blocks_ready_to_join_page_with_media_prompt():
    from plugins.google_meet.meet_bot import _classify_meet_ui

    result = _classify_meet_ui(
        (
            "mic Show more info videocam Show more info Ready to join? "
            "Alex Lee and Morgan Patel are in this call Join now "
            "Do you want people to see and hear you in the meeting? "
            "Continue without microphone and camera"
        ),
        in_call_control=True,
        url="https://meet.google.com/abc-defg-hij",
    )

    assert result["inCall"] is False
    assert result["preJoin"] is True


def test_classify_meet_ui_treats_host_wait_phrase_as_lobby():
    from plugins.google_meet.meet_bot import _classify_meet_ui

    result = _classify_meet_ui(
        (
            "Please wait until a meeting host brings you into the call "
            "Turn on microphone Turn on camera Leave call"
        ),
        leave=True,
        in_call_control=True,
        url="https://meet.google.com/abc-defg-hij",
    )

    assert result["inCall"] is False
    assert result["waitingLobby"] is True


def test_classify_meet_ui_blocks_landing_page_after_meet_error():
    from plugins.google_meet.meet_bot import _classify_meet_ui

    result = _classify_meet_ui(
        "Meet Secure video conferencing for everyone New meeting Join",
        in_call_control=True,
        url="https://meet.google.com/landing",
    )

    assert result["inCall"] is False
    assert result["landing"] is True


def test_classify_meet_ui_blocks_workspace_meet_product_redirect():
    from plugins.google_meet.meet_bot import _classify_meet_ui

    result = _classify_meet_ui(
        "Google Meet video meetings and calls for everyone",
        url="https://workspace.google.com/products/meet/",
    )

    assert result["inCall"] is False
    assert result["landing"] is True


def test_classify_meet_ui_blocks_could_not_start_video_call_error():
    from plugins.google_meet.meet_bot import _classify_meet_ui

    result = _classify_meet_ui(
        "Couldn't start the video call because of an error",
        in_call_control=True,
        url="https://meet.google.com/landing",
    )

    assert result["inCall"] is False
    assert result["callError"] is True


def test_classify_meet_ui_marks_return_home_denial_terminal():
    from plugins.google_meet.meet_bot import _classify_meet_ui

    result = _classify_meet_ui(
        (
            "You can't join this video call Return to home screen "
            "No one can join a meeting unless invited or admitted by the host "
            "Returning to home screen 45 seconds left"
        ),
        url="https://meet.google.com/abc-defg-hij",
    )

    assert result["denied"] is True
    assert result["terminalDenied"] is True


def test_classify_meet_ui_marks_policy_denial_terminal_without_join_click():
    from plugins.google_meet.meet_bot import _classify_meet_ui

    result = _classify_meet_ui(
        "No one can join a meeting unless invited or admitted by the host",
        url="https://meet.google.com/abc-defg-hij",
    )

    assert result["denied"] is True
    assert result["terminalDenied"] is True


def test_classify_meet_ui_preserves_in_call_signal_with_transient_call_error():
    from plugins.google_meet.meet_bot import _classify_meet_ui

    result = _classify_meet_ui(
        "Couldn't start the video call because of an error Meeting details",
        leave=True,
        in_call_control=True,
        url="https://meet.google.com/abc-defg-hij",
    )

    assert result["callError"] is True
    assert result["inCall"] is True


def test_compute_meet_phase_marks_join_attempt_as_stalled(tmp_path):
    from plugins.google_meet.meet_bot import _BotState, _compute_meet_phase

    state = _BotState(
        out_dir=tmp_path / "meet",
        meeting_id="abc-defg-hij",
        url="https://meet.google.com/abc-defg-hij",
    )
    state.set(join_attempted_at=100.0)

    phase, reason = _compute_meet_phase(state, now=205.0, stall_after=90.0)

    assert phase == "stalled"
    assert "no admission" in reason


def test_compute_meet_phase_reports_capturing_after_transcript(tmp_path):
    from plugins.google_meet.meet_bot import _BotState, _compute_meet_phase

    state = _BotState(
        out_dir=tmp_path / "meet",
        meeting_id="abc-defg-hij",
        url="https://meet.google.com/abc-defg-hij",
    )
    state.set(in_call=True, joined_at=100.0)
    state.record_caption("Alice", "hello")

    phase, reason = _compute_meet_phase(state, now=110.0, stall_after=90.0)

    assert phase == "capturing"
    assert reason is None


def test_apply_admission_probe_revokes_prejoin_false_positive(tmp_path):
    from plugins.google_meet.meet_bot import _BotState, _apply_admission_probe

    state = _BotState(
        out_dir=tmp_path / "meet",
        meeting_id="abc-defg-hij",
        url="https://meet.google.com/abc-defg-hij",
    )
    state.set(in_call=True, joined_at=100.0)

    admitted, terminal = _apply_admission_probe(
        state,
        {
            "inCall": False,
            "preJoin": True,
            "waitingLobby": False,
            "denied": False,
            "text": "Getting ready... You'll be able to join in just a moment",
        },
        now=105.0,
        lobby_deadline=400.0,
    )

    assert admitted is False
    assert terminal is False
    assert state.in_call is False
    assert state.joined_at is None
    assert state.phase == "joining"


def test_apply_admission_probe_ignores_denied_before_join_attempt(tmp_path):
    from plugins.google_meet.meet_bot import _BotState, _apply_admission_probe

    state = _BotState(
        out_dir=tmp_path / "meet",
        meeting_id="abc-defg-hij",
        url="https://meet.google.com/abc-defg-hij",
    )

    admitted, terminal = _apply_admission_probe(
        state,
        {
            "inCall": False,
            "waitingLobby": False,
            "denied": True,
            "preJoin": False,
            "text": "No one can join a meeting unless invited or admitted by the host",
        },
        now=105.0,
        lobby_deadline=400.0,
    )

    assert admitted is False
    assert terminal is False
    assert state.error is None
    assert state.leave_reason is None
    assert state.phase == "starting"


def test_apply_admission_probe_exits_on_terminal_denial_before_join_attempt(tmp_path):
    from plugins.google_meet.meet_bot import _BotState, _apply_admission_probe

    state = _BotState(
        out_dir=tmp_path / "meet",
        meeting_id="abc-defg-hij",
        url="https://meet.google.com/abc-defg-hij",
    )

    admitted, terminal = _apply_admission_probe(
        state,
        {
            "inCall": False,
            "waitingLobby": False,
            "denied": True,
            "terminalDenied": True,
            "preJoin": False,
            "text": "You can't join this video call Returning to home screen",
        },
        now=105.0,
        lobby_deadline=400.0,
    )

    assert admitted is False
    assert terminal is True
    assert state.error == "host denied admission"
    assert state.leave_reason == "denied"
    assert state.phase == "exited"


def test_apply_admission_probe_exits_when_meet_returns_to_landing(tmp_path):
    from plugins.google_meet.meet_bot import _BotState, _apply_admission_probe

    state = _BotState(
        out_dir=tmp_path / "meet",
        meeting_id="abc-defg-hij",
        url="https://meet.google.com/abc-defg-hij",
    )
    state.set(join_attempted_at=100.0, in_call=True, joined_at=105.0)

    admitted, terminal = _apply_admission_probe(
        state,
        {
            "inCall": False,
            "landing": True,
            "waitingLobby": False,
            "denied": False,
            "text": "Meet Secure video conferencing for everyone New meeting Join",
            "url": "https://meet.google.com/landing",
        },
        now=110.0,
        lobby_deadline=400.0,
    )

    assert admitted is False
    assert terminal is True
    assert state.in_call is False
    assert state.joined_at is None
    assert state.leave_reason == "meet_landing"
    assert "landing" in state.error
    assert state.phase == "exited"


def test_apply_admission_probe_exits_when_meet_redirects_to_landing_before_join(tmp_path):
    from plugins.google_meet.meet_bot import _BotState, _apply_admission_probe

    state = _BotState(
        out_dir=tmp_path / "meet",
        meeting_id="abc-defg-hij",
        url="https://meet.google.com/abc-defg-hij",
    )

    admitted, terminal = _apply_admission_probe(
        state,
        {
            "inCall": False,
            "landing": True,
            "waitingLobby": False,
            "denied": False,
            "text": "Google Meet video meetings and calls for everyone",
            "url": "https://workspace.google.com/products/meet/",
        },
        now=110.0,
        lobby_deadline=400.0,
    )

    assert admitted is False
    assert terminal is True
    assert state.in_call is False
    assert state.joined_at is None
    assert state.leave_reason == "meet_landing"
    assert "landing" in state.error
    assert state.phase == "exited"


def test_apply_admission_probe_tolerates_single_transient_call_error(tmp_path):
    """A one-off Meet "couldn't start the video call" flash must not kill a live session.

    Google Meet routinely shows this banner transiently while the call is in
    fact healthy (3 named participants, mic/cam off were observed in the live
    failure). A single observation must not be treated as terminal.
    """
    from plugins.google_meet.meet_bot import _BotState, _apply_admission_probe

    state = _BotState(
        out_dir=tmp_path / "meet",
        meeting_id="abc-defg-hij",
        url="https://meet.google.com/abc-defg-hij",
    )
    state.set(join_attempted_at=100.0, in_call=True, joined_at=105.0)

    admitted, terminal = _apply_admission_probe(
        state,
        {
            "inCall": False,
            "callError": True,
            "waitingLobby": False,
            "denied": False,
            "text": "Couldn't start the video call because of an error",
            "url": "https://meet.google.com/abc-defg-hij",
        },
        now=110.0,
        lobby_deadline=400.0,
    )

    assert terminal is False
    assert state.in_call is True
    assert state.joined_at == 105.0
    assert state.leave_reason is None
    assert state.phase != "exited"
    assert state.call_error_strikes == 1


def test_apply_admission_probe_keeps_call_alive_when_error_overlay_still_has_in_call_controls(
    tmp_path,
):
    from plugins.google_meet.meet_bot import _BotState, _apply_admission_probe

    state = _BotState(
        out_dir=tmp_path / "meet",
        meeting_id="abc-defg-hij",
        url="https://meet.google.com/abc-defg-hij",
    )
    state.set(join_attempted_at=100.0, in_call=True, joined_at=105.0)

    terminal = False
    for now in (110.0, 113.0, 116.0, 119.0):
        admitted, terminal = _apply_admission_probe(
            state,
            {
                "inCall": True,
                "callError": True,
                "waitingLobby": False,
                "denied": False,
                "preJoin": False,
                "text": "Couldn't start the video call because of an error Meeting details",
                "url": "https://meet.google.com/abc-defg-hij",
            },
            now=now,
            lobby_deadline=400.0,
        )
        assert admitted is True

    assert terminal is False
    assert state.in_call is True
    assert state.joined_at == 105.0
    assert state.leave_reason is None
    assert state.phase != "exited"
    assert state.call_error_strikes == 0


def test_apply_admission_probe_resets_call_error_strikes_when_cleared(tmp_path):
    """Strikes reset once Meet stops reporting the error, so a later flash starts fresh."""
    from plugins.google_meet.meet_bot import _BotState, _apply_admission_probe

    state = _BotState(
        out_dir=tmp_path / "meet",
        meeting_id="abc-defg-hij",
        url="https://meet.google.com/abc-defg-hij",
    )
    state.set(join_attempted_at=100.0, in_call=True, joined_at=105.0)

    error_probe = {
        "inCall": False,
        "callError": True,
        "text": "Couldn't start the video call because of an error",
        "url": "",
    }
    for now in (110.0, 113.0):
        _apply_admission_probe(
            state,
            error_probe,
            now=now,
            lobby_deadline=400.0,
        )
    assert state.call_error_strikes == 2

    # A clean in-call probe clears the count — the error was transient.
    _apply_admission_probe(
        state,
        {"inCall": True, "callError": False, "text": "Meeting details", "url": ""},
        now=116.0,
        lobby_deadline=400.0,
    )
    assert state.call_error_strikes == 0
    assert state.in_call is True


def test_apply_admission_probe_exits_on_persistent_call_start_error(tmp_path):
    """Before ever being admitted, a call error persisting past the strike limit
    is terminal — the call genuinely won't start."""
    from plugins.google_meet.meet_bot import _BotState, _apply_admission_probe

    state = _BotState(
        out_dir=tmp_path / "meet",
        meeting_id="abc-defg-hij",
        url="https://meet.google.com/abc-defg-hij",
    )
    state.set(join_attempted_at=100.0)  # never admitted

    probe = {
        "inCall": False,
        "callError": True,
        "waitingLobby": False,
        "denied": False,
        "text": "Couldn't start the video call because of an error",
        "url": "https://meet.google.com/landing",
    }

    terminal = False
    for now in (110.0, 113.0, 116.0):
        _, terminal = _apply_admission_probe(
            state,
            probe,
            now=now,
            lobby_deadline=400.0,
        )

    assert state.ever_admitted is False
    assert terminal is True
    assert state.in_call is False
    assert state.joined_at is None
    assert state.leave_reason == "meet_error"
    assert "error" in state.error
    assert state.phase == "exited"


def test_apply_admission_probe_ignores_call_error_after_caption_evidence(tmp_path):
    """Once captions have arrived, a transient call-error overlay is non-terminal.

    Admission alone is not enough because Meet may expose roster text while
    returning to an error page. Caption evidence proves the bot reached the
    target functionality, so persistent media-layer overlays should not end the
    transcript session.
    """
    from plugins.google_meet.meet_bot import _BotState, _apply_admission_probe

    state = _BotState(
        out_dir=tmp_path / "meet",
        meeting_id="abc-defg-hij",
        url="https://meet.google.com/abc-defg-hij",
    )
    state.set(join_attempted_at=100.0)

    admitted, _ = _apply_admission_probe(
        state,
        {"inCall": True, "callError": False, "text": "Meeting details", "url": ""},
        now=105.0,
        lobby_deadline=400.0,
    )
    assert admitted is True
    assert state.ever_admitted is True
    state.set(last_caption_at=106.0, transcript_lines=1)

    err = {
        "inCall": False,
        "callError": True,
        "text": "Couldn't start the video call because of an error",
        "url": "",
    }
    terminal = False
    for now in (108.0, 111.0, 114.0, 117.0, 120.0):
        _, terminal = _apply_admission_probe(
            state,
            err,
            now=now,
            lobby_deadline=400.0,
        )

    assert terminal is False
    assert state.leave_reason != "meet_error"
    assert state.call_error_strikes == 0


def test_apply_admission_probe_exits_on_persistent_call_error_after_false_admission(
    tmp_path,
):
    """A false-positive admission must not mask a persistent Meet error page."""
    from plugins.google_meet.meet_bot import _BotState, _apply_admission_probe

    state = _BotState(
        out_dir=tmp_path / "meet",
        meeting_id="abc-defg-hij",
        url="https://meet.google.com/abc-defg-hij",
    )
    state.set(join_attempted_at=100.0)

    admitted, terminal = _apply_admission_probe(
        state,
        {"inCall": True, "callError": False, "text": "Meeting details", "url": ""},
        now=105.0,
        lobby_deadline=400.0,
    )
    assert admitted is True
    assert terminal is False
    assert state.ever_admitted is True
    assert state.last_caption_at is None

    err = {
        "inCall": False,
        "callError": True,
        "text": (
            "Couldn't start the video call because of an error "
            "Returning to home screen in 60 seconds."
        ),
        "url": "https://meet.google.com/abc-defg-hij?pli=1",
    }
    for now in (108.0, 111.0, 114.0):
        _, terminal = _apply_admission_probe(
            state,
            err,
            now=now,
            lobby_deadline=400.0,
        )

    assert terminal is True
    assert state.in_call is False
    assert state.joined_at is None
    assert state.leave_reason == "meet_error"
    assert "error" in state.error
    assert state.phase == "exited"


def test_meet_proxy_env_pins_media_routing_args(monkeypatch):
    from plugins.google_meet.meet_bot import _apply_meet_proxy_args, _config_from_env

    args = []
    monkeypatch.setenv("HERMES_MEET_PROXY_SERVER", "http://proxy.example:8080")
    monkeypatch.delenv("HERMES_MEET_PROXY_BYPASS", raising=False)

    _apply_meet_proxy_args(_config_from_env(), args)

    assert args == [
        "--proxy-server=http://proxy.example:8080",
        "--proxy-bypass-list=74.125.250.0/24,74.125.247.128,142.250.82.0/24",
        "--force-webrtc-ip-handling-policy=disable_non_proxied_udp",
    ]


def test_meet_proxy_bypass_can_be_disabled_without_disabling_webrtc_policy(monkeypatch):
    from plugins.google_meet.meet_bot import _apply_meet_proxy_args, _config_from_env

    args = []
    monkeypatch.setenv("HERMES_MEET_PROXY_SERVER", "http://proxy.example:8080")
    monkeypatch.setenv("HERMES_MEET_PROXY_BYPASS", "")

    _apply_meet_proxy_args(_config_from_env(), args)

    assert args == [
        "--proxy-server=http://proxy.example:8080",
        "--force-webrtc-ip-handling-policy=disable_non_proxied_udp",
    ]


def test_transcribe_browser_config_uses_fake_device_and_fake_ui_flags_only(monkeypatch):
    from plugins.google_meet.meet_bot import _BotConfig, _build_browser_launch_config

    monkeypatch.delenv("HERMES_MEET_PROXY_SERVER", raising=False)

    chrome_args, permissions = _build_browser_launch_config(
        _BotConfig(realtime=False, proxy_server="")
    )

    assert "--use-fake-device-for-media-stream" in chrome_args
    assert "--use-fake-ui-for-media-stream" in chrome_args
    assert not any(
        arg.startswith("--use-file-for-fake-video-capture=") for arg in chrome_args
    )
    assert not any(
        arg.startswith("--use-file-for-fake-audio-capture=") for arg in chrome_args
    )
    assert chrome_args.count("--use-fake-ui-for-media-stream") == 1
    assert permissions == ["microphone", "camera"]


def test_realtime_browser_config_uses_real_audio_input(monkeypatch):
    from plugins.google_meet.meet_bot import _BotConfig, _build_browser_launch_config

    monkeypatch.delenv("HERMES_MEET_PROXY_SERVER", raising=False)

    chrome_args, permissions = _build_browser_launch_config(
        _BotConfig(realtime=True, proxy_server="")
    )

    assert "--use-fake-ui-for-media-stream" in chrome_args
    assert "--use-fake-device-for-media-stream" not in chrome_args
    assert permissions == ["microphone", "camera"]


@pytest.fixture
def browser_page():
    from pathlib import Path

    playwright = pytest.importorskip("playwright.sync_api")
    with playwright.sync_playwright() as runtime:
        if not Path(runtime.chromium.executable_path).is_file():
            pytest.skip("Google Meet browser contracts require Playwright Chromium")
        browser = runtime.chromium.launch(headless=True)
        page = browser.new_page()
        page.route("**/*", lambda route: route.abort())
        try:
            yield page
        finally:
            browser.close()


@pytest.mark.parametrize(
    "scenario",
    ["visible", "hover", "keyboard", "synthetic", "unverified", "prejoin", "enabled"],
)
def test_caption_activation_requires_browser_confirmation(
    browser_page, tmp_path, scenario
):
    from plugins.google_meet.meet_bot import _BotState, _retry_caption_enable

    page = browser_page
    page.set_content("""
        <button id="captions" aria-label="Turn on captions">Captions</button>
        <script>
        window.install = scenario => {
            const button = document.getElementById('captions');
            const enable = () => {
                if (scenario !== 'unverified') {
                    button.hidden = false;
                    button.setAttribute('aria-label', 'Turn off captions');
                }
            };
            button.hidden = !['visible', 'unverified', 'enabled'].includes(scenario);
            button.onclick = () => {
                if (button.getAttribute('aria-label') === 'Turn off captions') {
                    button.setAttribute('aria-label', 'Turn on captions');
                } else enable();
            };
            document.addEventListener('keydown', event => {
                if (event.key === 'ArrowDown' && scenario === 'hover') button.hidden = false;
                if (event.key === 'c' && scenario !== 'prejoin' &&
                    (scenario !== 'synthetic' || !event.isTrusted) &&
                    (scenario !== 'keyboard' || event.isTrusted)) enable();
            });
            if (scenario === 'enabled') enable();
        };
        </script>
    """)
    page.evaluate("window.install", scenario)
    state = _BotState(
        tmp_path / "meeting", "abc-defg-hij", "https://meet.google.com/abc-defg-hij"
    )
    state.set(in_call=scenario != "prejoin", join_attempted_at=100.0)
    enabled = scenario not in {"unverified", "prejoin"}
    assert _retry_caption_enable(page, state) is enabled
    assert state.captioning is enabled
    assert (
        page.locator("#captions").get_attribute("aria-label") == "Turn off captions"
    ) is enabled
    status = json.loads(state.status_path.read_text())
    assert status["captioning"] is enabled
    assert status["captionsEnabledAttempted"] is (scenario != "prejoin")


@pytest.mark.parametrize(
    "scenario", ["absent", "join", "ask", "text", "duplicate", "media_gate"]
)
def test_join_selects_visible_controls_and_preserves_lobby_state(
    browser_page, tmp_path, scenario
):
    from plugins.google_meet.meet_bot import _BotState, _click_join

    label = "Ask to join" if scenario in {"ask", "text"} else "Join now"
    tag = "span" if scenario == "text" else "button"
    html = "<script>window.joined = false;</script>"
    if scenario == "duplicate":
        html += '<button style="display:none">Join now</button>'
    if scenario != "absent":
        hidden = "hidden" if scenario == "media_gate" else ""
        html += (
            f'<{tag} id="join" {hidden} onclick="window.joined=true">{label}</{tag}>'
        )
    if scenario == "media_gate":
        html += """<button onclick="document.getElementById('join').hidden=false; this.remove()">Continue without microphone and camera</button>"""
    browser_page.set_content(html)
    state = _BotState(
        tmp_path / "meeting", "abc-defg-hij", "https://meet.google.com/abc-defg-hij"
    )
    assert _click_join(browser_page, state) is (scenario != "absent")
    assert browser_page.evaluate("window.joined") is (scenario != "absent")
    assert state.lobby_waiting is (scenario in {"ask", "text"})


@pytest.mark.parametrize(
    "realtime,route_ready,initial_on,controls_present",
    [
        (False, False, True, True),
        (False, False, False, True),
        (True, True, False, True),
        (True, False, False, True),
        (False, False, False, False),
    ],
)
def test_prejoin_media_is_verified_and_realtime_is_route_gated(
    browser_page, tmp_path, realtime, route_ready, initial_on, controls_present
):
    from plugins.google_meet.meet_bot import _BotState, _ensure_local_media_before_join

    html = "<script>window.toggled = [];</script>"
    if controls_present:
        action = "off" if initial_on else "on"
        # Hidden duplicate controls must not shadow the usable ones.
        for device in ("microphone", "camera"):
            html += f'<button hidden aria-label="Turn {action} {device}"></button>'
            html += f'<button id="{device}" aria-label="Turn {action} {device}">{device}</button>'
        html += """<script>
        for (const device of ['microphone', 'camera']) {
            document.getElementById(device).onclick = event => {
                const button = event.currentTarget;
                const next = button.getAttribute('aria-label').includes('Turn off') ? 'on' : 'off';
                button.setAttribute('aria-label', 'Turn ' + next + ' ' + device);
                window.toggled.push(device);
            };
        }
        </script>"""
    browser_page.set_content(html)
    state = _BotState(
        tmp_path / "meeting", "abc-defg-hij", "https://meet.google.com/abc-defg-hij"
    )
    safe = controls_present and (not realtime or route_ready)
    assert (
        _ensure_local_media_before_join(
            browser_page,
            state,
            realtime_enabled=realtime,
            realtime_route_ready=route_ready,
            attempts=1,
        )
        is safe
    )
    if safe:
        assert state.local_camera_on is False
        assert state.local_microphone_on is realtime
        assert (
            browser_page.locator("#camera").get_attribute("aria-label")
            == "Turn on camera"
        )
        assert browser_page.locator("#microphone").get_attribute("aria-label") == (
            "Turn off microphone" if realtime else "Turn on microphone"
        )
    else:
        assert state.exited is True
        assert state.in_call is False
        if realtime:
            assert "microphone" not in browser_page.evaluate("window.toggled")


@pytest.mark.parametrize("already_admitted", [False, True])
def test_drain_retries_captions_after_join_without_waiting_for_admission(
    browser_page, tmp_path, monkeypatch, already_admitted
):
    from plugins.google_meet import meet_bot

    browser_page.set_content("""
        <button aria-label="Turn on captions" onclick="this.setAttribute('aria-label','Turn off captions')">Captions</button>
        <button aria-label="Turn on microphone">Mic</button>
        <button aria-label="Turn on camera">Camera</button>
    """)
    state = meet_bot._BotState(
        tmp_path / "meeting", "abc-defg-hij", "https://meet.google.com/abc-defg-hij"
    )
    state.set(in_call=already_admitted, join_attempted_at=100.0)
    stop = {"stop": False}
    monkeypatch.setattr(meet_bot.time, "sleep", lambda _seconds: stop.update(stop=True))
    meet_bot._drain_loop(
        browser_page,
        meet_bot._BotConfig(
            guest_name="Bot", duration_s=None, lobby_timeout=300, realtime=False
        ),
        state,
        {"session": None},
        stop,
    )
    assert state.captioning is True
    assert browser_page.get_by_role("button", name="Turn off captions").is_visible()


def test_guest_name_uses_visible_placeholder_input(browser_page):
    from plugins.google_meet.meet_bot import _try_guest_name

    browser_page.set_content('<input placeholder="Your name">')
    _try_guest_name(browser_page, "Catchline Assistant")
    assert browser_page.locator("input").input_value() == "Catchline Assistant"


def test_prejoin_media_uses_status_and_keyboard_when_controls_are_hidden(tmp_path):
    from plugins.google_meet.meet_bot import _BotState, _ensure_local_media_before_join

    class Missing:
        first = property(lambda self: self)

        def count(self):
            return 0

        def is_visible(self):
            return False

    class Page:
        def __init__(self):
            self.media = {"localMicrophoneOn": True, "localCameraOn": True}
            self.keyboard = self

        def locator(self, *_args, **_kwargs):
            return Missing()

        def get_by_role(self, *_args, **_kwargs):
            return Missing()

        def evaluate(self, _script):
            return dict(self.media)

        def press(self, key):
            device = {"Control+D": "localMicrophoneOn", "Control+E": "localCameraOn"}[
                key
            ]
            self.media[device] = not self.media[device]

        def wait_for_timeout(self, _timeout):
            pass

    page = Page()
    state = _BotState(
        tmp_path / "meeting", "abc-defg-hij", "https://meet.google.com/abc-defg-hij"
    )
    assert _ensure_local_media_before_join(
        page, state, realtime_enabled=False, realtime_route_ready=False
    )
    assert page.media == {"localMicrophoneOn": False, "localCameraOn": False}
    assert state.local_microphone_on is False
    assert state.local_camera_on is False


def test_join_polls_for_asynchronously_rendered_button(browser_page, tmp_path):
    from plugins.google_meet.meet_bot import _BotConfig, _BotState, _join

    browser_page.set_content("""
        <button id="join" hidden onclick="window.joined=true">Join now</button>
        <script>
        window.joined = false;
        setTimeout(() => document.getElementById('join').hidden = false, 600);
        </script>
    """)
    state = _BotState(
        tmp_path / "meeting", "abc-defg-hij", "https://meet.google.com/abc-defg-hij"
    )
    assert _join(browser_page, _BotConfig(guest_name="Bot"), state, timeout=5)
    assert browser_page.evaluate("window.joined") is True


def test_transcribe_admission_keeps_microphone_and_camera_off(
    browser_page, tmp_path, monkeypatch
):
    from plugins.google_meet import meet_bot

    browser_page.set_content("""
        <button aria-label="Leave call">Leave</button>
        <button id="mic" aria-label="Turn on microphone" onclick="this.setAttribute('aria-label', 'Turn off microphone')">Mic</button>
        <button id="camera" aria-label="Turn on camera" onclick="this.setAttribute('aria-label', 'Turn off camera')">Camera</button>
        <button aria-label="Turn off captions">Captions</button>
    """)
    state = meet_bot._BotState(
        tmp_path / "meeting", "abc-defg-hij", "https://meet.google.com/abc-defg-hij"
    )
    state.set(join_attempted_at=100.0)
    stop = {"stop": False}
    monkeypatch.setattr(meet_bot.time, "sleep", lambda _seconds: stop.update(stop=True))
    meet_bot._drain_loop(
        browser_page,
        meet_bot._BotConfig(
            guest_name="Bot", realtime=False, duration_s=None, lobby_timeout=300
        ),
        state,
        {"session": None},
        stop,
    )
    assert state.in_call is True
    assert state.local_microphone_on is False
    assert state.local_camera_on is False
    assert (
        browser_page.locator("#mic").get_attribute("aria-label") == "Turn on microphone"
    )
    assert (
        browser_page.locator("#camera").get_attribute("aria-label") == "Turn on camera"
    )
