"""Regression tests for packaging metadata in pyproject.toml."""

from pathlib import Path
import tomllib


def _load_optional_dependencies():
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    with pyproject_path.open("rb") as handle:
        project = tomllib.load(handle)["project"]
    return project["optional-dependencies"]


def test_matrix_extra_linux_only_in_all():
    """mautrix[encryption] depends on python-olm which is upstream-broken on
    modern macOS (archived libolm, C++ errors with Clang 21+) and has no
    cp313 wheels.  The [matrix] extra is excluded from [all] and covered
    by LAZY_DEPS so it lazy-installs at first use on Linux only."""
    optional_dependencies = _load_optional_dependencies()

    assert "matrix" in optional_dependencies
    # Must NOT be in [all] at all — python-olm has no macOS/cp313 wheels.
    # Lazy-install on first use is the correct path.
    offending = [
        dep for dep in optional_dependencies["all"]
        if "matrix" in dep
    ]
    assert not offending, (
        f"[matrix] should not be in [all] — covered by LAZY_DEPS. Found: {offending}"
    )


def test_lazy_installable_extras_excluded_from_all():
    """Policy (2026-05-12): every extra that has a `LAZY_DEPS` entry
    in `tools/lazy_deps.py` must be excluded from [all].

    The lazy-install system exists so one quarantined PyPI release
    (e.g. mistralai 2.4.6) can't break every fresh install. Putting a
    backend in BOTH [all] and LAZY_DEPS defeats that — fresh installs
    eager-install it and inherit whatever's broken upstream.

    If you're tempted to add an opt-in backend to [all] for "convenience,"
    add it to `LAZY_DEPS` instead so it installs at first use.
    """
    optional_dependencies = _load_optional_dependencies()

    # Hard-coded mirror of the extras that are in LAZY_DEPS as of
    # 2026-05-12. This list intentionally duplicates rather than
    # imports tools/lazy_deps.py so the test stays a contract — if
    # someone adds a new lazy-install backend, they have to update
    # this list AND verify [all] doesn't contain it.
    lazy_covered_extras = {
        "anthropic", "bedrock",
        "exa", "firecrawl", "parallel-web",
        "fal",
        "edge-tts", "tts-premium",
        "voice",  # faster-whisper / sounddevice / numpy
        "modal", "daytona",
        "messaging", "slack", "matrix", "dingtalk", "feishu",
        "honcho", "hindsight",
        "mistral",  # mistralai — Voxtral STT/TTS, lazy-installed (stt.mistral / tts.mistral)
    }
    all_extra_specs = optional_dependencies["all"]
    for extra in lazy_covered_extras:
        offending = [
            spec for spec in all_extra_specs
            if f"hermes-agent[{extra}]" in spec
        ]
        assert not offending, (
            f"[{extra}] is in [all] but also in LAZY_DEPS. "
            f"Remove it from [all] in pyproject.toml — it lazy-installs "
            f"at first use. Found in [all]: {offending}"
        )


def test_messaging_extra_includes_qrcode_for_weixin_setup():
    optional_dependencies = _load_optional_dependencies()

    messaging_extra = optional_dependencies["messaging"]
    assert any(dep.startswith("qrcode") for dep in messaging_extra)


def test_dingtalk_extra_includes_qrcode_for_qr_auth():
    """DingTalk's QR-code device-flow auth (hermes_cli/dingtalk_auth.py)
    needs the qrcode package."""
    optional_dependencies = _load_optional_dependencies()

    dingtalk_extra = optional_dependencies["dingtalk"]
    assert any(dep.startswith("qrcode") for dep in dingtalk_extra)


def test_feishu_extra_includes_qrcode_for_qr_login():
    """Feishu's QR login flow (gateway/platforms/feishu.py) needs the
    qrcode package."""
    optional_dependencies = _load_optional_dependencies()

    feishu_extra = optional_dependencies["feishu"]
    assert any(dep.startswith("qrcode") for dep in feishu_extra)
