from pathlib import Path


def test_security_md_names_desktop_loopback_session_token_bootstrap() -> None:
    text = Path("SECURITY.md").read_text(encoding="utf-8")
    start = text.find("### 2.6")
    assert start != -1
    section = text[start:]
    nxt = section.find("\n## 3.", 1)
    if nxt != -1:
        section = section[:nxt]
    lowered = section.lower()
    assert "desktop" in lowered or "hermes serve" in lowered or "dashboard" in lowered
    # Pin the documented posture, not just loose keyword bags across the file.
    assert "__HERMES_SESSION_TOKEN__" in section
    assert "session token" in lowered
    assert "§3.2" in section
    assert "§3.1" in section
    assert any(s in lowered for s in ("html", "bootstrap", "unauthenticated"))
    assert any(s in lowered for s in ("loopback", "local", "os-level", "user account", "same user"))
    # Operator mitigation already named in §2.6 rule 5.
    assert "loopback" in lowered
    assert "0.0.0.0" in section or "non-loopback" in lowered
