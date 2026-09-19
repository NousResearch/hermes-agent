"""Contract test: install.sh propagates --hermes-home end to end (#89231).

``--hermes-home PATH`` used to be visible only to the installer's own
writes: ``HERMES_HOME`` was assigned but never exported, so the
``skills_sync.py`` child resolved the platform default, and the generated
launchers (hermes / hermes-agent / hermes-acp) carried no home at all —
every runtime invocation silently fell back to ``~/.hermes``. The result
was a split-brain install: config/.env/profiles under the requested home,
launcher and bundled skills under the platform default, with success
messages hardcoding ``~/.hermes`` so the logs looked correct either way.
"""
from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_SH = REPO_ROOT / "scripts" / "install.sh"


def test_hermes_home_is_exported_after_assignment() -> None:
    text = INSTALL_SH.read_text()
    assign = re.search(r'^HERMES_HOME="\$\{HERMES_HOME:-\$HOME/\.hermes\}"', text, re.M)
    assert assign, "install.sh must keep the HERMES_HOME default assignment"
    export = re.search(r"^export HERMES_HOME$", text, re.M)
    assert export, (
        "install.sh must export HERMES_HOME so child processes "
        "(skills_sync.py, setup wizard) resolve the same home (#89231)"
    )
    assert export.start() > assign.start(), (
        "the export must follow the default assignment (--hermes-home "
        "reassignments later in arg parsing keep the export attribute)"
    )


def test_every_launcher_template_bakes_in_hermes_home() -> None:
    """All six launcher heredocs (hermes/hermes-agent/hermes-acp, venv and
    no-venv variants) must export the install-time home as a default that
    the caller's environment can still override."""
    text = INSTALL_SH.read_text()
    templates = re.findall(
        r'cat > "\$command_link_dir/hermes[^"]*" <<EOF\n(.*?)EOF',
        text,
        re.S,
    )
    assert len(templates) >= 6, (
        f"expected >=6 launcher templates, found {len(templates)} — update "
        "this test alongside any new launcher"
    )
    for body in templates:
        assert 'export HERMES_HOME="\\${HERMES_HOME:-' in body, (
            "launcher template must bake in the install-time home with "
            "runtime override: "
            + body.splitlines()[0]
        )


def test_skills_sync_messages_report_the_real_home() -> None:
    """The sync success/info lines must print the requested home, not a
    hardcoded ~/.hermes that looks correct for every install."""
    text = INSTALL_SH.read_text()
    for msg in (
        r'log_info "Syncing bundled skills to \$HERMES_HOME/skills/ \.\.\."',
        r'log_success "Skills synced to \$HERMES_HOME/skills/"',
        r'log_success "Skills copied to \$HERMES_HOME/skills/"',
    ):
        assert re.search(msg, text), f"missing real-path message: {msg}"
    assert 'Skills synced to ~/.hermes/skills/' not in text, (
        "hardcoded ~/.hermes success message masks a split-brain install (#89231)"
    )
